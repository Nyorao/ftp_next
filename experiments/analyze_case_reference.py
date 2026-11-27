"""Post-training, validation-only middle-reference and position interventions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from torch.nn import functional as F
from freqssl.case_context import verify_dataset_geometry,reference_index,relative_position
from freqssl.context_cache import iter_cached_logits
from freqssl.context_diagnostics import region_masks,bootstrap_cases
from freqssl.data import _read_transposed
from freqssl.engine import seed_everything,state_hash,write_json
from freqssl.evaluation import prepare_validation,anatomical_diagnostics
from freqssl.inference import load_single_network
from spider_volume import volume_metrics,reconstruct,aggregate_cases
from experiments.profile_context_efficiency import ConvCounter


def shuffled_positions(values,case):
    rng=np.random.default_rng(20261112+int(hashlib.sha256(case.encode()).hexdigest()[:8],16))
    order=rng.permutation(len(values))
    if len(values)>1 and np.array_equal(order,np.arange(len(values))):
        order=np.roll(order,1)
    return np.asarray(values)[order].tolist()


def confusion_metrics(confusion):
    """Overlap on a native case subvolume; no averaging over individual slices."""
    cm=np.asarray(confusion,dtype=np.int64)
    gt,pred,hit=cm.sum(1),cm.sum(0),cm.diagonal()
    denominator=gt+pred
    dice=np.divide(200.*hit,denominator,out=np.zeros(20),where=denominator>0)
    present=np.arange(20)>0
    mean=lambda mask: float(dice[mask].mean()) if mask.any() else None
    result=dict(dice_percent=mean(present&(gt>0)),dice_union_present_percent=mean(present&(denominator>0)))
    for name,ids in (('vertebra',list(range(1,10))),('canal',[10]),('disc',list(range(11,20)))):
        total=gt[ids].sum()+pred[ids].sum()
        result[name+'_dice_percent']=float(200*cm[np.ix_(ids,ids)].sum()/total) if gt[ids].sum() else None
    wrong=sum(cm[np.ix_(ids,ids)].sum()-hit[ids].sum() for ids in (list(range(1,10)),list(range(11,20))))
    numbered=gt[1:10].sum()+gt[11:20].sum()
    result['same_tissue_wrong_level_percent']=float(100*wrong/numbered) if numbered else None
    result['absent_group_class_fp_percent']=float(100*pred[(gt==0)&present].sum()/max(1,gt.sum()))
    return result


def summarize_regions(cases):
    by_group=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for case in cases:
        buckets=defaultdict(list)
        for row in case['slices']:
            groups=['all',f"position_{min(4,int((row['position']+1)/2*5))}",
                'reference_duplicate' if row['reference_duplicate'] else 'reference_distinct',
                'endpoint' if row['endpoint'] else 'interior_slice',
                'distance_0_5mm' if row['distance_mm']<=5 else
                ('distance_5_15mm' if row['distance_mm']<=15 else 'distance_over15mm')]
            for group in groups:
                buckets[group].append(row)
        for group,rows in buckets.items():
            for condition in rows[0]['conditions']:
                cm=sum(np.asarray(r['native_confusion'][condition]) for r in rows)
                for metric,value in confusion_metrics(cm).items():
                    by_group[group][condition]['native_'+metric][case['case']]=value
                for region in rows[0]['conditions'][condition]:
                    values=[r['conditions'][condition][region] for r in rows]
                    n=sum(v['pixels'] for v in values)
                    if not n:
                        continue
                    for metric in ('errors','absent_slice_class_pixels','absent_volume_class_pixels'):
                        by_group[group][condition][region+'_'+metric+'_percent'][case['case']]=100*sum(v[metric] for v in values)/n
    return {g:{c:{k:dict(bootstrap_cases(v.values()),per_case=v) for k,v in metrics.items()}
               for c,metrics in conditions.items()} for g,conditions in by_group.items()}


def paired_analysis(candidate,baseline,candidate_condition='full',baseline_condition='full'):
    """Paired case differences, including native subvolumes and boundary FP."""
    a={c['case'] for c in candidate['metrics'][candidate_condition]['per_case']}
    b={c['case'] for c in baseline['metrics'][baseline_condition]['per_case']}
    if a!=b or len(a)!=candidate['cases'] or candidate['cases']!=baseline['cases']:
        raise ValueError('Paired analysis requires identical complete case inventories')
    result={}
    if candidate['regions'].keys()!=baseline['regions'].keys():
        raise ValueError('Paired analysis groups differ')
    for group,conditions in candidate['regions'].items():
        result[group]={}
        for metric,summary in conditions[candidate_condition].items():
            other=baseline['regions'][group][baseline_condition][metric]['per_case']
            if summary['per_case'].keys()!=other.keys():
                raise ValueError('Paired region case inventories differ')
            delta={case:value-other[case] for case,value in summary['per_case'].items()
                   if value is not None and other[case] is not None}
            result[group][metric]=dict(bootstrap_cases(delta.values()),per_case=delta)
    return result


def combine_seed_pairs(by_seed):
    from experiments.summarize_slice_context import clustered_summary
    first=next(iter(by_seed.values()))
    return {group:{metric:clustered_summary({s:v[group][metric]['per_case'] for s,v in by_seed.items()})
                   for metric in metrics} for group,metrics in first.items()}


@torch.inference_mode()
def analyze(checkpoint,output,device='cpu',expected_cases=21,geometry_manifest=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='running'))
    try:
        model,cfg,loading=load_single_network(checkpoint,device)
        seed_everything(cfg)
        before=state_hash(model)
        dataset,records=prepare_validation(cfg,'valid')
        if len(records)!=expected_cases:
            raise ValueError('Analysis requires the complete declared validation inventory')
        if dataset.geometry_receipt is None:
            positions,geometry=verify_dataset_geometry(dataset,geometry_manifest or cfg.context_geometry_manifest)
        else:
            positions,geometry=dataset.positions,dataset.geometry_receipt
        lookup={n:i for i,n in enumerate(dataset.names)}
        conditions=['full']
        if cfg.case_reference=='middle':
            conditions+=['reference_target','reference_quarter']
        if cfg.context_position=='relative':
            conditions+=['position_shuffled']
        cases=[]
        (output/'cases').mkdir()
        for record in records:
            case,names=record['case'],record['slice_names'];n=len(names)
            images=torch.stack([dataset.read_image(lookup[name]) for name in names])
            target=sitk.GetArrayFromImage(sitk.ReadImage(record['reference'])).astype(np.uint8)
            present=np.unique(target)
            targets=[];native_targets=[];shapes=[]
            for name in names:
                y=_read_transposed(dataset.root/'label'/name,torch.long)
                native_targets.append(y.numpy().astype(np.uint8))
                targets.append(F.interpolate(y[None,None].float(),size=cfg.image_size,mode='nearest')[0,0].numpy().astype(np.uint8))
                with Image.open(dataset.root/'image'/name) as img:
                    shapes.append(img.size)
            coords=[positions[name] for name in names]
            shuffled=shuffled_positions(coords,case)
            r=reference_index(n)
            rows=[dict(index=i,position=coords[i],distance_mm=abs(i-r)*geometry['cases'][case]['spacing_mm'],
                reference_duplicate=r in (max(0,i-1),i,min(n-1,i+1)),endpoint=i in (0,n-1),
                conditions={},native_confusion={}) for i in range(n)]
            metrics={};cost={}
            for condition in conditions:
                policy={'reference_target':'target','reference_quarter':'quarter'}.get(condition,'middle')
                p=shuffled if condition=='position_shuffled' else coords
                options={'read_position':lambda i:p[i]} if cfg.context_position=='relative' else {}
                stats={};counter=ConvCounter(model.backbone)
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                start=time.perf_counter()
                try:
                    logits=torch.cat([x for _,x in iter_cached_logits(model.backbone,n,lambda i:images[i],device,
                        cfg.eval_batch_size,reference_policy=policy,stats=stats,**options)])
                    prediction=logits.argmax(1).cpu().numpy().astype(np.uint8)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite diagnostic prediction')
                    del logits
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize(device)
                    elapsed=time.perf_counter()-start
                finally:
                    counter.close()
                cost[condition]=dict(seconds=elapsed,conv_macs=dict(counter.macs),cache=stats,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)
                native=[]
                for i,pred in enumerate(prediction):
                    gt=targets[i];masks=region_masks(gt)
                    masks.update(boundary_foreground=masks['boundary']&(gt>0),boundary_background=masks['boundary']&(gt==0))
                    scores={}
                    for region,mask in masks.items():
                        scores[region]=dict(pixels=int(mask.sum()),errors=int(((pred!=gt)&mask).sum()),
                            absent_slice_class_pixels=int(((pred>0)&~np.isin(pred,np.unique(gt))&mask).sum()),
                            absent_volume_class_pixels=int(((pred>0)&~np.isin(pred,present)&mask).sum()))
                    rows[i]['conditions'][condition]=scores
                    value=F.interpolate(torch.from_numpy(pred.astype(np.float32))[None,None],size=shapes[i],mode='nearest')[0,0]
                    restored=value.numpy().astype(np.uint8)
                    cm=np.bincount((20*native_targets[i].astype(np.int64)+restored).ravel(),minlength=400).reshape(20,20)
                    rows[i]['native_confusion'][condition]=cm.tolist()
                    native.append(restored.T.copy())
                volume=reconstruct(native,record['slice_axis'],record['undo_vertical_flip'],record['array_shape_zyx'])
                metrics[condition]=dict(case=case,classes=volume_metrics(volume,target,(1,1,1),surfaces=False),
                    anatomy=anatomical_diagnostics(volume,target),
                    absent_volume_class_voxels=int(((volume>0)&~np.isin(volume,present)).sum()))
            row=dict(case=case,slices=rows,metrics=metrics,cost=cost)
            write_json(output/'cases'/f'{case}.json',row);cases.append(row)
        assert state_hash(model)==before
        aggregates={c:aggregate_cases([r['metrics'][c] for r in cases]) for c in conditions}
        paired={}
        for condition in conditions[1:]:
            paired[condition]={}
            for metric in ('dice','dice_union_present'):
                full={r['case']:r[metric] for r in aggregates['full']['per_case']}
                delta={r['case']:100*(r[metric]-full[r['case']]) for r in aggregates[condition]['per_case']}
                paired[condition][metric]=dict(bootstrap_cases(delta.values()),per_case=delta)
        result=dict(status='completed',cases=len(cases),seed=cfg.seed,conditions=conditions,loading=loading,
            geometry=geometry,metrics=aggregates,paired_delta_pp=paired,regions=summarize_regions(cases),
            model_state_unchanged=True,training_updates=0,test_read=False,
            full_volume_seconds=sum(c['cost']['full']['seconds'] for c in cases),
            timing_scope='preloaded normalized inputs; model/cache/logit collection/argmax; excludes image IO and scoring; not isolated latency',
            region_scope='native_* uses native-grid case subvolume confusion; other region metrics use model-grid masks, with a 2-pixel boundary band',
            interpretation='Frozen-model input dependence, not anatomical causality; descriptive development analysis without multiplicity correction')
        result['paired_regions_delta_pp']={c:paired_analysis(result,result,c,'full') for c in conditions[1:]}
        write_json(output/'summary.json',result);write_json(output/'status.json',dict(status='completed'))
        return result
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--geometry-manifest',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    analyze(args.checkpoint,args.output_dir,args.device,geometry_manifest=str(args.geometry_manifest))
