"""Frozen-model four-slot controls and exact, padding-free fusion statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import SimpleITK as sitk
from PIL import Image
from freqssl.engine import seed_everything,state_hash,write_json
from freqssl.evaluation import prepare_validation,anatomical_diagnostics
from freqssl.inference import load_single_network
from freqssl.case_context import reference_index
from freqssl.context_diagnostics import bootstrap_cases
from spider_volume import reconstruct,volume_metrics,aggregate_cases

REPEATS=(1701,1702,1703)

def controls(names,positions,seed):
    """Image-inventory-only donor choices; same maps across model seeds."""
    rng=np.random.default_rng(seed);cases=sorted(names)
    if len(cases)<2: raise ValueError('Cross-case controls require at least two cases')
    order=list(rng.permutation(cases));shift=int(rng.integers(1,len(cases)))
    donor={c:order[(i+shift)%len(order)] for i,c in enumerate(order)}
    # One random reference per case/repetition, retained across all target slices.
    random_indices={c:[int(rng.choice([j for j in range(len(names[c])) if j!=reference_index(len(names[c]))]))
                      if len(names[c])>1 else 0]*len(names[c]) for c in cases}
    pairs=[(c,i) for c in cases for i in rng.permutation(len(names[c]))]
    largest=max(map(len,names.values()))
    if 2*largest>len(pairs): raise ValueError('Cannot permute positions without same-case donors')
    offset=int(rng.integers(largest,len(pairs)-largest+1))
    permuted=pairs[offset:]+pairs[:offset]
    mapping={c:{} for c in cases}
    for (c,i),(d,j) in zip(pairs,permuted):
        assert c!=d
        mapping[c][int(i)]=dict(donor_case=str(d),donor_index=int(j),position=float(positions[d][j]))
    return dict(reference_donor=donor,random_indices=random_indices,position_mapping=mapping)

def describe(values):
    values=np.asarray(values)
    return dict(mean=float(values.mean()),std=float(values.std()),
                quantiles=dict(zip(('p01','p05','p50','p95','p99'),map(float,np.quantile(values,[.01,.05,.5,.95,.99])))))

class FusionProbe:
    def __init__(self,model):
        self.model=model;self.handles=[];self.valid=0;self.values=[[] for _ in range(3)]
        self.residual=[[] for _ in range(3)]
        for i,layer in enumerate(model.context_fusion):
            self.handles.append(layer.register_forward_hook(self.hook(i)))

    def hook(self,index):
        def capture(layer,args,output):
            if not self.valid: return
            features=args[0][:self.valid]
            logits=layer.weights(features.flatten(1,2))
            if getattr(layer,'position_bias',None) is not None:
                logits=logits+layer.position_bias(F.interpolate(args[1][:self.valid],size=features.shape[-2:],mode='nearest'))
            w=logits.softmax(1)
            self.values[index].append(w.detach().cpu().numpy())
            if features.shape[1]==4:
                ref=layer.gamma*w[:,3,None]*(features[:,3]-features[:,1])
                ratio=ref.square().mean((1,2,3)).sqrt()/features[:,1].square().mean((1,2,3)).sqrt().clamp_min(1e-12)
                self.residual[index].extend(ratio.cpu().tolist())
        return capture

    def finish_case(self):
        stats=[]
        for index,layer in enumerate(self.model.context_fusion):
            w=np.concatenate(self.values[index]);gamma=float(layer.gamma)
            effective=gamma*w;effective[:,1]+=1-gamma
            stats.append(dict(stride=4*2**index,gamma=gamma,real_slices=len(w),
                weight=[describe(w[:,j]) for j in range(w.shape[1])],
                effective_coefficient=[describe(effective[:,j]) for j in range(w.shape[1])],
                reference_residual_to_center_rms=describe(self.residual[index]) if self.residual[index] else None))
        self.values=[[] for _ in range(3)];self.residual=[[] for _ in range(3)];return stats

    def close(self):
        for h in self.handles: h.remove()


@torch.inference_mode()
def probe(checkpoint,output,device='cpu',interventions=True,expected_cases=21):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='running'))
    model,cfg,loading=load_single_network(checkpoint,device);seed_everything(cfg);before=state_hash(model)
    dataset,records=prepare_validation(cfg)
    if len(records)!=expected_cases: raise ValueError('Incomplete validation inventory')
    groups={r['case']:r['slice_names'] for r in records};lookup={n:i for i,n in enumerate(dataset.names)}
    images={c:torch.stack([dataset.read_image(lookup[n]) for n in names]) for c,names in groups.items()}
    positions={c:[dataset.positions[n] for n in names] if cfg.context_position=='relative' else [0.]*len(names) for c,names in groups.items()}
    choices={s:controls(groups,positions,s) for s in REPEATS} if interventions else {}
    conditions=['full']
    if interventions and cfg.case_reference=='middle':
        conditions+=['reference_zero']+[f'{kind}_{s}' for kind in ('other_case','same_case_random') for s in REPEATS]
    if interventions and cfg.context_position=='relative': conditions += [f'cross_case_position_{s}' for s in REPEATS]
    write_json(output/'controls.json',choices)
    metrics={c:[] for c in conditions};fusion={};model_backbone=model.backbone
    collector=FusionProbe(model_backbone)
    try:
        for record in records:
            case=record['case'];names=groups[case];n=len(names);im=images[case];r=reference_index(n)
            shapes=[]
            for name in names:
                with Image.open(dataset.root/'image'/name) as img: shapes.append(img.size)
            target=sitk.GetArrayFromImage(sitk.ReadImage(record['reference']))
            for condition in conditions:
                predictions=[];changed=[]
                for start in range(0,n,8):
                    end=min(n,start+8);batch=[];ps=[]
                    for i in range(start,end):
                        value=im[[max(0,i-1),i,min(n-1,i+1)],0]
                        if cfg.case_reference=='middle':
                            ref=im[r]
                            if condition=='reference_zero': ref=torch.zeros_like(ref)
                            elif condition.startswith('other_case_'):
                                d=choices[int(condition.rsplit('_',1)[1])]['reference_donor'][case]
                                ref=images[d][reference_index(len(images[d]))]
                            elif condition.startswith('same_case_random_'):
                                j=choices[int(condition.rsplit('_',1)[1])]['random_indices'][case][i];ref=im[j]
                            value=torch.cat((value,ref))
                        p=positions[case][i]
                        if condition.startswith('cross_case_position_'):
                            p=choices[int(condition.rsplit('_',1)[1])]['position_mapping'][case][i]['position']
                        ps.append(p);changed.append(p!=positions[case][i]);batch.append(value)
                    x=torch.stack(batch).to(device)
                    collector.valid=end-start if condition=='full' else 0
                    options={'position':x.new_tensor(ps).reshape(-1,1,1,1)} if cfg.context_position=='relative' else {}
                    logits=model_backbone(x,**options)
                    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite logits')
                    pred=logits.argmax(1).float()[:,None]
                    for k,v in enumerate(pred):
                        plane=F.interpolate(v[None],size=shapes[start+k],mode='nearest')[0,0].cpu().numpy().astype(np.uint8)
                        predictions.append(plane.T.copy())
                if condition=='full': fusion[case]=collector.finish_case()
                volume=reconstruct(predictions,record['slice_axis'],record['undo_vertical_flip'],record['array_shape_zyx'])
                metrics[condition].append(dict(case=case,classes=volume_metrics(volume,target,(1,1,1),surfaces=False),
                    anatomy=anatomical_diagnostics(volume,target),position_value_changed_slices=sum(changed),slice_count=n))
        assert before==state_hash(model)
    finally: collector.close()
    aggregates={c:aggregate_cases(rows) for c,rows in metrics.items()}
    base={c['case']:c for c in aggregates['full']['per_case']};paired={}
    for condition in conditions[1:]:
        paired[condition]={}
        for metric in ('dice','dice_union_present'):
            delta={c['case']:100*(c[metric]-base[c['case']][metric]) for c in aggregates[condition]['per_case']}
            paired[condition][metric]=dict(bootstrap_cases(delta.values()),per_case=delta)
    averages={}
    for kind in ('other_case','same_case_random','cross_case_position'):
        keys=[c for c in conditions if c.startswith(kind+'_')]
        if not keys: continue
        averages[kind]={}
        for metric in ('dice','dice_union_present'):
            delta={case:float(np.mean([paired[c][metric]['per_case'][case] for c in keys])) for case in base}
            averages[kind][metric]=dict(bootstrap_cases(delta.values()),per_case=delta)
    weight_summary=[]
    for level in range(3):
        rows=[v[level] for v in fusion.values()];channels=len(rows[0]['weight'])
        weight_summary.append(dict(stride=4*2**level,gamma=rows[0]['gamma'],
            mean_w_case_equal=[float(np.mean([r['weight'][j]['mean'] for r in rows])) for j in range(channels)],
            mean_effective_case_equal=[float(np.mean([r['effective_coefficient'][j]['mean'] for r in rows])) for j in range(channels)],
            case_mean_weight_distribution=[describe([r['weight'][j]['mean'] for r in rows]) for j in range(channels)],
            mean_reference_residual_ratio=float(np.mean([r['reference_residual_to_center_rms']['mean'] for r in rows])) if channels==4 else None))
    result=dict(status='completed',checkpoint=str(checkpoint),loading=loading,cases=expected_cases,
        conditions=conditions,metrics=aggregates,paired_delta_pp=paired,random_control_mean_delta_pp=averages,
        fusion=weight_summary,per_case_fusion=fusion,model_state_unchanged=True,training_updates=0,test_read=False,
        interpretation='Input dependence only. w is input-dependent; gamma is one unconstrained scalar per scale, shared by all slots. Zero means zero standardized input, not zero encoded features.',
        position_shuffle='Global value-preserving permutation; every donor is another case. Repetitions are averaged within case before bootstrap.')
    write_json(output/'case_metrics.json',metrics);write_json(output/'summary.json',result)
    write_json(output/'status.json',dict(status='completed'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--weights-only',action='store_true');a=p.parse_args()
    probe(a.checkpoint,a.output_dir,a.device,not a.weights_only)
