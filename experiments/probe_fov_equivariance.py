"""Frozen-teacher canvas-position diagnostic; never trains or reads test data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from freqssl.anatomy import FAMILY_IDS
from freqssl.data import SpiderSliceDataset, _slice_names, _case_id
from freqssl.engine import seed_everything, state_hash, write_json
from freqssl.field_of_view import canvas_pair
from freqssl.inference import load_single_network


NETWORKS=('teacher1','teacher2','mean')
PROTOCOL=dict(train_labeled_samples=256,batch_size=8,content_size=160,
              ap_start=32,si_starts=[16,48],image_size=224,seed=42,
              geometric_interventions=2,gt_used_for_scoring_only=True,
              native_volume_metric=False,threshold_search=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(2**20),b''):h.update(data)
    return h.hexdigest()


def prepare(args):
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    checkpoint=args.checkpoint.resolve(strict=True)
    sources=list((ROOT/'freqssl').glob('*.py'))+[ROOT/'spider_volume.py',Path(__file__).resolve(),
                ROOT/'docs/fov_coordinate_20261029.md',ROOT/'tests/test_field_of_view.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    parity={name:sha(checkpoint.parent.parent/'code'/name)==hashes[name]
            for name in ('freqssl/models.py','freqssl/ordinal_model.py','freqssl/data.py')}
    if not all(parity.values()):raise ValueError('Teacher forward or original input pipeline changed')
    out.mkdir(parents=True,exist_ok=True)
    for source in sources:
        dest=out/'code'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    write_json(out/'manifest.json',dict(protocol=PROTOCOL,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),source_hashes=hashes,teacher_code_parity=parity,
        device=args.device,model_updates=0,test_read=False,unlabeled_annotations_read=False,
        diagnostic_after_valid_error_analysis=True))
    write_json(out/'status.json',dict(status='prepared'))
    command=[sys.executable,str(out/'code/experiments/probe_fov_equivariance.py'),
             '--output-dir',str(out),'--device',args.device]
    write_json(out/'command.json',command)
    print(shlex.join(command),flush=True)


def new_case():
    return dict(slices=0,foreground=0,changed=0,changed_tissue=0,changed_level=0,
        correct_to_wrong=0,wrong_to_correct=0,both_confident_changed=0,
        js_sum=0.,pad_pixels=0,pad_foreground=[0,0],
        confusion=[np.zeros((20,20),dtype=np.int64) for _ in range(2)],
        level_shift=np.zeros(17,dtype=np.int64))


def case_dice(confusion):
    target=confusion.sum(1);prediction=confusion.sum(0)
    selected=(target>0);selected[0]=False
    if not selected.any():return None
    dice=2*np.diag(confusion)[selected]/(target+prediction)[selected]
    return float(dice.mean())


@torch.inference_mode()
def evaluate(teachers,dataset,device,batch_size):
    cases={name:{} for name in NETWORKS}
    family=torch.tensor(FAMILY_IDS,device=device)
    for start in range(0,len(dataset),batch_size):
        samples=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
        x=torch.stack([s['image'] for s in samples]).to(device)
        gt=torch.cat([F.interpolate(s['label'][None,None].float(),(160,160),mode='nearest')
                      for s in samples])[:,0].long().to(device)
        views,places=canvas_pair(x)
        # Content transport is exact: neither prediction needs inverse resampling.
        if not torch.equal(places[0].extract(views[0]),places[1].extract(views[1])):
            raise AssertionError('The two interventions changed image content')
        whole=[]
        for view in views:
            a,b=[teacher(view) for teacher in teachers]
            whole.append(dict(teacher1=a,teacher2=b,mean=(a+b)/2))
        for name in NETWORKS:
            p,q=[place.extract(predictions[name]) for predictions,place in zip(whole,places)]
            mean=(p+q)/2
            js=.5*(p*(p.clamp_min(1e-30).log()-mean.clamp_min(1e-30).log())+
                    q*(q.clamp_min(1e-30).log()-mean.clamp_min(1e-30).log())).sum(1)
            cp,yp=p.max(1);cq,yq=q.max(1)
            fp,fq=family[yp],family[yq]
            padding=[]
            for full,place in zip(whole,places):
                outside=torch.ones_like(full[name][:,0],dtype=torch.bool)
                outside[:,place.ap_start:place.ap_start+place.size,place.si_start:place.si_start+place.size]=False
                padding.append(((full[name].argmax(1)>0)&outside).sum((1,2)))
            for i,sample in enumerate(samples):
                key=_case_id(sample['file_name']);row=cases[name].setdefault(key,new_case())
                mask=gt[i]>0;changed=(yp[i]!=yq[i])&mask
                same_numbered=(fp[i]==fq[i])&((fp[i]==1)|(fp[i]==3))&mask
                count=dict(slices=1,foreground=int(mask.sum()),changed=int(changed.sum()),
                    changed_tissue=int((changed&(fp[i]!=fq[i])).sum()),
                    changed_level=int((changed&same_numbered).sum()),
                    correct_to_wrong=int(((yp[i]==gt[i])&(yq[i]!=gt[i])&mask).sum()),
                    wrong_to_correct=int(((yp[i]!=gt[i])&(yq[i]==gt[i])&mask).sum()),
                    both_confident_changed=int((changed&(cp[i]>=.8)&(cq[i]>=.8)).sum()),
                    js_sum=float(js[i][mask].double().sum()),pad_pixels=224*224-160*160)
                for k,v in count.items():row[k]+=v
                for index,label in enumerate((yp[i],yq[i])):
                    row['confusion'][index]+=torch.bincount((gt[i]*20+label).flatten(),minlength=400).reshape(20,20).cpu().numpy()
                    row['pad_foreground'][index]+=int(padding[index][i])
                offsets=(yq[i]-yp[i])[same_numbered]+8
                row['level_shift']+=torch.bincount(offsets,minlength=17).cpu().numpy()
        if start==0:print('FIRST_BATCH',dataset.split,len(samples),flush=True)
    results={}
    for name,rows in cases.items():
        total={key:sum(row[key] for row in rows.values()) for key in
               ('slices','foreground','changed','changed_tissue','changed_level','correct_to_wrong',
                'wrong_to_correct','both_confident_changed','js_sum','pad_pixels')}
        total['pad_foreground']=[sum(row['pad_foreground'][i] for row in rows.values()) for i in range(2)]
        total['level_shift']={str(i-8):int(sum(row['level_shift'][i] for row in rows.values())) for i in range(17)}
        per_case={}
        for key,row in rows.items():
            row['grid_dice']=[case_dice(matrix) for matrix in row['confusion']]
            row['confusion']=[matrix.tolist() for matrix in row['confusion']]
            row['level_shift']={str(i-8):int(v) for i,v in enumerate(row['level_shift'])}
            per_case[key]=row
        total['grid_case_macro_dice']=[float(np.mean([row['grid_dice'][i] for row in rows.values()
                                         if row['grid_dice'][i] is not None])) for i in range(2)]
        total['cases']=len(rows)
        for key in ('changed','changed_tissue','changed_level','correct_to_wrong','wrong_to_correct'):
            total[key+'_foreground_fraction']=total[key]/max(1,total['foreground'])
        total['js_foreground_mean']=total['js_sum']/max(1,total['foreground'])
        results[name]=dict(total=total,per_case=per_case)
    return results


def run(args):
    out=args.output_dir.resolve();manifest=json.loads((out/'manifest.json').read_text())
    if ROOT!=out/'code' or args.device!=manifest['device']:raise ValueError('Use the prepared command')
    if json.loads((out/'status.json').read_text())['status']!='prepared':raise ValueError('Run already started')
    for name,digest in manifest['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    if sha(manifest['checkpoint'])!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    write_json(out/'status.json',dict(status='running'));started=time.monotonic()
    try:
        device=torch.device(args.device)
        a,cfg,_=load_single_network(manifest['checkpoint'],device,'teacher1')
        b,_,_=load_single_network(manifest['checkpoint'],device,'teacher2')
        if cfg.image_size!=224 or cfg.method!='dual_ordinal':raise ValueError('Unexpected teacher configuration')
        seed_everything(cfg)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        initial=[state_hash(a),state_hash(b)]
        pool=_slice_names(cfg.root_path,'train')[:cfg.max_samples][:cfg.labelnum]
        if len(pool)!=cfg.labelnum:raise ValueError('Original training label budget changed')
        selected=np.linspace(0,len(pool)-1,min(len(pool),PROTOCOL['train_labeled_samples']),dtype=int)
        names=[pool[i] for i in selected]
        train=SpiderSliceDataset(cfg.root_path,'train',224,names=names,labeled_names=names,augment=False)
        valid=SpiderSliceDataset(cfg.root_path,'valid',224,augment=False)
        if len(valid)!=418 or len({_case_id(n) for n in valid.names})!=21:
            raise ValueError('Validation inventory changed')
        write_json(out/'inputs.json',dict(labeled_training_files=names,valid_files=valid.names,
                   config=cfg.to_dict(),teacher_hashes=initial))
        scores={}
        for split,dataset in (('train',train),('valid',valid)):
            scores[split]=evaluate([a,b],dataset,device,PROTOCOL['batch_size'])
            write_json(out/f'{split}.json',scores[split]);print('DONE',split,flush=True)
        if initial!=[state_hash(a),state_hash(b)] or sha(manifest['checkpoint'])!=manifest['checkpoint_sha256']:
            raise RuntimeError('Frozen teacher changed')
        result=dict(status='completed',seconds=time.monotonic()-started,model_updates=0,test_read=False,
            native_volume_metric=False,parameters_unchanged=True,
            scores={split:{name:row['total'] for name,row in rows.items()} for split,rows in scores.items()})
        write_json(out/'results.json',result);write_json(out/'status.json',dict(status='completed'))
        print(json.dumps({s:{n:{k:v for k,v in r['total'].items() if k in
            ('changed_foreground_fraction','changed_level_foreground_fraction','grid_case_macro_dice')}
            for n,r in rows.items()} for s,rows in scores.items()}),flush=True)
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc)));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        if args.checkpoint is None:parser.error('Preparation requires --checkpoint')
        prepare(args)
    else:
        if args.checkpoint is not None:parser.error('Runtime uses the sealed checkpoint path')
        run(args)


if __name__=='__main__':main()
