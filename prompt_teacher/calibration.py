"""Offline case-held-out capability tables in the student's own environment."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from freqssl.config import load_config
from freqssl.data import build_training_data, SpiderSliceDataset
from freqssl.engine import load_model, seed_everything, write_json
from freqssl.prompt_supervision import PromptCache, file_sha256
from freqssl.prompt_revision import (prompt_bank, case_id, competence_eligible,
    tissue_probabilities, qualify_cell)
from freqssl.anatomy import FAMILY_IDS


class CalibrationCache:
    channels, version, geometry_mode, for_labeled = 17, 2, 'bilinear', True
    read = PromptCache.read

    def __init__(self, root, cfg, partition, source, variant):
        self.root=Path(root);self.image_size=cfg.image_size
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.entries=manifest['entries'];self.manifest=manifest
        expected=partition['calibration_files']
        if (manifest.get('status')!='completed' or manifest.get('format_version')!=2
                or manifest.get('purpose')!='calibration' or manifest.get('source')!=source
                or manifest.get('teacher_variant')!=variant
                or manifest.get('prompt_bank')!=prompt_bank(source)[:4]
                or manifest.get('training_split_sha256')!=file_sha256(cfg.training_split)
                or set(self.entries)!=set(expected) or manifest['labeled_files']!=partition['labeled_files']
                or manifest['unlabeled_files']!=partition['unlabeled_files']):
            raise ValueError('Calibration cache provenance mismatch')
        if file_sha256(manifest['teacher_checkpoint'])!=manifest['teacher_checkpoint_sha256']:
            raise ValueError('Calibration teacher checkpoint changed')
        for name,entry in self.entries.items():
            if entry['file']!=Path(name).stem+'.npz':raise ValueError('Unsafe cache path')
            if file_sha256(self.root/entry['file'])!=entry['sha256']:raise ValueError('Calibration cache changed')
            if file_sha256(Path(cfg.root_path)/'train/image'/name)!=entry['image_sha256']:raise ValueError('Calibration image changed')


def calibrate(cfg, reference, cache_root, output, source, variant, device):
    seed_everything(cfg);_,_,train_split=build_training_data(cfg)
    partition=json.loads(Path(cfg.training_split).read_text())
    names=partition['calibration_files'];cases=partition['calibration_cases']
    dataset=SpiderSliceDataset(cfg.root_path,'train',224,names=names,labeled_names=names,
        augment=True,seed=cfg.seed,slice_context='adaptive')
    dataset.prompt_cache=CalibrationCache(cache_root,cfg,partition,source,variant)
    if set(names)&set(train_split['labeled_files']+train_split['unlabeled_files']):raise ValueError('Calibration leakage')
    label_hashes={name:file_sha256(Path(cfg.root_path)/'train/label'/name) for name in names}
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError(output)
    started=time.monotonic();tables={};details={};count_rows={};checkpoints={};total_fg=total_valid=0
    family=torch.tensor(FAMILY_IDS,device=device)
    for step in range(500,6000,500):
        path=reference/f'step_{step}.pt';model,ref_cfg,payload=load_model(path,device)
        if ref_cfg.training_split!=cfg.training_split or payload['step']!=step:
            raise ValueError('Reference checkpoint does not match held-out protocol')
        if json.loads((reference/'split.json').read_text())!=train_split:raise ValueError('Reference split changed')
        del payload;model.eval();counts=np.zeros((len(cases),4,2,3),dtype=np.int64)
        valid_counts=np.zeros(len(cases),dtype=np.int64);case_indices={c:i for i,c in enumerate(cases)}
        with torch.inference_mode():
            for geometry_id in range(4):
                dataset.set_epoch(geometry_id)
                for start in range(0,len(dataset),4):
                    samples=[dataset[i] for i in range(start,min(start+4,len(dataset)))]
                    images=torch.stack([s['image'] for s in samples]).to(device)
                    targets=torch.stack([s['prompt_targets'] for s in samples]).to(device)
                    gt=family[torch.stack([s['label'] for s in samples]).to(device)]
                    ema=model.teachers[0](images).float().softmax(1)
                    tissue,trusted,valid,edge=(targets[:,i] for i in range(4))
                    tissue=tissue.long();edge=edge.long();eligible=competence_eligible(ema,tissue,trusted,valid)
                    ep=tissue_probabilities(ema).argmax(1)
                    for i,s in enumerate(samples):
                        ci=case_indices[case_id(s['file_name'])];valid_counts[ci]+=int(valid[i].sum())
                        for c in range(4):
                            for e in range(2):
                                mask=eligible[i]&(tissue[i]==c)&(edge[i]==e)
                                counts[ci,c,e]+=[int(mask.sum()),int((mask&(tissue[i]==gt[i])).sum()),int((mask&(ep[i]==gt[i])).sum())]
        table=np.zeros((4,2),dtype=bool);cells={}
        for c in range(4):
            for e in range(2):
                cell=qualify_cell(counts[:,c,e].tolist(),c);table[c,e]=cell['enabled'];cells[f'{c}:{e}']=cell
        selected_fg=int((counts[:,1:,:,0]*table[None,1:]).sum());denominator=int(valid_counts.sum())
        total_fg+=selected_fg;total_valid+=denominator
        tables[str(step)]=table.tolist();details[str(step)]={'cells':cells,'foreground_pixels':selected_fg,
            'valid_pixels':denominator,'foreground_coverage':selected_fg/denominator}
        count_rows[str(step)]={'cases':cases,'counts':counts.tolist(),'valid_pixels':valid_counts.tolist()}
        checkpoints[str(step)]={'path':str(path.resolve()),'sha256':file_sha256(path)}
        write_json(output/'status.json',dict(status='running',step=step,foreground_coverage=selected_fg/denominator))
        print('GATE',source,variant,step,table.tolist(),'foreground',selected_fg/denominator,flush=True)
        del model
    result=dict(format_version=1,status='completed',source=source,variant=variant,prompt_bank=prompt_bank(source)[:4],
        tables=tables,details=details,case_counts=count_rows,mean_foreground_guidance_coverage=total_fg/total_valid,
        training_split_sha256=file_sha256(cfg.training_split),cache_manifest_sha256=file_sha256(Path(cache_root)/'manifest.json'),
        calibration_label_sha256=label_hashes,
        reference_checkpoints=checkpoints,geometry_epochs=[0,1,2,3],bootstrap_seed=42,bootstrap_samples=2000,
        thresholds={'minimum_cases':5,'minimum_pixels':1000,'foreground_precision':.95,'background_precision':.99,
                    'advantage':.02,'bootstrap_lower_bound':0},seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),test_evaluated=False,
        limitation='Offline pilot capability table; frozen during full-label refit and student training. Not a probability guarantee.')
    write_json(output/'gate.json',result);write_json(output/'result.json',result)
    write_json(output/'status.json',dict(status='completed'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=['text','neutral'],required=True);p.add_argument('--variant',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    calibrate(load_config(a.config),a.reference,a.cache,a.output,a.source,a.variant,torch.device(a.device))

if __name__=='__main__':main()
