"""Paired prompt-robust teacher training and image-only revision caches.

FairVLM-inspired prompt variation and PromptSRC-inspired cosine anchoring are
adaptations, not reproductions. Both use the existing native BiomedCLIP blocks.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import seed_everything, state_hash, write_json, recover_log
from freqssl.evaluation import prepare_validation, _audit_raw_case
from freqssl.objectives import segmentation_loss
from freqssl.prompt_revision import prompt_bank, VARIANTS, unpack_template_probabilities, case_id
from freqssl.prompt_supervision import file_sha256
from freqssl.anatomy import FAMILY_IDS
from .data import TissueDataset, image_seed, supervision_fingerprint
from .model import build_teacher, MODEL_ID
from .run import protocol, save_teacher, evaluate_teacher, source_hashes


class PromptViews:
    def __init__(self, model, source):
        from open_clip import get_tokenizer
        self.prompts = prompt_bank(source)
        tokenizer = get_tokenizer('hf-hub:' + MODEL_ID)
        device = model.tokens.device
        self.tokens = torch.stack([tokenizer(p, context_length=256) for p in self.prompts]).to(device)
        with torch.no_grad():
            self.anchors = F.normalize(model.base.text(self.tokens.flatten(0, 1)).float(), dim=-1).reshape(6, 4, -1)
        assert not self.anchors.requires_grad

    def choices(self, batch, step, seed, variant):
        if variant == 'T0':
            return np.zeros((batch, 2), dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 7349]))
        return np.stack([rng.choice(4, size=2, replace=False) for _ in range(batch)])


def balanced_js(logits_a, logits_b, labels):
    log_a, log_b = logits_a.float().log_softmax(1), logits_b.float().log_softmax(1)
    a, b = log_a.exp(), log_b.exp()
    log_m = ((a + b) * .5).clamp_min(1e-8).log()
    pixels = .5 * ((a * (log_a - log_m)).sum(1) + (b * (log_b - log_m)).sum(1))
    return torch.stack([pixels[labels == c].mean() for c in range(4) if (labels == c).any()]).mean()


def paired_loss(model, views, images, labels, variant, step, seed=42):
    choices = views.choices(len(images), step, seed, variant)
    logits, anchors = [], []
    for view in range(2):
        indices = torch.as_tensor(choices[:, view], device=images.device)
        with torch.random.fork_rng(devices=[images.device.index] if images.is_cuda else []):
            torch.manual_seed(image_seed(f'revision-fit:{step}', seed))
            prediction, text = model(images, prompt_tokens=views.tokens[indices], return_text_features=True)
        logits.append(prediction)
        anchors.append((1 - (text * views.anchors[indices]).sum(-1)).mean())
    segmentation = .5 * sum(segmentation_loss(p, labels) for p in logits)
    js = balanced_js(*logits, labels)
    anchor = .5 * sum(anchors)
    loss = segmentation + (.1 * js if variant != 'T0' else 0.) + (.1 * anchor if variant == 'T2' else 0.)
    return loss, {'segmentation': float(segmentation.detach()), 'js': float(js.detach()),
        'anchor': float(anchor.detach()), 'prompt_ids': choices.tolist()}


def recipe_for(cfg, source, variant):
    return {**protocol(cfg, source), 'revision': 2, 'variant': variant, 'prompt_bank': prompt_bank(source),
        'prompt_sampling': 'stateless SeedSequence(seed,step,7349); two views with common MC random numbers',
        'js_weight': .1 if variant != 'T0' else 0., 'anchor_weight': .1 if variant == 'T2' else 0.,
        'training_split_sha256': file_sha256(cfg.training_split) if cfg.training_split else None}


def fit(cfg, source, variant, output, device, resume=None, stop_after=None):
    cfg = replace(cfg, prompt_source='none', prompt_cache='', prompt_gate='', prompt_route='legacy', slice_context='none')
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise FileExistsError(output)
    seed_everything(cfg)
    _, _, split = build_training_data(cfg)
    validation, records = prepare_validation(cfg)
    model, metadata = build_teacher(source, device, cfg.seed)
    base_hash = state_hash(model.base)
    provenance = {**metadata, 'foundation_sha256': file_sha256(metadata['checkpoint']),
        'initial_frozen_state_sha256': base_hash,
        'initial_adaptation_sha256': {k: state_hash(getattr(model, k)) for k in ('adapters','mask_head','upscale')},
        'source_hashes': source_hashes(), 'unlabeled_annotations_opened': False, 'test_evaluated': False,
        **supervision_fingerprint(cfg.root_path, split)}
    recipe = recipe_for(cfg, source, variant)
    views = PromptViews(model, source)
    training = TissueDataset(cfg.root_path, 'train', split['labeled_files'], metadata['mean'], metadata['std'], augment=True, seed=cfg.seed)
    valid = TissueDataset(cfg.root_path, 'valid', validation.names, metadata['mean'], metadata['std'], image_only=True)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.0003, weight_decay=.0001)
    step, best = 0, -1.
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['recipe'] != recipe or saved['provenance'] != provenance or json.loads((output/'split.json').read_text()) != split:
            raise ValueError('Revision teacher resume provenance changed')
        model.load_adaptation(saved['adaptation']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, best = saved['step'], saved['best_score']; recover_log(output, step)
    else:
        for name, obj in [('recipe',recipe),('provenance',provenance),('split',split),('validation_audit',records)]:
            write_json(output/f'{name}.json',obj)
    model.train(); per_epoch = len(training)//4
    if not per_epoch: raise ValueError('Insufficient authorized teacher samples')
    end = min(6000, step + stop_after) if stop_after else 6000
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    try:
        with (output/'metrics.jsonl').open('a',buffering=1) as log:
            while step < end:
                epoch, offset = divmod(step,per_epoch); training.set_epoch(epoch)
                indices = np.random.RandomState(cfg.seed+epoch).permutation(len(training))
                for batch_index in range(offset,per_epoch):
                    samples = [training[int(i)] for i in indices[batch_index*4:batch_index*4+4]]
                    images = torch.stack([s['image'] for s in samples]).to(device)
                    labels = torch.stack([s['label'] for s in samples]).to(device)
                    lr = .0003*(1-step/6000)**.9
                    for group in optimizer.param_groups: group['lr']=lr
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = paired_loss(model, views, images, labels, variant, step, cfg.seed)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite revision teacher loss')
                    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()),12.,error_if_nonfinite=True)
                    optimizer.step(); step+=1
                    entry=dict(metrics,kind='train',step=step,loss=float(loss.detach()),lr=lr,grad_norm=float(norm))
                    log.write(json.dumps(entry,allow_nan=False)+'\n')
                    if step==1 or step%25==0:
                        print(f'REVISION {variant}/{source} step={step}/6000 loss={entry["loss"]:.5f}',flush=True)
                        write_json(output/'status.json',dict(status='running',step=step,variant=variant,source=source,elapsed_seconds=time.monotonic()-started))
                    if step%500==0:
                        scores=evaluate_teacher(model,valid,records,device,cfg.seed)
                        write_json(output/f'valid_step_{step}.json',scores)
                        log.write(json.dumps(dict(kind='validation',step=step,dice=scores['dice']))+'\n')
                        if scores['dice']>best:
                            best=scores['dice'];save_teacher(output/'best.pt',model,optimizer,recipe,step,best,provenance)
                        print(f'VALID {variant}/{source} step={step} dice={scores["dice"]:.6f}',flush=True)
                    if step%100==0 or step==end:save_teacher(output/'last.pt',model,optimizer,recipe,step,best,provenance)
                    if step>=end:break
        if state_hash(model.base)!=base_hash:raise AssertionError('Frozen foundation changed')
        result=dict(status='completed' if step==6000 else 'paused',step=step,variant=variant,source=source,
            best_valid_tissue_dice=best,seconds_this_invocation=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),frozen_weights_unchanged=True,
            trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),test_evaluated=False)
        write_json(output/'result.json',result);write_json(output/'status.json',result)
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',step=step,error=repr(error)));raise


def load_revision(cfg, source, variant, checkpoint, device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['recipe'] != recipe_for(cfg,source,variant):raise ValueError('Teacher recipe changed')
    _,_,split=build_training_data(replace(cfg,prompt_source='none',prompt_cache=''))
    if json.loads((checkpoint.parent/'split.json').read_text())!=split:raise ValueError('Teacher split changed')
    fp=supervision_fingerprint(cfg.root_path,split)
    if any(saved['provenance'].get(k)!=v for k,v in fp.items()):raise ValueError('Teacher label provenance changed')
    model,meta=build_teacher(source,device,cfg.seed)
    if file_sha256(meta['checkpoint'])!=saved['provenance']['foundation_sha256']:raise ValueError('Foundation changed')
    model.load_adaptation(saved['adaptation']);model.eval()
    return model,meta,PromptViews(model,source),saved,split


@torch.inference_mode()
def predict_templates(model,views,images,seed):
    probs=[]
    for template in range(4):
        with torch.random.fork_rng(devices=[images.device.index] if images.is_cuda else []):
            torch.manual_seed(seed)
            tokens=views.tokens[template][None].expand(len(images),-1,-1)
            probs.append(torch.stack([model(images,prompt_tokens=tokens).float().softmax(1) for _ in range(2)]).mean(0))
    return torch.stack(probs,1)


def calibration_records(cfg,partition):
    allowed=set(partition['authorized_labeled_files']);names=partition['calibration_files']
    if not set(names)<=allowed:raise ValueError('Unauthorized calibration labels')
    groups={c:[n for n in names if case_id(n)==c] for c in partition['calibration_cases']}
    return [_audit_raw_case(cfg.root_path,Path(cfg.root_path).parent/'mha','train',c,ns) for c,ns in groups.items()]


@torch.inference_mode()
def export(cfg,source,variant,checkpoint,output,device,calibration=False):
    seed_everything(cfg);model,meta,views,saved,split=load_revision(cfg,source,variant,checkpoint,device)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError(output)
    partition=json.loads(Path(cfg.training_split).read_text()) if calibration else None
    names=partition['calibration_files'] if calibration else split['unlabeled_files']
    ds=TissueDataset(cfg.root_path,'train',names,meta['mean'],meta['std'],image_only=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    if calibration:
        records=calibration_records(cfg,partition);scores={}
        for key,template in [('canonical',0),('heldout_4',4),('heldout_5',5)]:
            model.tokens=views.tokens[template]
            scores[key]=evaluate_teacher(model,ds,records,device,cfg.seed)
            print('CALIBRATION',variant,source,key,scores[key]['dice'],flush=True)
        model.tokens=views.tokens[0]
        write_json(output/'robustness.json',scores)
    entries={};accepted=np.zeros(4,dtype=np.int64);predicted=accepted.copy()
    for start in range(0,len(ds),4):
        samples=[ds[i] for i in range(start,min(start+4,len(ds)))];ns=[s['file_name'] for s in samples]
        images=torch.stack([s['image'] for s in samples]).to(device)
        if len(images)<4:images=torch.cat([images,images.new_zeros((4-len(images),*images.shape[1:]))])
        q=predict_templates(model,views,images,image_seed('|'.join(ns),cfg.seed))
        packed=torch.cat([q.flatten(1,2),q.new_ones((len(images),1,224,224))],1)
        targets=unpack_template_probabilities(packed)
        for i,name in enumerate(ns):
            file=Path(name).stem+'.npz';temp=output/(file+'.tmp')
            with temp.open('wb') as handle:np.savez_compressed(handle,template_probabilities=q[i].cpu().numpy().astype(np.float16),valid=np.ones((224,224),dtype=bool))
            temp.replace(output/file)
            entries[name]={'file':file,'sha256':file_sha256(output/file),'image_sha256':file_sha256(Path(cfg.root_path)/'train/image'/name)}
            target=targets[i,0].long().cpu().numpy();mask=targets[i,1].bool().cpu().numpy()
            accepted+=np.bincount(target[mask],minlength=4);predicted+=np.bincount(target.ravel(),minlength=4)
        if start%100==0:
            write_json(output/'status.json',dict(status='running',slices=len(entries),total=len(ds)))
            print('CACHE_REVISION',source,variant,len(entries),len(ds),flush=True)
    manifest=dict(format_version=2,status='completed',source=source,teacher_variant=variant,prompts=prompt_bank(source)[0],
        prompt_bank=prompt_bank(source)[:4],samples_per_template=2,mc_samples=8,root_path=split['root_path'],
        labeled_files=split['labeled_files'],unlabeled_files=split['unlabeled_files'],image_size=224,family_ids=list(FAMILY_IDS),
        entries=entries,purpose='calibration' if calibration else 'unlabeled_training',
        calibration_files=names if calibration else [],training_split_sha256=file_sha256(cfg.training_split) if cfg.training_split else None,
        teacher_checkpoint=str(checkpoint.resolve()),teacher_checkpoint_sha256=file_sha256(checkpoint),teacher_step=saved['step'],
        foundation=meta,unlabeled_annotations_opened=False,test_evaluated=False,
        accepted_pixels_per_tissue=accepted.tolist(),predicted_pixels_per_tissue=predicted.tolist(),coverage=float(accepted.sum()/predicted.sum()),
        seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),**supervision_fingerprint(cfg.root_path,split))
    write_json(output/'manifest.json',manifest);write_json(output/'status.json',dict(status='completed',slices=len(entries)))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','cache','calibration-cache'])
    p.add_argument('--config',required=True);p.add_argument('--source',choices=['text','neutral'],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=load_config(a.config);device=torch.device(a.device)
    if a.stage=='fit':fit(cfg,a.source,a.variant,a.output,device,a.resume,a.stop_after)
    else:export(cfg,a.source,a.variant,a.checkpoint,a.output,device,a.stage=='calibration-cache')

if __name__=='__main__':main()
