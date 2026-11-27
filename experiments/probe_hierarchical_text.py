"""Paired, short calibration probe; not a segmentation benchmark or full MT run.

Six arms see the same four labeled training slices for 32 updates. Four slices
from other training cases diagnose transfer. A separate real mixed-batch MT
smoke test exercises the normal training engine, EMA and unlabeled path.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from torch.nn import functional as F
from freqssl.anatomy import family_labels
from freqssl.config import ExperimentConfig
from freqssl.engine import seed_everything, make_optimizer, train_step, snapshot_sources, state_hash
from freqssl.hierarchical_text import tissue_alignment_loss, global_alignment_loss, centered_global_alignment_loss
from freqssl.models import StudentTeacherSystem
from freqssl.objectives import segmentation_loss
from freqssl.prompt_supervision import file_sha256


def dice_scores(pred,truth,nclasses):
    scores = {}
    for c in range(1,nclasses):
        p,t = pred==c,truth==c
        scores[str(c)] = (float(2*(p&t).sum()/(p.sum()+t.sum())) if t.any() else None)
    present = [v for v in scores.values() if v is not None]
    return dict(mean=sum(present)/len(present) if present else None,per_class=scores)


@torch.no_grad()
def diagnose(net,batch,hierarchical):
    # This model uses GroupNorm, no dropout/BatchNorm. Collect alignment on the
    # same training execution path before/after, without parameter updates.
    net.train()
    if hierarchical:
        logits,aux = net.forward_with_alignment(batch['image'])
    else:
        logits,aux = net(batch['image']),None
    result = dict(segmentation_loss=float(segmentation_loss(logits,batch['label'])),
        slice_pooled_fg_present_dice=dice_scores(logits.argmax(1),batch['label'],20))
    if aux is not None:
        similarity = F.normalize(aux['global_embedding'],dim=-1) @ batch['global_target'].T
        mask = torch.eye(len(similarity),dtype=torch.bool,device=similarity.device)
        result.update(tissue_ce=float(tissue_alignment_loss(aux['tissue_logits'],batch['label'])),
            tissue_class_balanced_ce=float(tissue_alignment_loss(aux['tissue_logits'],batch['label'],'class')),
            tissue_ce_per_stage=[float(tissue_alignment_loss([p],batch['label'])) for p in aux['tissue_logits']],
            global_cosine=float(similarity.diag().mean()),
            global_pair_margin=float(similarity.diag().mean()-similarity[~mask].mean()),
            global_pair_top1=float((similarity.argmax(1)==torch.arange(len(similarity),device=similarity.device)).float().mean()),
            centered_global_cosine=1-float(centered_global_alignment_loss(aux['global_embedding'],batch['global_target'])),
            gates=net.text_pyramid.gates.tanh().tolist(),
            mean_layer_mixtures=[v.mean(0).tolist() for v in aux['layer_mixtures']],
            tissue_fg_dice_per_stage=[dice_scores(F.interpolate(p,size=batch['label'].shape[-2:],
                mode='bilinear',align_corners=False).argmax(1),family_labels(batch['label']),4) for p in aux['tissue_logits']])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--suite',choices=('initial','balanced'),default='initial')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 2:
        raise ValueError('Need at least two updates')
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the requested device')
    args.output.mkdir(parents=True)
    snapshot_sources(args.output)
    manifest = json.loads((args.assets/'manifest.json').read_text())
    if file_sha256(args.assets/'probe_batches.pt') != manifest['probe']['batch_sha256']:
        raise ValueError('Probe batches changed')
    batches = torch.load(args.assets/'probe_batches.pt',weights_only=True)
    batches = {n:{k:v.to(device) for k,v in b.items()} for n,b in batches.items()}
    base = ExperimentConfig(**manifest['probe']['config']).validate()
    seed_everything(base)
    arms = [
        ('visual_baseline','none','grouped',0.,0.),
        ('hierarchical_unaligned','text','grouped',0.,0.),
        ('hierarchical_dense','text','grouped',.1,0.),
        ('hierarchical_aligned','text','grouped',.1,.05),
        ('last_layer_aligned','text','last',.1,.05),
        ('neutral_aligned','neutral','grouped',.1,.05),
    ]
    if args.suite=='balanced':
        arms = [
            ('hierarchical_balanced','text','grouped',.1,.05),
            ('hierarchical_balanced_centered','text','grouped',.1,.05),
            ('last_layer_balanced_centered','text','last',.1,.05),
            ('neutral_balanced_centered','neutral','grouped',.1,.05),
        ]
    result = dict(status='running',protocol='fixed labeled calibration, separate MT integration check',
        updates_per_arm=args.steps,labeled_slices_per_update=4,seed=base.seed,
        heldout_is_training_subset=True,validation_or_test_content_used=False,
        full_training_started=False,segmentation_efficacy_established=False,
        suite=args.suite,fixed_lr=base.base_lr,amp=False,device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        dataset=manifest['probe'],arms={})
    def save():
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    start = time.monotonic()
    expected_parameters = None
    initial_visual = None
    with (args.output/'trace.jsonl').open('w',buffering=1) as log:
        for name,source,layer_mode,dense_weight,global_weight in arms:
            hierarchical = source!='none'
            balance = 'class' if args.suite=='balanced' else 'pixel'
            centered_weight = .05 if name.endswith('_centered') else 0.
            cfg = replace(base,internal_text=source,text_bank=str((args.assets/(source+'.pt')).resolve()) if hierarchical else '',
                text_fusion_layout='hierarchical' if hierarchical else 'e4',text_layer_mode=layer_mode,
                text_alignment_weight=dense_weight,text_alignment_balance=balance).validate()
            model = StudentTeacherSystem(cfg).to(device)
            net = model.students[0]
            # Verify actual initialized weights, not merely random seeds.
            visual = {k:v.detach().cpu().clone() for k,v in net.named_parameters()
                      if not k.startswith(('text_pyramid.','global_alignment.'))}
            if initial_visual is None:
                initial_visual = visual
            assert all(torch.equal(v,initial_visual[k]) for k,v in visual.items())
            if hierarchical:
                params = {k:v.detach().cpu().clone() for k,v in net.named_parameters()}
                if expected_parameters is None:
                    expected_parameters = params
                assert all(torch.equal(v,expected_parameters[k]) for k,v in params.items())
                frozen = net.text_pyramid.hidden_tokens.clone()
            record = dict(source=source,layer_mode=layer_mode,dense_weight=dense_weight,global_weight=global_weight,
                tissue_balance=balance,centered_weight=centered_weight,
                parameter_count=sum(p.numel() for p in net.parameters()),
                before={n:diagnose(net,b,hierarchical) for n,b in batches.items()})
            optimizer = make_optimizer(model,cfg)
            train = batches['train']
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                if hierarchical:
                    logits,aux = net.forward_with_alignment(train['image'])
                    dense = tissue_alignment_loss(aux['tissue_logits'],train['label'],balance)
                    global_loss = global_alignment_loss(aux['global_embedding'],train['global_target'])
                    centered = centered_global_alignment_loss(aux['global_embedding'],train['global_target'])
                else:
                    logits = net(train['image'])
                    dense = global_loss = centered = logits.new_zeros(())
                loss = (segmentation_loss(logits,train['label']) + dense_weight*dense
                        + global_weight*global_loss + centered_weight*centered)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),cfg.grad_clip,error_if_nonfinite=True)
                optimizer.step()
                model.update_teachers()
                entry = dict(arm=name,step=step+1,loss=float(loss),tissue_ce=float(dense),
                             global_loss=float(global_loss),centered_loss=float(centered))
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                if step==0 or (step+1)%16==0:
                    print(json.dumps(entry),flush=True)
            record['after'] = {n:diagnose(net,b,hierarchical) for n,b in batches.items()}
            if hierarchical:
                assert torch.equal(frozen,net.text_pyramid.hidden_tokens)
                assert torch.equal(frozen,model.teachers[0].text_pyramid.hidden_tokens)
            assert all(p.grad is None for p in model.teachers.parameters())
            record['frozen_text_and_ema_verified'] = True
            record['final_student_hash'] = state_hash(net)
            result['arms'][name] = record
            save()
            del optimizer,model,net,logits,loss
            if hierarchical:
                del aux,frozen,params
            gc.collect()
            if device.type=='cuda':
                torch.cuda.empty_cache()
    # Normal MT engine, genuine labeled/unlabeled samples. Zero warmup and
    # threshold expose the U path in three steps, without claiming benchmark parity.
    from freqssl.data import build_training_data
    from torch.utils.data._utils.collate import default_collate
    cfg = replace(base,internal_text='text',text_bank=str((args.assets/'text.pt').resolve()),
        text_fusion_layout='hierarchical',text_alignment_weight=.1,
        text_alignment_balance='class' if args.suite=='balanced' else 'pixel',
        warmup_steps=0,ramp_steps=0,confidence=0.,background_confidence=0.)
    dataset,sampler,_ = build_training_data(cfg)
    mixed = default_collate([dataset[i] for i in next(iter(sampler))])
    assert (mixed['label'][~mixed['is_labeled']]==-100).all()
    model = StudentTeacherSystem(cfg).to(device)
    optimizer = make_optimizer(model,cfg)
    result['mt_smoke'] = [train_step(model,optimizer,mixed,cfg,device,i) for i in range(3)]
    result.update(status='completed',seconds=time.monotonic()-start,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        initial_visual_parameters_equal=True,all_text_arm_parameters_equal=True)
    save()
    print(json.dumps(dict(status='completed',output=str(args.output),seconds=result['seconds'])),flush=True)


if __name__ == '__main__':
    main()
