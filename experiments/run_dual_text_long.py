"""Frozen paired MT / context N0 / full N7 training on one 200-epoch schedule."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

from experiments.run_dual_text import arm_config as dual_config, interventions
from experiments.run_encoder_text import read, write_csv, percent, compare
from experiments.run_hierarchical_text import initialize_device, class_name
from experiments.train_long_v3 import resolve_long_config, run_long
from freqssl.augment import strong_view, aligned_cutmix
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import write_json, state_hash
from freqssl.hierarchical_text import load_hierarchical_bank
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import file_sha256
from spider_volume import aggregate_cases

ARMS = ('MT', 'N0', 'N7')
LABELS = dict(MT='单切片纯MT', N0='三片上下文，无文本', N7='三片上下文，完整文本模块')


def arm_config(base, arm, bank, output):
    if arm not in ARMS:
        raise ValueError('Unknown long-training arm')
    cfg = dual_config(base, 'N0' if arm == 'MT' else arm, bank, output)
    return replace(cfg, slice_context='none' if arm == 'MT' else 'adaptive', max_seconds=86400).validate()


def center(images):
    return images[:, 1:2] if images.shape[1] == 3 else images


def verify_pairing(configs, epochs=(0, 99, 100, 199)):
    """Read actual augmented samples and compare entire sampled epoch orders."""
    reference = None
    receipts = {}
    for arm in ARMS:
        cfg = configs[arm]
        dataset, sampler, split = build_training_data(cfg)
        rows = []
        for epoch in epochs:
            dataset.set_epoch(epoch); sampler.set_epoch(epoch)
            order = list(iter(sampler))
            batch = default_collate([dataset[index] for index in order[0]])
            step = epoch * len(sampler)
            strong = strong_view(batch['image'], cfg.seed+step*104729, cfg.noise_std, cfg.gain_std)
            unlabeled = strong[~batch['is_labeled']]
            # A shared dummy target verifies the CutMix donor/rectangle too.
            target = torch.ones(len(unlabeled), 20, *unlabeled.shape[-2:]) / 20
            mixed, mixed_target = aligned_cutmix(unlabeled, target, cfg.seed+step*104729+17, cfg.cutmix_probability)
            rows.append(dict(epoch=epoch, order=order, image=center(batch['image']), label=batch['label'],
                labeled=batch['is_labeled'], strong=center(strong), mixed=center(mixed), target=mixed_target))
        if reference is None:
            reference = (split, rows)
        else:
            if split != reference[0]:
                raise AssertionError('Paired training split changed')
            for row, other in zip(rows, reference[1]):
                if row['order'] != other['order']:
                    raise AssertionError('Paired sampler order changed')
                for key in ('image', 'label', 'labeled', 'strong', 'mixed', 'target'):
                    torch.testing.assert_close(row[key], other[key], atol=0, rtol=0)
        digest = hashlib.sha256()
        for row in rows:
            digest.update(json.dumps(row['order']).encode())
            for key in ('image','label','labeled','strong','mixed','target'):
                digest.update(row[key].contiguous().numpy().tobytes())
        receipts[arm] = digest.hexdigest()
    return dict(checked_zero_based_epochs=list(epochs), center_data_augmentation_cutmix_exact=True,
        full_epoch_sampling_order_exact=True, hashes=receipts, steps_per_epoch=len(sampler),
        split=reference[0], test_images_or_labels_read=False)


def prepare(config, banks, output, epochs=(100, 200), eval_every_epochs=5):
    base = load_config(config)
    if (base.method!='mean_teacher' or base.slice_context!='adaptive' or base.internal_text!='none'
            or base.prompt_source!='none' or base.region_weight or base.pseudo_dice_weight or base.text_alignment_weight):
        raise ValueError('Use the original adaptive-context MT protocol as base')
    torch.set_num_threads(base.cpu_threads)
    output, banks = Path(output).resolve(), Path(banks).resolve()
    resolved, steps_per_epoch, split = resolve_long_config(base, output, epochs, eval_every_epochs)
    _, provenance = load_hierarchical_bank(banks/'text.pt', 'text')
    if shutil.disk_usage(output.parent).free < 6*1024**3:
        raise RuntimeError('Need at least 6 GiB free for the three long runs')
    output.mkdir(parents=True, exist_ok=False)
    (output/'banks').mkdir(); (output/'configs').mkdir()
    shutil.copy2(banks/'text.pt', output/'banks/text.pt')
    configs = {a:arm_config(resolved,a,output/'banks/text.pt',output/'runs'/a/'training') for a in ARMS}
    check_epochs = sorted(set((0,epochs[0]-1, min(epochs[0], epochs[-1]-1), epochs[-1]-1)))
    pairing = verify_pairing(configs, check_epochs)
    write_json(output/'pairing.json', pairing)
    parameters, init_hashes = {}, {}
    visual = context = None
    probe = torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32)
    for arm,cfg in configs.items():
        (output/'configs'/f'{arm}.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
        before = torch.get_rng_state().clone()
        model = StudentTeacherSystem(cfg).students[0]
        if not torch.equal(before,torch.get_rng_state()):
            raise AssertionError('Initialization changed caller RNG')
        named = dict(model.named_parameters())
        if visual is None:
            visual = {k:v.detach().clone() for k,v in named.items()}
        for name,value in visual.items():
            if name not in named or not torch.equal(named[name],value):
                raise AssertionError('Shared visual initialization differs')
        with torch.no_grad():
            if arm=='N0':
                context = {k:v.detach().clone() for k,v in named.items()}
                zero_logits = model(probe)
            if arm=='N7':
                if not all(torch.equal(named[k],v) for k,v in context.items()):
                    raise AssertionError('Context initialization differs')
                torch.testing.assert_close(model(probe),zero_logits,atol=0,rtol=0)
        parameters[arm] = dict(total=sum(p.numel() for p in model.parameters()),
            participating_training=sum(p.numel() for p in model.parameters() if p.requires_grad))
        init_hashes[arm] = state_hash(model)
    code = output/'code'
    for folder in ('freqssl','experiments','tests'):
        shutil.copytree(ROOT/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train.py','evaluate.py','spider_volume.py'):
        shutil.copy2(ROOT/name,code/name)
    write_json(output/'protocol.json',dict(arms=list(ARMS),epochs=list(epochs),steps_per_epoch=steps_per_epoch,
        steps_per_arm=resolved.max_steps,total_steps=3*resolved.max_steps,eval_every_epochs=eval_every_epochs,
        eval_every=resolved.eval_every,seed=base.seed,parameters=parameters,initial_hashes=init_hashes,
        milestones={str(e):e*steps_per_epoch for e in epochs},initialization='fresh paired seed',
        epoch_definition='one pass through labeled slices; independent cycling unlabeled pool',
        learning_rate='base_lr*(1-completed_steps/final_steps)**0.9; no reset at milestone',
        intermediate_is_independent_short_schedule=False,max_seconds_per_invocation=86400,
        text_bank=provenance,selection='teacher1 case-equal GT-present foreground Dice; earlier ties',
        primary_contrast='N7-N0',primary_threshold_pp=.3,union_non_decreasing_required=True,
        other_contrasts=['N0-MT','N7-MT'],test_evaluated=False,multiple_seeds=False,
        post_training=dict(arm='N7',checkpoints=['best','last'],states=['full','zero_e2','zero_e3','zero_both','class_mismatch'],
            permutation=[3,0,1,2],anchors_unchanged=True,full_reproduction_required=True),
        pairing_receipt='pairing.json',no_automatic_retry=True,physical_gpu=2))
    paths = [output/'protocol.json',output/'pairing.json',*[p for folder in (code,output/'configs',output/'banks') for p in sorted(folder.rglob('*')) if p.is_file()]]
    write_json(output/'frozen_sources.json',{str(p.relative_to(output)):file_sha256(p) for p in paths})
    write_json(output/'status.json',dict(status='prepared',training_started=False,arms=list(ARMS)))


def add_score(tables, common, score):
    if aggregate_cases(score['cases']) != {k:score[k] for k in ('overall','per_class','per_case')}:
        raise ValueError('Saved metrics disagree with case scores')
    tables['overall'].append(dict(**common,mean_dice_pct=percent(score['overall']['dice']),
        mean_iou_pct=percent(score['overall']['iou']),union_dice_pct=percent(score['overall']['dice_union_present']),valid_cases=len(score['cases'])))
    for row in score['per_class']:
        tables['per_class'].append(dict(**common,class_id=row['class_id'],label=class_name(row['class_id']),
            dice_pct=percent(row['dice']),iou_pct=percent(row['iou']),union_dice_pct=percent(row['dice_union_present']),
            gt_present_cases=row['gt_present_cases'],missed_gt_cases=row['missed_gt_cases'],fp_absent_gt_cases=row['fp_absent_gt_cases']))
    background = []
    for case in score['cases']:
        cm=np.asarray(case['anatomy']['confusion_gt_rows_pred_columns'],dtype=np.int64)
        g,p,t=int(cm[0].sum()),int(cm[:,0].sum()),int(cm[0,0])
        if g: background.append((2*t/(g+p),t/(g+p-t)))
    tables['per_class'].append(dict(**common,class_id=0,label='背景',
        dice_pct=100*float(np.mean([x[0] for x in background])) if background else None,
        iou_pct=100*float(np.mean([x[1] for x in background])) if background else None,gt_present_cases=len(background)))
    for row in score['per_case']:
        tables['per_case'].append(dict(**common,case=row['case'],dice_pct=percent(row['dice']),
            iou_pct=percent(row['iou']),union_dice_pct=percent(row['dice_union_present'])))
    for tissue in ('vertebra','canal','disc'):
        values=[c['anatomy']['tissue_dice'][tissue] for c in score['cases']]
        values=[v for v in values if v is not None]
        tables['merged_tissue'].append(dict(**common,tissue=tissue,
            dice_pct=100*float(np.mean(values)) if values else None,gt_present_cases=len(values)))


def decisions(arms):
    primary=compare(arms,'N7','N0',.3)
    others=[compare(arms,c,b,0.) for c,b in (('N0','MT'),('N7','MT'))]
    for row in others:
        row.pop('threshold_pp'); row.pop('passed')
    return dict(primary=primary,other_contrasts=others,confirmatory_efficacy_claim=False,
                components_independently_identified=False)


def summarize(output):
    output=Path(output); protocol=read(output/'protocol.json'); spe=protocol['steps_per_epoch']
    tables={k:[] for k in ('overall','per_class','per_case','merged_tissue','validation_history',
        'paired_case_deltas','comparisons','resources','training_diagnostics','training_window_summary')}
    milestones={str(e):{} for e in protocol['epochs']}; scores={}; reference=None; complete=[]
    for arm in ARMS:
        directory=output/'runs'/arm; training=directory/'training'
        path=directory/'results.json'
        if not path.exists(): continue
        result=read(path)
        if result['status']=='completed': complete.append(arm)
        split=read(directory/'split.json')
        if reference is None: reference=split
        if reference!=split: raise ValueError('Long-run data splits differ')
        invocations=[read(p) for p in sorted((directory/'invocations').glob('*.json'))]
        for epoch in protocol['epochs']:
            key=str(epoch)
            if key not in result['milestones']: continue
            receipt=result['milestones'][key]; folder=directory/'epochs'/f'{epoch:04d}'
            if any(file_sha256(v['path'])!=v['sha256'] for v in receipt['checkpoints'].values()):
                raise ValueError('Milestone checkpoint changed')
            best=read(folder/'best_valid.json')['teacher1']; exact=read(folder/'at_epoch_valid.json')['teacher1']
            milestones[key][arm]=dict(best_step=receipt['best_checkpoint_step'],last_step=receipt['step'],
                best=best['overall'],last=exact['overall'])
            for kind,score,step in (('best',best,receipt['best_checkpoint_step']),('exact',exact,receipt['step'])):
                if sorted(c['case'] for c in score['cases'])!=sorted(split['validation_cases']):
                    raise ValueError('Incomplete milestone validation')
                scores[arm,epoch,kind]=score
                add_score(tables,dict(arm=arm,milestone_epoch=epoch,checkpoint=kind,step=step,checkpoint_epoch=step/spe),score)
            costs=[x for x in invocations if x['start_step']<epoch*spe]
            tables['resources'].append(dict(arm=arm,milestone_epoch=epoch,**protocol['parameters'][arm],
                seconds=sum(x['seconds'] for x in costs),invocations=len(costs),failed_invocations=sum(x['status']=='failed' for x in costs),
                peak_allocated_bytes=max((x['peak_allocated_bytes'] or 0 for x in costs),default=0),
                peak_reserved_bytes=max((x['peak_reserved_bytes'] or 0 for x in costs),default=0)))
        for path in sorted(training.glob('valid_step_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
            step=int(path.stem.split('_')[-1]); score=read(path)['teacher1']['overall']
            tables['validation_history'].append(dict(arm=arm,step=step,epoch=step/spe,
                mean_dice_pct=percent(score['dice']),mean_iou_pct=percent(score['iou']),union_dice_pct=percent(score['dice_union_present'])))
        logs=[]
        if (training/'metrics.jsonl').exists():
            with (training/'metrics.jsonl').open() as handle:
                logs=[r for r in map(json.loads,handle) if r.get('kind')=='train']
        if logs and [r['step'] for r in logs]!=list(range(1,logs[-1]['step']+1)):
            raise ValueError('Missing or duplicate training updates')
        cfg=load_config(output/'configs'/f'{arm}.yaml')
        if arm in complete and (not logs or logs[-1]['step']!=protocol['steps_per_arm']):
            raise ValueError('Completed run has wrong update budget')
        for r in logs:
            if r['step']!=1 and r['step']%cfg.log_every: continue
            row=dict(arm=arm,step=r['step'],epoch=r['step']/spe,scope='sampled_at_log_interval',loss=r['loss'],
                alignment_loss=r.get('text_alignment_loss'),pseudo_coverage=r.get('coverage_s1'))
            for i,s in enumerate((2,3,4)):
                row[f'E{s}_gate']=r.get('text_gates',[None]*3)[i]
                row[f'E{s}_alignment_loss']=r.get('text_alignment_loss_per_stage',[None]*3)[i]
                for c in range(4): row[f'E{s}_tissue_fraction_{c}']=r.get('text_tissue_prediction_fractions',[[None]*4]*3)[i][c]
            row.update(r.get('encoder_text_diagnostics',{})); tables['training_diagnostics'].append(row)
        for epoch in protocol['epochs']:
            if str(epoch) not in result['milestones']: continue
            data=[r for r in logs if r['step']<=epoch*spe]
            for window,records in (('through_milestone',data),('last_500_steps',data[-500:])):
                row=dict(arm=arm,milestone_epoch=epoch,scope='every_training_step',window=window,samples=len(records),
                    first_step=records[0]['step'],last_step=records[-1]['step'])
                for key in ('loss','text_alignment_loss','coverage_s1','unsupervised_weight'):
                    values=[r[key] for r in records if key in r]
                    row[key]=float(np.mean(values)) if values else None
                tables['training_window_summary'].append(row)
    decision={}
    for epoch in protocol['epochs']:
        arms=milestones[str(epoch)]
        if len(arms)==len(ARMS):
            decision[str(epoch)]=decisions(arms)
            for relation,row in [('primary',decision[str(epoch)]['primary'])]+[('descriptive',r) for r in decision[str(epoch)]['other_contrasts']]:
                tables['comparisons'].append(dict(milestone_epoch=epoch,relation=relation,**row))
        for candidate,control in (('N7','N0'),('N0','MT'),('N7','MT')):
            if candidate not in arms or control not in arms: continue
            for kind in ('best','exact'):
                ref={r['case']:r for r in scores[control,epoch,kind]['per_case']}
                for row in scores[candidate,epoch,kind]['per_case']:
                    other=ref[row['case']]
                    tables['paired_case_deltas'].append(dict(milestone_epoch=epoch,checkpoint=kind,candidate=candidate,control=control,case=row['case'],
                        dice_delta_pp=100*(row['dice']-other['dice']),iou_delta_pp=100*(row['iou']-other['iou']),
                        union_delta_pp=100*(row['dice_union_present']-other['dice_union_present'])))
    post=read(output/'interventions/result.json') if (output/'interventions/result.json').exists() else dict(status='pending')
    status='completed' if len(complete)==3 and post['status']=='completed' else 'training_completed' if len(complete)==3 else 'partial'
    result=dict(status=status,completed_arms=complete,milestones=milestones,decision=decision,
        post_training_interventions=post,test_evaluated=False,multiple_seeds=False)
    write_json(output/'results.json',result)
    report=output/'metrics_report'; report.mkdir(exist_ok=True)
    for name,rows in tables.items(): write_csv(report/f'{name}.csv',rows)
    lines=[f"# MT / N0 / N7：{protocol['epochs'][-1]}轮配对实验",'',
        f"状态：{status}；完成{len(complete)}/3组。每组{protocol['steps_per_arm']}步，合计{protocol['total_steps']}步。",
        '主指标：病例等权、GT存在前景Mean Dice。所有检查点使用完整验证集，不评估测试集。','',
        '| 里程碑 | 组别 | 最佳步 | 最佳Dice% | 最佳IoU% | 最佳Union% | 准确轮次Dice% | 准确轮次IoU% | 准确轮次Union% |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for epoch,arms in milestones.items():
        for arm,row in arms.items():
            b,l=row['best'],row['last']
            lines.append(f"| {epoch} | {arm} {LABELS[arm]} | {row['best_step']} | {100*b['dice']:.4f} | {100*b['iou']:.4f} | {100*b['dice_union_present']:.4f} | {100*l['dice']:.4f} | {100*l['iou']:.4f} | {100*l['dice_union_present']:.4f} |")
    for epoch,d in decision.items():
        p=d['primary']; lines += ['',f"第{epoch}轮：N7−N0最佳Dice {p['delta_dice_pp']:+.4f}pp，Union {p['delta_union_pp']:+.4f}pp；工程筛查通过={p['passed']}；最佳/末步排序反转={p['rank_reversal']}。"]
    lines += ['', '中间快照使用完整训练预算的学习率，不是独立短训练。单种子结果不能证明统计显著性或分别确定路由、差分的贡献。',
        '全步均值见training_window_summary.csv；采样诊断见training_diagnostics.csv。耗时累计所有已记录调用，显存取峰值；三组更新预算相同但计算量不同。',
        f"训练后固定检查点干预：{post['status']}；详见interventions目录。"]
    (report/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def execute(output,device):
    output=Path(output)
    if ROOT!=output/'code': raise RuntimeError('Execute the frozen runner')
    started=datetime.now(timezone.utc).isoformat(); arm=None; phase='preflight'
    with (output/'screen.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if any(file_sha256(output/p)!=h for p,h in read(output/'frozen_sources.json').items()):
                raise ValueError('Frozen code/config/bank/protocol changed')
            protocol=read(output/'protocol.json'); phase='training'
            for arm in ARMS:
                directory=output/'runs'/arm
                if (directory/'results.json').exists() and read(directory/'results.json')['status']=='completed': continue
                write_json(output/'status.json',dict(status='running',phase=phase,arm=arm,pid=os.getpid(),device=str(device),started_utc=started))
                initialize_device(device)
                cfg=load_config(output/'configs'/f'{arm}.yaml')
                run_long(cfg,device,directory,tuple(protocol['epochs']),protocol['eval_every_epochs'],
                    resume=(directory/'protocol.json').exists(),on_milestone=lambda epoch:summarize(output))
                summarize(output); gc.collect()
                if torch.device(device).type=='cuda': torch.cuda.empty_cache()
            phase='fixed_checkpoint_interventions'; initialize_device(device)
            interventions(output,device,run_directory=output/'runs/N7/training')
            result=summarize(output)
            write_json(output/'status.json',dict(status='completed',decision=result['decision'],post_training_interventions=result['post_training_interventions']))
            write_json(output/'exit.json',dict(exit_code=0,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat()))
        except BaseException as error:
            write_json(output/'status.json',dict(status='failed',phase=phase,arm=arm,error=repr(error)))
            write_json(output/'exit.json',dict(exit_code=1,error=repr(error),finished_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v3/context_efficiency_reference.yaml')
    parser.add_argument('--banks',type=Path,default=ROOT/'outputs/hierarchical_text_assets_20261108')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--epochs',type=int,nargs='+',default=[100,200])
    parser.add_argument('--eval-every-epochs',type=int,default=5)
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--execute-frozen',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args(); output=args.output.resolve()
    if args.summarize_only: summarize(output)
    elif args.execute_frozen: execute(output,args.device)
    else:
        if not args.resume: prepare(args.config,args.banks,output,tuple(args.epochs),args.eval_every_epochs)
        if not args.prepare_only:
            subprocess.run([sys.executable,'-u',str(output/'code/experiments/run_dual_text_long.py'),
                '--output',str(output),'--device',args.device,'--execute-frozen'],check=True,cwd=output/'code')
        else: print(f'Prepared: {output}',flush=True)


if __name__=='__main__': main()
