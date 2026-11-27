"""Freeze and execute a paired, four-arm hierarchical-text MT validation screen.

Every arm starts from the same visual initialization and uses the same complete
training/validation protocol. Global foundation-image losses are not enabled.
The supervisor writes reports as arms finish and stops on the first failure.
"""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
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
from freqssl.config import load_config
from freqssl.engine import run_experiment, write_json
from freqssl.hierarchical_text import load_hierarchical_bank
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import file_sha256


ARMS = ('hierarchical', 'context', 'neutral', 'last_layer')
ARM_LABELS = dict(hierarchical='分层组织文本', context='视觉基线',
                  neutral='分层中性文本', last_layer='重复末层组织文本')


def read(path):
    return json.loads(path.read_text())


def prepare(config, banks, output):
    cfg = load_config(config)
    torch.set_num_threads(cfg.cpu_threads)
    if (cfg.method != 'mean_teacher' or cfg.slice_context != 'adaptive'
            or cfg.internal_text != 'none' or cfg.prompt_source != 'none'
            or cfg.region_weight or cfg.pseudo_dice_weight or cfg.text_alignment_weight):
        raise ValueError('Start from the unmodified adaptive-context MT config')
    banks, output = Path(banks).resolve(), Path(output).resolve()
    metadata = {s:load_hierarchical_bank(banks/(s+'.pt'),s)[1] for s in ('text','neutral')}
    if metadata['text']['provenance'] != metadata['neutral']['provenance']:
        raise ValueError('Paired banks must have identical pretrained provenance')
    output.mkdir(parents=True,exist_ok=False)
    (output/'banks').mkdir()
    (output/'configs').mkdir()
    for source in ('text','neutral'):
        shutil.copy2(banks/(source+'.pt'),output/'banks'/(source+'.pt'))
    parameters = {}
    reference_visual, reference_all = None, None
    for arm in ARMS:
        source = 'none' if arm=='context' else 'neutral' if arm=='neutral' else 'text'
        current = replace(cfg,output_dir=str(output/'runs'/arm),internal_text=source,
            text_bank='' if source=='none' else str(output/'banks'/(source+'.pt')),
            text_fusion_layout='e4' if source=='none' else 'hierarchical',
            text_layer_mode='last' if arm=='last_layer' else 'grouped',
            text_alignment_weight=0. if source=='none' else .1,
            text_alignment_balance='pixel' if source=='none' else 'class').validate()
        (output/'configs'/(arm+'.yaml')).write_text(yaml.safe_dump(current.to_dict(),sort_keys=False))
        net = StudentTeacherSystem(current).students[0]
        named = {k:v.detach().clone() for k,v in net.named_parameters()}
        visual = {k:v for k,v in named.items() if not k.startswith(('text_pyramid.','global_alignment.'))}
        if reference_visual is None:
            reference_visual = visual
        if not all(torch.equal(v,reference_visual[k]) for k,v in visual.items()):
            raise ValueError('Visual initialization mismatch')
        if source!='none':
            if reference_all is None:
                reference_all = named
            if not all(torch.equal(v,reference_all[k]) for k,v in named.items()):
                raise ValueError('Text-arm initialization mismatch')
        parameters[arm] = sum(p.numel() for p in net.parameters())
    code = output/'code'
    for folder in ('freqssl','experiments','tests'):
        shutil.copytree(ROOT/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train.py','evaluate.py','spider_volume.py'):
        shutil.copy2(ROOT/name,code/name)
    write_json(output/'protocol.json',dict(
        arms=list(ARMS),steps_per_arm=cfg.max_steps,eval_every=cfg.eval_every,seed=cfg.seed,
        initialization='fresh, identical visual parameters; all text-arm parameters equal',
        initialization_checked=True,parameters=parameters,
        data='unchanged label budget and all complete native validation volumes',
        selection='best teacher1 case-weighted foreground Mean Dice; earlier step on ties',
        fusion='E4/D3/D2; text layers 9-12/5-8/1-4; shared K/V; dim64',
        loss='original MT plus 0.1 class-balanced soft tissue CE for all text arms',
        global_alignment_enabled=False,global_head_used=False,
        threshold_vs_context_pp=.3,threshold_vs_neutral_pp=.1,threshold_vs_last_layer_pp=.1,
        union_dice_must_not_decrease=True,confirmation='single-seed exploratory screen only',
        required_followup='multiple seeds and no-text tissue-head control for attribution',
        text_bank_provenance=metadata,test_evaluated=False))
    paths = [output/'protocol.json',*[p for parent in (code,output/'configs',output/'banks')
                                   for p in sorted(parent.rglob('*')) if p.is_file()]]
    write_json(output/'frozen_sources.json',{str(p.relative_to(output)):file_sha256(p) for p in paths})
    write_json(output/'status.json',dict(status='prepared',training_started=False,arms=list(ARMS)))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_name(label):
    return '背景' if label==0 else '椎管' if label==10 else f'椎体{label}' if label<10 else f'椎间盘{label-10}'


def percent(value):
    return None if value is None else 100*value


def initialize_device(device):
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable')
        # reset_peak_memory_stats does not create a CUDA context in torch 2.0.
        # Force allocation on the selected device before calling the allocator API.
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def decision(arms,protocol):
    candidate = arms['hierarchical']['overall']
    comparisons = {}
    for control,threshold in (('context',protocol['threshold_vs_context_pp']),
                              ('neutral',protocol['threshold_vs_neutral_pp']),
                              ('last_layer',protocol['threshold_vs_last_layer_pp'])):
        other = arms[control]['overall']
        delta = 100*(candidate['dice']-other['dice'])
        union_ok = candidate['dice_union_present'] >= other['dice_union_present']
        comparisons[control] = dict(delta_dice_pp=delta,threshold_pp=threshold,
                                    union_non_decreasing=union_ok,passed=delta>=threshold and union_ok)
    return dict(comparisons=comparisons,passed_single_seed_screen=all(v['passed'] for v in comparisons.values()),
                confirmatory_efficacy_claim=False)


def summarize(output):
    from experiments.run_slice_context import compact_run, selected_validation, paired_cases
    from spider_volume import aggregate_cases
    arms, scores, overall, classes, cases, tissues, curves = {}, {}, [], [], [], [], []
    diagnostics = []
    split_reference = case_reference = None
    for arm in ARMS:
        directory = output/'runs'/arm
        if not (directory/'result.json').exists() or read(directory/'result.json')['status']!='completed':
            continue
        current = compact_run(directory)
        _,score = selected_validation(directory)
        if aggregate_cases(score['cases']) != {k:score[k] for k in ('overall','per_class','per_case')}:
            raise ValueError('Saved aggregate metrics mismatch')
        split = read(directory/'split.json')
        case_ids = [v['case'] for v in score['cases']]
        if split_reference is None:
            split_reference,case_reference = split,case_ids
        if split!=split_reference or case_ids!=case_reference:
            raise ValueError('Arms have different training or validation cases')
        saved = torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        gates = saved['model'].get('teachers.0.text_pyramid.gates')
        current['teacher_text_gates'] = None if gates is None else gates.tanh().tolist()
        current['trainable_parameters'] = read(directory/'manifest.json')['trainable_parameters']
        arms[arm],scores[arm] = current,score
        final = read(directory/'result.json')['final_validation']['teacher1']['overall']
        current['final_overall'] = final
        overall.append(dict(arm=arm,completed_steps=current['step'],selected_step=current['best_step'],
            mean_dice_pct=percent(score['overall']['dice']),mean_iou_pct=percent(score['overall']['iou']),
            union_dice_pct=percent(score['overall']['dice_union_present']),
            final_step_dice_pct=percent(final['dice']),final_step_iou_pct=percent(final['iou']),
            final_step_union_dice_pct=percent(final['dice_union_present']),valid_cases=score['overall']['cases'],
            parameters=current['trainable_parameters']))
        for row in score['per_class']:
            classes.append(dict(arm=arm,class_id=row['class_id'],label=class_name(row['class_id']),
                dice_pct=percent(row['dice']),iou_pct=percent(row['iou']),
                union_dice_pct=percent(row['dice_union_present']),
                gt_present_cases=row['gt_present_cases'],missed_gt_cases=row['missed_gt_cases'],
                fp_absent_gt_cases=row['fp_absent_gt_cases']))
        background=[]
        for item in score['cases']:
            cm=np.asarray(item['anatomy']['confusion_gt_rows_pred_columns'],dtype=np.int64)
            gt,pred,tp=int(cm[0].sum()),int(cm[:,0].sum()),int(cm[0,0])
            if gt:
                background.append((2*tp/(gt+pred),tp/(gt+pred-tp)))
        classes.append(dict(arm=arm,class_id=0,label=class_name(0),
            dice_pct=100*float(np.mean([v[0] for v in background])) if background else None,
            iou_pct=100*float(np.mean([v[1] for v in background])) if background else None,
            union_dice_pct=None,gt_present_cases=len(background),missed_gt_cases=None,fp_absent_gt_cases=None))
        for row in score['per_case']:
            cases.append(dict(arm=arm,case=row['case'],dice_pct=percent(row['dice']),iou_pct=percent(row['iou']),
                              union_dice_pct=percent(row['dice_union_present'])))
        for tissue in ('vertebra','canal','disc'):
            values=[c['anatomy']['tissue_dice'][tissue] for c in score['cases']]
            values=[v for v in values if v is not None]
            tissues.append(dict(arm=arm,tissue=tissue,dice_pct=100*float(np.mean(values)) if values else None,
                                gt_present_cases=len(values)))
        for p in sorted(directory.glob('valid_step_*.json'),key=lambda v:int(v.stem.split('_')[-1])):
            v=read(p)['teacher1']['overall']
            curves.append(dict(arm=arm,step=int(p.stem.split('_')[-1]),mean_dice_pct=percent(v['dice']),
                               mean_iou_pct=percent(v['iou']),union_dice_pct=percent(v['dice_union_present'])))
        cfg = read(directory/'config.json')
        with (directory/'metrics.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('kind') != 'train' or not (row['step'] == 1 or row['step'] % cfg['log_every'] == 0):
                    continue
                entry = dict(arm=arm,step=row['step'],loss=row['loss'],
                    text_alignment_loss=row.get('text_alignment_loss'),
                    pseudo_coverage=row.get('coverage_s1'),unsupervised_weight=row['unsupervised_weight'])
                for stage,name in enumerate(('E4','D3','D2')):
                    entry[name+'_gate'] = row.get('text_gates',[None]*3)[stage]
                    entry[name+'_alignment_loss'] = row.get('text_alignment_loss_per_stage',[None]*3)[stage]
                    for k in range(4):
                        entry[f'{name}_layer_weight_{k+1}'] = row.get('text_layer_mixtures',[[None]*4]*3)[stage][k]
                        entry[f'{name}_tissue_fraction_{k}'] = row.get('text_tissue_prediction_fractions',[[None]*4]*3)[stage][k]
                diagnostics.append(entry)
    complete=len(arms)==len(ARMS)
    result=dict(status='completed' if complete else 'partial',arms=arms,test_evaluated=False,multiple_seeds=False,
                metric_policy='GT-present foreground class mean per case, then equal case mean')
    if complete:
        result['decision']=decision(arms,read(output/'protocol.json'))
        result['paired_cases']={a:paired_cases(scores['hierarchical'],scores[a]) for a in ARMS if a!='hierarchical'}
    write_json(output/'results.json',result)
    if overall:
        report=output/'metrics_report'
        report.mkdir(exist_ok=True)
        for name,rows in (('overall',overall),('per_class',classes),('per_case',cases),('merged_tissue',tissues),
                          ('validation_history',curves),('training_diagnostics',diagnostics)):
            write_csv(report/(name+'.csv'),rows)
        if complete:
            deltas = [dict(control=control,case=case,dice_delta_pp=delta)
                      for control,comparison in result['paired_cases'].items()
                      for case,delta in comparison['delta_pp'].items()]
            write_csv(report/'paired_case_deltas.csv',deltas)
        lines=['# 分层文本融合验证结果','',f'状态：{result["status"]}；完成{len(arms)}/4组。',
               '单种子完整valid；使用各组最佳EMA教师，未评估test。',
               '主Mean Dice先在病例内平均GT存在的前景类别，再对病例等权平均；背景不参与。','',
               '| 组别 | 最佳步 | Mean Dice % | Mean IoU % | Union Dice % | 最后一步Dice % |',
               '|---|---:|---:|---:|---:|---:|']
        for row in overall:
            lines.append(f'| {ARM_LABELS[row["arm"]]} | {row["selected_step"]} | {row["mean_dice_pct"]:.4f} | '
                         f'{row["mean_iou_pct"]:.4f} | {row["union_dice_pct"]:.4f} | {row["final_step_dice_pct"]:.4f} |')
        lines += ['','## 最后一步结果','',
                  '| 组别 | 步数 | Mean Dice % | Mean IoU % | Union Dice % |',
                  '|---|---:|---:|---:|---:|']
        for row in overall:
            lines.append(f'| {ARM_LABELS[row["arm"]]} | {row["completed_steps"]} | {row["final_step_dice_pct"]:.4f} | '
                         f'{row["final_step_iou_pct"]:.4f} | {row["final_step_union_dice_pct"]:.4f} |')
        lines += ['','## 逐类结果','','每格为Dice / IoU（%），N为GT存在该类别的病例数。相对编号不代表已核实的绝对椎体解剖名称。',
                  '','| ID | 类别 | N | '+' | '.join(ARM_LABELS[a] for a in arms)+' |',
                  '|---:|---|---:|'+''.join('---:|' for _ in arms)]
        lookup={(v['arm'],v['class_id']):v for v in classes}
        for c in range(20):
            rows=[lookup[a,c] for a in arms]
            cells=['—' if r['dice_pct'] is None else f'{r["dice_pct"]:.2f} / {r["iou_pct"]:.2f}' for r in rows]
            lines.append(f'| {c} | {class_name(c)} | {rows[0]["gt_present_cases"]} | '+' | '.join(cells)+' |')
        if complete:
            lines += ['',f'预声明筛查通过：{result["decision"]["passed_single_seed_screen"]}。',
                      '即使筛查通过，仍需多随机种子及无文本组织辅助头对照，才能归因于文本语义和分层设计。']
            for control,comparison in result['decision']['comparisons'].items():
                lines.append(f'- 分层组织文本相对{ARM_LABELS[control]}：{comparison["delta_dice_pp"]:+.4f}个百分点；'
                             f'Union不下降：{comparison["union_non_decreasing"]}；该对照通过：{comparison["passed"]}。')
        lines += ['','逐病例、合并组织、训练诊断和验证曲线见同目录CSV；缺失指标留空，未计算HD95/ASD。']
        (report/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def execute(output,device):
    if ROOT!=output/'code':
        raise RuntimeError('Execute the frozen runner')
    if any(file_sha256(output/p)!=h for p,h in read(output/'frozen_sources.json').items()):
        raise ValueError('Frozen protocol/source/config/bank changed')
    started=datetime.now(timezone.utc).isoformat()
    with (output/'screen.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for arm in ARMS:
                cfg=load_config(output/'configs'/(arm+'.yaml'))
                directory=Path(cfg.output_dir)
                if (directory/'result.json').exists() and read(directory/'result.json')['status']=='completed':
                    continue
                checkpoint=directory/'last.pt'
                write_json(output/'status.json',dict(status='running',arm=arm,pid=os.getpid(),device=device,started_utc=started))
                initialize_device(device)
                run_experiment(cfg,device,resume=checkpoint if checkpoint.exists() else None)
                summarize(output)
                gc.collect()
                if torch.device(device).type=='cuda':
                    torch.cuda.empty_cache()
            result=summarize(output)
            write_json(output/'status.json',dict(status='completed',decision=result['decision']))
            write_json(output/'exit.json',dict(exit_code=0,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat()))
        except BaseException as error:
            write_json(output/'status.json',dict(status='failed',arm=arm,error=repr(error)))
            write_json(output/'exit.json',dict(exit_code=1,error=repr(error),finished_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v3/context_efficiency_reference.yaml')
    parser.add_argument('--banks',type=Path,default=ROOT/'outputs/hierarchical_text_assets_20261108')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--execute-frozen',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.summarize_only:
        summarize(output)
        return
    if args.execute_frozen:
        execute(output,args.device)
        return
    if not args.prepare_only and args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
    if not args.resume:
        prepare(args.config,args.banks,output)
    if not (output/'frozen_sources.json').exists():
        raise ValueError('No prepared screen')
    if not args.prepare_only:
        subprocess.run([sys.executable,'-u',str(output/'code/experiments/run_hierarchical_text.py'),
                        '--output',str(output),'--device',args.device,'--execute-frozen'],check=True,cwd=output/'code')
    else:
        print(f'Prepared: {output}',flush=True)


if __name__=='__main__':
    main()
