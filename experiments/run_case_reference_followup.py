"""Fixed-step follow-up: frozen seed42 diagnostics plus A/B seeds43 and44."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from freqssl.config import ExperimentConfig
from freqssl.engine import run_experiment,write_json
from freqssl.context_diagnostics import bootstrap_cases
from experiments.probe_case_reference import probe

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def csv_write(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):
    return dict(mean=float(np.mean(v)),sample_std=float(np.std(v,ddof=1)) if len(v)>1 else None,n_seeds=len(v))

def decomposition(valid,seed,arm):
    cases=[];classes=[]
    for row in valid['cases']:
        gt=[c for c in row['classes'] if c['gt_voxels']>0]
        absent=[c for c in row['classes'] if not c['gt_voxels'] and c['pred_voxels']>0]
        k,m=len(gt),len(absent);s=sum(c['dice'] for c in gt)
        cases.append(dict(seed=seed,arm=arm,case=row['case'],gt_present_classes=k,absent_gt_predicted_classes=m,
            sum_gt_present_dice=s,primary_dice_percent=100*s/k,union_present_percent=100*s/(k+m),
            absent_class_predicted_voxels=sum(c['pred_voxels'] for c in absent)))
    for j,base in enumerate(valid['cases'][0]['classes']):
        rows=[c['classes'][j] for c in valid['cases']]
        d=[r['dice']*100 for r in rows if r['dice'] is not None]
        u=[r['dice_union_present']*100 for r in rows if r['dice_union_present'] is not None]
        classes.append(dict(seed=seed,arm=arm,class_id=base['class_id'],class_name=base['class_name'],
            dice_percent=float(np.mean(d)) if d else None,union_present_percent=float(np.mean(u)) if u else None,
            gt_present_cases=sum(r['gt_voxels']>0 for r in rows),fp_absent_gt_cases=sum(r['false_positive_absent_class'] for r in rows),
            absent_class_predicted_voxels=sum(r['pred_voxels'] for r in rows if not r['gt_voxels']),
            primary_overall_contribution_pp=sum(100*(r['dice'] or 0)/c['gt_present_classes'] for r,c in zip(rows,cases))/len(cases),
            union_overall_contribution_pp=sum(100*(r['dice_union_present'] or 0)/(c['gt_present_classes']+c['absent_gt_predicted_classes']) for r,c in zip(rows,cases))/len(cases)))
    assert abs(sum(c['primary_overall_contribution_pp'] for c in classes)-valid['overall']['dice']*100)<1e-9
    assert abs(sum(c['union_overall_contribution_pp'] for c in classes)-valid['overall']['dice_union_present']*100)<1e-9
    return cases,classes

def summarize(out):
    manifest=read(out/'manifest.json');base=Path(manifest['baseline']);runs={};case_rows=[];class_rows=[]
    for seed in (42,43,44):
        for arm in ('A','B') if seed!=42 else ('A','B','C'):
            directory=base/f'screen/{arm}' if seed==42 else out/f'training/seed{seed}/{arm}'
            if not (directory/'result.json').exists() or read(directory/'result.json')['status']!='completed': continue
            assert read(directory/'result.json')['step']==6000
            valid=read(directory/'valid_step_6000.json')['teacher1']
            if valid['overall']['cases']!=21: raise ValueError('Incomplete validation')
            runs[f'{seed}/{arm}']=dict(directory=str(directory),overall=valid['overall'],checkpoint='last.pt',step=6000)
            cases,classes=decomposition(valid,seed,arm);case_rows+=cases;class_rows+=classes
    paired={};means={}
    for seed in (42,43,44):
        if not all(f'{seed}/{a}' in runs for a in 'AB'): continue
        values={a:{r['case']:r for r in case_rows if r['seed']==seed and r['arm']==a} for a in 'AB'}
        if values['A'].keys()!=values['B'].keys(): raise ValueError('Pairing mismatch')
        paired[str(seed)]={}
        for metric in ('primary_dice_percent','union_present_percent'):
            d={c:values['B'][c][metric]-values['A'][c][metric] for c in values['A']}
            paired[str(seed)][metric]=dict(bootstrap_cases(d.values()),per_case=d)
    for arm in 'AB':
        rows=[r for k,r in runs.items() if k.endswith('/'+arm)]
        means[arm]={m:stats([100*r['overall'][m] for r in rows]) for m in ('dice','dice_union_present')}
    cluster={}
    for metric in ('primary_dice_percent','union_present_percent'):
        seeds=list(paired)
        cases=paired[seeds[0]][metric]['per_case']
        d={c:float(np.mean([paired[s][metric]['per_case'][c] for s in seeds])) for c in cases}
        cluster[metric]=dict(bootstrap_cases(d.values()),per_case=d,
            per_seed_mean=[paired[s][metric]['mean'] for s in seeds],
            seed_difference_stats=stats([paired[s][metric]['mean'] for s in seeds]))
    complete=len(paired)==3
    result=dict(status='completed' if complete else 'partial',selection='final EMA at fixed step6000; valid development set, not test',
        runs=runs,mean_std=means,paired_by_seed=paired,case_bootstrap_after_seed_average=cluster,test_read=False)
    write_json(out/'fixed_step_results.json',result)
    csv_write(out/'per_class_dice.csv',class_rows);csv_write(out/'union_decomposition.csv',case_rows)
    aggregate_classes=[]
    for arm in 'AB':
        for cid in range(1,20):
            rows=[r for r in class_rows if r['arm']==arm and r['class_id']==cid]
            row=dict(arm=arm,class_id=cid,class_name=rows[0]['class_name'],n_seeds=len(rows))
            for key in ('dice_percent','union_present_percent','primary_overall_contribution_pp','union_overall_contribution_pp'):
                vals=[r[key] for r in rows if r[key] is not None];s=stats(vals) if vals else {'mean':None,'sample_std':None}
                row[key+'_mean']=s['mean'];row[key+'_sample_std']=s['sample_std']
            aggregate_classes.append(row)
    csv_write(out/'per_class_mean_std.csv',aggregate_classes)
    lines=['# 固定步数的病例参考片补充实验','',
        '所有比较使用 6000 步最终 EMA；不使用 best.pt 选择最终成绩。仍为开发验证集，未使用 test。',
        f"当前已完成 A/B 配对种子：{', '.join(paired)}。"+('三个种子已齐。' if complete else '其余种子仍待完成，不能把当前均值视为三种子结论。'),'',
        '| seed | 组 | Dice % | union-present % |','|---|---|---:|---:|']
    for k,r in runs.items():
        seed,arm=k.split('/');lines.append(f"| {seed} | {arm} | {100*r['overall']['dice']:.4f} | {100*r['overall']['dice_union_present']:.4f} |")
    if complete:
        lines+=['','| 组 | 主 Dice 均值±样本标准差 % | union-present 均值±样本标准差 % |','|---|---:|---:|']
        for arm,v in means.items():
            lines.append(f"| {arm} | {v['dice']['mean']:.4f} ± {v['dice']['sample_std']:.4f} | {v['dice_union_present']['mean']:.4f} ± {v['dice_union_present']['sample_std']:.4f} |")
        for metric,v in cluster.items(): lines.append(f"\nB−A {metric}：{v['mean']:+.4f} pp，病例 bootstrap 95% 区间 {v['ci95']}。")
    lines+=['','病例 bootstrap 先在同一病例内平均配对种子差值，再重采样 21 个病例；不把 63 个病例×种子当作独立样本。种子标准差使用 ddof=1。',
        '逐类结果见 per_class_dice.csv、per_class_mean_std.csv；病例级构成见 union_decomposition.csv。',
        '对每例，K 为 GT 存在类别数，M 为 GT 缺失但有预测的类别数，S 为 GT 存在类别 Dice 之和：主 Dice=S/K，union-present=S/(K+M)。空 GT 且空预测不计；缺失类别的任何非空预测计零分。',
        'CSV 中各类 overall_contribution_pp 的和精确等于病例等权整体指标，不能用逐类均值的简单平均代替整体指标。','']
    (out/'report.md').write_text('\n'.join(lines))
    return result

def prepare(out,baseline,device):
    if out.exists(): raise FileExistsError(out)
    old=read(baseline/'manifest.json')
    for p,h in old['source_hashes'].items():
        if sha(baseline/'code'/p)!=h: raise ValueError(f'Baseline source changed: {p}')
    shutil.copytree(baseline/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('experiments/probe_case_reference.py','experiments/run_case_reference_followup.py',
                     'tests/test_case_reference_followup.py','docs/case_reference_followup_20261116.md'):
        shutil.copy2(ROOT/relative,out/'code'/relative)
    configs={}
    for seed in (43,44):
        for arm in 'AB':
            cfg=ExperimentConfig(**read(baseline/f'screen/{arm}/config.json'))
            cfg=replace(cfg,seed=seed,output_dir=str(out/f'training/seed{seed}/{arm}')).validate()
            configs[f'{seed}/{arm}']=cfg.to_dict()
    files=[p for p in (out/'code').rglob('*') if p.is_file()]
    origins={f'{a}/{kind}':sha(baseline/f'screen/{a}/{kind}.pt') for a in 'ABC' for kind in ('best','last')}
    m=dict(baseline=str(baseline),device=device,configs=configs,
        baseline_checkpoint_hashes=origins,source_hashes={str(p.relative_to(out/'code')):sha(p) for p in files},
        fixed_steps=6000,seeds=[42,43,44],selection='last.pt EMA only',test_read=False,
        repeated_control_seeds=[1701,1702,1703],note='Extension to previously failed screen explicitly requested by user; no gate and no 30000-step runs')
    write_json(out/'manifest.json',m);write_json(out/'status.json',dict(status='prepared'))
    summarize(out)

def worker(out):
    m=read(out/'manifest.json');baseline=Path(m['baseline']);device=m['device']
    if ROOT!=out/'code': raise ValueError('Use frozen worker')
    if read(out/'status.json')['status']!='prepared': raise ValueError('Already started')
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        for p,h in m['source_hashes'].items():
            if sha(ROOT/p)!=h: raise ValueError(f'Changed source {p}')
        for key,h in m['baseline_checkpoint_hashes'].items():
            arm,kind=key.split('/')
            if sha(baseline/f'screen/{arm}/{kind}.pt')!=h: raise ValueError('Changed input checkpoint')
        with (out/'preflight.log').open('w') as f:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_case_reference_followup.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        for arm in 'BC':
            for kind in ('best','last'):
                write_json(out/'status.json',dict(status='diagnostics',active=f'{arm}/{kind}',pid=os.getpid()))
                s=probe(baseline/f'screen/{arm}/{kind}.pt',out/f'diagnostics/{arm}_{kind}',device,kind=='last')
                step=s['loading']['checkpoint_step'];v=read(baseline/f'screen/{arm}/valid_step_{step}.json')['teacher1']['overall']
                for metric in ('dice','dice_union_present'):
                    if abs(s['metrics']['full']['overall'][metric]-v[metric])>1e-10: raise ValueError('Diagnostic baseline mismatch')
        # Keep the already logged student trajectory distinct from saved EMA values.
        trajectories=[];saved=[]
        for arm in 'ABC':
            for line in (baseline/f'screen/{arm}/metrics.jsonl').read_text().splitlines():
                r=json.loads(line)
                if 'context_gamma' in r: trajectories.append(dict(arm=arm,network='student',step=r['step'],**{f'gamma_stride{s}':g for s,g in zip((4,8,16),r['context_gamma'])}))
            for kind in ('best','last'):
                c=torch.load(baseline/f'screen/{arm}/{kind}.pt',map_location='cpu',weights_only=True)
                saved.append(dict(arm=arm,checkpoint=kind,step=c['step'],network='EMA teacher',
                    **{f'gamma_stride{s}':float(c['model'][f'teachers.0.context_fusion.{i}.gamma']) for i,s in enumerate((4,8,16))}))
        csv_write(out/'gamma_student_trajectory.csv',trajectories);csv_write(out/'gamma_saved_ema.csv',saved)
        write_json(out/'diagnostics_status.json',dict(status='completed',validation_cases=21,baseline_predictions_reproduced=True))
        for name,values in m['configs'].items():
            write_json(out/'status.json',dict(status='training',active=name,pid=os.getpid()))
            cfg=ExperimentConfig(**values).validate()
            run_experiment(cfg,device)
            # Final checkpoints are evaluated by the unchanged frozen training engine.
            summarize(out)
        write_json(out/'status.json',dict(status='completed',exit_code=0))
    except BaseException as e:
        write_json(out/'status.json',dict(status='failed',error=repr(e),exit_code=1));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'outputs/case_reference_20261112');p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args();out=a.output_dir.resolve()
    if a.worker: worker(out);return
    prepare(out,a.baseline.resolve(),a.device)
    if a.execute:
        command=[sys.executable,str(out/'code/experiments/run_case_reference_followup.py'),'--output-dir',str(out),'--worker']
        with (out/'run.log').open('w') as f:
            process=subprocess.Popen(command,cwd=out/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(out/'launch.json',dict(pid=process.pid,command=command,CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),started_unix=time.time()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(out))))
    else: print(json.dumps(dict(status='prepared',output=str(out))))

if __name__=='__main__': main()
