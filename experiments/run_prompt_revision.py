"""Frozen, fail-closed teacher screening followed by five-arm student ablation."""
from __future__ import annotations
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from freqssl.config import load_config,ExperimentConfig
from freqssl.data import build_training_data
from freqssl.engine import write_json
from freqssl.prompt_supervision import file_sha256
from freqssl.prompt_revision import make_partition,prompt_bank,teacher_screen,choose_teacher
from experiments.run_prompt_context import Process,done,resume_checkpoint
from experiments.run_slice_context import compact_run,selected_validation,paired_cases,benchmark_checkpoint

DEFAULT_OUTPUT=ROOT/'outputs/spine_prompt_revision_20261104'
HISTORY=ROOT/'outputs/spine_prompt_context_20261103'


def save_config(path,cfg):
    value=yaml.safe_dump(cfg.to_dict(),sort_keys=False)
    if path.exists() and path.read_text()!=value:raise ValueError(f'Frozen configuration changed: {path}')
    path.write_text(value)


def freeze(output,config,history=HISTORY):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);dest=output/'code'
    if dest.exists():raise FileExistsError(dest)
    cfg=replace(load_config(config),output_dir=str(output))
    _,_,split=build_training_data(cfg);partition=make_partition(split)
    if len(partition['labeled_files'])!=1223 or len(partition['calibration_files'])!=273:raise ValueError('Unexpected SPIDER budget')
    write_json(output/'calibration_split.json',partition)
    save_config(output/'full.yaml',cfg)
    save_config(output/'pilot.yaml',replace(cfg,training_split=str(output/'calibration_split.json')))
    for folder in ('freqssl','prompt_teacher','experiments','configs','tests','docs'):
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train.py','evaluate.py','spider_volume.py'):shutil.copy2(ROOT/name,dest/name)
    hashes={str(p.relative_to(dest)):file_sha256(p) for p in dest.rglob('*') if p.is_file()}
    write_json(output/'frozen_sources.json',hashes)
    protocol={'revision':2,'config':cfg.to_dict(),'history':str(history.resolve()),'teacher_variants':['T0','T1','T2'],
        'student_arms':['S0','S1','S2','S3','S4'],'prompt_bank_text':prompt_bank('text'),'prompt_bank_neutral':prompt_bank('neutral'),
        'teacher_steps':6000,'teacher_batch':4,'paired_views':2,'js_weight':.1,'anchor_weight':.1,
        'calibration_cases':partition['calibration_cases'],'calibration_slices':273,'pilot_labeled_slices':1223,
        'calibration_split_sha256':file_sha256(output/'calibration_split.json'),
        'teacher_screen':{'canonical_noninferior':True,'max_tissue_drop':.005,'max_heldout_prompt_drop':.01,
            'minimum_foreground_coverage':.001,'T2_minimum_advantage_over_T1':.001},
        'competence':{'minimum_cases':5,'minimum_pixels':1000,'fg_accuracy':.95,'bg_accuracy':.99,'minimum_advantage':.02,
            'case_bootstrap_lower_above':0,'bootstrap_samples':2000,'bootstrap_seed':42,'geometry_epochs':[0,1,2,3]},
        'student_gates':{'S3-S0':.003,'S3-S4':.001,'S3-S2':.001,'union_non_decreasing':True},
        'full_refit_gate_transfer':'Pilot gates are frozen; no post-refit recalibration on now-trained cases.',
        'test_evaluated':False,'automatic_confirmation':False,'stop_if_teacher_screen_fails':True}
    write_json(output/'protocol.json',protocol)
    oldcfg=ExperimentConfig(**json.loads((history/'students/context/config.json').read_text())).to_dict()
    now=cfg.to_dict();oldcfg.pop('output_dir');now.pop('output_dir')
    if oldcfg!=now:raise ValueError('S0 protocol cannot be reused')
    write_json(output/'S0_reference.json',{'directory':str(history/'students/context'),
        'best_sha256':file_sha256(history/'students/context/best.pt'),'config_matches_except_output':True})
    diagnostic=Path('/tmp/spine_prompt_diagnostic_20261104_v5p3lell')
    if diagnostic.exists():
        for p in diagnostic.rglob('*.json'):
            target=output/'prior_diagnostics'/p.relative_to(diagnostic);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    return dest


def summarize_arm(directory,device):
    arm=compact_run(directory);_,valid=selected_validation(directory);rows=valid['cases']
    arm['anatomy']={k:float(np.mean([r['anatomy']['tissue_dice'][k] for r in rows if r['anatomy']['tissue_dice'][k] is not None])) for k in ('vertebra','disc','canal')}
    arm['anatomy']['same_tissue_wrong_level_fraction']=sum(r['anatomy']['same_tissue_wrong_level_pixels'] for r in rows)/sum(r['anatomy']['numbered_gt_pixels'] for r in rows)
    arm['absent_class_false_predictions']=[{'case':r['case'],'class_id':c['class_id'],'pred_voxels':c['pred_voxels']} for r in rows for c in r['classes'] if c['false_positive_absent_class']]
    metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    metrics=[m for m in metrics if m.get('kind')=='train' and m['step']>1800]
    arm['supervision_after_ramp']={key:float(np.mean([m[key] for m in metrics if key in m])) for key in ('coverage_s1','fine_coverage_s1','tissue_coverage_s1','prompt_reliable_coverage_s1','prompt_conflict_coverage_s1') if any(key in m for m in metrics)}
    path=directory/'volume_benchmark.json'
    arm['inference']=json.loads(path.read_text()) if path.exists() else benchmark_checkpoint(directory/'best.pt',device,path)
    arm['inference'].pop('metrics',None)
    return arm,valid


def student_decisions(arms,neutral_gate):
    def passes(ref,delta):
        a,b=arms['S3']['overall'],arms[ref]['overall']
        return a['dice']-b['dice']>=delta and a['dice_union_present']>=b['dice_union_present']
    active=any(any(any(row) for row in table) for table in neutral_gate['tables'].values())
    base,semantics,gate=passes('S0',.003),passes('S4',.001),passes('S2',.001)
    return {'improves_context':base,'text_pipeline_advantage':semantics,'neutral_gate_nonzero':active,
        'text_semantics_worth_further_validation':semantics and active,'competence_contribution':gate,
        'complete_design_passed':base and semantics and active and gate,'confirmatory_efficacy_claim':False}


def summarize(output,device):
    protocol=json.loads((output/'protocol.json').read_text());history=Path(protocol['history']);arms={};valid={}
    for name in ('S0','S1','S2','S3','S4'):
        directory=history/'students/context' if name=='S0' else output/'students'/name
        arms[name],valid[name]=summarize_arm(directory,device)
    neutral=json.loads((output/'gates/neutral/gate.json').read_text());decision=student_decisions(arms,neutral)
    _,oldtext=selected_validation(history/'students/text')
    comparisons={f'{a}-{b}':paired_cases(valid[a],valid[b]) for a,b in [('S1','S0'),('S2','S1'),('S3','S2'),('S3','S0'),('S3','S4')]}
    comparisons['S1-legacy_text']=paired_cases(valid['S1'],oldtext)
    phases={}
    for p in output.glob('teachers/*/result.json'):phases[str(p.relative_to(output))]=json.loads(p.read_text())
    for p in output.glob('caches/*/manifest.json'):
        m=json.loads(p.read_text());phases[str(p.relative_to(output))]={k:m[k] for k in ('seconds','peak_allocated_bytes','coverage','teacher_step')}
    result={'status':'completed','arms':arms,'comparisons':comparisons,'decision':decision,'phase_costs':phases,
        'teacher_screen':json.loads((output/'teacher_screen.json').read_text()),'test_evaluated':False,'multi_seed_confirmation_performed':False}
    write_json(output/'results.json',result)
    lines=['# 提示稳健教师与能力门控：第二轮结果','','单种子 valid 筛查，非确认性结论。','',
        '| 组别 | 最佳步 | 主 Dice% | union-present Dice% |','|---|---:|---:|---:|']
    for k,v in arms.items():lines.append(f'| {k} | {v["best_step"]} | {100*v["overall"]["dice"]:.4f} | {100*v["overall"]["dice_union_present"]:.4f} |')
    lines+=['',f'完整设计通过：{decision["complete_design_passed"]}。',
        '分模块判断、病例差值、组织 Dice、编号错误、缺失类别误检、覆盖率与开销详见 results.json。',
        '门控来源于独立病例校准运行，正式全标签重训后未重新校准；不是可靠性保证。',
        '若中性门控全部关闭，文本与中性差异仅代表流程差异。未访问 test，未自动追加预算。']
    (output/'report.md').write_text('\n'.join(lines)+'\n');write_json(output/'status.json',dict(status='completed',decision=decision,test_evaluated=False))


def pipeline(args):
    output=args.output.resolve()
    if ROOT.resolve()!=(output/'code').resolve():raise ValueError('Formal pipeline requires frozen code')
    hashes=json.loads((output/'frozen_sources.json').read_text())
    if any(file_sha256(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Frozen code changed')
    protocol=json.loads((output/'protocol.json').read_text());history=Path(protocol['history'])
    compatibility=json.loads((output/'preflight/compatibility.json').read_text())
    if not compatibility['split_exact'] or compatibility['student_ema_updates_exact_at_steps']!=[0,601,1801]:raise ValueError('Missing S0 compatibility evidence')
    s0=json.loads((output/'S0_reference.json').read_text())
    if file_sha256(history/'students/context/best.pt')!=s0['best_sha256']:raise ValueError('S0 checkpoint changed')
    for receipt in output.glob('process_*.json'):
        p=json.loads(receipt.read_text());cmd=Path(f'/proc/{p["pid"]}/cmdline')
        if cmd.exists() and str(output).encode() in cmd.read_bytes():raise RuntimeError(f'Existing child: {p["stage"]}')
    logs=output/'logs';logs.mkdir(exist_ok=True)
    full=load_config(output/'full.yaml');pilot=load_config(output/'pilot.yaml');reference=output/'reference_context'
    pending=None
    def launch(command,stage,gpu):
        write_json(output/'status.json',dict(status='running',stage=stage,pid=os.getpid(),test_evaluated=False))
        return Process(command,logs/f'{stage}.log',gpu,stage,output)
    def student(name,cfg):
        directory=Path(cfg.output_dir)
        if done(directory):return None
        config_path=output/f'{name}.yaml';save_config(config_path,cfg)
        cmd=[args.student_python,'train.py','--config',str(config_path),'--device','cuda:0']
        checkpoint=resume_checkpoint(directory)
        if checkpoint:cmd+=['--resume',str(checkpoint)]
        return launch(cmd,name,args.student_gpu)
    def teacher(name,cfg,source,variant):
        directory=output/'teachers'/name
        if done(directory):return directory
        config_path=output/('pilot.yaml' if cfg.training_split else 'full.yaml')
        cmd=[args.teacher_python,'-m','prompt_teacher.revision','fit','--config',str(config_path),'--source',source,
             '--variant',variant,'--output',str(directory),'--device','cuda:0']
        checkpoint=resume_checkpoint(directory)
        if checkpoint:cmd+=['--resume',str(checkpoint)]
        launch(cmd,'teacher_'+name,args.teacher_gpu).wait();return directory
    def fresh_stage(path):
        if path.exists() and any(path.iterdir()):
            rec=output/'recovery';rec.mkdir(exist_ok=True);path.rename(rec/f'{path.name}_{time.time_ns()}')
    def cache(name,cfg,source,variant,teacher_dir,calibration=False):
        directory=output/('calibration_caches' if calibration else 'caches')/name
        if done(directory,'manifest.json'):return directory
        fresh_stage(directory)
        cmd=[args.teacher_python,'-m','prompt_teacher.revision','calibration-cache' if calibration else 'cache',
            '--config',str(output/('pilot.yaml' if cfg.training_split else 'full.yaml')),'--source',source,'--variant',variant,
            '--checkpoint',str(teacher_dir/'best.pt'),'--output',str(directory),'--device','cuda:0']
        launch(cmd,'cache_'+name,args.teacher_gpu).wait();return directory
    def calibrate(name,source,variant,cache_dir):
        directory=output/'gates'/name
        if not done(directory):
            fresh_stage(directory)
            cmd=[args.student_python,'-m','prompt_teacher.calibration','--config',str(output/'pilot.yaml'),
                '--reference',str(reference),'--cache',str(cache_dir),'--source',source,'--variant',variant,
                '--output',str(directory),'--device','cuda:0']
            launch(cmd,'gate_'+name,args.student_gpu).wait()
        return json.loads((directory/'gate.json').read_text())
    try:
        pending=student('reference_context',replace(pilot,output_dir=str(reference),checkpoint_every=500))
        robustness={};gates={};validation={}
        for variant in ('T0','T1','T2'):
            td=teacher('pilot_'+variant,pilot,'text',variant)
            cc=cache(variant,pilot,'text',variant,td,True)
            if pending:pending.wait();pending=None
            gates[variant]=calibrate(variant,'text',variant,cc)
            robustness[variant]=json.loads((cc/'robustness.json').read_text())
            validation[variant]=json.loads((td/'result.json').read_text())['best_valid_tissue_dice']
        screens={v:teacher_screen(robustness['T0'],robustness[v],gates[v]) for v in ('T1','T2')}
        selected=choose_teacher(screens,validation)
        screening=dict(status='completed',selected=selected,checks=screens,robustness=robustness,valid_scores=validation,
            gates={v:{'foreground_coverage':gates[v]['mean_foreground_guidance_coverage'],'path':str(output/'gates'/v/'gate.json')} for v in gates})
        write_json(output/'teacher_screen.json',screening)
        if selected is None:
            result=dict(status='completed',outcome='teacher_screen_failed',teacher_screen=screening,
                student_experiments_started=False,test_evaluated=False)
            write_json(output/'results.json',result);write_json(output/'status.json',result)
            lines=['# 第二轮教师筛查结果','','T1、T2 均未通过预先固定筛查条件，按计划停止。未启动 S1–S4 学生实验。','',
                '| 教师 | valid 组织 Dice% | 校准原句 Dice% | 留出句式 4 Dice% | 留出句式 5 Dice% | 有效前景覆盖% |','|---|---:|---:|---:|---:|---:|']
            for v in ('T0','T1','T2'):
                r=robustness[v];lines.append(f'| {v} | {100*validation[v]:.4f} | {100*r["canonical"]["dice"]:.4f} | {100*r["heldout_4"]["dice"]:.4f} | {100*r["heldout_5"]["dice"]:.4f} | {100*gates[v]["mean_foreground_guidance_coverage"]:.4f} |')
            lines+=['','逐项失败原因见 teacher_screen.json，逐病例和阶段统计见 gates/。未访问 test。']
            (output/'report.md').write_text('\n'.join(lines)+'\n');return
        td=teacher('pilot_neutral',pilot,'neutral',selected)
        cc=cache('neutral',pilot,'neutral',selected,td,True);calibrate('neutral','neutral',selected,cc)
        for source in ('text','neutral'):
            td=teacher('full_'+source,full,source,selected);cache(source,full,source,selected,td)
        for phase in ('pilot','full'):
            a=json.loads((output/'teachers'/('pilot_'+selected if phase=='pilot' else 'full_text')/'provenance.json').read_text())
            b=json.loads((output/'teachers'/f'{phase}_neutral'/'provenance.json').read_text())
            for k in ('foundation_sha256','initial_frozen_state_sha256','initial_adaptation_sha256','labeled_annotation_sha256','labeled_image_sha256'):
                if a[k]!=b[k]:raise ValueError(f'Neutral teacher pairing differs: {phase}/{k}')
        specifications=[('S1','text',history/'caches/text','preserve_ema',''),
            ('S2','text',output/'caches/text','preserve_ema',''),
            ('S3','text',output/'caches/text','competence',str(output/'gates'/selected/'gate.json')),
            ('S4','neutral',output/'caches/neutral','competence',str(output/'gates/neutral/gate.json'))]
        for name,source,cp,route,gate in specifications:
            pending=student(name,replace(full,output_dir=str(output/'students'/name),prompt_source=source,
                prompt_cache=str(cp),prompt_route=route,prompt_gate=gate))
            if pending:pending.wait();pending=None
        launch([args.student_python,'experiments/run_prompt_revision.py','--output',str(output),'--summarize'],
               'summary',args.student_gpu).wait()
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error),pid=os.getpid(),
            reference_pid=pending.process.pid if pending and pending.process.poll() is None else None,test_evaluated=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--config',type=Path,default=ROOT/'configs/v3/prompt_context.yaml');p.add_argument('--history',type=Path,default=HISTORY)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--teacher-python',default=os.environ.get('TEACHER_PYTHON','/home/zhp/.conda/envs/medclipseg/bin/python'))
    p.add_argument('--student-python',default=os.environ.get('STUDENT_PYTHON','/home/zhp/.conda/envs/mamba/bin/python'))
    p.add_argument('--teacher-gpu',default='0');p.add_argument('--student-gpu',default='1');a=p.parse_args()
    if a.freeze_only:print(freeze(a.output,a.config,a.history),flush=True)
    elif a.summarize:summarize(a.output.resolve(),torch.device('cuda:0'))
    else:
        with (a.output/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pipeline(a)

if __name__=='__main__':main()
