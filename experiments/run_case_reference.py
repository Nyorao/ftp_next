"""Prepare by default; explicitly launch a frozen A/B/C screen and gated confirmation."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from freqssl.config import ExperimentConfig,load_config
from freqssl.engine import run_experiment,write_json,seed_everything
from freqssl.data import build_training_data
from freqssl.evaluation import prepare_validation
from experiments.run_slice_context import compact_run,selected_validation,paired_cases
from experiments.analyze_case_reference import analyze,paired_analysis,combine_seed_pairs

ARMS={'A':('none','none'),'B':('middle','none'),'C':('none','relative')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_candidate(overall):
    base=overall['A']
    passed=[a for a in ('B','C') if overall[a]['dice']-base['dice']>=.003 and
            overall[a]['dice_union_present']>=base['dice_union_present']]
    # C has the lower declared forward compute; deterministic exact-score tie break.
    return max(passed,key=lambda a:(overall[a]['dice'],a=='C')) if passed else None


def confirmed(dice,union):
    return len(dice)==len(union)==3 and np.mean(dice)>=.3 and sum(v>0 for v in dice)>=2 and np.mean(union)>=0


def config_for(base,arm,directory,seed=42,steps=6000):
    reference,position=ARMS[arm]
    return replace(base,case_reference=reference,context_position=position,seed=seed,
        output_dir=str(directory),max_steps=steps,eval_every=500 if steps==6000 else 1000,
        warmup_steps=600 if steps==6000 else 1500,ramp_steps=1200 if steps==6000 else 3000,
        max_seconds=14400 if steps==6000 else 86400).validate()


@torch.inference_mode()
def architecture_profiles(base):
    from freqssl.models import StudentTeacherSystem
    from experiments.profile_context_efficiency import ConvCounter
    previous_threads=torch.get_num_threads();profiles={}
    torch.set_num_threads(base.cpu_threads)
    try:
        with torch.random.fork_rng(devices=[]):
            for arm in ARMS:
                cfg=config_for(base,arm,Path(base.output_dir)/arm)
                model=StudentTeacherSystem(cfg).students[0].train()
                counter=ConvCounter(model)
                try:
                    model(torch.zeros(1,4 if arm=='B' else 3,cfg.image_size,cfg.image_size),
                        **({'position':torch.zeros(1,1,cfg.image_size,cfg.image_size)} if arm=='C' else {}))
                finally:
                    counter.close()
                profiles[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),
                    parameters_by_component={n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()},
                    fusion_weight_shapes=[list(f.weights.weight.shape) for f in model.context_fusion],
                    student_forward_conv_macs=sum(counter.macs.values()),conv_macs_by_component=dict(counter.macs),
                    scope='one target at 224x224; training-mode forward without padding, gradients or updates; Conv2d only, not training-step FLOPs')
    finally:
        torch.set_num_threads(previous_threads)
    for row in profiles.values():
        row['extra_parameters_vs_A']=row['parameters']-profiles['A']['parameters']
        row['forward_conv_mac_increase_percent']=100*(row['student_forward_conv_macs']/profiles['A']['student_forward_conv_macs']-1)
    return profiles


def prepare(args):
    out=args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    base=load_config(args.config)
    if (base.method!='mean_teacher' or base.slice_context!='adaptive' or base.base_channels!=32 or
            base.max_channels!=256 or base.image_size!=224 or base.labelnum!=1496 or base.max_samples!=3732 or
            base.batch_size!=8 or base.labeled_bs!=4 or base.internal_text!='none' or base.prompt_source!='none' or
            base.region_weight or base.pseudo_dice_weight or base.training_split or
            base.case_reference!='none' or base.context_position!='none'):
        raise ValueError('Use the declared unchanged adaptive Mean Teacher baseline protocol')
    geometry=Path(base.context_geometry_manifest).resolve(strict=True)
    out.mkdir(parents=True)
    shutil.copy2(geometry,out/'geometry.json')
    base=replace(base,context_geometry_manifest=str(out/'geometry.json'))
    files=[*sorted((ROOT/'freqssl').glob('*.py')),*sorted((ROOT/'experiments').glob('*.py')),
           *sorted((ROOT/'tests').glob('*.py')),*sorted((ROOT/'configs/v3').glob('*.yaml')),
           ROOT/'spider_volume.py',ROOT/'train.py',ROOT/'evaluate.py',ROOT/'predict_spine.py',
           ROOT/'docs/case_reference_20261112.md']
    for path in files:
        dst=out/'code'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst)
    (out/'configs').mkdir()
    configs={}
    for arm in ARMS:
        cfg=config_for(base,arm,out/'screen'/arm)
        write_json(out/'configs'/f'{arm}.json',cfg.to_dict());configs[arm]=cfg.to_dict()
    manifest=dict(schema_version=1,created_unix=time.time(),base=base.to_dict(),configs=configs,
        geometry_sha256=sha(out/'geometry.json'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
        device=args.device,confirm_on_pass=not args.screen_only,
        protocol=dict(seeds_screen=[42],steps_screen=6000,validation_cases=21,
            confirmation_seeds=[42,43,44],confirmation_steps=30000,
            selection='B/C Dice >= A + .003, union-present >= A; best Dice wins, exact tie favors C',
            confirmation='mean Dice gain >= .3pp; >=2/3 positive seeds; mean union-present nondecreasing',
            test_read=False,reference_labels_used=False,
            position_bins=5,distance_bins_mm=[5,15],bootstrap_samples=10000,
            annotation_budget=1496,image_pool=3732,confirmation_warmup_steps=1500,confirmation_ramp_steps=3000,
            intervention_scope='frozen-model dependence; no anatomical causality; not a trained comparison of reference policies'))
    write_json(out/'manifest.json',manifest)
    write_json(out/'architecture.json',architecture_profiles(base))
    command=[sys.executable,str(out/'code/experiments/run_case_reference.py'),'--output-dir',str(out),'--worker']
    write_json(out/'command.json',command)
    write_json(out/'status.json',dict(status='prepared',training_started=False))
    return command


def report(out,result):
    lines=['# 病例中部参考片：三组配对实验','','仅valid；主指标为原生三维Dice。筛查与确认是不同预算，不能混用。','',
        '| 组 | 最佳步数 | Dice% | union-present% | 参数 | 总耗时s | 峰值显存bytes |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in result.get('screen',{}).items():
        manifest=json.loads((Path(row['directory'])/'manifest.json').read_text())
        lines.append(f"| {arm} | {row['best_step']} | {100*row['overall']['dice']:.4f} | "
            f"{100*row['overall']['dice_union_present']:.4f} | {manifest['trainable_parameters']} | {row.get('seconds_this_invocation')} | {row.get('peak_allocated_bytes')} |")
    lines+=['',f"候选：{result.get('selected')}；运行状态：{result['status']}。",
        '没有完整21例×指定种子的结果不标记为确认通过。']
    if 'confirmation' in result:
        c=result['confirmation']
        lines += ['',f"确认主Dice差值/pp：{c['dice_delta_pp']}；union-present差值/pp：{c['union_delta_pp']}。",
                  f"确认通过：{c['passed']}。"]
    lines+=['','总耗时包含数据校验、训练、验证和保存；峰值显存覆盖整个运行。',
        '参数与未缓存学生前向MAC见architecture.json；整卷推理MAC、耗时、参考片预编码与尾批填充见各analysis目录。',
        '逐病例配对差值与bootstrap区间见results.json；干预与区域统计见各analysis目录。',
        '分析中参考片替换和位置打乱是推理分布干预。新增容量和影像曝光不能仅靠这三组完全排除。',
        '多个探索性分组未作多重比较校正，验证集结果不等于独立测试结论。','']
    (out/'report.md').write_text('\n'.join(lines))


def worker(out):
    out=out.resolve();m=json.loads((out/'manifest.json').read_text())
    if ROOT!=out/'code':
        raise ValueError('Execute frozen command.json')
    if json.loads((out/'status.json').read_text())['status']!='prepared':
        raise ValueError('Run already started; use a fresh output directory')
    result=dict(status='running',screen={},test_read=False,training_started=False)
    write_json(out/'status.json',dict(status='preflight',pid=os.getpid()))
    try:
        for path,digest in m['source_hashes'].items():
            if sha(ROOT/path)!=digest:
                raise ValueError(f'Frozen source changed: {path}')
        if sha(out/'geometry.json')!=m['geometry_sha256']:
            raise ValueError('Geometry manifest changed')
        device=torch.device(m['device'])
        if device.type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('Formal experiment requires CUDA; no automatic CPU fallback')
        torch.cuda.set_device(device)
        torch.empty(1,device=device)
        free,total=torch.cuda.mem_get_info(device)
        write_json(out/'device.json',dict(device=str(device),name=torch.cuda.get_device_name(device),
            free_bytes=free,total_bytes=total,torch_version=str(torch.__version__)))
        # Same-code numerical/data controls before spending any training budget.
        with (out/'preflight.log').open('w') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_case_context.py','-v'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        base=ExperimentConfig(**m['base']).validate();seed_everything(base)
        inventories=[]
        for arm in ARMS:
            cfg=ExperimentConfig(**m['configs'][arm]).validate()
            training,_,split=build_training_data(cfg)
            valid,records=prepare_validation(cfg,'valid')
            if len(records)!=21:
                raise ValueError('Expected full 21-case validation')
            inventories.append((split['labeled_files'],split['unlabeled_files'],[(r['case'],r['slice_names']) for r in records]))
            if arm=='C':
                write_json(out/'geometry_preflight.json',dict(train=training.geometry_receipt,valid=valid.geometry_receipt))
        if any(i!=inventories[0] for i in inventories):
            raise ValueError('A/B/C data inventories differ')
        def run(arm,cfg):
            directory=Path(cfg.output_dir)
            write_json(out/'status.json',dict(status='running',active_run=str(directory.relative_to(out)),pid=os.getpid()))
            result['training_started']=True
            run_experiment(cfg,str(device))
            row=compact_run(directory)
            if row['overall']['cases']!=21:
                raise ValueError('Refuse incomplete validation summary')
            return row
        def diagnose(directory,destination,row):
            summary=analyze(directory/'best.pt',destination,str(device),geometry_manifest=str(out/'geometry.json'))
            for metric in ('dice','dice_union_present'):
                if abs(summary['metrics']['full']['overall'][metric]-row['overall'][metric])>1e-10:
                    raise ValueError('Analysis prediction differs from the selected validation checkpoint')
            return summary
        for arm in ARMS:
            cfg=ExperimentConfig(**m['configs'][arm]).validate()
            result['screen'][arm]=run(arm,cfg)
            write_json(out/'results.json',result);report(out,result)
        selected=choose_candidate({a:r['overall'] for a,r in result['screen'].items()})
        result['selected']=selected
        a_valid=selected_validation(out/'screen/A')[1]
        result['screen_paired']={a:paired_cases(selected_validation(out/f'screen/{a}')[1],a_valid) for a in ('B','C')}
        summaries={arm:diagnose(out/f'screen/{arm}',out/f'analysis/screen/{arm}',result['screen'][arm]) for arm in ARMS}
        result['screen_analysis_paired']={f'{a}_minus_{b}':paired_analysis(summaries[a],summaries[b])
            for a,b in (('B','A'),('C','A'),('B','C'))}
        write_json(out/'results.json',result);report(out,result)
        if selected and m['confirm_on_pass']:
            runs={};dice=[];union=[];paired={};analysis_pairs={}
            for seed in (42,43,44):
                summaries={}
                for arm in ('A',selected):
                    cfg=config_for(base,arm,out/f'confirmation/seed{seed}/{arm}',seed,30000)
                    runs[f'{seed}/{arm}']=run(arm,cfg)
                    summaries[arm]=diagnose(Path(cfg.output_dir),out/f'analysis/confirmation/seed{seed}/{arm}',runs[f'{seed}/{arm}'])
                analysis_pairs[str(seed)]=paired_analysis(summaries[selected],summaries['A'])
                first,last=[runs[f'{seed}/{a}']['overall'] for a in ('A',selected)]
                dice.append(100*(last['dice']-first['dice']))
                union.append(100*(last['dice_union_present']-first['dice_union_present']))
                paired[str(seed)]=paired_cases(selected_validation(out/f'confirmation/seed{seed}/{selected}')[1],
                    selected_validation(out/f'confirmation/seed{seed}/A')[1])
                write_json(out/'confirmation_progress.json',dict(runs=runs,dice_delta_pp=dice,union_delta_pp=union,
                    paired=paired,analysis_paired=analysis_pairs))
            from experiments.summarize_slice_context import clustered_summary
            result['confirmation']=dict(runs=runs,dice_delta_pp=dice,union_delta_pp=union,paired=paired,
                mean_case_delta=clustered_summary({s:v['delta_pp'] for s,v in paired.items()}),
                analysis_paired=analysis_pairs,analysis_across_seeds=combine_seed_pairs(analysis_pairs),
                passed=bool(confirmed(dice,union)))
        result['status']='completed'
        write_json(out/'results.json',result);report(out,result)
        write_json(out/'status.json',dict(status='completed',selected=selected,
            confirmation_performed='confirmation' in result,training_started=True,exit_code=0))
    except BaseException as error:
        result.update(status='failed',error=repr(error),exit_code=1)
        write_json(out/'results.json',result);write_json(out/'status.json',result);report(out,result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v3/case_reference.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--screen-only',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='Explicitly start detached GPU training after preparation')
    mode.add_argument('--launch-prepared',action='store_true',help='Explicitly launch an existing prepared snapshot')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.output_dir);return
    if args.launch_prepared:
        if json.loads((args.output_dir/'status.json').read_text())['status']!='prepared':
            raise ValueError('Only an unstarted prepared snapshot can be launched')
        command=json.loads((args.output_dir/'command.json').read_text())
    else:
        command=prepare(args)
    if args.execute or args.launch_prepared:
        with (args.output_dir/'run.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        write_json(args.output_dir/'launch.json',dict(pid=process.pid,command=command,started_unix=time.time()))
        print(json.dumps(dict(status='launched',pid=process.pid,output=str(args.output_dir.resolve()))))
    else:
        print(json.dumps(dict(status='prepared',training_started=False,command=command)))


if __name__=='__main__':
    main()
