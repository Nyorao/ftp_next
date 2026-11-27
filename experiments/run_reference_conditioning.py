"""Prepare the six-arm reference study; formal GPU work requires an explicit launch."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from freqssl.config import ExperimentConfig, load_config
from freqssl.data import build_training_data
from freqssl.engine import run_experiment, seed_everything, write_json
from freqssl.evaluation import prepare_validation
from freqssl.models import StudentTeacherSystem
from freqssl.reference_conditioning import si_coordinates, row_prior
from freqssl.reference_data import build_prediction_cache, file_hash, ReferencePredictionCache
from freqssl.reference_metrics import WorkCounter, case_mean
from experiments.analyze_reference_conditioning import analyze


ARMS = {
    'P1': ('row', 'random', False), 'P1-GT': ('row', 'random', True),
    'KV-pos': ('position', 'random', False), 'KV': ('kv', 'random', False),
    'KV-fixed': ('kv', 'fixed', False), 'KV-GT': ('kv', 'random', True),
}
SEEDS = (42, 43, 44)


def default_baselines():
    return {str(seed): str(ROOT / ('outputs/case_reference_20261112/screen/A' if seed == 42 else
        f'outputs/case_reference_followup_20261116/training/seed{seed}/A')) for seed in SEEDS}


def config_for(base, arm, output, seed, cache, teacher, teacher_hash):
    kind, sampling, gt = ARMS['KV' if arm == 'KV-cycle' else arm]
    return replace(base, output_dir=str(output), seed=seed, reference_condition=kind,
        reference_sampling=sampling, reference_gt=gt, reference_cycle_weight=.1 if arm == 'KV-cycle' else 0.,
        reference_teacher=str(teacher), reference_teacher_sha256=teacher_hash,
        reference_cache=str(cache), max_steps=6000, eval_every=500, warmup_steps=600,
        ramp_steps=1200, max_seconds=14400).validate()


def choose_candidate(rows):
    baseline = rows['A']['overall']
    passed = [arm for arm in ARMS if rows[arm]['overall']['dice'] - baseline['dice'] >= .003
              and rows[arm]['overall']['dice_union_present'] >= baseline['dice_union_present']]
    return min(passed, key=lambda arm: (-rows[arm]['overall']['dice'], rows[arm]['pipeline_macs'], list(ARMS).index(arm))) if passed else None


def confirmation_arms(winner):
    return list(dict.fromkeys(['P1', 'KV-pos', 'KV', winner])) if winner else []


def confirmation_result(runs, arm):
    if not all(str(seed) in runs and arm in runs[str(seed)] for seed in SEEDS):
        return dict(complete=False, passed=False)
    dice = [100*(runs[str(s)][arm]['overall']['dice']-runs[str(s)]['A']['overall']['dice']) for s in SEEDS]
    union = [100*(runs[str(s)][arm]['overall']['dice_union_present']-runs[str(s)]['A']['overall']['dice_union_present']) for s in SEEDS]
    return dict(complete=True, passed=bool(np.mean(dice) >= .3 and sum(v > 0 for v in dice) >= 2 and np.mean(union) >= 0),
                dice_delta_pp=dice, union_delta_pp=union,
                dice_mean_percent=float(100*np.mean([runs[str(s)][arm]['overall']['dice'] for s in SEEDS])),
                dice_std_percent=float(100*np.std([runs[str(s)][arm]['overall']['dice'] for s in SEEDS], ddof=1)))


def cycle_eligible(runs):
    if not confirmation_result(runs, 'KV')['passed']:
        return False
    for other in ('P1', 'KV-pos'):
        if not all(other in runs[str(s)] for s in SEEDS):
            return False
        gain = np.mean([runs[str(s)]['KV']['overall']['dice'] - runs[str(s)][other]['overall']['dice'] for s in SEEDS])
        if any(runs[str(s)][arm]['numbering']['conditional_wrong_level_percent'] is None
               for s in SEEDS for arm in ('KV', other)):
            return False
        wrong = [runs[str(s)]['KV']['numbering']['conditional_wrong_level_percent'] -
                 runs[str(s)][other]['numbering']['conditional_wrong_level_percent'] for s in SEEDS]
        if gain < .001 or np.mean(wrong) > 0:
            return False
    return True


def cycle_screen_passed(rows):
    cycle, base = rows['KV-cycle'], rows['KV']
    if any(row['numbering']['conditional_wrong_level_percent'] is None for row in (cycle, base)):
        return False
    return (cycle['overall']['dice'] - base['overall']['dice'] >= .001 and
            cycle['overall']['dice_union_present'] >= base['overall']['dice_union_present'] and
            cycle['numbering']['conditional_wrong_level_percent'] <= base['numbering']['conditional_wrong_level_percent'])


def paired_summary(runs, arm, control='A'):
    seeds = [str(s) for s in SEEDS if str(s) in runs and arm in runs[str(s)] and control in runs[str(s)]]
    if not seeds:
        return None
    analyses = {(seed, group): json.loads(Path(runs[seed][group]['analysis']).read_text())
                for seed in seeds for group in (arm, control)}
    metrics = ('dice', 'dice_union_present', 'wrong_level_percent', 'conditional_wrong_level_percent', 'tissue_coverage_percent')
    output = {}
    for metric in metrics:
        differences = {}
        for seed in seeds:
            per_group = {}
            for group in (arm, control):
                result = analyses[seed, group]
                if metric in ('dice', 'dice_union_present'):
                    per_group[group] = {r['case']: 100*r[metric] for r in result['metrics']['full']['per_case']}
                else:
                    per_group[group] = {r['case']: r['conditions']['full']['numbering'][metric] for r in result['cases']}
            if set(per_group[arm]) != set(per_group[control]):
                raise ValueError('Paired comparison requires identical complete case inventories')
            differences[seed] = {c:per_group[arm][c]-per_group[control][c] for c in per_group[arm]
                                if per_group[arm][c] is not None and per_group[control][c] is not None}
        # Use the common support across seeds, never turn missing rates into zeros.
        cases = sorted(set.intersection(*(set(v) for v in differences.values())))
        if not cases:
            output[metric] = dict(mean_pp=None, cases=0); continue
        values = np.asarray([np.mean([differences[s][c] for s in seeds]) for c in cases])
        generator = np.random.default_rng(20261125)
        bootstrap = values[generator.integers(len(values), size=(10000, len(values)))].mean(1)
        output[metric] = dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(bootstrap, [.025, .975]).tolist(),
            cases=len(cases), seeds=seeds, per_seed= differences, per_case_mean=dict(zip(cases, values.tolist())))
    return output


def validate_baseline(directory, seed, base):
    directory = Path(directory)
    checkpoint = directory / 'last.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved = ExperimentConfig(**payload['config']).validate()
    if payload['step'] != 6000 or saved.seed != seed or saved.reference_condition != 'none':
        raise ValueError('Baseline must be the corresponding seed, final step 6000 A')
    ignore = {'output_dir', 'seed', 'context_geometry_manifest', 'max_seconds'}
    if {k:v for k,v in saved.to_dict().items() if k not in ignore} != {k:v for k,v in base.to_dict().items() if k not in ignore}:
        raise ValueError(f'Baseline protocol differs: {directory}')
    result = json.loads((directory / 'result.json').read_text())
    validation = json.loads((directory / 'valid_step_6000.json').read_text())['teacher1']
    if result['status'] != 'completed' or result['step'] != 6000 or validation['overall']['cases'] != 21:
        raise ValueError('Baseline final validation is incomplete')
    split = json.loads((directory / 'split.json').read_text())
    return dict(directory=str(directory.resolve()), checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=file_hash(checkpoint), split_sha256=file_hash(directory / 'split.json'),
        labeled_files=split['labeled_files'], unlabeled_files=split['unlabeled_files'],
        validation_cases=split['validation_cases'], overall=validation['overall'],
        source_training_seconds=result.get('seconds_this_invocation'), source_training_steps=6000)


def verify_legacy_baseline(snapshot):
    script = '''import json, torch
from freqssl.config import ExperimentConfig
from freqssl.models import StudentTeacherSystem
from freqssl.engine import train_step, make_optimizer, state_hash, seed_everything
c=ExperimentConfig(root_path='/unused',method='mean_teacher',slice_context='adaptive',image_size=32,base_channels=4,max_channels=32,labelnum=4,max_samples=8,batch_size=4,labeled_bs=2,seed=42,max_steps=3,warmup_steps=0,ramp_steps=0,cutmix_probability=1.,cpu_threads=2).validate()
seed_everything(c);m=StudentTeacherSystem(c);o=make_optimizer(m,c)
g=torch.Generator().manual_seed(91)
b={'image':torch.randn(4,3,32,32,generator=g),'label':torch.cat((torch.randint(20,(2,32,32),generator=g),torch.full((2,32,32),-100))),'is_labeled':torch.tensor([True,True,False,False])}
initial=state_hash(m.students[0]); losses=[]
for s in range(2):losses.append(train_step(m,o,b,c,torch.device('cpu'),s)['loss'])
print(json.dumps({'initial':initial,'student':state_hash(m.students[0]),'teacher':state_hash(m.teachers[0]),'losses':losses}))'''
    receipts = []
    for root in (Path(snapshot), ROOT):
        process = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True, check=True, timeout=120)
        receipts.append(json.loads(process.stdout))
    if receipts[0] != receipts[1]:
        raise ValueError('Disabled reference path does not reproduce frozen A training')
    return dict(exact=True, steps=2, synthetic=True, reference_snapshot=str(snapshot), result=receipts[0])


@torch.inference_mode()
def architecture_profiles(base, configs):
    profiles = {}
    torch.set_num_threads(base.cpu_threads)
    for arm, cfg in [('A', base)] + [(a, ExperimentConfig(**v)) for a, v in configs.items()]:
        seed_everything(cfg)
        model = StudentTeacherSystem(cfg).students[0].train()
        images = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
        reference = dict(image=images[:, :1], probabilities=torch.full((1, 20, cfg.image_size, cfg.image_size), .05),
            si=si_coordinates(1, images.shape[-2:]), valid=torch.ones_like(images[:, :1]))
        reference['prior'] = row_prior(reference['probabilities'])
        counter = WorkCounter(model)
        model(images, **({'reference':reference} if arm != 'A' else {}))
        profiles[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
            parameters_by_component={n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()},
            target_training_forward=counter.finish())
    for arm, profile in profiles.items():
        profile['deployment_weight_parameters'] = profile['parameters'] + (profiles['A']['parameters'] if arm != 'A' else 0)
    return dict(arms=profiles, scope='One unpadded target forward; source precomputation and cycle excluded here, measured separately. ' + WorkCounter.__doc__)


def prepare(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    base = load_config(args.config)
    if (base.method != 'mean_teacher' or base.slice_context != 'adaptive' or base.reference_condition != 'none'
            or base.case_reference != 'none' or base.context_position != 'none'
            or (base.image_size, base.base_channels, base.max_channels, base.labelnum, base.max_samples,
                base.batch_size, base.labeled_bs, base.max_steps, base.warmup_steps, base.ramp_steps) !=
               (224,32,256,1496,3732,8,4,6000,600,1200)):
        raise ValueError('Use the unchanged 6000-step adaptive A baseline configuration')
    baselines = json.loads(args.baselines.read_text()) if args.baselines else default_baselines()
    baselines = {str(s):validate_baseline(baselines[str(s)], s, base) for s in SEEDS}
    inventories = [(b['labeled_files'], b['unlabeled_files'], b['validation_cases']) for b in baselines.values()]
    if any(v != inventories[0] for v in inventories):
        raise ValueError('Baseline seeds have different data inventories')
    baseline_replay = verify_legacy_baseline(args.legacy_snapshot)
    training, _, split = build_training_data(base)
    if (split['labeled_files'], split['unlabeled_files'], split['validation_cases']) != inventories[0]:
        raise ValueError('Current data inventory differs from existing A')
    out.mkdir(parents=True)
    geometry = Path(base.context_geometry_manifest).resolve(strict=True)
    shutil.copy2(geometry, out / 'geometry.json')
    base = replace(base, context_geometry_manifest=str(out / 'geometry.json'))
    files = [*sorted((ROOT / 'freqssl').glob('*.py')), *sorted((ROOT / 'experiments').glob('*.py')),
             *sorted((ROOT / 'tests').glob('*.py')), *sorted((ROOT / 'configs/v3').rglob('*.yaml')),
             ROOT / 'spider_volume.py', ROOT / 'train.py', ROOT / 'evaluate.py', ROOT / 'predict_spine.py',
             ROOT / 'docs/reference_conditioning_20261125.md']
    for path in files:
        destination = out / 'code' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    configs = {}
    for seed in SEEDS:
        directory = out / 'configs' / f'seed{seed}'; directory.mkdir(parents=True)
        configs[str(seed)] = {}
        source = baselines[str(seed)]
        for arm in [*ARMS, 'KV-cycle']:
            cfg = config_for(base, arm, out / 'training' / f'seed{seed}' / arm, seed,
                out / 'cache' / f'seed{seed}', source['checkpoint'], source['checkpoint_sha256'])
            write_json(directory / (arm + '.json'), cfg.to_dict()); configs[str(seed)][arm] = cfg.to_dict()
    protocol = dict(screen_seeds=[42], confirmation_seeds=list(SEEDS), steps=6000,
        expected_cases=21, selection='final EMA only; +0.3pp Dice and union nondecreasing',
        confirmation_arms='P1, KV-pos, KV plus winning arm if distinct',
        reference_selection='maximum argmax canal area, middle-distance then smaller index ties; all-zero -> middle',
        random_reference_pool='foreground area > 0 and >= 10% case maximum; exclude target when possible',
        gt_curriculum='0.5*max(0,1-step/3000); whitelist only; never self-reference GT',
        cycle='conditional follow-up only; pure KV; weight 0.1*ramp; no intermediate detach',
        windows=[2,1], cache_dtype='float16, normalized to float32 on read',
        bootstrap_samples=10000, test_read=False, extra_annotation_budget=0)
    manifest = dict(version=2, created_unix=time.time(), base=base.to_dict(), baselines=baselines,
        configs=configs, protocol=protocol, device=args.device, screen_only=args.screen_only,
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in files},
        geometry_sha256=file_hash(out / 'geometry.json'))
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'baseline_replay.json', baseline_replay)
    write_json(out / 'architecture.json', architecture_profiles(base, {a:configs['42'][a] for a in ARMS}))
    command = [sys.executable, str(out / 'code/experiments/run_reference_conditioning.py'), '--output-dir', str(out), '--worker']
    write_json(out / 'command.json', command)
    write_json(out / 'status.json', dict(status='prepared', training_started=False))
    return command


def validate_snapshot(out, manifest):
    if ROOT != out / 'code':
        raise ValueError('Use the frozen command.json, not a mutable source worker')
    for name, digest in manifest['source_hashes'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if file_hash(out / 'geometry.json') != manifest['geometry_sha256']:
        raise ValueError('Frozen geometry changed')
    for baseline in manifest['baselines'].values():
        if file_hash(baseline['checkpoint']) != baseline['checkpoint_sha256']:
            raise ValueError('Frozen source teacher changed')


def ensure_cache(out, manifest, seed):
    base = replace(ExperimentConfig(**manifest['base']), seed=seed)
    seed_everything(base)
    training, _, split = build_training_data(base)
    validation, _ = prepare_validation(base)
    source = manifest['baselines'][str(seed)]
    if split['labeled_files'] != source['labeled_files'] or split['unlabeled_files'] != source['unlabeled_files']:
        raise ValueError('Reference generation data differs from frozen budget')
    root = out / 'cache' / f'seed{seed}'
    if not root.exists():
        required = (len(training)+len(validation)) * 20 * base.image_size**2 * 2 + 3 * 1024**3
        if shutil.disk_usage(out).free < required:
            raise OSError(f'Insufficient free space for one seed cache plus 3 GiB reserve ({required} bytes)')
        root.mkdir(parents=True)
        write_json(root / 'owner.json', dict(output=str(out), seed=seed, temporary_reference_cache=True))
        for split, dataset in (('train', training), ('valid', validation)):
            build_prediction_cache(source['checkpoint'], dataset, root / split, manifest['device'], base.eval_batch_size)
    for split, dataset in (('train', training), ('valid', validation)):
        ReferencePredictionCache(root / split, dataset, source['checkpoint_sha256'])
    return root


def release_cache(out, seed):
    root = out / 'cache' / f'seed{seed}'
    if not root.exists():
        return
    owner = json.loads((root / 'owner.json').read_text())
    if owner != dict(output=str(out), seed=seed, temporary_reference_cache=True):
        raise ValueError('Refuse to remove an unowned reference cache')
    destination = out / 'cache_manifests' / f'seed{seed}'; destination.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid'):
        path = root / split / 'manifest.json'
        if not path.exists():
            raise ValueError('Incomplete cache retained for inspection')
        shutil.copy2(path, destination / (split + '.json'))
    shutil.rmtree(root)


def report(out, result):
    lines = ['# Reference conditioning v2', '', 'Fixed step 6000 final EMA; development validation only. No test evaluation.', '',
             '| Seed | Arm | Dice % | Union % | Conditional wrong-level % | Pipeline MACs |',
             '|---|---|---:|---:|---:|---:|']
    for seed, rows in result.get('runs', {}).items():
        for arm, row in rows.items():
            wrong = row['numbering']['conditional_wrong_level_percent']
            lines.append(f"| {seed} | {arm} | {100*row['overall']['dice']:.4f} | {100*row['overall']['dice_union_present']:.4f} | {wrong} | {row['pipeline_macs']:.0f} |")
    lines += ['', f"Status: {result['status']}; screen winner: {result.get('selected')}; training started: {result.get('training_started', False)}.",
              '', 'MACs include Conv/Linear and attention products; normalization, interpolation, softmax and backward are excluded.',
              'Pipeline costs include source whole-volume selection for conditional models. A source scans performed for diagnostics are excluded from A deployment cost.',
              'Numbering/coverage, distance strata, reference interventions, attention mass, gates and paired case bootstrap intervals are saved alongside the report.',
              'Window-concentrated attention and cycle consistency do not establish correct anatomical numbering.', '']
    if 'confirmation' in result:
        for arm, row in result['confirmation'].items():
            lines.append(f"{arm}: {json.dumps(row, ensure_ascii=False)}")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def training_summary(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    values = dict(seconds=result['seconds_this_invocation'], peak_allocated_bytes=result.get('peak_allocated_bytes'),
        reference_exposures=0, reference_gt_available=0, reference_gt_used=0, self_pairs=0, cycle_samples=0)
    choices = {}
    with (directory / 'metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'train':
                continue
            names = row.get('reference_names', [])
            values['reference_exposures'] += len(names)
            values['reference_gt_available'] += row.get('reference_gt_available', 0)
            values['reference_gt_used'] += row.get('reference_gt_used', 0)
            values['self_pairs'] += row.get('reference_self_pairs', 0)
            values['cycle_samples'] += row.get('cycle_samples', 0)
            for name in names:
                choices[name] = choices.get(name, 0) + 1
    values['reference_counts'] = choices
    return values


def plot_key_comparison(out, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds = [str(s) for s in SEEDS if str(s) in runs and all(a in runs[str(s)] for a in ('KV','KV-pos'))]
    if not seeds:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10,3.5), constrained_layout=True)
    for arm in ('KV', 'KV-pos'):
        per_case = {}
        for seed in seeds:
            data = json.loads(Path(runs[seed][arm]['analysis']).read_text())
            for case in data['cases']:
                per_case.setdefault(case['case'], []).append(case['attention'])
        for level, axis in enumerate(axes):
            curves = [np.mean([np.asarray(run[level]['mass']).mean(0) for run in values], axis=0)
                      for values in per_case.values()]
            offsets = next(iter(per_case.values()))[0][level]['offsets']
            axis.plot(offsets, np.mean(curves, axis=0), 'o-', label=arm)
            axis.set(xlabel='Reference SI minus query SI (feature rows)', ylabel='Case-equal attention mass', title=f'Stride {8*2**level}')
            axis.legend()
    figure.savefig(out / 'attention_key_ablation.png', dpi=180)
    figure.savefig(out / 'attention_key_ablation.svg')
    plt.close(figure)


def worker(out, rebuild_cache=None):
    out = out.resolve(); manifest = json.loads((out / 'manifest.json').read_text())
    validate_snapshot(out, manifest)
    if rebuild_cache is not None:
        if not torch.cuda.is_available() or torch.device(manifest['device']).type != 'cuda':
            raise RuntimeError('Reference cache rebuilding requires CUDA in the formal protocol')
        return ensure_cache(out, manifest, rebuild_cache)
    if json.loads((out / 'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Worker already started; no implicit retry or overwrite')
    result = dict(status='running', training_started=False, runs={}, test_read=False)
    try:
        write_json(out / 'status.json', dict(status='preflight', pid=os.getpid()))
        if not torch.cuda.is_available() or torch.device(manifest['device']).type != 'cuda':
            raise RuntimeError('Formal work requires CUDA; no automatic CPU fallback')
        with (out / 'preflight.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_reference_conditioning.py', '-v'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        torch.cuda.set_device(manifest['device'])
        free, total = torch.cuda.mem_get_info(manifest['device'])
        write_json(out / 'device.json', dict(name=torch.cuda.get_device_name(manifest['device']), free_bytes=free,
            total_bytes=total, torch_version=str(torch.__version__)))
        def save():
            write_json(out / 'results.json', result); report(out, result)
        def run(seed, arm):
            rows = result['runs'].setdefault(str(seed), {})
            if arm in rows:
                return
            source = manifest['baselines'][str(seed)]
            if arm == 'A':
                checkpoint = Path(source['checkpoint'])
            else:
                cfg = ExperimentConfig(**manifest['configs'][str(seed)][arm]).validate()
                result['training_started'] = True
                write_json(out / 'status.json', dict(status='running', seed=seed, arm=arm, pid=os.getpid()))
                run_experiment(cfg, manifest['device'])
                checkpoint = Path(cfg.output_dir) / 'last.pt'
            destination = out / 'analysis' / f'seed{seed}' / arm
            summary = analyze(checkpoint, destination, manifest['device'], expected_cases=21,
                reference_cache=out / 'cache' / f'seed{seed}', source_checkpoint=source['checkpoint'], interventions=arm != 'A')
            expected = json.loads((checkpoint.parent / 'valid_step_6000.json').read_text())['teacher1']['overall']
            actual = summary['metrics']['full']['overall']
            if actual['cases'] != 21 or any(abs(actual[k]-expected[k]) > 1e-10 for k in ('dice','dice_union_present')):
                raise ValueError('Diagnostic inference must reproduce final validation exactly')
            macs = [case['conditions']['full']['cost']['total_counted_macs'] +
                    (case['source_cost']['conv_macs'] if arm != 'A' else 0) for case in summary['cases']]
            rows[arm] = dict(overall=actual, numbering=summary['numbering']['full'], analysis=str(destination / 'summary.json'),
                checkpoint=str(checkpoint), pipeline_macs=float(np.mean(macs)), training=training_summary(checkpoint.parent),
                pipeline_model_seconds=case_mean([c['conditions']['full']['cost']['model_seconds'] +
                    (c['source_cost']['seconds'] if arm != 'A' else 0) for c in summary['cases']]))
            save()
        ensure_cache(out, manifest, 42)
        run(42, 'A')
        for arm in ARMS:
            run(42, arm)
        result['selected'] = choose_candidate(result['runs']['42'])
        result['confirmation_arms'] = confirmation_arms(result['selected'])
        save(); release_cache(out, 42)
        if result['selected'] and not manifest['screen_only']:
            for seed in (43, 44):
                ensure_cache(out, manifest, seed); run(seed, 'A')
                for arm in result['confirmation_arms']:
                    run(seed, arm)
                release_cache(out, seed)
            result['confirmation'] = {arm:confirmation_result(result['runs'], arm) for arm in result['confirmation_arms']}
            result['cycle_eligible'] = cycle_eligible(result['runs'])
            save()
            if result['cycle_eligible']:
                ensure_cache(out, manifest, 42); run(42, 'KV-cycle'); release_cache(out, 42)
                result['cycle_screen_passed'] = cycle_screen_passed(result['runs']['42'])
                if result['cycle_screen_passed']:
                    for seed in (43,44):
                        ensure_cache(out, manifest, seed); run(seed, 'KV-cycle'); release_cache(out, seed)
        comparisons = [(arm, 'A') for arm in ARMS] + [('KV','KV-pos'), ('KV','P1'), ('KV-pos','P1'),
            ('KV','KV-fixed'), ('KV-GT','KV'), ('P1-GT','P1'), ('KV-cycle','KV')]
        result['paired'] = {a + '_minus_' + b: paired_summary(result['runs'], a, b) for a, b in comparisons}
        plot_key_comparison(out, result['runs'])
        result['status'] = 'completed'; save()
        write_json(out / 'status.json', dict(status='completed', training_started=result['training_started'], exit_code=0))
    except BaseException as error:
        result.update(status='failed', error=repr(error), exit_code=1)
        write_json(out / 'results.json', result); report(out, result); write_json(out / 'status.json', result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v3/case_reference.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--baselines', type=Path, help='JSON mapping seeds 42/43/44 to completed A directories')
    parser.add_argument('--legacy-snapshot', type=Path, default=ROOT / 'outputs/case_reference_20261112/code')
    parser.add_argument('--screen-only', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--launch-prepared', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--rebuild-cache', type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    if args.worker or args.rebuild_cache is not None:
        return worker(args.output_dir, args.rebuild_cache)
    if args.launch_prepared:
        if json.loads((args.output_dir / 'status.json').read_text())['status'] != 'prepared':
            raise ValueError('Only an unstarted prepared experiment can be launched')
        command = json.loads((args.output_dir / 'command.json').read_text())
    else:
        command = prepare(args)
    if args.execute or args.launch_prepared:
        with (args.output_dir / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.output_dir / 'launch.json', dict(pid=process.pid, command=command, started_unix=time.time()))
        print(json.dumps(dict(status='launched', pid=process.pid, output=str(args.output_dir.resolve()))))
    else:
        print(json.dumps(dict(status='prepared', training_started=False, output=str(args.output_dir.resolve()), command=command)))


if __name__ == '__main__':
    main()
