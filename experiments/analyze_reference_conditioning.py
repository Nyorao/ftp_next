"""Fixed-final-EMA reference interventions; GT is only used for scoring."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from freqssl.case_context import verify_dataset_geometry
from freqssl.context_cache import iter_cached_logits
from freqssl.data import _read_transposed
from freqssl.engine import write_json, state_hash, seed_everything
from freqssl.evaluation import prepare_validation
from freqssl.inference import load_single_network
from freqssl.reference_data import ReferencePredictionCache, file_hash
from freqssl.reference_inference import iter_reference_logits, frozen_source_predictions
from freqssl.reference_metrics import numbering_metrics, distance_bin, case_mean, WorkCounter, aggregate_attention
from spider_volume import reconstruct, volume_metrics, aggregate_cases
from experiments.analyze_case_reference import confusion_metrics


CONTROL_SEEDS = (1701, 1702, 1703)


def controls(groups, selected):
    cases = sorted(groups)
    result = {}
    for seed in CONTROL_SEEDS:
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(cases))
        donors = {case: order[(i+1) % len(order)] for i, case in enumerate(order)} if len(cases) > 1 else {}
        random_indices = {}
        for case in cases:
            choices = [i for i in range(len(groups[case])) if i != selected[case]]
            random_indices[case] = int(rng.choice(choices)) if choices else selected[case]
        result[str(seed)] = dict(donors=donors, random_indices=random_indices)
    return result


def write_plots(output, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = [case for case in cases if case['attention']]
    if not available:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    scatter, panels = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for level, (axis, panel) in enumerate(zip(axes, panels)):
        observations = [case['attention'][level] for case in available]
        x = observations[0]['offsets']
        mass = np.mean([np.asarray(row['mass']).mean(0) for row in observations], axis=0)
        uniform = np.mean([row['uniform_mass'] for row in observations], axis=0)
        axis.plot(x, mass, 'o-', label='Learned attention')
        axis.plot(x, uniform, '--', label='Uniform valid window')
        axis.set(xlabel='Reference SI minus query SI (feature rows)', ylabel='Attention mass', title=f'Stride {8 * 2**level}')
        axis.legend()
        points = [(np.mean(case['attention'][level]['mean_abs_delta']),
                   case['conditions']['full']['numbering']['conditional_wrong_level_percent']) for case in available]
        points = [(x, y) for x, y in points if y is not None]
        if points:
            panel.scatter(*zip(*points), s=24)
        panel.set(xlabel='Mean absolute row displacement', ylabel='Conditional wrong-level rate (%)', title=f'Stride {8 * 2**level}; cases')
    for name, fig in (('attention_mass', figure), ('attention_numbering', scatter)):
        fig.savefig(output / (name + '.png'), dpi=180)
        fig.savefig(output / (name + '.svg'))
        plt.close(fig)


def intervention_summary(result):
    """Average repeated interventions within case before resampling cases."""
    conditions = set(result['metrics']) - {'full'}
    families = {name:[name] for name in ('middle','disabled','row_shift') if name in conditions}
    for prefix in ('same_case_random', 'other_case'):
        names = [f'{prefix}_{seed}' for seed in CONTROL_SEEDS if f'{prefix}_{seed}' in conditions]
        if names:
            families[prefix] = names
    lookup = {condition:{row['case']:row for row in value['per_case']}
              for condition, value in result['metrics'].items()}
    summaries = {}
    for family, repeats in families.items():
        summaries[family] = {}
        for metric in ('dice', 'dice_union_present', 'conditional_wrong_level_percent'):
            differences = {}
            for case in result['cases']:
                if metric == 'conditional_wrong_level_percent':
                    base = case['conditions']['full']['numbering'][metric]
                    values = [case['conditions'][condition]['numbering'][metric] for condition in repeats]
                    factor = 1
                else:
                    base = lookup['full'][case['case']][metric]
                    values = [lookup[condition][case['case']][metric] for condition in repeats]
                    factor = 100
                if base is not None and all(value is not None for value in values):
                    differences[case['case']] = factor * (float(np.mean(values)) - base)
            values = np.asarray(list(differences.values()))
            if len(values):
                rng = np.random.default_rng(20261125)
                boot = values[rng.integers(len(values), size=(10000,len(values)))].mean(1)
                summaries[family][metric] = dict(mean_delta_pp=float(values.mean()),
                    ci95_pp=np.quantile(boot,[.025,.975]).tolist(), per_case_delta_pp=differences,
                    repeats=len(repeats), independent_cases=len(values))
    return summaries


def summarize_groups(cases):
    buckets = {}
    for case in cases:
        for condition, row in case['conditions'].items():
            for group, values in row['groups'].items():
                for metric, value in values.items():
                    if metric.endswith('_percent') and value is not None:
                        buckets.setdefault((condition, group, metric), {})[case['case']] = float(value)
    output = []
    for (condition, group, metric), values in sorted(buckets.items()):
        array = np.asarray(list(values.values()))
        rng = np.random.default_rng(20261125)
        boot = array[rng.integers(len(array), size=(10000, len(array)))].mean(1)
        low, high = np.quantile(boot, [.025, .975])
        output.append(dict(condition=condition, group=group, metric=metric, mean_percent=float(array.mean()),
            ci95_lower=float(low), ci95_upper=float(high), cases=len(array), per_case=values))
    return output


@torch.inference_mode()
def analyze(checkpoint, output, device='cpu', *, expected_cases=21, interventions=True,
            reference_cache=None, source_checkpoint=None, measure_source=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='running'))
    try:
        model, cfg, receipt = load_single_network(checkpoint, device)
        seed_everything(cfg)
        conditional = cfg.reference_condition != 'none'
        source_checkpoint = source_checkpoint or cfg.reference_teacher
        cache_root = Path(reference_cache or cfg.reference_cache)
        if not source_checkpoint:
            raise ValueError('A diagnostics require the matching frozen source checkpoint/cache')
        # Audit and scoring may read validation labels; reference/cache reads cannot.
        dataset, records = prepare_validation(cfg)
        cache = getattr(dataset, 'reference_cache', None)
        if cache is None:
            cache = ReferencePredictionCache(cache_root / 'valid', dataset, file_hash(source_checkpoint))
        _, geometry = verify_dataset_geometry(dataset, cfg.context_geometry_manifest)
        if len(records) != expected_cases:
            raise ValueError('Refuse partial validation diagnostics')
        mapping = controls(cache.groups, cache.selected)
        write_json(output / 'controls.json', mapping)
        lookup = {name: i for i, name in enumerate(dataset.names)}
        before = state_hash(model)
        cases, by_condition = [], {}
        for record in records:
            case, names = record['case'], record['slice_names']
            n, r = len(names), cache.selected[case]
            images = torch.stack([dataset.read_image(lookup[name]) for name in names])
            source_cost = None
            if measure_source:
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                fresh, source_cost = frozen_source_predictions(source_checkpoint, n, lambda i: images[i],
                    device, cache.manifest['teacher_sha256'], cfg.eval_batch_size)
                for i in range(n):
                    torch.testing.assert_close(fresh[i], cache.read(names[i]), rtol=0, atol=0)
            conditions = ['full']
            if conditional and interventions:
                conditions += ['middle', 'disabled', 'row_shift']
                for seed in CONTROL_SEEDS:
                    conditions.append(f'same_case_random_{seed}')
                    if len(records) > 1:
                        conditions.append(f'other_case_{seed}')
            entry = dict(case=case, middle_index=(n-1)//2, selected_reference=r,
                         source_cost=source_cost, attention=[], conditions={})
            # Metadata and comparison strata are fixed to the full-input reference.
            spacing = geometry['cases'][case]['spacing_mm']
            for condition in conditions:
                ref_index, ref_image, ref_probability = r, None, None
                if condition == 'middle':
                    ref_index = (n-1)//2
                elif condition.startswith('same_case_random_'):
                    ref_index = mapping[condition.rsplit('_', 1)[1]]['random_indices'][case]
                elif condition.startswith('other_case_'):
                    donor = mapping[condition.rsplit('_', 1)[1]]['donors'][case]
                    donor_name = cache.groups[donor][cache.selected[donor]]
                    ref_image, ref_probability = dataset.read_image(lookup[donor_name]), cache.read(donor_name)
                adapters = getattr(model.backbone, 'reference_adapters', [])
                for adapter in adapters:
                    if hasattr(adapter, 'observe'):
                        adapter.observations.clear(); adapter.observe = condition == 'full'
                counter = WorkCounter(model.backbone)
                stats, slices, group_cms, slice_rows = {}, [], {}, []
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                started = time.monotonic()
                if conditional:
                    iterator = iter_reference_logits(model.backbone, n, lambda i: images[i], lambda i: cache.read(names[i]),
                        device, cfg.eval_batch_size, reference_index=ref_index, reference_image=ref_image,
                        reference_probability=ref_probability, disable=condition == 'disabled',
                        row_shift=condition == 'row_shift', stats=stats)
                else:
                    iterator = iter_cached_logits(model.backbone, n, lambda i: images[i], device, cfg.eval_batch_size, stats=stats)
                # Keep scoring and PNG-label IO outside the timed model iterator.
                predictions = []
                for _, logits in iterator:
                    predictions.extend(logits.argmax(1).cpu().unbind(0))
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                cost = dict(counter.finish(), model_seconds=elapsed, **stats,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)
                for i, (name, prediction) in enumerate(zip(names, predictions)):
                    with Image.open(dataset.root / 'image' / name) as image:
                        shape = image.size
                    pred = F.interpolate(prediction[None, None].float(), size=shape, mode='nearest')[0, 0].numpy().astype(np.uint8)
                    target_slice = _read_transposed(dataset.root / 'label' / name, torch.long).numpy()
                    cm = np.bincount(target_slice.ravel() * 20 + pred.ravel(), minlength=400).reshape(20, 20)
                    slices.append(pred.T.copy())
                    middle_distance, ref_distance = abs(i-(n-1)//2), abs(i-r)
                    groups = ['all', 'middle_' + distance_bin(middle_distance*spacing),
                              'reference_' + distance_bin(ref_distance*spacing)]
                    if i != r:
                        groups.append('exclude_selected_reference')
                    if i != ref_index or ref_image is not None:
                        groups.append('exclude_actual_reference')
                    for group in groups:
                        group_cms[group] = group_cms.get(group, np.zeros((20, 20), dtype=np.int64)) + cm
                    wrong = (pred != target_slice)
                    fg = target_slice > 0
                    slice_rows.append(dict(index=i, distance_to_middle_slices=middle_distance,
                        distance_to_middle_mm=middle_distance*spacing, distance_to_reference_slices=ref_distance,
                        distance_to_reference_mm=ref_distance*spacing,
                        foreground_pixels=int(fg.sum()), foreground_errors=int((wrong & fg).sum()),
                        background_pixels=int((~fg).sum()), background_errors=int((wrong & ~fg).sum())))
                reference = sitk.ReadImage(record['reference']); target = sitk.GetArrayFromImage(reference)
                native = reconstruct(slices, record['slice_axis'], record['undo_vertical_flip'], target.shape)
                classes = volume_metrics(native, target, tuple(reversed(reference.GetSpacing())), surfaces=False)
                metrics = dict(case=case, classes=classes)
                by_condition.setdefault(condition, []).append(metrics)
                entry['conditions'][condition] = dict(numbering=numbering_metrics(group_cms['all']),
                    groups={g:dict(confusion_metrics(cm), **numbering_metrics(cm), confusion=cm.tolist(),
                        foreground_error_percent=float(100*(cm[1:].sum()-np.trace(cm[1:,1:]))/cm[1:].sum()) if cm[1:].sum() else None,
                        background_error_percent=float(100*cm[0,1:].sum()/cm[0].sum()) if cm[0].sum() else None)
                        for g, cm in group_cms.items()},
                    slice_regions=slice_rows, cost=cost, actual_reference=ref_index if ref_image is None else 'other_case')
                if condition == 'full':
                    for adapter in adapters:
                        if hasattr(adapter, 'observations'):
                            entry['attention'].append(aggregate_attention(adapter.observations, adapter.window))
                for adapter in adapters:
                    if hasattr(adapter, 'observe'):
                        adapter.observe = False; adapter.observations.clear()
            cases.append(entry)
            write_json(output / 'status.json', dict(status='running', completed_cases=len(cases), total_cases=len(records)))
        result = dict(checkpoint=receipt, checkpoint_sha256=file_hash(checkpoint), cases=cases,
            metrics={name:dict(aggregate_cases(rows), cases=rows) for name, rows in by_condition.items()},
            numbering={condition:{metric:case_mean([c['conditions'][condition]['numbering'][metric] for c in cases])
                for metric in ('wrong_level_percent', 'conditional_wrong_level_percent', 'tissue_coverage_percent')}
                for condition in by_condition},
            model_state_unchanged=state_hash(model) == before,
            grouping='middle and selected-reference strata fixed before interventions; actual-reference exclusion separately labeled',
            timing_scope='preloaded target images; model and reference-bank IO; source model load/scan timed separately; excludes scoring',
            mac_scope=WorkCounter.__doc__, test_read=False)
        if not result['model_state_unchanged']:
            raise ValueError('Diagnostics modified model state')
        result['interventions'] = intervention_summary(result)
        result['groups'] = summarize_groups(cases)
        write_json(output / 'summary.json', result)
        with (output / 'distance_and_regions.csv').open('w', newline='') as stream:
            fields = ['condition','group','metric','mean_percent','ci95_lower','ci95_upper','cases']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
            writer.writerows(result['groups'])
        with (output / 'numbering.csv').open('w', newline='') as stream:
            fields = ['case', 'condition', 'group', 'wrong_level_percent', 'conditional_wrong_level_percent', 'tissue_coverage_percent',
                      'numbered_gt', 'tissue_correct', 'wrong_one', 'wrong_multiple']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for case in cases:
                for condition, values in case['conditions'].items():
                    for group, row in values['groups'].items():
                        writer.writerow(dict(case=case['case'], condition=condition, group=group, **{k:row[k] for k in fields[3:]}))
        with (output / 'attention.csv').open('w', newline='') as stream:
            fields = ['case', 'stride', 'head', 'delta_y', 'mass', 'uniform_mass', 'gate', 'residual_rms_ratio', 'conditional_wrong_level_percent']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for case in cases:
                for level, row in enumerate(case['attention']):
                    for head, mass in enumerate(row['mass']):
                        for i, delta in enumerate(row['offsets']):
                            writer.writerow(dict(case=case['case'], stride=8*2**level, head=head, delta_y=delta,
                                mass=mass[i], uniform_mass=row['uniform_mass'][i], gate=row['gate'],
                                residual_rms_ratio=row['residual_rms_ratio'],
                                conditional_wrong_level_percent=case['conditions']['full']['numbering']['conditional_wrong_level_percent']))
        write_plots(output, cases)
        write_json(output / 'status.json', dict(status='completed', cases=len(cases), exit_code=0))
        return result
    except BaseException as error:
        write_json(output / 'status.json', dict(status='failed', error=repr(error), exit_code=1))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    analyze(args.checkpoint, args.output_dir, args.device)


if __name__ == '__main__':
    main()
