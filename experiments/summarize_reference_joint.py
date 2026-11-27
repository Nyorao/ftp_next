"""Fixed-step EMA comparison for the joint KV-pos study (development valid only).

Reads ``valid_step_<step>.json`` (teacher1, all 21 valid cases) of A, KV-pos,
NB-pos and JOINT-pos for each seed listed in the plan, then writes
summary.json / summary.md / per_case.csv. Missing seeds are reported as
partial, never filled in. The decision thresholds come from the plan file and
are engineering gates, not significance tests.

    python experiments/summarize_reference_joint.py
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from freqssl.paths import expand
from freqssl.reference_metrics import numbering_metrics, case_mean

ARMS = ('A', 'KV-pos', 'NB-pos', 'JOINT-pos')
COMPARISONS = (('JOINT-pos', 'A'), ('KV-pos', 'A'), ('NB-pos', 'A'), ('JOINT-pos', 'KV-pos'),
               ('JOINT-pos', 'NB-pos'), ('NB-pos', 'KV-pos'))
METRICS = ('dice', 'dice_union_present')


def load_run(directory, step):
    directory = Path(directory)
    validation = directory / f'valid_step_{step}.json'
    if not validation.exists():
        return None
    result = directory / 'result.json'
    if result.exists():
        status = json.loads(result.read_text())
        if status.get('status') != 'completed' or status.get('step') != step:
            return None
    teacher = json.loads(validation.read_text())['teacher1']
    config = json.loads((directory / 'config.json').read_text()) if (directory / 'config.json').exists() else {}
    split = json.loads((directory / 'split.json').read_text()) if (directory / 'split.json').exists() else {}
    cases = {row['case']: row for row in teacher['per_case']}
    wrong = {row['case']: numbering_metrics(row['anatomy']['confusion_gt_rows_pred_columns'])
             ['conditional_wrong_level_percent'] for row in teacher.get('cases', [])}
    return dict(directory=str(directory.resolve()), overall=teacher['overall'], cases=cases,
                conditional_wrong_level=wrong, seed=config.get('seed'),
                sources=config.get('reference_sources', 'case' if config.get('reference_condition', 'none') != 'none' else None),
                teacher_sha256=split.get('reference_conditioning', {}).get('teacher_sha256'))


def bootstrap(values, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261127)
    means = values[rng.integers(len(values), size=(samples, len(values)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def compare(runs, seeds, arm, control):
    seeds = [s for s in seeds if runs[s].get(arm) and runs[s].get(control)]
    if not seeds:
        return None
    output = dict(seeds=seeds, complete=len(seeds) == len(runs))
    for metric in METRICS:
        per_seed = [100 * (runs[s][arm]['overall'][metric] - runs[s][control]['overall'][metric]) for s in seeds]
        cases = sorted(set.intersection(*(set(runs[s][arm]['cases']) & set(runs[s][control]['cases']) for s in seeds)))
        # Average seeds inside each case first; cases, not seeds x cases, are resampled.
        paired = [np.mean([100 * (runs[s][arm]['cases'][c][metric] - runs[s][control]['cases'][c][metric])
                           for s in seeds]) for c in cases]
        output[metric] = dict(per_seed_pp=dict(zip(map(str, seeds), per_seed)), mean_pp=float(np.mean(per_seed)),
                              positive_seeds=int(sum(v > 0 for v in per_seed)), cases=len(cases),
                              case_bootstrap_ci95_pp=bootstrap(paired) if cases else None)
    wrong = [case_mean(runs[s][arm]['conditional_wrong_level'].values()) for s in seeds]
    base = [case_mean(runs[s][control]['conditional_wrong_level'].values()) for s in seeds]
    output['conditional_wrong_level_mean_delta_pp'] = (float(np.mean([a - b for a, b in zip(wrong, base)]))
                                                       if all(v is not None for v in wrong + base) else None)
    return output


def decide(runs, seeds, comparisons, thresholds):
    complete = {arm: all(runs[s].get(arm) for s in seeds) for arm in ARMS}
    if not all(complete.values()):
        return dict(status='partial', complete_arms=complete,
                    note='Decision needs all four arms on every seed in the plan.')
    mean = {arm: float(np.mean([runs[s][arm]['overall']['dice'] for s in seeds])) for arm in ARMS}
    best = max(('KV-pos', 'NB-pos'), key=lambda arm: mean[arm])
    def gate(row, threshold):
        return bool(row['dice']['mean_pp'] >= threshold and
                    row['dice']['positive_seeds'] >= thresholds['min_positive_seeds'] and
                    row['dice_union_present']['mean_pp'] >= 0)
    best_row = comparisons[f'JOINT-pos_minus_{best}']
    return dict(status='complete', best_single=best, mean_dice_percent={a: 100 * v for a, v in mean.items()},
                joint_beats_best_single=gate(best_row, thresholds['joint_vs_best_single_dice_pp']),
                joint_beats_a=gate(comparisons['JOINT-pos_minus_A'], thresholds['joint_vs_a_dice_pp']),
                rule=(f"JOINT minus best single: mean Dice >= +{thresholds['joint_vs_best_single_dice_pp']}pp, "
                      f">= {thresholds['min_positive_seeds']} positive seeds, mean union-present delta >= 0. "
                      f"JOINT minus A: same with +{thresholds['joint_vs_a_dice_pp']}pp."))


def report(summary):
    lines = ['# Joint KV-pos study: fixed-step EMA summary', '',
             f"Step {summary['step']}, teacher1, development valid only (no test). Seeds: {summary['seeds']}.", '',
             '| Seed | Arm | Dice % | Union % | Cond. wrong-level % | Run |', '|---|---|---:|---:|---:|---|']
    for seed, rows in summary['runs'].items():
        for arm in ARMS:
            row = rows.get(arm)
            if row is None:
                lines.append(f'| {seed} | {arm} | missing | | | |'); continue
            wrong = case_mean(row['conditional_wrong_level'].values())
            lines.append(f"| {seed} | {arm} | {100*row['overall']['dice']:.4f} | "
                         f"{100*row['overall']['dice_union_present']:.4f} | "
                         f"{'n/a' if wrong is None else f'{wrong:.3f}'} | `{row['directory']}` |")
    lines += ['', '| Comparison | Seeds | Dice delta pp per seed | Mean | 95% case CI | Union mean pp |',
              '|---|---|---|---:|---|---:|']
    for name, row in summary['comparisons'].items():
        if row is None:
            lines.append(f'| {name} | none | | | | |'); continue
        dice, union = row['dice'], row['dice_union_present']
        ci = dice['case_bootstrap_ci95_pp']
        lines.append(f"| {name} | {row['seeds']} | " + ', '.join(f'{v:+.3f}' for v in dice['per_seed_pp'].values()) +
                     f" | {dice['mean_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {union['mean_pp']:+.3f} |")
    lines += ['', f"Decision: `{json.dumps(summary['decision'], ensure_ascii=False)}`", '',
              'Thresholds are engineering gates fixed in the plan before results; they are not significance tests.',
              'Case CIs resample the 21 validation cases after averaging seeds within each case.']
    for note in summary['notes']:
        lines.append(f'- {note}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/plans/reference_joint_20261127.yaml')
    args = parser.parse_args(argv)
    plan = expand(yaml.safe_load(args.plan.read_text()))
    step, seeds = int(plan['step']), [int(s) for s in plan['seeds']]
    runs, notes = {s: {} for s in seeds}, []
    for arm in ARMS:
        for seed in seeds:
            candidates = [Path(p) for p in plan['arms'][arm][seed]]
            found = [(path, load_run(path, step)) for path in candidates]
            found = [(path, run) for path, run in found if run is not None]
            if not found:
                continue
            runs[seed][arm] = found[0][1]
            if found[0][1]['seed'] not in (None, seed):
                raise ValueError(f'{found[0][0]} has seed {found[0][1]["seed"]}, expected {seed}')
            if arm == 'KV-pos' and len(found) > 1:
                a, b = (100 * run['overall']['dice'] for _, run in found[:2])
                notes.append(f'seed {seed} KV-pos rerun {a:.4f}% vs earlier run {b:.4f}% '
                             f'({"identical" if a == b else f"differ by {a-b:+.4f}pp"}); using the rerun.')
    for seed in seeds:
        hashes = {arm: run['teacher_sha256'] for arm, run in runs[seed].items()
                  if arm != 'A' and run['teacher_sha256']}
        if len(set(hashes.values())) > 1:
            raise ValueError(f'seed {seed}: arms used different frozen teachers {hashes}')
        for arm in ARMS:
            if arm not in runs[seed]:
                notes.append(f'seed {seed} {arm}: no completed step-{step} run found.')
    comparisons = {f'{a}_minus_{b}': compare(runs, seeds, a, b) for a, b in COMPARISONS}
    summary = dict(step=step, seeds=seeds, test_read=False, runs={str(s): runs[s] for s in seeds},
                   comparisons=comparisons, decision=decide(runs, seeds, comparisons, plan['decision']), notes=notes)
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    text = report(summary)
    (output / 'summary.md').write_text(text)
    with (output / 'per_case.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'arm', 'case', 'dice', 'dice_union_present', 'conditional_wrong_level_percent'])
        for seed in seeds:
            for arm, run in runs[seed].items():
                for case, row in sorted(run['cases'].items()):
                    writer.writerow([seed, arm, case, row['dice'], row['dice_union_present'],
                                     run['conditional_wrong_level'].get(case)])
    print(text)


if __name__ == '__main__':
    main()
