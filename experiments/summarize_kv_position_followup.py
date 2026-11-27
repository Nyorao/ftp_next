"""Summarize saved interventions and new prediction-only KV-pos diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

SEEDS = (42, 43, 44)
REPEATS = (1701, 1702, 1703)
FAMILIES = {
    'A': ('full',),
    'full': ('full',),
    'disabled': ('disabled',),
    'middle': ('middle',),
    'same_case_random': tuple(f'same_case_random_{r}' for r in REPEATS),
    'other_case': tuple(f'other_case_{r}' for r in REPEATS),
}
IDS = {'vertebra': list(range(1, 10)), 'disc': list(range(11, 20))}


def average(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def rate(count, base):
    return float(100 * count / base) if base else None


def stats(cm, family='all'):
    cm = np.asarray(cm)
    groups = list(IDS.values()) if family == 'all' else [IDS[family]]
    gt = covered = wrong_one = wrong_more = plus = minus = 0
    for ids in groups:
        block = cm[np.ix_(ids, ids)]
        gt += int(cm[ids].sum())
        covered += int(block.sum())
        plus += int(np.diag(block, 1).sum())
        minus += int(np.diag(block, -1).sum())
        d = np.abs(np.subtract.outer(ids, ids))
        wrong_one += int(block[d == 1].sum())
        wrong_more += int(block[d > 1].sum())
    return dict(gt=gt, covered=covered, wrong_one=wrong_one, wrong_multiple=wrong_more,
                plus_one=plus, minus_one=minus,
                one_of_gt_pct=rate(wrong_one, gt), multiple_of_gt_pct=rate(wrong_more, gt),
                wrong_of_gt_pct=rate(wrong_one+wrong_more, gt),
                one_conditional_pct=rate(wrong_one, covered),
                multiple_conditional_pct=rate(wrong_more, covered),
                conditional_wrong_pct=rate(wrong_one+wrong_more, covered),
                coverage_pct=rate(covered, gt))


def metrics(cm):
    cm = np.asarray(cm)
    gt, pred, hit = cm.sum(1)[1:], cm.sum(0)[1:], cm.diagonal()[1:]
    valid, union = gt > 0, gt + pred > 0
    return dict(dice_pct=float((200*hit[valid]/(gt+pred)[valid]).mean()) if valid.any() else None,
                iou_pct=float((100*hit[valid]/(gt+pred-hit)[valid]).mean()) if valid.any() else None,
                union_dice_pct=float((200*hit[union]/(gt+pred)[union]).mean()) if union.any() else None,
                absent_class_fp_count=int(((gt == 0) & (pred > 0)).sum()),
                absent_class_fp_voxels=int(pred[gt == 0].sum()),
                missed_class_count=int(((gt > 0) & (pred == 0)).sum()))


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values):
    x = np.asarray(values)
    rng = np.random.default_rng(20261127)
    return np.quantile(x[rng.integers(len(x), size=(10000, len(x)))].mean(1), [.025, .975]).tolist()


def stored(run):
    return {(seed, arm): {entry['case']:entry for entry in read(
        run / f'analysis/seed{seed}/{arm}/summary.json')['cases']}
        for seed in SEEDS for arm in ('A', 'KV-pos')}


def entries(saved, seed, family, conditions):
    arm = 'A' if family == 'A' else 'KV-pos'
    return {case: [entry['conditions'][c]['groups']['all']['confusion'] for c in conditions]
            for case, entry in saved[seed, arm].items()}


def coarse_shift(cm, family, direction):
    value = stats(cm, family)
    return rate(value[direction], value['covered'])


def audit(out, saved):
    control_rows, tissue_rows, gate_rows = [], [], []
    for seed in SEEDS:
        full = entries(saved, seed, 'full', ('full',))
        for family, repeats in FAMILIES.items():
            cases = entries(saved, seed, family, repeats)
            for case, cms in cases.items():
                base = metrics(full[case][0])
                values = [dict(**metrics(cm), **stats(cm)) for cm in cms]
                row = {key: average(v[key] for v in values) for key in values[0]}
                control_rows.append(dict(seed=seed, condition=family, case=case, repeats=len(cms),
                                         dice_delta_vs_full_pp=row['dice_pct']-base['dice_pct'], **row))
                for tissue in ('all', 'vertebra', 'disc'):
                    values = [stats(cm, tissue) for cm in cms]
                    tissue_rows.append(dict(seed=seed, condition=family, case=case, tissue=tissue,
                                            **{key: average(v[key] for v in values) for key in values[0]}))
        for case, entry in saved[seed, 'KV-pos'].items():
            for stride, row in zip((8, 16), entry['attention'], strict=True):
                gate_rows.append(dict(seed=seed, case=case, stride=stride, gate=row['gate'],
                                      residual_rms_ratio=row['residual_rms_ratio']))

    csv_write(out/'interventions_by_case.csv', control_rows)
    csv_write(out/'tissue_numbering_by_case.csv', tissue_rows)
    csv_write(out/'gates_by_case.csv', gate_rows)
    controls, tissues, gates = [], [], []
    for seed in SEEDS:
        for family in FAMILIES:
            rows = [r for r in control_rows if r['seed']==seed and r['condition']==family]
            controls.append(dict(seed=seed, condition=family, cases=len(rows),
                                 **{k:average(r[k] for r in rows) for k in (
                                     'dice_pct', 'iou_pct', 'union_dice_pct', 'dice_delta_vs_full_pp',
                                     'conditional_wrong_pct', 'wrong_of_gt_pct', 'coverage_pct')},
                                 dice_delta_case_ci95=bootstrap([r['dice_delta_vs_full_pp'] for r in rows]),
                                 absent_class_fp_pairs=sum(r['absent_class_fp_count'] for r in rows),
                                 absent_class_fp_voxels=sum(r['absent_class_fp_voxels'] for r in rows)))
            for tissue in ('all', 'vertebra', 'disc'):
                rows = [r for r in tissue_rows if r['seed']==seed and r['condition']==family and r['tissue']==tissue]
                tissues.append(dict(seed=seed, condition=family, tissue=tissue,
                                    **{k:average(r[k] for r in rows) for k in (
                                        'one_of_gt_pct','multiple_of_gt_pct','one_conditional_pct',
                                        'multiple_conditional_pct','conditional_wrong_pct','coverage_pct')},
                                    wrong_one_voxels=sum(r['wrong_one'] for r in rows),
                                    wrong_multiple_voxels=sum(r['wrong_multiple'] for r in rows)))
        for stride in (8, 16):
            rows = [r for r in gate_rows if r['seed']==seed and r['stride']==stride]
            gates.append(dict(seed=seed, stride=stride, gate=average(r['gate'] for r in rows),
                              rms_mean=average(r['residual_rms_ratio'] for r in rows),
                              rms_median=float(np.median([r['residual_rms_ratio'] for r in rows])),
                              rms_max=max(r['residual_rms_ratio'] for r in rows)))
    csv_write(out/'interventions.csv', controls)
    csv_write(out/'tissue_numbering.csv', tissues)
    csv_write(out/'gates.csv', gates)

    compare = []
    for case in saved[43, 'KV-pos']:
        a = saved[43, 'A'][case]['conditions']['full']['groups']['all']['confusion']
        kv = saved[43, 'KV-pos'][case]['conditions']['full']['groups']['all']['confusion']
        ma, mk = metrics(a), metrics(kv)
        sa, sk = stats(a), stats(kv)
        row = dict(case=case, a_dice_pct=ma['dice_pct'], kv_pos_dice_pct=mk['dice_pct'],
                   delta_dice_pp=mk['dice_pct']-ma['dice_pct'],
                   a_wrong_conditional_pct=sa['conditional_wrong_pct'],
                   kv_pos_wrong_conditional_pct=sk['conditional_wrong_pct'],
                   a_coverage_pct=sa['coverage_pct'], kv_pos_coverage_pct=sk['coverage_pct'],
                   a_absent_class_fp_count=ma['absent_class_fp_count'],
                   kv_pos_absent_class_fp_count=mk['absent_class_fp_count'],
                   kv_pos_wrong_one_share_pct=rate(sk['wrong_one'], sk['wrong_one']+sk['wrong_multiple']))
        for tissue in IDS:
            for arm, cm in (('a', a), ('kv_pos', kv)):
                t = stats(cm, tissue)
                for sign in ('plus_one', 'minus_one'):
                    row[f'{arm}_{tissue}_{sign}_covered_pct'] = rate(t[sign], t['covered'])
        compare.append(row)
    compare.sort(key=lambda row: row['delta_dice_pp'])
    csv_write(out/'seed43_cases.csv', compare)
    return compare, controls, tissues, gates


def correlation(rows, field, exclude=None):
    pairs = [(r['teacher_conditional_wrong_pct'], r[field]) for r in rows
             if r['case'] != exclude and r['teacher_conditional_wrong_pct'] is not None and r[field] is not None]
    if len(pairs) < 3:
        return dict(n=len(pairs), pearson=None, spearman=None)
    x, y = np.asarray(pairs).T
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), pearson=None, spearman=None)
    return dict(n=len(x), pearson=float(pearsonr(x,y).statistic), spearman=float(spearmanr(x,y).statistic))


def additional(out, compare):
    extra = {seed: read(out / f'seed{seed}/detail.json') for seed in SEEDS}
    for seed, detail in extra.items():
        if (len(detail['cases']) != 21 or detail['test_read'] or detail['validated_cache_slices'] != 418
                or read(out/f'seed{seed}/status.json')['status'] != 'completed'):
            raise ValueError(f'Incomplete diagnostics for seed {seed}')
    refs, ensembles, oracles, shifts = [], [], [], []
    for seed, detail in extra.items():
        for entry in detail['cases']:
            case = entry['case']
            reference = stats(entry['source_reference']['confusion'])
            single = stats(entry['ensembles']['top1']['confusion'])
            oscores = entry['oracle_excluding_reference']
            nonref = stats(oscores['pred']['confusion'])
            refs.append(dict(seed=seed, case=case, selected_reference=entry['selected_reference'],
                             **{f'teacher_{k}': v for k, v in reference.items()},
                             volume_conditional_wrong_pct=single['conditional_wrong_pct'],
                             nonref_conditional_wrong_pct=nonref['conditional_wrong_pct']))
            for k in (1,3,5):
                cm = entry['ensembles'][f'top{k}']['confusion']
                target_macs = sum(v['macs'] for v in entry['target_costs'][:k])
                target_seconds = sum(v['seconds'] for v in entry['target_costs'][:k])
                ensembles.append(dict(seed=seed, case=case, k=k, **metrics(cm), **stats(cm),
                                      references=','.join(map(str,entry['ranked_indices'][:k])),
                                      pipeline_macs=target_macs+entry['source_cost']['conv_macs'],
                                      target_seconds=target_seconds))
            for condition, score in oscores.items():
                cm = score['confusion']
                oracles.append(dict(seed=seed, case=case, condition=condition, **metrics(cm), **stats(cm)))
            if seed == 43:
                a_cms = np.asarray(entry['slice_confusions']['A'])
                a_outside = a_cms.sum(0)-a_cms[entry['selected_reference']]
                oracles.append(dict(seed=seed, case=case, condition='A', **metrics(a_outside), **stats(a_outside)))
                for arm, cms in entry['slice_confusions'].items():
                    for tissue in IDS:
                        for index, cm in enumerate(cms):
                            shifts.append(dict(case=case, arm=arm, tissue=tissue, slice_index=index,
                                               **stats(cm, tissue)))
    csv_write(out/'teacher_reference_vs_volume.csv', refs)
    csv_write(out/'ensemble_by_case.csv', ensembles)
    csv_write(out/'seed43_slice_numbering.csv', shifts)
    diagnostic_dir = out/'gt_oracle_diagnostic_only'
    diagnostic_dir.mkdir(exist_ok=True)
    csv_write(diagnostic_dir/'by_case.csv', oracles)

    correlations = []
    worst = compare[0]['case']
    for seed in SEEDS:
        rows = [r for r in refs if r['seed'] == seed]
        for field in ('volume_conditional_wrong_pct', 'nonref_conditional_wrong_pct'):
            for exclude in (None, worst):
                correlations.append(dict(seed=seed, outcome=field, excluded_case=exclude,
                                         **correlation(rows, field, exclude)))
    csv_write(out/'teacher_correlations.csv', correlations)

    esummary, osummary = [], []
    for seed in SEEDS:
        base = {r['case']:r for r in ensembles if r['seed']==seed and r['k']==1}
        for k in (1,3,5):
            rows = [r for r in ensembles if r['seed']==seed and r['k']==k]
            delta = [r['dice_pct']-base[r['case']]['dice_pct'] for r in rows]
            esummary.append(dict(seed=seed, k=k, dice_delta_pp=average(delta),
                                 dice_delta_ci95=bootstrap(delta), **{m:average(r[m] for r in rows) for m in (
                                     'dice_pct','iou_pct','union_dice_pct','conditional_wrong_pct','coverage_pct',
                                     'pipeline_macs','target_seconds')}))
        pred = {r['case']:r for r in oracles if r['seed']==seed and r['condition']=='pred'}
        gt = {r['case']:r for r in oracles if r['seed']==seed and r['condition']=='gt'}
        a = [r for r in oracles if r['seed']==seed and r['condition']=='A']
        delta = [gt[c]['dice_pct']-pred[c]['dice_pct'] for c in pred]
        best_case = max(pred, key=lambda case: gt[case]['dice_pct']-pred[case]['dice_pct'])
        osummary.append(dict(seed=seed, pred_dice=average(r['dice_pct'] for r in pred.values()),
                             gt_dice=average(r['dice_pct'] for r in gt.values()),
                             a_dice_on_same_support=average(r['dice_pct'] for r in a),
                             dice_delta_pp=average(delta), dice_delta_ci95=bootstrap(delta),
                             pred_wrong=average(r['conditional_wrong_pct'] for r in pred.values()),
                             gt_wrong=average(r['conditional_wrong_pct'] for r in gt.values()),
                             worst_case=worst, worst_case_pred_dice=pred[worst]['dice_pct'],
                             worst_case_gt_dice=gt[worst]['dice_pct'],
                             worst_case_pred_wrong=pred[worst]['conditional_wrong_pct'],
                             worst_case_gt_wrong=gt[worst]['conditional_wrong_pct'],
                             largest_gain_case=best_case,
                             largest_gain_pp=gt[best_case]['dice_pct']-pred[best_case]['dice_pct']))
    csv_write(out/'ensemble.csv', esummary)
    csv_write(diagnostic_dir/'summary.csv', osummary)

    across = []
    for k in (1,3,5):
        rows = [r for r in esummary if r['k']==k]
        dice = [r['dice_pct'] for r in rows]
        case_delta = []
        for case in sorted({r['case'] for r in ensembles}):
            case_delta.append(average(
                next(r['dice_pct'] for r in ensembles if (r['case'],r['seed'],r['k'])==(case,s,k))
                - next(r['dice_pct'] for r in ensembles if (r['case'],r['seed'],r['k'])==(case,s,1))
                for s in SEEDS))
        across.append(dict(k=k, mean_dice=average(dice), sd_dice=float(np.std(dice,ddof=1)),
                           mean_delta_pp=average(case_delta), delta_case_ci95=bootstrap(case_delta),
                           mean_iou=average(r['iou_pct'] for r in rows),
                           mean_union_dice=average(r['union_dice_pct'] for r in rows),
                           mean_wrong_conditional=average(r['conditional_wrong_pct'] for r in rows),
                           mean_coverage=average(r['coverage_pct'] for r in rows),
                           mean_pipeline_macs=average(r['pipeline_macs'] for r in rows),
                           mean_target_seconds=average(r['target_seconds'] for r in rows)))
    csv_write(out/'ensemble_three_seed.csv', across)

    evidence = []
    for row in compare:
        case = row['case']
        for tissue in IDS:
            for arm in ('A','KV_pos'):
                rr = [r for r in shifts if r['case']==case and r['tissue']==tissue and r['arm']==arm and r['covered']]
                if rr:
                    evidence.append(dict(case=case, tissue=tissue, arm=arm, slices_with_coverage=len(rr),
                        plus_one_majority_slices=sum(r['plus_one']*2>r['covered'] for r in rr),
                        minus_one_majority_slices=sum(r['minus_one']*2>r['covered'] for r in rr),
                        plus_one_shift_pct=rate(sum(r['plus_one'] for r in rr),sum(r['covered'] for r in rr)),
                        minus_one_shift_pct=rate(sum(r['minus_one'] for r in rr),sum(r['covered'] for r in rr))))
    csv_write(out/'seed43_shift_evidence.csv', evidence)
    return correlations, esummary, across, evidence, osummary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--archived-only', action='store_true')
    args = p.parse_args()
    saved = stored(args.run)
    compare, controls, tissues, gates = audit(args.out, saved)
    print('Worst seed43 cases:', [(r['case'], round(r['delta_dice_pp'],3)) for r in compare[:5]])
    if not args.archived_only:
        correlations, ensemble, across, shifts, _ = additional(args.out, compare)
        dump(args.out/'summary.json', dict(seed43_cases=compare, interventions=controls, tissue_numbering=tissues,
             gates=gates, correlations=correlations, ensemble=ensemble, ensemble_three_seed=across,
             seed43_shift_evidence=shifts))
        print('Ensemble across seeds:', across)


if __name__ == '__main__':
    main()
