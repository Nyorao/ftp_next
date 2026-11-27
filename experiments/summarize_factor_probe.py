"""Recompute a completed probe's fixed gate and paired validation diagnostics.

This is analysis only: no model loading, label reading, fitting, threshold search,
or checkpoint selection. Smoke runs remain explicitly ineligible for screening.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES = ('teacher1', 'teacher2', 'mean', 'flat', 'factor_tissue', 'factor_level', 'factor')
LABELS = dict(teacher1='教师A', teacher2='教师B', mean='原始算术平均', flat='完整类别门',
              factor_tissue='仅组织门', factor_level='仅编号门', factor='完整因子门')


def paired_comparison(first, second, *, seed=20261028, samples=10000):
    def index(rows):
        result = {row['case']: float(row['dice']) for row in rows}
        if len(result) != len(rows) or not result or not all(0 <= v <= 1 for v in result.values()):
            raise ValueError('Expected unique cases with finite Dice in [0,1]')
        return result
    a, b = index(first), index(second)
    if set(a) != set(b):
        raise ValueError('Paired comparison requires identical case identifiers')
    cases = sorted(a)
    differences = 100*np.array([a[case]-b[case] for case in cases])
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(cases), (samples, len(cases)))].mean(1)
    return dict(mean_pp=float(differences.mean()),
                case_bootstrap_95_pp=np.quantile(means, [.025, .975]).tolist(),
                wins=int((differences > 1e-10).sum()), losses=int((differences < -1e-10).sum()),
                ties=int((np.abs(differences) <= 1e-10).sum()),
                differences_pp=dict(zip(cases, differences.tolist())),
                seed=seed, resamples=samples, training_seed_uncertainty_included=False)


def summarize(run):
    run = Path(run).resolve()
    filenames = ('manifest.json', 'status.json', 'results.json', 'native_valid.json',
                 'valid_diagnostics.json', 'train_diagnostics.json')
    payload = {name: json.loads((run/name).read_text()) for name in filenames}
    status, result, manifest = [payload[name] for name in ('status.json', 'results.json', 'manifest.json')]
    if status['status'] != 'completed' or result['status'] != 'completed':
        raise ValueError('Probe is not complete; no partial-result interpretation')
    if (result['test_read'] or manifest['test_read'] or not result['teacher_parameters_unchanged']
            or manifest['segmentation_networks_trained'] or manifest['unlabeled_annotations_read']):
        raise ValueError('Probe isolation contract not satisfied')
    if result['final_step'] != manifest['protocol']['steps']:
        raise ValueError('Update budget mismatch')
    scores, diagnostic, train = [payload[name] for name in ('native_valid.json', 'valid_diagnostics.json', 'train_diagnostics.json')]
    cases = result['valid_cases']
    if not manifest['smoke'] and (cases != 21 or result['valid_slices'] != 418):
        raise ValueError('Full probe must cover the fixed 21 cases / 418 slices')
    dice = {}
    for name in NAMES:
        rows = scores[name]['per_case']
        # Also checks duplicate IDs and range before using the case mean.
        paired_comparison(rows, scores['mean']['per_case'], samples=1)
        if len(rows) != cases:
            raise ValueError(f'Incomplete validation cases: {name}')
        value = float(np.mean([row['dice'] for row in rows]))
        for recorded in (scores[name]['overall']['dice'], result['dice'][name]):
            if abs(value-recorded) > 1e-12:
                raise ValueError(f'Per-case mean disagrees with stored Dice: {name}')
        dice[name] = value
    nll = {name: diagnostic['nll'][name]['foreground_nll_sum'] /
                diagnostic['nll'][name]['foreground_pixels'] for name in NAMES}
    for name in NAMES:
        if abs(nll[name]-result['foreground_nll'][name]) > 1e-12:
            raise ValueError(f'NLL sum/count disagrees with stored result: {name}')
    gain = 100*(dice['factor']-dice['mean'])
    criteria = dict(minimum_dice_gain=gain >= manifest['protocol']['minimum_gain_pp'],
                    beats_flat_gate=dice['factor'] > dice['flat'],
                    foreground_nll_not_worse=nll['factor'] <= nll['mean']+1e-6)
    passed = not manifest['smoke'] and all(criteria.values())
    if criteria != result['screening_criteria'] or passed != result['screening_passed']:
        raise ValueError('Recomputed fixed screening gate disagrees with stored gate')
    comparisons = {f'{a}_vs_{b}': paired_comparison(scores[a]['per_case'], scores[b]['per_case'])
                   for a, b in (('factor', 'mean'), ('factor', 'flat'), ('flat', 'mean'),
                                ('factor_tissue', 'mean'), ('factor_level', 'mean'))}
    calibration = {}
    for split, source in (('train', train), ('valid', diagnostic)):
        rows = []
        for mode, heads in (('flat', ('完整类别',)), ('factor', ('组织', '椎体编号', '椎间盘编号'))):
            raw = source['calibration'][mode]
            for i, head in enumerate(heads):
                w, zero = raw['weight'][i], raw['zero_mse_sum'][i]
                rows.append(dict(mode=mode, head=head,
                    mse=raw['mse_sum'][i]/w if w else None,
                    zero_mse=zero/w if w else None,
                    mse_over_zero=raw['mse_sum'][i]/zero if zero else None,
                    descent_alignment=raw['descent_alignment_sum'][i]/w if w else None,
                    direction_accuracy=raw['correct_direction'][i]/raw['informative'][i]
                        if raw['informative'][i] else None,
                    informative_pixels=raw['informative'][i]))
        calibration[split] = rows
    summary = dict(smoke=manifest['smoke'], screening_passed=passed, criteria=criteria,
        single_network_efficacy_verified=False, factor_gain_over_mean_pp=gain,
        comparisons=comparisons, calibration=calibration,
        input_sha256={name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in filenames},
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    lines = ['# 冻结教师优势融合探针', '',
        '**软件短测；不得用于效果筛查。**' if manifest['smoke'] else
        ('**通过预设冻结教师筛查；尚未验证单网络分割收益。**' if passed else
         '**未通过预设冻结教师筛查，不据此启动完整分割训练。**'), '',
        f"固定最终第 {result['final_step']} 步，{cases} 例 / {result['valid_slices']} 切片 valid；未读取 test，未更新分割教师。", '',
        '| 融合 | 原生3D Dice % | 相对平均 pp | 前景NLL | 组织NLL | 条件编号NLL |',
        '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        row = diagnostic['nll'][name]
        lines.append(f"| {LABELS[name]} | {100*dice[name]:.3f} | {100*(dice[name]-dice['mean']):+.3f} | {nll[name]:.5f} | {row['tissue_nll']:.5f} | {row['conditional_nll']:.5f} |")
    lines += ['', 'NLL 为224²网格上的像素汇总：组织NLL在全部真值前景上计算，条件编号NLL仅在椎体/椎间盘上计算，二者不能直接相加成表中前景NLL。Dice沿用原生3D逐病例等权指标。仅组织门/仅编号门是同一个完整router的推理干预，不是分别训练的单因素模型。', '',
              '## 固定条件与逐病例差异', '',
              f"完整方案比平均至少增加 {manifest['protocol']['minimum_gain_pp']:.1f}pp：{criteria['minimum_dice_gain']}；超过完整类别门：{criteria['beats_flat_gate']}；前景NLL不增加：{criteria['foreground_nll_not_worse']}。", '',
              '| 比较 | 平均差 pp | 病例重采样95%区间 pp | 胜/负/平 |',
              '|---|---:|---:|---:|']
    for name, row in comparisons.items():
        low, high = row['case_bootstrap_95_pp']
        lines.append(f"| {name} | {row['mean_pp']:+.3f} | [{low:+.3f}, {high:+.3f}] | {row['wins']}/{row['losses']}/{row['ties']} |")
    lines += ['', '区间只反映固定教师、固定训练种子的验证病例差异，不包含训练随机性。组织/编号单项是机制诊断；不能在完整方案失败后改选单项并宣称通过。', '',
              '## 优势预测是否泛化', '',
              '| 数据 | 预测头 | MSE / 零预测MSE | 与真实下降方向的平均乘积 | 方向一致率 % |',
              '|---|---|---:|---:|---:|']
    for split, rows in calibration.items():
        for row in rows:
            fmt = lambda v: 'NA' if v is None else f'{v:.6g}'
            acc = row['direction_accuracy']
            lines.append(f"| {split} | {row['mode']}/{row['head']} | {fmt(row['mse_over_zero'])} | {fmt(row['descent_alignment'])} | {fmt(None if acc is None else 100*acc)} |")
    lines += ['', 'MSE比值小于1表示优于始终预测零调整；乘积为正只表示平均局部方向有利，不保证有限门值或Dice改善。方向一致率只统计真实优势绝对值至少0.001的像素；训练探针为固定抽取的至多256张未增强标注切片，与valid的病例和类别分布不同。教师已学过这些训练标签，训练表现不能替代未见患者泛化。', '',
              '## 互补信息诊断', '']
    opportunity = diagnostic['opportunity']
    for key, value in opportunity.items():
        lines.append(f'- `{key}`：{value:,}')
    lines += ['', 'oracle_local_direction 使用真值计算局部一步调整，只是不可部署的诊断，也不是全局最优上界。它没有参与训练、选模或上述方法Dice。', '',
              '本探针比较的是冻结双教师输出融合；即使通过，也不等于把收益传入单个学生。factor按每头支持分别归一化，flat按全部加权像素归一化，比较同时包含组织重加权的影响。后续实验须保留平均软蒸馏对照，固定教师、初始化、像素选择规则、更新预算和单网络选模；单独归因门解耦还需要匹配有效像素权重的flat对照。', '',
              '输入哈希、完整逐病例差异和重算结果见 `analysis_summary.json`。']
    return summary, '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    summary, report = summarize(args.run)
    (args.run/'analysis_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (args.run/'report.md').write_text(report)
    print(json.dumps({k: summary[k] for k in ('smoke', 'screening_passed', 'factor_gain_over_mean_pp')}))


if __name__ == '__main__':
    main()
