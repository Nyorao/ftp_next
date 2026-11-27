"""Recompute transport screening from voxel counts and audit counts, no models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize_factor_probe import paired_comparison


LABELS = dict(teacher1='教师A（未更新）', teacher2='教师B（未更新）', mean='原始平均',
              transport_mean_a='A锚定搬运平均', transport_mean_b='B锚定搬运平均',
              abstain_mean_a='A锚定弃用平均')


def summarize(run):
    run = Path(run)
    names = ('manifest.json', 'status.json', 'results.json', 'native_valid.json', 'valid_audit.json', 'train_audit.json')
    data = {name: json.loads((run/name).read_text()) for name in names}
    manifest, result = data['manifest.json'], data['results.json']
    if data['status.json']['status'] != 'completed' or result['status'] != 'completed':
        raise ValueError('Only completed runs may be summarized')
    if (result['test_read'] or manifest['test_read'] or manifest['unlabeled_annotations_read']
            or result['model_updates'] or not result['teacher_parameters_unchanged']):
        raise ValueError('Frozen probe isolation contract violated')
    if not manifest['smoke'] and (result['valid_cases'] != 21 or result['valid_slices'] != 418):
        raise ValueError('Incomplete validation inventory')
    scores = data['native_valid.json']; paired = {}; dice = {}
    for name in LABELS:
        rows = []
        for case in scores[name]['cases']:
            values = []
            for c in case['classes']:
                if c['class_id'] == 0 or c['gt_voxels'] == 0: continue
                value = 2*c['intersection_voxels']/(c['gt_voxels']+c['pred_voxels'])
                if abs(value-c['dice']) > 1e-12: raise ValueError('Class Dice disagrees with voxel counts')
                values.append(value)
            if not values: raise ValueError('No GT foreground for a validation case')
            rows.append(dict(case=case['case'], dice=float(np.mean(values))))
        if len(rows) != result['valid_cases']: raise ValueError('Case count mismatch')
        recorded = {row['case']: row['dice'] for row in scores[name]['per_case']}
        if len(recorded) != len(rows) or any(abs(row['dice']-recorded[row['case']]) > 1e-12 for row in rows):
            raise ValueError('Case means disagree with voxel-derived scores')
        paired[name] = rows; dice[name] = float(np.mean([row['dice'] for row in rows]))
        if abs(dice[name]-result['dice'][name]) > 1e-12: raise ValueError('Overall Dice mismatch')
    counts = data['valid_audit.json']['counts']; old = counts['a']['peer']; new = counts['a']['transport']
    information = {}
    for direction, methods in counts.items():
        peer, moved = methods['peer'], methods['transport']
        for row in methods.values():
            if row['correct_foreground']+row['wrong_foreground'] != row['selected_foreground']:
                raise ValueError('Pseudo foreground counts inconsistent')
            if (row['retained_peer_correct_foreground']+row['peer_correct_to_wrong']+
                    row['peer_correct_to_abstain'] != peer['correct_foreground']):
                raise ValueError('Partition of originally correct targets inconsistent')
        # With the fixed >0.5 confidence thresholds, transport never increases
        # the maximum class probability or changes an accepted background/canal argmax.
        # Therefore any accepted corrected label came from a previously
        # accepted numbered target. The own-only and both-wrong sets partition
        # the peer-wrong -> correct transitions.
        recovered = moved['peer_wrong_to_correct']-moved['correct_only_own_correct']
        if recovered < 0: raise ValueError('Correction partition inconsistent')
        information[direction] = dict(both_teachers_wrong_to_correct=recovered,
            own_correct_peer_wrong_redundant_correct=moved['correct_only_own_correct'],
            original_peer_only_correct=peer['correct_only_peer_correct'],
            retained_peer_only_correct=moved['correct_only_peer_correct'])
    p = manifest['protocol']; gain = 100*(dice['transport_mean_a']-dice['mean'])
    criteria = dict(minimum_dice_gain=gain >= p['minimum_gain_pp'],
        beats_abstention=dice['transport_mean_a'] > dice['abstain_mean_a'],
        retains_original_correct=new['retained_peer_correct_foreground'] >= p['minimum_retention']*old['correct_foreground'],
        does_not_increase_wrong_foreground=new['wrong_foreground'] <= old['wrong_foreground'])
    passed = not manifest['smoke'] and all(criteria.values())
    if criteria != result['criteria'] or passed != result['screening_passed']:
        raise ValueError('Recomputed fixed gate disagrees with runtime')
    comparisons = {f'{a}_vs_{b}': paired_comparison(paired[a], paired[b], seed=20261029)
        for a, b in [('transport_mean_a', 'mean'), ('transport_mean_a', 'abstain_mean_a'), ('transport_mean_b', 'mean')]}
    lines = ['# 冻结教师编号搬运筛查', '',
        '**软件短测，不用于效果判断。**' if manifest['smoke'] else
        ('**通过最低机制筛查；尚未证明单网络训练收益。**' if passed else '**未通过预设机制筛查。**'), '',
        f"共 {result['valid_cases']} 例 / {result['valid_slices']} 张valid，无模型更新、无test读取。", '',
        '| 输出 | 原生3D Dice % | 相对原平均 pp |', '|---|---:|---:|']
    for name, label in LABELS.items():
        lines.append(f"| {label} | {100*dice[name]:.3f} | {100*(dice[name]-dice['mean']):+.3f} |")
    lines += ['', '后四行使用双教师预测。主部署网络A没有更新；融合改善不能当作单网络改善。绝对编号来自接收教师，不能据此断言接收教师编号正确。', '',
        f"非零偏移发生于 {result['nonzero_offset_slices']} 张valid切片 / {len(result['nonzero_offset_cases'])} 个病例。", '',
        '| 比较 | 平均差 pp | 病例重采样95%区间 pp | 胜/负/平 |', '|---|---:|---:|---:|']
    for name, row in comparisons.items():
        lo, hi = row['case_bootstrap_95_pp']
        lines.append(f"| {name} | {row['mean_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['wins']}/{row['losses']}/{row['ties']} |")
    lines += ['', '上述区间只反映固定模型的病例差异，不含训练种子或后续学生训练的不确定性；valid已用于开发。', '',
        '| 接收方向 | 目标 | 正确前景目标数 | 错误前景目标数 | 原正确保留率 % | 原正确改错 | 原错误改对 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for direction in ('a', 'b'):
        for method in ('peer', 'abstain', 'transport'):
            c = counts[direction][method]
            retention = c['retained_peer_correct_foreground']/counts[direction]['peer']['correct_foreground']
            lines.append(f"| {direction.upper()} | {method} | {c['correct_foreground']:,} | {c['wrong_foreground']:,} | {100*retention:.3f} | {c['peer_correct_to_wrong']:,} | {c['peer_wrong_to_correct']:,} |")
    lines += ['', '伪目标统计使用224²网格和原置信阈值。前景指被接受的预测前景，错误数包含GT背景上的误报；原正确保留率按同一批原先正确且被接受的目标逐像素计算。表中计数不是原生体素数。', '',
        '| 接收方向 | 两教师均错→目标正确 | 接收方本就正确的重复目标 | 原同伴独有正确 | 保留的同伴独有正确 |',
        '|---|---:|---:|---:|---:|']
    for direction, row in information.items():
        lines.append(f"| {direction.upper()} | {row['both_teachers_wrong_to_correct']:,} | {row['own_correct_peer_wrong_redundant_correct']:,} | {row['original_peer_only_correct']:,} | {row['retained_peer_only_correct']:,} |")
    lines += ['', '纯整体偏移的搬运目标可以等于接收教师自己的原硬标签。新纠正与重复监督必须区分；即使最低门槛通过，仍须对照相同接受掩码下改用自身EMA标签，才能判断是否超出自训练。上表的两者均错恢复由同伴错误改正数减去接收方原本正确区域的改正数得到，适用于本轮不归一、不增大概率且置信阈值高于0.5的搬运规则。', '',
        '固定条件：', '']
    for name, value in criteria.items(): lines.append(f'- `{name}`：{value}')
    lines += ['', '未将B方向或短测改选为主要结果，也没有按本轮结果修改偏移阈值。原标签的文字名称沿用历史文件，以上比较仅按数值类别ID，不将相对ID解释为确定的解剖L1/L5。', '',
        'Verification Status: ANALYZED。原生Dice从交集/预测/真值体素数重算并核对，门槛按预先冻结协议重算；本报告不重新执行推理。输入哈希、逐病例差及脚本哈希见 `analysis_summary.json`。']
    summary = dict(screening_passed=passed, smoke=manifest['smoke'], criteria=criteria, dice=dice,
        gain_over_mean_pp=gain, comparisons=comparisons, information_transfer=information,
        single_network_efficacy_verified=False,
        input_sha256={name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in names},
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'analysis_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    return {k: summary[k] for k in ('screening_passed', 'smoke', 'gain_over_mean_pp')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(summarize(args.run)))
