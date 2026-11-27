"""Case-paired summaries for the frozen context probe (never trains a model)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from freqssl.context_diagnostics import CONDITIONS, REGIONS, bootstrap_cases, utility_contrasts
from freqssl.engine import write_json


def average(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def correlation(x, y):
    pairs = [(a,b) for a,b in zip(x,y) if a is not None and b is not None and np.isfinite(a+b)]
    if len(pairs) < 5:
        return None
    a, b = np.asarray(pairs).T
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    return float(spearmanr(a, b).statistic)


def clustered_summary(by_seed):
    """Seeds share patients: average seeds within each case before bootstrapping."""
    cases = sorted({case for rows in by_seed.values() for case in rows})
    per_case = {case:average([rows.get(case) for rows in by_seed.values()]) for case in cases}
    per_seed = {seed:average(rows.values()) for seed,rows in by_seed.items()}
    result = bootstrap_cases(per_case.values())
    result.update(per_seed=per_seed, per_case=per_case,
                  scope='exploratory development-validation case variability; seeds are not independent cases')
    return result


def signal_consistent(summary, expected_sign=None):
    """Exploratory signal rule; not a confirmatory significance or novelty test."""
    ci, mean = summary.get('ci95'), summary.get('mean')
    values = [v for v in summary.get('per_seed', {}).values() if v is not None]
    if not ci or mean is None or len(values) != 3 or summary['n_cases'] < 10:
        return False
    sign = expected_sign if expected_sign is not None else (1 if mean > 0 else -1)
    return (ci[0] > 0 if sign > 0 else ci[1] < 0) and sum(v*sign > 0 for v in values) >= 2


def region_error(rows, condition, region):
    values = [r['conditions'][condition]['regions'][region] for r in rows]
    count = sum(v['pixels'] for v in values)
    return sum(v['errors'] for v in values)/count if count else None


def groups(row):
    return ['all', 'endpoint' if row['endpoint'] else 'interior_slice',
            f"position_{row['physical_position_bin']}", f"spacing_{row['spacing_bin']}",
            'gap3_complete' if row['both_gap3_real'] else 'gap3_clipped']


def case_associations(case, region):
    rows = [r for r in case['slices'] if r['both_adjacent_real'] and
            r['conditions']['full']['regions'][region]['pixels'] > 0]
    data = defaultdict(list)
    for row in rows:
        e = {c:row['conditions'][c]['regions'][region]['errors']/row['conditions'][c]['regions'][region]['pixels']
             for c in CONDITIONS[:4]}
        u = utility_contrasts(e)
        fusion = [f for f in row['fusion'][region] if f is not None]
        if not fusion:
            continue
        data['weight_difference'].append(average([f['weights'][0]-f['weights'][2] for f in fusion]))
        data['residual_difference'].append(average([f['neighbor_residual_rms'][0]-f['neighbor_residual_rms'][1] for f in fusion]))
        data['direction_utility'].append(u['previous_minus_next'])
        data['position'].append(row['physical_position'])
        data['physical_direction_utility'].append(u['previous_minus_next'] *
            (1 if row['previous_physical_direction'] == 'left' else -1))
        for stride, f in zip((4,8,16), row['fusion'][region]):
            data[f'weight_difference_stride{stride}'].append(f['weights'][0]-f['weights'][2] if f else None)
            data[f'effective_difference_stride{stride}'].append(
                f['effective_coefficients'][0]-f['effective_coefficients'][2] if f else None)
    answer = {
        'position_vs_physical_direction_utility':correlation(data['position'], data['physical_direction_utility']),
        'weight_vs_direction_utility':correlation(data['weight_difference'], data['direction_utility']),
        'residual_vs_direction_utility':correlation(data['residual_difference'], data['direction_utility'])}
    for stride in (4,8,16):
        for name in ('weight', 'effective'):
            answer[f'{name}_stride{stride}_vs_direction_utility'] = correlation(
                data[f'{name}_difference_stride{stride}'], data['direction_utility'])
    for side, intervention in (('previous','no_previous'), ('next','no_next')):
        usefulness, mismatch, shift, texture, quality, entropies = [], [], [], [], [], []
        for row in rows:
            full = row['conditions']['full']['regions'][region]
            ablated = row['conditions'][intervention]['regions'][region]
            usefulness.append((ablated['errors']-full['errors'])/full['pixels'])
            relationships = row['feature_relationships'][side][region]
            mismatch.append(relationships['cosine_distance'])
            shift.append(relationships['local_match_gain'])
            texture.append(row[side+'_quality']['laplacian_variance'])
            quality.append(row[side+'_quality']['high_frequency_mad'])
            entropies.append(full['entropy_sum']/full['pixels'])
        for name, x in [('mismatch',mismatch),('local_match_gain',shift),('laplacian',texture),
                        ('high_frequency',quality),('full_entropy',entropies)]:
            answer[f'{side}_{name}_vs_utility'] = correlation(x, usefulness)
    return answer


def direction_candidates(associations, contrasts, paired, complete):
    def association(name, region='foreground', sign=None):
        return signal_consistent(associations.get(region, {}).get(name, {}), sign)
    position = association('position_vs_physical_direction_utility')
    distance = any(signal_consistent(paired[c]['dice'], 1) and paired[c]['dice_union_present']['mean'] >= 0
                   for c in ('gap2','gap3'))
    reliability = any(association(f'{side}_{name}_vs_utility', sign=-1)
                      for side in ('previous','next') for name in ('mismatch','high_frequency','full_entropy'))
    interaction = contrasts.get('interior_slice', {}).get('foreground', {}).get('interaction', {})
    complement = signal_consistent(interaction)
    alignment = any(association(f'{side}_local_match_gain_vs_utility', 'boundary', -1)
                    for side in ('previous','next'))
    definitions = [
        ('position', '位置相关的方向贡献', position,
         '至少10病例的逐例位置—物理方向效用相关：病例bootstrap区间不跨0，至少2/3种子同向。',
         '扫描相对位置不是解剖位置；病例间视野、结构出现/消失及输入替换均可能解释关联。',
         '训练内容权重与内容＋位置权重的匹配对照；位置只取自影像元数据。'),
        ('range', '物理距离与上下文范围', distance,
         'gap2或gap3完整体积主Dice差值的病例区间为正，至少2/3种子正向，union-present不下降。',
         '远邻属于冻结模型未训练的输入分布；未改善不能排除从头训练后的价值。',
         '先比较近邻三片、远邻三片与五片；自适应选择再对照最佳固定范围。'),
        ('reliability', '邻片可靠性与负面贡献', reliability,
         '特征差异、纹理代理或预测熵与邻片效用存在跨种子稳定的负关联。',
         '纹理和熵并非校准质量；关联可能来自器官边界、难例和背景比例。',
         '先在有标注训练病例交叉验证效用可预测性，再设计可靠性融合，禁止拟合验证标签。'),
        ('complementarity', '互补性与空间对应', complement or alignment,
         '双侧交互项稳定非零，或局部匹配增益与边界邻片效用稳定负相关。',
         '非线性解码器和替换分布可产生交互；局部余弦匹配不是解剖配准。',
         '交互证据支持差异特征对照；错配证据支持局部对应对照，两者不同时添加。')]
    result, selected = [], 0
    for key, name, supported, rule, alternative, next_step in definitions:
        supported = bool(supported and complete)
        chosen = supported and selected < 2
        selected += chosen
        result.append(dict(id=key, name=name, signal_supported=supported,
            selected_for_method_design=bool(chosen), evidence_rule=rule,
            status='exploratory_signal' if supported else 'hypothesis_only',
            alternative_explanation=alternative, next_step=next_step,
            training_authorized_by_this_report=False))
    return result


def additional_diagnostics(records, metrics):
    harmful, false_positives = {}, {}
    for region in ('foreground','boundary','interior'):
        harmful[region] = {}
        for side, condition in (('previous','no_previous'),('next','no_next')):
            by_seed = {}
            for seed,cases in records.items():
                by_seed[seed] = {}
                for case in cases:
                    rows = [r for r in case['slices'] if r['both_adjacent_real'] and
                            r['conditions']['full']['regions'][region]['pixels']]
                    by_seed[seed][case['case']] = average([
                        float(r['conditions'][condition]['regions'][region]['errors'] <
                              r['conditions']['full']['regions'][region]['errors']) for r in rows])
            harmful[region][side] = clustered_summary(by_seed)
    for condition in CONDITIONS:
        false_positives[condition] = {}
        for name in ('absent_class_pixels','slice_absent_class_pixels'):
            # Fractions on the same diagnostic grid; not native volume counts.
            false_positives[condition][name] = clustered_summary({seed:{case['case']:100*sum(
                r['conditions'][condition]['regions']['all'][name] for r in case['slices']) /
                sum(r['conditions'][condition]['regions']['all']['pixels'] for r in case['slices'])
                for case in cases} for seed,cases in records.items()})
    per_seed_winners, per_case = {}, defaultdict(dict)
    for seed, values in metrics.items():
        lookup = {condition:{r['case']:r['dice'] for r in values[condition]['per_case']}
                  for condition in ('full','gap2','gap3')}
        counts = {condition:0 for condition in lookup}
        for case in lookup['full']:
            best = max(lookup, key=lambda c:lookup[c][case])
            counts[best] += 1
            per_case[case][seed] = best
        per_seed_winners[seed] = counts
    return dict(harmful_slice_fraction=harmful, diagnostic_grid_false_positive_percent=false_positives,
        gamma_by_seed={seed:cases[0]['gamma'] for seed,cases in records.items()},
        range_winner_counts_by_seed=per_seed_winners,
        range_winner_by_case=per_case,
        range_winner_unanimous_cases=sum(len(set(v.values()))==1 for v in per_case.values()),
        range_winner_note='Retrospective best of three by validation labels; descriptive only, not a deployable selector.',
        reversed_physical_order_cases=[r['case'] for r in next(iter(records.values()))
                                       if r['geometry']['reverse_physical_order']])


def summarize(out):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    status = json.loads((out/'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Refuse to rank incomplete/failed runs')
    records, metrics = {}, {}
    for seed in manifest['checkpoints']:
        directory = out/f'seed{seed}'
        receipt = json.loads((directory/'receipt.json').read_text())
        if receipt['status'] != 'completed' or not receipt['model_state_unchanged']:
            raise ValueError('Missing successful frozen-model receipt')
        records[seed] = [json.loads(p.read_text()) for p in sorted((directory/'cases').glob('*.json'))]
        metrics[seed] = json.loads((directory/'metrics.json').read_text())
    inventory = [{r['case'] for r in cases} for cases in records.values()]
    if any(value != inventory[0] for value in inventory):
        raise ValueError('Cannot pair different case inventories')
    complete = set(records) == {'42','43','44'} and not manifest['max_cases']
    paired = {}
    for condition in CONDITIONS[1:]:
        paired[condition] = {}
        for metric in ('dice','dice_union_present'):
            by_seed = {}
            for seed, values in metrics.items():
                reference = {r['case']:r[metric] for r in values['full']['per_case']}
                by_seed[seed] = {r['case']:100*(r[metric]-reference[r['case']])
                                for r in values[condition]['per_case']}
            paired[condition][metric] = clustered_summary(by_seed)
        paired[condition]['absent_class_voxel_change'] = clustered_summary({seed:{r['case']:
            r['metrics'][condition]['native_absent_class_voxels']-r['metrics']['full']['native_absent_class_voxels']
            for r in cases} for seed,cases in records.items()})
    raw_contrasts = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(dict))))
    raw_associations = defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    distance_cases = defaultdict(dict)
    flat = []
    for seed, cases in records.items():
        for case in cases:
            strata = defaultdict(list)
            for row in case['slices']:
                for group in groups(row):
                    strata[group].append(row)
                for condition in CONDITIONS:
                    for region, score in row['conditions'][condition]['regions'].items():
                        flat.append(dict(seed=int(seed),case=case['case'],index=row['index'],
                            position=row['position'],physical_position=row['physical_position'],
                            position_bin=row['physical_position_bin'],spacing_mm=row['spacing_mm'],
                            spacing_bin=row['spacing_bin'],endpoint=row['endpoint'],
                            both_gap3_real=row['both_gap3_real'],condition=condition,region=region,**score))
            for group, rows in strata.items():
                for region in REGIONS:
                    errors = {c:region_error(rows,c,region) for c in CONDITIONS}
                    if errors['full'] is None:
                        continue
                    values = utility_contrasts(errors)
                    values.update({f'{c}_error_change':errors[c]-errors['full'] for c in ('swapped','gap2','gap3')})
                    for name, value in values.items():
                        raw_contrasts[group][region][name][seed][case['case']] = 100*value
            for region in ('foreground','boundary','interior'):
                for name, value in case_associations(case, region).items():
                    raw_associations[region][name][seed][case['case']] = value
            common = [r for r in case['slices'] if r['both_gap3_real']]
            distance_cases[seed][case['case']] = dict(spacing_mm=case['geometry']['spacing_mm'],
                error={c:region_error(common,c,'foreground') for c in ('full','gap2','gap3')})
    contrasts = {group:{region:{name:clustered_summary(values) for name,values in names.items()}
                       for region,names in regions.items()} for group,regions in raw_contrasts.items()}
    associations = {region:{name:clustered_summary(values) for name,values in names.items()}
                    for region,names in raw_associations.items()}
    distance_associations = {}
    for condition in ('gap2','gap3'):
        distance_associations[condition] = {seed:correlation([r['spacing_mm'] for r in cases.values()],
            [r['error'][condition]-r['error']['full'] if r['error'][condition] is not None else None
             for r in cases.values()]) for seed,cases in distance_cases.items()}
    candidates = direction_candidates(associations, contrasts, paired, complete)
    result = dict(schema_version=1, complete_three_seed_validation=complete, seeds=list(records),
        cases=len(inventory[0]), conditions=list(CONDITIONS), native_paired_delta_pp=paired,
        grid_error_contrasts_pp=contrasts, within_case_spearman=associations,
        distance_common_interior=distance_cases, spacing_vs_gap_error_spearman=distance_associations,
        diagnostics=additional_diagnostics(records, metrics),
        directions=candidates, model_updates=0, test_read=False,
        selection_limit='At most two exploratory directions; no architecture/training is justified by a heatmap alone.',
        statistical_scope='Descriptive development analysis. No multiplicity adjustment; no independent confirmation. '
                          'Average seeds within case before paired case bootstrap; pixels are not independent samples.')
    write_json(out/'summary.json', result)
    with (out/'slice_regions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    render_report(out, result, metrics, records)
    render_figures(out, result, records)
    return result


def fmt(summary):
    if summary['mean'] is None:
        return 'NA'
    lo, hi = summary['ci95']
    return f"{summary['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def render_report(out, result, metrics, records):
    lines = ['# 邻切片贡献图谱：冻结教师诊断', '', '## Material Passport', '',
        '来源：既有最佳EMA教师、完整valid影像与标注、经影像校验的几何记录；AI辅助实现与分析。',
        f"状态：{'完整三种子诊断' if result['complete_three_seed_validation'] else '仅链路短测，不能筛选方向'}；"
        f"{result['cases']}例，seeds={','.join(result['seeds'])}。训练更新0；未读test。", '',
        '## 原生三维结果', '', '| seed | 条件 | Dice% | union-present% | 缺失类别误检对数 |',
        '|---|---|---:|---:|---:|']
    for seed, conditions in metrics.items():
        for name in CONDITIONS:
            o = conditions[name]['overall']
            lines.append(f"| {seed} | {name} | {100*o['dice']:.4f} | {100*o['dice_union_present']:.4f} | {o['fp_absent_gt_class_case_pairs']} |")
    lines += ['', '正差表示该条件优于完整三片。区间按病例配对bootstrap；先在同病例内平均种子，未把63条种子×病例视为63个独立病例。', '',
              '| 条件相对full | 主Dice变化/pp [95%区间] | union-present变化/pp [95%区间] |', '|---|---:|---:|']
    for name, values in result['native_paired_delta_pp'].items():
        lines.append(f"| {name} | {fmt(values['dice'])} | {fmt(values['dice_union_present'])} |")
    lines += ['', '## 贡献与位置', '',
        '以下为224²诊断网格错误率变化，非原生三维Dice。正贡献表示移除相应信息后错误增加；交互项为双侧联合收益减去两个单侧收益。', '',
        '| 区域（非端点切片） | 上一片条件贡献/pp | 下一片条件贡献/pp | 联合贡献/pp | 交互项/pp |', '|---|---:|---:|---:|---:|']
    for region in ('foreground','vertebra','disc','canal','boundary','interior'):
        values = result['grid_error_contrasts_pp'].get('interior_slice',{}).get(region)
        if values:
            lines.append('| '+region+' | '+' | '.join(fmt(values[name]) for name in
                         ('previous_given_next','next_given_previous','joint','interaction'))+' |')
    lines += ['', '各相对位置五分组、物理间距组、端点复制组及各尺度权重见summary.json和逐病例JSON。',
              '', '## 机制证据与反例', '',
              '相关系数先在病例内计算，再对病例等权汇总。区间反映本验证集病例变异，种子固定。', '',
              '| 逐例相关（区域） | 平均ρ [95%区间] | 各seed平均ρ | 正/负相关病例数（种子均值） |',
              '|---|---:|---|---:|']
    associations = result['within_case_spearman']
    for region, names in (
        ('foreground', ['position_vs_physical_direction_utility', 'weight_vs_direction_utility',
            'residual_vs_direction_utility', *[f'{side}_{proxy}_vs_utility'
                for side in ('previous','next') for proxy in ('mismatch','high_frequency','full_entropy')]]),
        ('boundary', ['previous_local_match_gain_vs_utility','next_local_match_gain_vs_utility'])):
        for name in names:
            s = associations[region][name]
            seed_text = ', '.join(f'{seed}: {value:+.3f}' if value is not None else f'{seed}: NA'
                                  for seed,value in s['per_seed'].items())
            lines.append(f"| {name} ({region}) | {fmt(s)} | {seed_text} | "
                         f"{s.get('positive_cases',0)}/{s.get('negative_cases',0)} |")
    audit = result['diagnostics']
    lines += ['', 'γ（stride4/8/16）：'+json.dumps(audit['gamma_by_seed'])+'。',
        '方向反转病例（仅分析时校正）：'+str(audit['reversed_physical_order_cases'])+'。', '',
        '| 区域 | 上一片有害切片比例 [95%区间] | 下一片有害切片比例 [95%区间] |', '|---|---:|---:|']
    for region, values in audit['harmful_slice_fraction'].items():
        lines.append(f"| {region} | {fmt(values['previous'])} | {fmt(values['next'])} |")
    lines += ['', '有害定义为替换后该区域错误计数严格减少；排除端点，先算病例内比例，忽略零差异切片的有害计数。',
        '各seed事后最优范围病例数：'+json.dumps(audit['range_winner_counts_by_seed'])+'。',
        f"各seed最优范围一致的病例数：{audit['range_winner_unanimous_cases']}/{result['cases']}。"
        '这是用验证标签得到的描述，不是可部署选择器，也不作为性能估计。',
        '共同内部切片中间距与远邻错误变化的病例间Spearman：'+
        json.dumps(result['spacing_vs_gap_error_spearman'])+'。', '',
        '| 输入条件 | 整例缺失类别误检占全部网格像素% | 当前片缺失类别误检占全部网格像素% |', '|---|---:|---:|']
    for condition, values in audit['diagnostic_grid_false_positive_percent'].items():
        lines.append(f"| {condition} | {fmt(values['absent_class_pixels'])} | {fmt(values['slice_absent_class_pixels'])} |")
    lines += [
              '', '## 方向排序（探索性）', '']
    for i, item in enumerate(result['directions'],1):
        lines += [f"### {i}. {item['name']}", '',
            f"状态：{item['status']}；进入最多两个方法设计候选：{item['selected_for_method_design']}。",
            f"判据：{item['evidence_rule']}", f"替代解释：{item['alternative_explanation']}",
            f"后续验证：{item['next_step']}", '']
    lines += ['## 解释与后续训练边界', '',
        '- 替换与远邻输入是分布外干预；当前结果不等于解剖因果关系，也不能否定适配训练后的效果。',
        '- 保留历史采集顺序；物理左右只用于汇总。相对位置是扫描范围坐标，不是解剖坐标。',
        '- 权重与gamma共同决定残差；gamma可为负，不将softmax权重直接解释为贡献概率。',
        '- 纹理、特征距离和局部匹配都是代理量；边界为模型网格的两像素带，不是毫米边界指标。',
        '- 多个探索性分组与相关分析未进行多重比较校正；区间不构成独立确认或通用新颖性证明。',
        '- 历史CUDA分数与本次CPU分数允许预声明的0.01pp单病例误差；实际差值逐例记录在historical_parity.json。',
        '- 后续仅在冻结具体方法后启动：seed42/6000步筛查；通过后42/43/44各30000步，比较当前adaptive。',
        '- 确认门槛：平均主Dice至少+0.3pp、至少2/3种子正向、平均union-present不下降；验证标签不可用于训练门控。',
        '', '## 图与数据', '', '![原生三维差值](native_deltas.png)', '',
        '![位置与方向贡献](position_utility.png)', '',
        '数据：summary.json、slice_regions.csv、seed*/cases/*.json；NPZ保存局部预测、输入和权重图。', '']
    (out/'report.md').write_text('\n'.join(lines))


def render_figures(out, result, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':180})
    fig, ax = plt.subplots(figsize=(8,4))
    names = list(CONDITIONS[1:])
    for offset, metric, color in [(-.12,'dice','#276A9E'),(.12,'dice_union_present','#C77C33')]:
        means = np.array([result['native_paired_delta_pp'][n][metric]['mean'] for n in names])
        cis = np.array([result['native_paired_delta_pp'][n][metric]['ci95'] for n in names])
        ax.errorbar(np.arange(len(names))+offset,means,yerr=np.maximum(0,np.stack((means-cis[:,0],cis[:,1]-means))),
                    fmt='o',capsize=3,label=metric,color=color)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set_xticks(np.arange(len(names)),names,rotation=20)
    ax.set_ylabel('Native 3D Dice change vs full (pp)')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'native_deltas.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,3.8))
    for ax, region in zip(axes,('foreground','boundary')):
        for seed, cases in records.items():
            values = [[] for _ in range(5)]
            for case in cases:
                for b in range(5):
                    rows = [r for r in case['slices'] if r['physical_position_bin']==b and r['both_adjacent_real']]
                    prev, nxt = region_error(rows,'no_previous',region), region_error(rows,'no_next',region)
                    if prev is not None:
                        sign = 1 if rows[0]['previous_physical_direction']=='left' else -1
                        values[b].append(100*(prev-nxt)*sign)
            ax.plot(np.arange(5),[average(v) for v in values],marker='o',label=f'seed {seed}')
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set_xticks(range(5),['0-.2','.2-.4','.4-.6','.6-.8','.8-1'])
        ax.set_xlabel('Right-to-left relative scan position'); ax.set_title(region)
        ax.set_ylabel('Left minus right context utility (error pp)')
    axes[1].legend(frameon=False); fig.tight_layout(); fig.savefig(out/'position_utility.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    summarize(parser.parse_args().output_dir)
