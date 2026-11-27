"""One post-training audit of the EXACT proposed pseudo-supervision mask.

Loads a completed probe's frozen source, teachers and routers. Ground truth is
used only to score validation predictions. It never fits a model, changes a
threshold or reads test. The full-image preregistered screening gate is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


METHODS = ('mean', 'flat', 'factor_tissue', 'factor_level', 'factor')
REGIONS = ('selected_all', 'selected_gt_foreground', 'selected_numbered',
           'selected_only_a_correct', 'selected_only_b_correct', 'selected_both_wrong',
           'ignored_gt_foreground')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_audit(run):
    run = Path(run).resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    result = json.loads((run/'results.json').read_text())
    if result['status'] != 'completed' or json.loads((run/'status.json').read_text())['status'] != 'completed':
        raise ValueError('Audit only a completed probe; do not inspect partial training results')
    if (run/'supervision_audit.json').exists():
        raise FileExistsError('Existing audit is preserved; do not rerun it for threshold tuning')
    if any(name == 'freqssl' or name.startswith('freqssl.') for name in sys.modules):
        raise RuntimeError('Run this audit in a fresh process to import only the frozen package')
    for name, digest in manifest['source_hashes'].items():
        if sha(run/'code'/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Original teacher checkpoint changed')
    spec = importlib.util.spec_from_file_location('frozen_factor_probe', run/'code/experiments/probe_factor_routing.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from freqssl.factor_routing import pack_soft_targets

    payload = torch.load(run/'routers.pt', map_location='cpu', weights_only=True)
    if (payload['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or payload['protocol'] != manifest['protocol']
            or payload['final_step'] != manifest['protocol']['steps']):
        raise ValueError('Router weights do not match the final fixed protocol')
    device = torch.device(manifest['device'])
    teacher_a, cfg, _ = probe.load_single_network(manifest['checkpoint'], device, 'teacher1')
    teacher_b, _, _ = probe.load_single_network(manifest['checkpoint'], device, 'teacher2')
    probe.seed_everything(cfg)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = manifest['protocol']
    routers = {}
    for mode in ('flat', 'factor'):
        router = probe.AdvantageRouter(mode, protocol['width'])
        router.load_state_dict(payload['routers'][mode], strict=True)
        routers[mode] = router.requires_grad_(False).eval().to(device)
    model = probe.ProbeModel([teacher_a, teacher_b], routers, protocol['gate_scale']).eval()
    initial_hash = probe.state_hash(model)
    valid, records = probe.prepare_validation(cfg, 'valid')
    if manifest['smoke']:
        records = records[:1]
        valid.names = list(records[0]['slice_names'])
        valid.labeled_names = set(valid.names)
    if len(records) != result['valid_cases'] or len(valid) != result['valid_slices']:
        raise ValueError('Audit validation coverage differs from the completed probe')
    thresholds = dict(confidence=cfg.confidence, background_confidence=cfg.background_confidence,
                      background_weight=cfg.background_weight)
    values = {region: {method: dict(weight=0., nll_sum=0., correct_weight=0.,
                         logit_gradient_alignment_sum=0., target_true_probability_sum=0.)
                       for method in METHODS} for region in REGIONS}
    counts = dict(all_pixels=0, gt_foreground=0, selected=0, selected_gt_foreground=0,
                  selected_pseudo_foreground=0, selected_pseudo_wrong_tissue=0)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(valid), protocol['eval_batch_size']):
            images, labels = probe.batch(valid, list(range(start, min(start+protocol['eval_batch_size'], len(valid)))), device)
            predictions = model.probabilities(images)
            a, b, mean = predictions['teacher1'], predictions['teacher2'], predictions['mean']
            packed = pack_soft_targets(a, b, mean, **thresholds)
            weight = packed[:, 20]
            selected = weight > 0
            fg = labels > 0
            truth = probe.family_labels(labels)
            pseudo_family = probe.family_probabilities(mean).argmax(1)
            ca, cb = a.argmax(1) == labels, b.argmax(1) == labels
            regions = dict(selected_all=weight, selected_gt_foreground=weight*fg,
                selected_numbered=weight*((truth == 1) | (truth == 3)),
                selected_only_a_correct=weight*(ca & ~cb & fg),
                selected_only_b_correct=weight*(~ca & cb & fg),
                selected_both_wrong=weight*(~ca & ~cb & fg),
                ignored_gt_foreground=((~selected) & fg).float())
            counts['all_pixels'] += labels.numel()
            counts['gt_foreground'] += int(fg.sum())
            counts['selected'] += int(selected.sum())
            counts['selected_gt_foreground'] += int((selected & fg).sum())
            counts['selected_pseudo_foreground'] += int((selected & (pseudo_family > 0)).sum())
            counts['selected_pseudo_wrong_tissue'] += int((selected & (pseudo_family != truth)).sum())
            # Logit-space proxy at the current teacher-A distribution; this is
            # not a parameter-gradient or post-training Dice guarantee.
            true_gradient = a-F.one_hot(labels, 20).permute(0, 3, 1, 2).to(a)
            for method in METHODS:
                q = predictions[method]
                probability = q.gather(1, labels[:, None])[:, 0]
                nll = -probability.clamp_min(1e-30).log()
                correct = q.argmax(1) == labels
                alignment = (true_gradient*(a-q)).sum(1)
                for region, weights in regions.items():
                    row = values[region][method]
                    row['weight'] += float(weights.double().sum())
                    row['nll_sum'] += float((nll*weights).double().sum())
                    row['correct_weight'] += float((correct*weights).double().sum())
                    row['logit_gradient_alignment_sum'] += float((alignment*weights).double().sum())
                    row['target_true_probability_sum'] += float((probability*weights).double().sum())
    model.clear()
    if probe.state_hash(model) != initial_hash:
        raise RuntimeError('Audit unexpectedly changed teachers or routers')
    for methods in values.values():
        for row in methods.values():
            denominator = row['weight']
            for field, total in (('nll', 'nll_sum'), ('weighted_accuracy', 'correct_weight'),
                                 ('logit_gradient_alignment', 'logit_gradient_alignment_sum'),
                                 ('target_true_probability', 'target_true_probability_sum')):
                row[field] = row[total]/denominator if denominator else None
        for row in methods.values():
            row['nll_delta_vs_mean'] = row['nll']-methods['mean']['nll'] if row['nll'] is not None else None
    audit = dict(status='completed', smoke=manifest['smoke'], split='valid',
        grid=list(valid.image_size), valid_cases=len(records), valid_slices=len(valid),
        thresholds=thresholds, counts=counts, regions=values,
        mask_is_identical_for_all_methods=True, parameters_unchanged=True,
        test_read=False, threshold_search=False, student_updates=0,
        original_screening_passed=result['screening_passed'], screening_gate_changed=False,
        prediction_device=str(device), tf32=False, seconds=time.monotonic()-started,
        source_sha256=sha(__file__), input_sha256={name: sha(run/name)
            for name in ('manifest.json', 'results.json', 'routers.pt')})
    probe.write(run/'supervision_audit.json', audit)
    lines = ['# 固定伪监督区域审计', '',
        '这是冻结验证预测的补充诊断，未修改原筛查条件；没有训练学生。' +
        (' 本次输入为软件短测，不能推断效果。' if manifest['smoke'] else ''), '',
        f"原阈值：前景{cfg.confidence}、背景{cfg.background_confidence}、背景权重{cfg.background_weight}；所有方法共用中性平均教师的组织置信掩码。", '',
        f"保留真值前景 {counts['selected_gt_foreground']:,} / {counts['gt_foreground']:,} 像素；所有保留像素中组织误分 {counts['selected_pseudo_wrong_tissue']:,} / {counts['selected']:,}。", '',
        '| 区域 | 方法 | 加权NLL | 相对平均NLL差 | Logit梯度方向内积 |',
        '|---|---|---:|---:|---:|']
    for region in REGIONS:
        for method in ('mean', 'flat', 'factor'):
            row = values[region][method]
            fmt = lambda value: 'NA' if value is None else f'{value:.6g}'
            lines.append(f"| {region} | {method} | {fmt(row['nll'])} | {fmt(row['nll_delta_vs_mean'])} | {fmt(row['logit_gradient_alignment'])} |")
    lines += ['', 'NLL差越负越好；梯度内积为正仅表示在当前教师A的logit空间，软目标梯度与真值CE梯度局部同向。它不含网络Jacobian、Dice项或优化器，不是参数更新后收益的证明。', '',
        '`ignored_gt_foreground` 使用单位权重报告被排除区域，其数值不进入实际伪监督损失。其他区域使用实际像素权重，因此背景误判的真值前景也按0.25计入。未施加CutMix或选择新阈值。', '',
        '组织门和编号门单项的完整统计、源文件及输入哈希见 `supervision_audit.json`。']
    (run/'supervision_report.md').write_text('\n'.join(lines)+'\n')
    return dict(valid_cases=len(records), selected_gt_foreground=counts['selected_gt_foreground'],
                factor_selected_fg_nll_delta=values['selected_gt_foreground']['factor']['nll_delta_vs_mean'],
                original_screening_passed=result['screening_passed'], screening_gate_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.run)), flush=True)


if __name__ == '__main__':
    main()
