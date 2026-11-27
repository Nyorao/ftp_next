"""Native confusion metrics and actual forward MAC accounting."""
from collections import defaultdict

import numpy as np
import torch


def numbering_metrics(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.shape != (20, 20) or (cm < 0).any():
        raise ValueError('Expected a nonnegative 20 by 20 confusion matrix')
    result, totals = {}, np.zeros(4, dtype=np.int64)
    for name, ids in (('vertebra', np.arange(1, 10)), ('disc', np.arange(11, 20))):
        block = cm[np.ix_(ids, ids)]
        gt, covered = cm[ids].sum(), block.sum()
        wrong = covered - np.trace(block)
        distance = np.abs(ids[:, None] - ids[None])
        one = block[distance == 1].sum()
        totals += (gt, covered, wrong, one)
        result[name] = dict(numbered_gt=int(gt), tissue_correct=int(covered), wrong=int(wrong),
            wrong_one=int(one), wrong_multiple=int(wrong-one),
            wrong_level_percent=100*wrong/gt if gt else None,
            conditional_wrong_level_percent=100*wrong/covered if covered else None,
            tissue_coverage_percent=100*covered/gt if gt else None)
    gt, covered, wrong, one = totals
    result.update(numbered_gt=int(gt), tissue_correct=int(covered), wrong=int(wrong),
        wrong_one=int(one), wrong_multiple=int(wrong-one),
        wrong_level_percent=100*wrong/gt if gt else None,
        conditional_wrong_level_percent=100*wrong/covered if covered else None,
        tissue_coverage_percent=100*covered/gt if gt else None)
    return result


def distance_bin(distance):
    return '0_5mm' if distance <= 5 else ('5_15mm' if distance <= 15 else 'over15mm')


def case_mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


class WorkCounter:
    """Counts Conv/Linear and both attention matmuls, including padded work.

    Does not count normalization, activations, positional math, indexing,
    interpolation, softmax, backward or optimizer operations; not total FLOPs.
    """
    def __init__(self, model):
        self.counts = defaultdict(int)
        self.handles = []
        self.attentions = [m for m in model.modules() if hasattr(m, 'attention_macs')]
        self.before = [m.attention_macs for m in self.attentions]
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                def conv(layer, args, out):
                    self.counts['conv_macs'] += out.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size).item()
                self.handles.append(module.register_forward_hook(conv))
            elif isinstance(module, torch.nn.Linear):
                def linear(layer, args, out):
                    self.counts['linear_macs'] += out.numel() * layer.in_features
                self.handles.append(module.register_forward_hook(linear))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        values = dict(self.counts)
        values['attention_matmul_macs'] = sum(m.attention_macs - n for m, n in zip(self.attentions, self.before))
        values['total_counted_macs'] = sum(values.values())
        return values


def aggregate_attention(observations, window):
    observations = [row for row in observations if row['queries'] > 0]
    if not observations:
        return None
    n = sum(row['queries'] for row in observations)
    mass = sum(np.asarray(row['mass']) for row in observations) / n
    uniform = sum(np.asarray(row['uniform_mass']) for row in observations) / n
    return dict(offsets=list(range(-window, window+1)), mass=mass.tolist(), uniform_mass=uniform.tolist(),
        queries=n, mean_abs_delta=(sum(np.asarray(row['mean_abs_delta']) for row in observations)/n).tolist(),
        entropy=(sum(np.asarray(row['entropy_sum']) for row in observations)/n).tolist(),
        edge_mass=(mass[:, 0]+mass[:, -1]).tolist(),
        gate=observations[0]['gate'], residual_rms_ratio=case_mean([row['residual_rms_ratio'] for row in observations]))
