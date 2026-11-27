"""Freeze two existing teachers, learn tiny advantage heads on L, evaluate valid.

Preparation freezes source and protocol before any model execution. Run the
emitted command from that snapshot. Test data is never an evaluation option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from freqssl.anatomy import family_labels, family_probabilities
from freqssl.data import SpiderSliceDataset, _slice_names
from freqssl.engine import seed_everything, state_hash
from freqssl.evaluation import evaluate_volumes, prepare_validation
from freqssl.factor_routing import (AdvantageRouter, advantage_loss, advantage_targets,
                                   factor_fusion, flat_fusion)
from freqssl.inference import load_single_network


PROTOCOL = dict(seed=42, steps=800, batch_size=4, eval_batch_size=8,
                lr=.001, weight_decay=.0001, width=16, gate_scale=2.,
                grad_clip=5., background_weight=.25, train_audit_samples=256,
                minimum_gain_pp=.3, direction_threshold=.001)
NETWORKS = ('teacher1', 'teacher2', 'mean', 'flat', 'factor_tissue', 'factor_level', 'factor')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing probe: {output}')
    checkpoint = args.checkpoint.resolve(strict=True)
    sources = list((ROOT/'freqssl').glob('*.py')) + [ROOT/'spider_volume.py', Path(__file__).resolve(),
               ROOT/'docs/factor_advantage_20261028.md', ROOT/'tests/test_factor_routing.py']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    teacher_code = checkpoint.parent.parent/'code'
    parity = {}
    # Same teacher forward, input preparation and native volume metric as the
    # previous completed experiment. New fusion is the only forward addition.
    for name in ('freqssl/models.py', 'freqssl/ordinal_model.py', 'freqssl/data.py',
                 'freqssl/evaluation.py', 'spider_volume.py'):
        old = teacher_code/name
        parity[name] = old.is_file() and sha(old) == source_hashes[name]
    if not args.smoke and not all(parity.values()):
        raise ValueError(f'Frozen teacher/runtime mismatch: {parity}')
    protocol = dict(PROTOCOL)
    if args.smoke:
        protocol.update(steps=2, batch_size=2, train_audit_samples=4)
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = output/'code'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    source_hashes=source_hashes, teacher_runtime_parity=parity,
                    protocol=protocol, smoke=args.smoke, device=args.device,
                    test_read=False, unlabeled_annotations_read=False,
                    segmentation_networks_trained=False, checkpoint_selection='fixed final router step',
                    prepared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(output/'manifest.json', manifest)
    write(output/'status.json', dict(status='prepared', step=0, performance_verified=False))
    command = [sys.executable, str(output/'code/experiments/probe_factor_routing.py'),
               '--output-dir', str(output), '--device', args.device]
    write(output/'command.json', command)
    print(shlex.join(command), flush=True)


class ProbeModel(nn.Module):
    def __init__(self, teachers, routers, gate_scale):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.routers = nn.ModuleDict(routers)
        self.gate_scale = gate_scale
        self._images = None
        self._probabilities = None

    def clear(self):
        self._images = self._probabilities = None

    def probabilities(self, images):
        if images is not self._images:
            a, b = [teacher(images) for teacher in self.teachers]
            gates = self.routers['factor'](a, b) * self.gate_scale
            tissue = gates.clone(); tissue[:, 1:] = 0
            level = gates.clone(); level[:, 0] = 0
            self._probabilities = dict(teacher1=a, teacher2=b, mean=(a+b)/2,
                flat=flat_fusion(a, b, self.routers['flat'](a, b)*self.gate_scale),
                factor_tissue=factor_fusion(a, b, tissue), factor_level=factor_fusion(a, b, level),
                factor=factor_fusion(a, b, gates))
            self._images = images
        return self._probabilities

    def predict_proba(self, images, network):
        return self.probabilities(images)[network]


def labeled_data(cfg, smoke=False, augment=True):
    names = _slice_names(cfg.root_path, 'train')[:cfg.max_samples][:cfg.labelnum]
    if len(names) != cfg.labelnum:
        raise ValueError('Training inventory no longer matches the original label budget')
    if smoke:
        names = names[:8]
    return SpiderSliceDataset(cfg.root_path, 'train', cfg.image_size, names=names,
                              labeled_names=names, augment=augment, seed=cfg.seed)


def batch(dataset, indices, device):
    samples = [dataset[int(i)] for i in indices]
    images = torch.stack([s['image'] for s in samples]).to(device)
    labels = torch.cat([F.interpolate(s['label'][None, None].float(), size=images.shape[-2:],
                                     mode='nearest') for s in samples])[:, 0].long().to(device)
    return images, labels


@torch.no_grad()
def diagnose(model, dataset, indices, device, protocol):
    """224-grid calibration/opportunity statistics, explicitly not native Dice."""
    model.eval(); model.clear()
    stats = {mode: dict(weight=[0.]*channels, mse_sum=[0.]*channels,
                         zero_mse_sum=[0.]*channels, descent_alignment_sum=[0.]*channels,
                         informative=[0]*channels, correct_direction=[0]*channels)
             for mode, channels in (('flat', 1), ('factor', 3))}
    nll = {name: dict(foreground_nll_sum=0., foreground_pixels=0,
                      tissue_nll_sum=0., conditional_nll_sum=0., numbered_pixels=0)
           for name in (*NETWORKS, 'oracle_local_direction')}
    opportunity = dict(foreground_pixels=0, both_wrong=0, only_a_correct=0, only_b_correct=0,
                       factor_recovers_both_wrong=0, mean_recovers_both_wrong=0,
                       oracle_recovers_both_wrong=0)
    for start in range(0, len(indices), protocol['eval_batch_size']):
        images, labels = batch(dataset, indices[start:start+protocol['eval_batch_size']], device)
        output = dict(model.probabilities(images))
        a, b = output['teacher1'], output['teacher2']
        truth = family_labels(labels)
        for mode in ('flat', 'factor'):
            prediction = model.routers[mode](a, b)
            target, mask = advantage_targets(a, b, labels, mode=mode)
            weight = mask.float() * torch.where(labels[:, None] == 0, protocol['background_weight'], 1.)
            informative = mask & (target.abs() >= protocol['direction_threshold'])
            values = dict(weight=weight.sum((0, 2, 3)),
                          mse_sum=((prediction-target).square()*weight).sum((0, 2, 3)),
                          zero_mse_sum=(target.square()*weight).sum((0, 2, 3)),
                          descent_alignment_sum=(prediction*target*weight).sum((0, 2, 3)),
                          informative=informative.sum((0, 2, 3)),
                          correct_direction=((prediction*target > 0)&informative).sum((0, 2, 3)))
            for key, values_tensor in values.items():
                stats[mode][key] = [x+y for x, y in zip(stats[mode][key], values_tensor.cpu().tolist())]
            if mode == 'factor':
                # This output reads validation GT and is ONLY a diagnostic.
                output['oracle_local_direction'] = factor_fusion(a, b, target*protocol['gate_scale'])
        fg = labels > 0
        numbered = (truth == 1) | (truth == 3)
        for name, probabilities in output.items():
            fine = -probabilities.gather(1, labels[:, None])[:, 0].clamp_min(1e-30).log()
            tissue = -family_probabilities(probabilities).gather(1, truth[:, None])[:, 0].clamp_min(1e-30).log()
            row = nll[name]
            row['foreground_nll_sum'] += float(fine[fg].double().sum())
            row['foreground_pixels'] += int(fg.sum())
            row['tissue_nll_sum'] += float(tissue[fg].double().sum())
            row['conditional_nll_sum'] += float((fine-tissue)[numbered].double().sum())
            row['numbered_pixels'] += int(numbered.sum())
        ca, cb = a.argmax(1) == labels, b.argmax(1) == labels
        wrong = ~ca & ~cb & fg
        opportunity['foreground_pixels'] += int(fg.sum())
        opportunity['both_wrong'] += int(wrong.sum())
        opportunity['only_a_correct'] += int((ca & ~cb & fg).sum())
        opportunity['only_b_correct'] += int((cb & ~ca & fg).sum())
        for field, name in (('factor', 'factor'), ('mean', 'mean'), ('oracle', 'oracle_local_direction')):
            opportunity[f'{field}_recovers_both_wrong'] += int((wrong & (output[name].argmax(1) == labels)).sum())
    model.clear()
    for row in stats.values():
        for name in ('mse', 'zero_mse', 'descent_alignment'):
            row[name] = [s/n if n else None for s, n in zip(row[name+'_sum'], row['weight'])]
        row['direction_accuracy'] = [s/n if n else None for s, n in zip(row['correct_direction'], row['informative'])]
    for row in nll.values():
        row['foreground_nll'] = row['foreground_nll_sum']/max(1, row['foreground_pixels'])
        row['tissue_nll'] = row['tissue_nll_sum']/max(1, row['foreground_pixels'])
        row['conditional_nll'] = row['conditional_nll_sum']/max(1, row['numbered_pixels'])
    return dict(grid=list(dataset.image_size), slices=len(indices), calibration=stats, nll=nll,
                opportunity=opportunity, native_dice=False, oracle_is_deployable=False)


def run(args):
    output = args.output_dir.resolve()
    manifest = json.loads((output/'manifest.json').read_text())
    if json.loads((output/'status.json').read_text())['status'] != 'prepared':
        raise FileExistsError('Probe already started; preserve it and use a new output directory')
    if ROOT != output/'code':
        raise ValueError('Run the prepared source snapshot, using command.json')
    if args.device != manifest['device']:
        raise ValueError('Device changed since preparation; prepare a separate probe')
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Prepared source changed: {name}')
    if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed since preparation')
    protocol = manifest['protocol']; device = torch.device(args.device)
    started = time.monotonic()
    write(output/'status.json', dict(status='running', step=0, performance_verified=False))
    step = 0
    try:
        first, cfg, info = load_single_network(manifest['checkpoint'], device, 'teacher1')
        second, _, _ = load_single_network(manifest['checkpoint'], device, 'teacher2')
        if not manifest['smoke'] and not (cfg.method == 'dual_ordinal' and not cfg.ordinal_use_routing
                and cfg.image_size == 224 and cfg.labelnum == 1496 and cfg.max_samples == 3732
                and cfg.seed == 42 and info['checkpoint_step'] == 4500):
            raise ValueError('Full probe must use the preregistered hard_peer checkpoint/protocol')
        seed_everything(cfg)
        # Fixed strict FP32 for every probe arm, including the mean control.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        hashes = [state_hash(first), state_hash(second)]
        routers = {}
        for mode in ('flat', 'factor'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(protocol['seed'])
                routers[mode] = AdvantageRouter(mode, protocol['width']).to(device)
        optimizers = {name: torch.optim.AdamW(router.parameters(), lr=protocol['lr'],
                                              weight_decay=protocol['weight_decay'])
                      for name, router in routers.items()}
        dataset = labeled_data(cfg, manifest['smoke'])
        if len(dataset) < protocol['batch_size']:
            raise ValueError('Labeled probe pool is smaller than one complete batch')
        write(output/'training_inputs.json', dict(labeled_files=dataset.names,
            dataset_config=cfg.to_dict(), annotations_read='only listed labeled training files',
            router_parameters={m: sum(p.numel() for p in r.parameters()) for m, r in routers.items()},
            teacher_initial_hashes=hashes, torch_version=torch.__version__, tf32=False,
            protocol=protocol))
        epoch = 0
        while step < protocol['steps']:
            dataset.set_epoch(epoch)
            indices = np.random.RandomState(protocol['seed']+epoch).permutation(len(dataset))
            for start in range(0, len(indices)-protocol['batch_size']+1, protocol['batch_size']):
                step += 1
                images, labels = batch(dataset, indices[start:start+protocol['batch_size']], device)
                with torch.no_grad():
                    a, b = first(images), second(images)
                metrics = dict(step=step, epoch=epoch)
                for mode, router in routers.items():
                    router.train(); optimizers[mode].zero_grad(set_to_none=True)
                    target, mask = advantage_targets(a, b, labels, mode=mode)
                    loss = advantage_loss(router(a, b), target, mask, labels, protocol['background_weight'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'{mode}: nonfinite probe loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(router.parameters(), protocol['grad_clip'], error_if_nonfinite=True)
                    optimizers[mode].step()
                    metrics[mode] = dict(loss=float(loss.detach()), grad_norm=float(norm))
                if step == 1 or step % 25 == 0 or step == protocol['steps']:
                    with (output/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics, allow_nan=False)+'\n')
                    write(output/'status.json', dict(status='running', step=step, performance_verified=False))
                    print(json.dumps(metrics), flush=True)
                if step >= protocol['steps']:
                    break
            epoch += 1
        torch.save(dict(routers={m: r.state_dict() for m, r in routers.items()}, protocol=protocol,
                        checkpoint_sha256=manifest['checkpoint_sha256'], final_step=step), output/'routers.pt')
        model = ProbeModel([first, second], routers, protocol['gate_scale']).eval()
        write(output/'status.json', dict(status='evaluating_valid', step=step, performance_verified=False))
        valid, records = prepare_validation(cfg, 'valid')
        if manifest['smoke']:
            records = records[:1]
            valid.names = list(records[0]['slice_names'])
            valid.labeled_names = set(valid.names)
        write(output/'validation_reconstruction_audit.json', records)
        scores = evaluate_volumes(model, valid, records, device, networks=NETWORKS,
                                  batch_size=protocol['eval_batch_size'])
        write(output/'native_valid.json', scores)
        valid_diagnostics = diagnose(model, valid, list(range(len(valid))), device, protocol)
        write(output/'valid_diagnostics.json', valid_diagnostics)
        train = labeled_data(cfg, manifest['smoke'], augment=False)
        indices = np.linspace(0, len(train)-1, min(len(train), protocol['train_audit_samples']), dtype=int).tolist()
        train_diagnostics = diagnose(model, train, indices, device, protocol)
        write(output/'train_diagnostics.json', dict(sample_names=[train.names[i] for i in indices], **train_diagnostics))
        after = [state_hash(first), state_hash(second)]
        if hashes != after or any(p.grad is not None for t in model.teachers for p in t.parameters()):
            raise RuntimeError('Frozen teacher parameters or gradients changed')
        if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
            raise RuntimeError('Teacher checkpoint changed while running')
        dice = {name: row['overall']['dice'] for name, row in scores.items()}
        fg_nll = {name: row['foreground_nll'] for name, row in valid_diagnostics['nll'].items()}
        gain = 100*(dice['factor']-dice['mean'])
        criteria = dict(minimum_dice_gain=gain >= protocol['minimum_gain_pp'],
                        beats_flat_gate=dice['factor'] > dice['flat'],
                        foreground_nll_not_worse=fg_nll['factor'] <= fg_nll['mean']+1e-6)
        result = dict(status='completed', final_step=step, smoke=manifest['smoke'],
            valid_cases=len(records), valid_slices=len(valid), dice=dice,
            foreground_nll=fg_nll, factor_gain_over_mean_pp=gain,
            screening_criteria=criteria, screening_passed=not manifest['smoke'] and all(criteria.values()),
            single_network_efficacy_verified=False, teacher_parameters_unchanged=True,
            teacher_final_hashes=after, test_read=False, seconds=time.monotonic()-started)
        write(output/'results.json', result)
        write(output/'status.json', dict(status='completed', step=step,
            screening_passed=result['screening_passed'], performance_verified=False))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write(output/'status.json', dict(status='failed', step=step, error=repr(exc), performance_verified=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two updates, eight train slices, one valid case; no efficacy claim')
    args = parser.parse_args()
    if args.prepare_only:
        if args.checkpoint is None:
            parser.error('--prepare-only requires --checkpoint')
        prepare(args)
    else:
        if args.checkpoint is not None or args.smoke:
            parser.error('Run uses only the sealed manifest; --checkpoint/--smoke belong to preparation')
        run(args)


if __name__ == '__main__':
    main()
