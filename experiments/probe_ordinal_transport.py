"""Frozen, prediction-only ordinal transport probe; valid only, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from freqssl.data import SpiderSliceDataset, _slice_names, _case_id
from freqssl.engine import seed_everything, state_hash, write_json
from freqssl.evaluation import prepare_validation, evaluate_volumes
from freqssl.inference import load_single_network
from freqssl.ordinal_transport import cross_targets, anchored_mean


PROTOCOL = dict(batch_size=8, train_samples=256, confidence=.8, minimum_pixels=16,
                minimum_support=.6, minimum_levels=2, background_confidence=.95,
                background_weight=.25, minimum_gain_pp=.3, minimum_retention=.98)
NETWORKS = ('teacher1', 'teacher2', 'mean', 'transport_mean_a', 'transport_mean_b', 'abstain_mean_a')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def prepare(args):
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose a new output directory')
    checkpoint = args.checkpoint.resolve(strict=True)
    sources = list((ROOT/'freqssl').glob('*.py')) + [ROOT/'spider_volume.py', Path(__file__).resolve(),
        ROOT/'docs/ordinal_transport_20261029.md', ROOT/'tests/test_ordinal_transport.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    parity = {name: sha(checkpoint.parent.parent/'code'/name) == hashes[name] for name in
        ('freqssl/models.py', 'freqssl/ordinal_model.py', 'freqssl/data.py', 'freqssl/evaluation.py', 'spider_volume.py')}
    if not all(parity.values()): raise ValueError('Existing teacher/input/evaluation runtime changed')
    out.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = out/'code'/source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    protocol = dict(PROTOCOL)
    if args.smoke: protocol['train_samples'] = 4
    write_json(out/'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source_hashes=hashes, protocol=protocol, teacher_runtime_parity=parity,
        device=args.device, smoke=args.smoke, test_read=False, unlabeled_annotations_read=False,
        prepared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    write_json(out/'status.json', dict(status='prepared', performance_verified=False))
    command = [sys.executable, str(out/'code/experiments/probe_ordinal_transport.py'),
               '--output-dir', str(out), '--device', args.device]
    write_json(out/'command.json', command); print(shlex.join(command), flush=True)


def abstain_changes(peer, moved):
    return peer * (peer.argmax(1) == moved.argmax(1))[:, None]


class ProbeModel(nn.Module):
    def __init__(self, teachers, protocol):
        super().__init__(); self.teachers = nn.ModuleList(teachers)
        self.options = {k: protocol[k] for k in
                        ('confidence', 'minimum_pixels', 'minimum_support', 'minimum_levels')}
        self.clear()

    def clear(self):
        self._images = self._output = self.targets = self.evidence = None

    def probabilities(self, images):
        if images is not self._images:
            a, b = [teacher(images) for teacher in self.teachers]
            to_a, to_b, evidence = cross_targets(a, b, **self.options)
            shift = evidence['offset']; drop_a = abstain_changes(b, to_a)
            self.targets = dict(a=dict(peer=b, transport=to_a, abstain=drop_a),
                                b=dict(peer=a, transport=to_b, abstain=abstain_changes(a, to_b)))
            self.evidence = evidence
            self._output = dict(teacher1=a, teacher2=b, mean=(a+b)/2,
                transport_mean_a=anchored_mean(a, b, to_a, shift),
                transport_mean_b=anchored_mean(b, a, to_b, -shift),
                abstain_mean_a=anchored_mean(a, b, drop_a, shift))
            self._images = images
        return self._output

    def predict_proba(self, images, network):
        return self.probabilities(images)[network]


def hard_selection(target, protocol):
    certainty, label = target.max(1)
    threshold = torch.where(label == 0, protocol['background_confidence'], protocol['confidence'])
    return label, certainty >= threshold


def new_counts():
    return dict(pixels=0, gt_foreground=0, selected=0, selected_foreground=0,
                correct_foreground=0, wrong_foreground=0, retained_peer_correct_foreground=0,
                peer_correct_to_wrong=0, peer_correct_to_abstain=0, peer_wrong_to_correct=0,
                selected_only_own_correct=0, correct_only_own_correct=0,
                selected_only_peer_correct=0, correct_only_peer_correct=0)


@torch.inference_mode()
def audit(model, dataset, device, protocol):
    model.eval(); model.clear()
    counts = {direction: {method: new_counts() for method in ('peer', 'transport', 'abstain')}
              for direction in ('a', 'b')}
    slices = []
    for start in range(0, len(dataset), protocol['batch_size']):
        samples = [dataset[i] for i in range(start, min(start+protocol['batch_size'], len(dataset)))]
        images = torch.stack([s['image'] for s in samples]).to(device)
        output = model.probabilities(images)
        # Labels enter ONLY this scoring block, after prediction/transport.
        gt = torch.cat([F.interpolate(s['label'][None, None].float(), images.shape[-2:], mode='nearest')
                        for s in samples])[:, 0].long().to(device)
        for direction, own_name in (('a', 'teacher1'), ('b', 'teacher2')):
            targets = model.targets[direction]
            original, accepted = hard_selection(targets['peer'], protocol)
            peer_correct = original == gt
            own_correct = output[own_name].argmax(1) == gt
            old_correct = peer_correct & accepted & (original > 0)
            old_wrong = ~peer_correct & accepted & (original > 0)
            only_own = own_correct & ~peer_correct & (gt > 0)
            only_peer = peer_correct & ~own_correct & (gt > 0)
            for method, target in targets.items():
                label, selected = hard_selection(target, protocol)
                correct = label == gt; foreground = label > 0
                values = dict(pixels=gt.numel(), gt_foreground=int((gt > 0).sum()),
                    selected=int(selected.sum()), selected_foreground=int((selected & foreground).sum()),
                    correct_foreground=int((selected & foreground & correct).sum()),
                    wrong_foreground=int((selected & foreground & ~correct).sum()),
                    retained_peer_correct_foreground=int((selected & correct & old_correct).sum()),
                    peer_correct_to_wrong=int((selected & ~correct & old_correct).sum()),
                    peer_correct_to_abstain=int((~selected & old_correct).sum()),
                    peer_wrong_to_correct=int((selected & correct & old_wrong).sum()),
                    selected_only_own_correct=int((selected & only_own).sum()),
                    correct_only_own_correct=int((selected & correct & only_own).sum()),
                    selected_only_peer_correct=int((selected & only_peer).sum()),
                    correct_only_peer_correct=int((selected & correct & only_peer).sum()))
                for key, value in values.items(): counts[direction][method][key] += value
        evidence = model.evidence
        for i, sample in enumerate(samples):
            slices.append(dict(name=sample['file_name'], case=_case_id(sample['file_name']),
                offset=int(evidence['offset'][i]), proposed_offset=int(evidence['proposed_offset'][i]),
                support=float(evidence['support'][i]), levels=int(evidence['supported_levels'][i]),
                correspondence_pixels=int(evidence['correspondence_pixels'][i]),
                votes=evidence['scores'][i].cpu().tolist()))
        if start == 0: print('FIRST_BATCH', dataset.split, len(samples), flush=True)
    model.clear()
    for directions in counts.values():
        old = directions['peer']['correct_foreground']
        for row in directions.values():
            row['retention_of_original_correct'] = row['retained_peer_correct_foreground']/old if old else None
            row['foreground_precision'] = row['correct_foreground']/row['selected_foreground'] if row['selected_foreground'] else None
    return dict(split=dataset.split, native_volume_metric=False, slices=slices, counts=counts,
        nonzero_offset_slices=sum(row['offset'] != 0 for row in slices),
        nonzero_offset_cases=sorted({row['case'] for row in slices if row['offset'] != 0}))


def run(args):
    out = args.output_dir.resolve(); manifest = json.loads((out/'manifest.json').read_text())
    if ROOT != out/'code' or args.device != manifest['device']: raise ValueError('Use the sealed command')
    if json.loads((out/'status.json').read_text())['status'] != 'prepared': raise ValueError('Run already started')
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT/name) != digest: raise ValueError(f'Sealed source changed: {name}')
    if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    protocol = manifest['protocol']; device = torch.device(args.device); started = time.monotonic()
    write_json(out/'status.json', dict(status='running', pid=os.getpid(), performance_verified=False))
    try:
        a, cfg, info = load_single_network(manifest['checkpoint'], device, 'teacher1')
        b, _, _ = load_single_network(manifest['checkpoint'], device, 'teacher2')
        if not (cfg.method == 'dual_ordinal' and not cfg.ordinal_use_routing and cfg.image_size == 224
                and cfg.labelnum == 1496 and cfg.max_samples == 3732 and cfg.seed == 42
                and info['checkpoint_step'] == 4500): raise ValueError('Unexpected teacher/budget')
        seed_everything(cfg); torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        initial = [state_hash(a), state_hash(b)]
        model = ProbeModel([a, b], protocol).eval()
        pool = _slice_names(cfg.root_path, 'train')[:cfg.max_samples][:cfg.labelnum]
        if len(pool) != cfg.labelnum: raise ValueError('Training inventory changed')
        selected = np.linspace(0, len(pool)-1, min(len(pool), protocol['train_samples']), dtype=int)
        names = [pool[i] for i in selected]
        train = SpiderSliceDataset(cfg.root_path, 'train', cfg.image_size, names=names,
                                   labeled_names=names, augment=False, seed=cfg.seed)
        valid, records = prepare_validation(cfg, 'valid')
        if len(valid) != 418 or len(records) != 21: raise ValueError('Validation inventory changed')
        if manifest['smoke']:
            records = records[:1]; valid.names = list(records[0]['slice_names']); valid.labeled_names = set(valid.names)
        write_json(out/'inputs.json', dict(train_names=names, valid_names=valid.names, teacher_hashes=initial, config=cfg.to_dict()))
        write_json(out/'validation_reconstruction_audit.json', records)
        train_audit = audit(model, train, device, protocol); write_json(out/'train_audit.json', train_audit)
        valid_audit = audit(model, valid, device, protocol); write_json(out/'valid_audit.json', valid_audit)
        model.clear()
        scores = evaluate_volumes(model, valid, records, device, networks=NETWORKS, batch_size=protocol['batch_size'])
        write_json(out/'native_valid.json', scores); model.clear()
        if initial != [state_hash(a), state_hash(b)] or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen teachers changed')
        if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']: raise RuntimeError('Checkpoint changed during run')
        dice = {name: value['overall']['dice'] for name, value in scores.items()}
        targets = valid_audit['counts']['a']; old, new = targets['peer'], targets['transport']
        gain = 100 * (dice['transport_mean_a'] - dice['mean'])
        criteria = dict(minimum_dice_gain=gain >= protocol['minimum_gain_pp'],
            beats_abstention=dice['transport_mean_a'] > dice['abstain_mean_a'],
            retains_original_correct=new['retained_peer_correct_foreground'] >= protocol['minimum_retention'] * old['correct_foreground'],
            does_not_increase_wrong_foreground=new['wrong_foreground'] <= old['wrong_foreground'])
        result = dict(status='completed', smoke=manifest['smoke'], seconds=time.monotonic()-started,
            valid_cases=len(records), valid_slices=len(valid), dice=dice, gain_over_mean_pp=gain,
            criteria=criteria, screening_passed=not manifest['smoke'] and all(criteria.values()),
            nonzero_offset_slices=valid_audit['nonzero_offset_slices'],
            nonzero_offset_cases=valid_audit['nonzero_offset_cases'],
            teacher_parameters_unchanged=True, model_updates=0, test_read=False,
            single_network_efficacy_verified=False)
        write_json(out/'results.json', result)
        write_json(out/'status.json', dict(status='completed', screening_passed=result['screening_passed'], performance_verified=False))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(out/'status.json', dict(status='failed', error=repr(exc), performance_verified=False)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        if args.checkpoint is None: parser.error('--prepare-only requires --checkpoint')
        prepare(args)
    else:
        if args.checkpoint is not None or args.smoke: parser.error('Use only the frozen runtime command')
        run(args)


if __name__ == '__main__': main()
