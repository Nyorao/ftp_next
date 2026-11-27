"""Freeze and run a validation-only, three-seed context contribution atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from freqssl.context_diagnostics import (CONDITIONS, encode_volume, feature_relationship,
    image_quality, observed_regions, predict_encoded, region_masks, slice_metadata,
    score_logits, context_indices)
from freqssl.data import _read_transposed
from freqssl.engine import seed_everything, state_hash, write_json
from freqssl.evaluation import anatomical_diagnostics, prepare_validation
from freqssl.geometry import canonicalize
from freqssl.inference import load_single_network, prepare_slice, read_volume
from spider_volume import aggregate_cases, reconstruct, volume_metrics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seed')
    reference = args.reference_dir.resolve(strict=True)
    geometry = args.geometry_manifest.resolve(strict=True)
    checkpoints = {}
    for seed in args.seeds:
        checkpoint = reference / f'confirmation/seed{seed}/adaptive/best.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected = checkpoint.parent / f"valid_step_{payload['step']}.json"
        if payload['config']['seed'] != seed or payload['config']['slice_context'] != 'adaptive':
            raise ValueError('Wrong checkpoint configuration')
        checkpoints[str(seed)] = dict(path=str(checkpoint), sha256=sha(checkpoint),
            step=payload['step'], historical_metrics=str(expected), historical_sha256=sha(expected))
    sources = [*sorted((ROOT/'freqssl').glob('*.py')), ROOT/'spider_volume.py',
               ROOT/'experiments/probe_slice_context.py', ROOT/'experiments/summarize_slice_context.py',
               ROOT/'tests/test_context_diagnostics.py', ROOT/'docs/context_contribution_20261105.md']
    # Establish that the learned forward architecture is unchanged from the training snapshot.
    parity = {name: sha(ROOT/name) == sha(reference/'code'/name)
              for name in ('freqssl/models.py', 'freqssl/slice_context.py')}
    if not all(parity.values()):
        raise ValueError(f'Frozen training architecture differs: {parity}')
    out.mkdir(parents=True)
    for source in sources:
        dest = out/'code'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    manifest = dict(schema_version=1, checkpoints=checkpoints, geometry_manifest=str(geometry),
        geometry_sha256=sha(geometry), source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        architecture_parity=parity, device=args.device, batch_size=8, jobs=args.jobs,
        max_cases=args.max_cases, historical_tolerance=args.historical_tolerance,
        protocol=dict(conditions=list(CONDITIONS), seeds=args.seeds, split='valid', model_updates=0,
            checkpoint_selection='existing best teacher1; never selected by this probe',
            position_bins=5, spacing_bins_mm=[3.8, 4.6], boundary_radius_model_pixels=2,
            bootstrap_samples=10000, bootstrap_seed=20261105, physical_direction='LPS left/right',
            test_read=False, train_labels_read=False, labels_used_for_scoring_only=True,
            historical_tolerance_note='absolute Dice; CPU/CUDA arithmetic need not be bit-identical',
            inference='one shared encoder pass per real slice per seed; seven decoder conditions',
            interpretation='frozen-model sensitivity under interventions, not anatomical causality',
            candidate_rule='exploratory ranking only; no automatic training from validation associations'))
    write_json(out/'manifest.json', manifest)
    write_json(out/'status.json', dict(status='prepared'))
    command = [sys.executable, str(out/'code/experiments/probe_slice_context.py'),
               '--output-dir', str(out), '--worker']
    write_json(out/'command.json', command)
    return command


def verify_geometry(record, verified, images, image_size):
    """Bind geometry to image bytes and verify every ordered PNG input, without labels."""
    if verified['split'] != 'valid' or verified['case'] != record['case']:
        raise ValueError('Wrong geometry record')
    array, plan, receipt = read_volume(verified['converted_image'], verified)
    canonical = canonicalize(array, plan)
    reverse = 0 in plan['flip_canonical_axes']
    ordered = canonical[::-1] if reverse else canonical
    if len(ordered) != len(images):
        raise ValueError('Geometry slice count mismatch')
    for i, value in enumerate(ordered):
        if not torch.equal(prepare_slice(value, image_size), images[i]):
            raise ValueError(f"PNG/geometry input order differs: {record['case']} slice {i}")
    return dict(case=record['case'], image_sha256=receipt['input_array_sha256'],
                ordered_png_inputs_exact=True, spacing_mm=plan['canonical_spacing_mm'][0],
                reverse_physical_order=reverse, annotation_files_read=0)


@torch.inference_mode()
def analyze_case(backbone, dataset, indices, record, geo, device, output):
    started = time.monotonic()
    names = record['slice_names']
    images = torch.stack([dataset.read_image(indices[name]) for name in names])
    quality = [image_quality(image[0].numpy()) for image in images]
    encoded = encode_volume(backbone, images.to(device))
    target_native = sitk.GetArrayFromImage(sitk.ReadImage(record['reference'])).astype(np.uint8)
    volume_present = np.unique(target_native)
    targets, shapes = [], []
    for name in names:
        label = _read_transposed(dataset.root/'label'/name, torch.long)
        with Image.open(dataset.root/'image'/name) as image:
            shapes.append(image.size)
        targets.append(F.interpolate(label[None, None].float(), size=images.shape[-2:],
                                     mode='nearest')[0, 0].long().numpy())
    targets = np.stack(targets)
    count = len(names)
    gamma = [float(layer.gamma) for layer in backbone.context_fusion]
    rows = []
    for i in range(count):
        metadata = slice_metadata(i, count, geo['spacing_mm'], geo['reverse_physical_order'])
        relationships = {}
        for side, delta in (('previous', -1), ('next', 1)):
            other = min(count-1, max(0, i+delta))
            maps = feature_relationship(encoded[2][i], encoded[2][other])
            masks = region_masks(targets[i])
            relationships[side] = {}
            for region in ('all', 'foreground', 'boundary', 'interior'):
                occ = F.adaptive_avg_pool2d(torch.from_numpy(masks[region].astype(np.float32))[None, None],
                                           encoded[2].shape[-2:])[0, 0].to(device)
                denom = float(occ.sum())
                relationships[side][region] = {k: float((v*occ).sum()/denom) if denom else None
                                              for k, v in maps.items()}
        rows.append(dict(file_name=names[i], **metadata, image_quality=quality[i],
            previous_quality=quality[max(0, i-1)], next_quality=quality[min(count-1, i+1)],
            feature_relationships=relationships, conditions={}))
    grid_predictions = np.empty((len(CONDITIONS), count, *targets.shape[-2:]), dtype=np.uint8)
    all_weights = [[] for _ in gamma]
    metrics = {}
    for c, condition in enumerate(CONDITIONS):
        native_slices = []
        for start, logits, observations in predict_encoded(backbone, encoded, condition, observe=c == 0):
            predictions = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for b, prediction in enumerate(predictions):
                i = start + b
                grid_predictions[c, i] = prediction
                _, scores = score_logits(logits[b], targets[i], grid_predictions[0, i], volume_present)
                selected = context_indices(i, count, condition)
                rows[i]['conditions'][condition] = dict(regions=scores, input_indices=list(selected),
                    signed_distances_mm=[(j-i)*rows[i]['signed_mm_per_index'] for j in selected],
                    distinct_input_slices=len(set(selected)),
                    context_clipped=(condition in ('full', 'swapped', 'gap2', 'gap3') and
                        not (int(condition[-1]) if condition.startswith('gap') else 1) <= i <
                        count-(int(condition[-1]) if condition.startswith('gap') else 1)))
                if c == 0:
                    rows[i]['fusion'] = observed_regions(observations, b, targets[i], gamma)
                restored = F.interpolate(torch.from_numpy(prediction.astype(np.float32))[None, None],
                                         size=shapes[i], mode='nearest')[0, 0]
                native_slices.append(restored.numpy().astype(np.uint8).T.copy())
            if c == 0:
                for level, (weights, _) in enumerate(observations):
                    all_weights[level].append(weights.numpy())
        prediction_native = reconstruct(native_slices, record['slice_axis'],
                                        record['undo_vertical_flip'], record['array_shape_zyx'])
        classes = volume_metrics(prediction_native, target_native, (1, 1, 1), surfaces=False)
        metrics[condition] = dict(case=record['case'], classes=classes,
            anatomy=anatomical_diagnostics(prediction_native, target_native),
            native_absent_class_voxels=int(((prediction_native > 0) &
                ~np.isin(prediction_native, volume_present)).sum()),
            native_changed_voxels_from_full=0 if c == 0 else int((prediction_native != full_native).sum()))
        if c == 0:
            full_native = prediction_native
    result = dict(case=record['case'], geometry=geo, gamma=gamma, metrics=metrics,
                  slices=rows, seconds=time.monotonic()-started)
    # Compact, local-only maps support qualitative checks without rerunning inference.
    np.savez_compressed(output.with_suffix('.npz'), predictions=grid_predictions, target=targets.astype(np.uint8),
        images=images[:, 0].numpy().astype(np.float16),
        **{f'weights_stride{stride}':np.concatenate(parts).astype(np.float16)
           for stride, parts in zip((4, 8, 16), all_weights)})
    write_json(output, result)
    return result


def frozen_manifest(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if ROOT != out/'code':
        raise ValueError('Execute the frozen command.json')
    for relative, expected in manifest['source_hashes'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    if sha(manifest['geometry_manifest']) != manifest['geometry_sha256']:
        raise ValueError('Geometry manifest changed')
    return manifest


def run_seed(out, seed):
    manifest = frozen_manifest(out)
    checkpoint = manifest['checkpoints'][str(seed)]
    directory = out/f'seed{seed}'
    (directory/'cases').mkdir(parents=True)
    write_json(directory/'status.json', dict(status='running', seed=seed, cases_done=0))
    try:
        device = torch.device(manifest['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device unavailable; no automatic device substitution')
        geometry = {r['case']:r for r in json.loads(Path(manifest['geometry_manifest']).read_text())['records']
                    if r['split'] == 'valid'}
        if sha(checkpoint['path']) != checkpoint['sha256'] or sha(checkpoint['historical_metrics']) != checkpoint['historical_sha256']:
            raise ValueError('Checkpoint or historical reference changed')
        model, cfg, loading = load_single_network(checkpoint['path'], device, 'teacher1')
        seed_everything(cfg)
        initial = state_hash(model)
        dataset, records = prepare_validation(cfg, split='valid')
        indices = {name:i for i, name in enumerate(dataset.names)}
        if set(geometry) != {r['case'] for r in records}:
            raise ValueError('Validation geometry inventory mismatch')
        write_json(directory/'inputs.json', dict(config=cfg.to_dict(), loading=loading, cases=records,
            torch_version=torch.__version__, device=str(device),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
        selected = records[:manifest['max_cases']] if manifest['max_cases'] else records
        case_results, parity, geo_receipts = [], [], {}
        expected_cases = {r['case']:r for r in json.loads(Path(checkpoint['historical_metrics']).read_text())['teacher1']['cases']}
        for number, record in enumerate(selected, 1):
            case = record['case']
            images = torch.stack([dataset.read_image(indices[n]) for n in record['slice_names']])
            geo_receipts[case] = verify_geometry(record, geometry[case], images, cfg.image_size)
            write_json(directory/'geometry_verification.json', geo_receipts)
            result = analyze_case(model.backbone, dataset, indices, record, geo_receipts[case], device,
                                  directory/'cases'/f'{case}.json')
            current = aggregate_cases([result['metrics']['full']])['overall']
            historical = aggregate_cases([expected_cases[case]])['overall']
            delta = {key:current[key]-historical[key] for key in ('dice', 'dice_union_present')}
            parity.append(dict(case=case, delta=delta))
            write_json(directory/'historical_parity.json', dict(tolerance=manifest['historical_tolerance'], cases=parity))
            if any(abs(value) > manifest['historical_tolerance'] for value in delta.values()):
                raise ValueError(f'Historical validation drift exceeds predeclared tolerance: {case}: {delta}')
            case_results.append(result)
            write_json(directory/'status.json', dict(status='running', seed=int(seed), cases_done=number,
                cases_total=len(selected), latest_case=case))
            print(json.dumps(dict(seed=int(seed), case=case, done=number, total=len(selected),
                seconds=round(result['seconds'], 2), full_dice=current['dice'], historical_delta=delta)), flush=True)
        if state_hash(model) != initial:
            raise AssertionError('Frozen network state was modified')
        write_json(directory/'metrics.json', {condition:aggregate_cases([r['metrics'][condition] for r in case_results])
                                              for condition in CONDITIONS})
        receipt = dict(status='completed', model_state_unchanged=True,
            model_state_sha256=initial, cases=len(case_results), test_read=False, model_updates=0)
        write_json(directory/'receipt.json', receipt)
        write_json(directory/'status.json', receipt)
    except Exception as error:
        write_json(directory/'status.json', dict(status='failed', error=repr(error)))
        raise


def run_seed_processes(out, seeds, jobs):
    """Independent seed processes write only their own subdirectories."""
    pending, active = list(seeds), []
    try:
        while pending or active:
            while pending and len(active) < jobs:
                seed = pending.pop(0)
                command = [sys.executable, str(ROOT/'experiments/probe_slice_context.py'),
                           '--output-dir', str(out), '--seed-worker', str(seed)]
                active.append((seed, subprocess.Popen(command)))
            for seed, process in list(active):
                code = process.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'Seed {seed} exited with code {code}; see seed status')
                active.remove((seed, process))
            if active:
                time.sleep(.2)
    finally:
        for _, process in active:
            if process.poll() is None:
                process.terminate()
        for _, process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def run_worker(args):
    out = args.output_dir.resolve()
    manifest = frozen_manifest(out)
    if json.loads((out/'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Run already started; use a new directory, do not overwrite results')
    write_json(out/'status.json', dict(status='running', started_unix=time.time()))
    try:
        run_seed_processes(out, manifest['checkpoints'], manifest['jobs'])
        geo_receipts = {}
        for seed in manifest['checkpoints']:
            current = json.loads((out/f'seed{seed}/geometry_verification.json').read_text())
            if geo_receipts and current != geo_receipts:
                raise ValueError('Seed geometry verification differs')
            geo_receipts = current
        write_json(out/'geometry_verification.json', geo_receipts)
        write_json(out/'status.json', dict(status='completed', smoke=bool(manifest['max_cases']),
                                         seeds=[int(s) for s in manifest['checkpoints']]))
        from experiments.summarize_slice_context import summarize
        summarize(out)
    except Exception as error:
        write_json(out/'status.json', dict(status='failed', error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'outputs/spine_slice_context_20261030_r2')
    parser.add_argument('--geometry-manifest', type=Path, default=ROOT/'data/spider_image_geometry_manifest.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--jobs', type=int, default=1, help='Concurrent CPU seed processes (1..3)')
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42,43,44), default=[42,43,44])
    parser.add_argument('--max-cases', type=int, default=0, help='Smoke only; 0 means all validation cases')
    parser.add_argument('--historical-tolerance', type=float, default=1e-4)
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed-worker', type=int, choices=(42,43,44), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seed_worker is not None:
        run_seed(args.output_dir.resolve(), args.seed_worker)
        return
    if args.worker:
        run_worker(args)
        return
    if args.max_cases < 0 or not 0 <= args.historical_tolerance <= .001 or args.max_seconds < 1:
        parser.error('Invalid case limit, parity tolerance, or timeout')
    if not 1 <= args.jobs <= 3 or (args.jobs > 1 and args.device != 'cpu'):
        parser.error('Concurrent seeds require CPU and jobs in 1..3')
    command = prepare(args)
    print(json.dumps(dict(command=command, prepared=args.prepare_only)), flush=True)
    if not args.prepare_only:
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait(timeout=args.max_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Only stop the process group created by this invocation, including seed children.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            write_json(args.output_dir/'status.json', dict(status='interrupted_or_timed_out', max_seconds=args.max_seconds))
            raise
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
