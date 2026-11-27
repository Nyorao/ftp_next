"""Validation-only follow-up for the three completed KV-pos checkpoints."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    # Import the original experiment snapshot for bitwise comparison.
    sys.path.insert(0, str(run / 'code'))
    for seed in args.seeds:
        diagnose(run, args.out.resolve(), seed)


def diagnose(run, out, seed):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from freqssl.config import ExperimentConfig
    from freqssl.context_cache import iter_cached_logits
    from freqssl.data import _read_transposed
    from freqssl.engine import seed_everything, state_hash, write_json
    from freqssl.evaluation import prepare_validation
    from freqssl.inference import load_single_network
    from freqssl.reference_data import build_prediction_cache, file_hash
    from freqssl.reference_inference import iter_reference_logits
    from freqssl.reference_metrics import WorkCounter, numbering_metrics
    from experiments.analyze_case_reference import confusion_metrics

    device = 'cuda:0'
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one GPU for each diagnostic worker')
    dest = out / f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=False)
    status = dest / 'status.json'
    manifest = json.loads((run / 'manifest.json').read_text())
    source = manifest['baselines'][str(seed)]
    checkpoint = run / f'training/seed{seed}/KV-pos/last.pt'
    saved = {}
    for arm in ('A', 'KV-pos'):
        result = json.loads((run / f'analysis/seed{seed}/{arm}/summary.json').read_text())
        expected = source['checkpoint_sha256'] if arm == 'A' else file_hash(checkpoint)
        if result['checkpoint_sha256'] != expected:
            raise ValueError(f'{arm}: checkpoint differs from saved diagnostics')
        saved[arm] = {row['case']: row for row in result['cases']}

    try:
        write_json(status, dict(status='rebuilding_validation_cache', completed_cases=0))
        base = replace(ExperimentConfig(**manifest['base']), seed=seed)
        seed_everything(base)
        dataset, _ = prepare_validation(base)
        cache_root = dest / 'temporary_cache'
        rebuilt = build_prediction_cache(source['checkpoint'], dataset, cache_root / 'valid',
                                         device, base.eval_batch_size)
        original = json.loads((run / f'cache_manifests/seed{seed}/valid.json').read_text())
        if rebuilt['teacher_sha256'] != original['teacher_sha256'] or rebuilt['slices'] != original['slices']:
            raise ValueError('Rebuilt validation cache differs from original')
        model, cfg, receipt = load_single_network(checkpoint, device, 'teacher1')
        if cfg.reference_condition != 'position' or receipt['checkpoint_step'] != 6000:
            raise ValueError('Expected final KV-pos EMA')
        seed_everything(cfg)
        dataset, records = prepare_validation(replace(cfg, reference_cache=str(cache_root)))
        cache = dataset.reference_cache
        if len(records) != 21 or any({r['case'] for r in records} != set(rows) for rows in saved.values()):
            raise ValueError('Expected the original 21 validation cases')
        a_model, _, _ = load_single_network(source['checkpoint'], device, 'teacher1') if seed == 43 else (None, None, None)
        lookup = {name: i for i, name in enumerate(dataset.names)}
        before = state_hash(model)
        cases = []

        def score(cm):
            return dict(confusion_metrics(cm), numbering=numbering_metrics(cm), confusion=cm.tolist())

        def on_native_grid(prediction, truth):
            return F.interpolate(prediction[None, None].float(), size=truth.shape,
                                 mode='nearest')[0, 0].to(torch.long)

        def slice_cm(prediction, truth):
            restored = on_native_grid(prediction, truth)
            return np.bincount((truth * 20 + restored).numpy().ravel(), minlength=400).reshape(20, 20)

        for record in records:
            case, names = record['case'], record['slice_names']
            n = len(names)
            r = cache.selected[case]
            middle = (n - 1) // 2
            ranks = sorted(range(n), key=lambda i: (-original['slices'][names[i]]['canal_area'],
                                                    abs(i-middle), i))[:5]
            if len(ranks) < 5 or ranks[0] != r or saved['KV-pos'][case]['selected_reference'] != r:
                raise ValueError(f'{case}: reference ranking or slice inventory mismatch')
            images = torch.stack([dataset.read_image(lookup[name]) for name in names])
            targets = [_read_transposed(dataset.root / 'label' / name, torch.long) for name in names]
            # Rank references using only the immutable image-only teacher cache.
            # GT substitution happens separately and stays on the selected slice.
            gt_small = F.interpolate(targets[r][None, None].float(), size=cfg.image_size,
                                     mode='nearest')[0, 0].long()
            oracle = F.one_hot(gt_small, 20).permute(2, 0, 1).float()
            reference_probs = {i: cache.read(names[i]) for i in ranks}

            def predict(index, override=None):
                counter = WorkCounter(model.backbone)
                torch.cuda.synchronize(device)
                started = time.monotonic()
                prob = torch.empty((n, 20, cfg.image_size, cfg.image_size), dtype=torch.float32)
                labels = torch.empty((n, cfg.image_size, cfg.image_size), dtype=torch.long)
                try:
                    for start, logits in iter_reference_logits(
                            model.backbone, n, images.__getitem__, reference_probs.__getitem__,
                            device, cfg.eval_batch_size, reference_index=index,
                            reference_probability=override):
                        prob[start:start + len(logits)] = logits.float().softmax(1).cpu()
                        labels[start:start + len(logits)] = logits.argmax(1).cpu()
                finally:
                    cost = counter.finish()
                torch.cuda.synchronize(device)
                return prob, labels, dict(macs=cost['total_counted_macs'], seconds=time.monotonic()-started)

            def cm_by_slice(predictions):
                return np.stack([slice_cm(prediction, truth)
                                 for prediction, truth in zip(predictions, targets, strict=True)])

            scores, costs = {}, []
            accumulator = None
            for rank, index in enumerate(ranks, 1):
                probs, labels, cost = predict(index)
                costs.append(cost)
                if accumulator is None:
                    accumulator = probs
                else:
                    accumulator.add_(probs)
                    del probs
                if rank in (1, 3, 5):
                    # Dividing by the ensemble size leaves argmax unchanged.
                    cms = cm_by_slice(labels if rank == 1 else accumulator.argmax(1))
                    scores[f'top{rank}'] = score(cms.sum(0))
                    if rank == 1:
                        full_cms = cms
            expected = np.asarray(saved['KV-pos'][case]['conditions']['full']['groups']['all']['confusion'])
            if not np.array_equal(full_cms.sum(0), expected):
                raise ValueError(f'{case}: recomputed single-reference confusion differs')
            del accumulator
            probs, labels, _ = predict(r, oracle)
            oracle_cms = cm_by_slice(labels)
            del probs
            teacher_ref_cm = slice_cm(reference_probs[r].argmax(0), targets[r])
            outside = np.arange(n) != r
            origin = full_cms[outside].sum(0)
            oracle_outside = oracle_cms[outside].sum(0)
            row = dict(case=case, selected_reference=r, ranked_indices=ranks, slices=n,
                       ranked_canal_pixels=[original['slices'][names[i]]['canal_area'] for i in ranks],
                       source_reference=score(teacher_ref_cm), ensembles=scores,
                       source_cost=saved['KV-pos'][case]['source_cost'], target_costs=costs,
                       oracle_excluding_reference=dict(pred=score(origin), gt=score(oracle_outside)))
            if a_model is not None:
                a_predictions = []
                for _, logits in iter_cached_logits(a_model.backbone, n, images.__getitem__,
                                                     device, cfg.eval_batch_size):
                    a_predictions.extend(logits.argmax(1).cpu().unbind(0))
                a_cms = cm_by_slice(a_predictions)
                a_expected = saved['A'][case]['conditions']['full']['groups']['all']['confusion']
                if not np.array_equal(a_cms.sum(0), a_expected):
                    raise ValueError(f'{case}: recomputed A confusion differs')
                row['slice_confusions'] = dict(A=a_cms.tolist(), KV_pos=full_cms.tolist())
            cases.append(row)
            write_json(status, dict(status='running', completed_cases=len(cases), total_cases=len(records)))
            print(seed, case, f'{len(cases)}/{len(records)}', flush=True)

        if state_hash(model) != before:
            raise ValueError('Diagnostics changed model state')
        write_json(dest / 'detail.json', dict(seed=seed, checkpoint_sha256=file_hash(checkpoint),
                   teacher_sha256=original['teacher_sha256'], validated_cache_slices=len(original['slices']),
                   test_read=False, cases=cases))
        shutil.rmtree(cache_root)
        write_json(status, dict(status='completed', completed_cases=len(cases)))
    except BaseException as error:
        write_json(status, dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
