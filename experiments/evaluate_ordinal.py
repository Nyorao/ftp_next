"""Evaluate the preselected OAR-Spine checkpoints using their frozen runtime.

This command is invoked only after a request to inspect completed experiments.
It never trains or picks a checkpoint/network using test scores.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-test', type=Path, required=True)
    parser.add_argument('--baseline-code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    source = run/'code'
    sys.path.insert(0, str(source))
    import torch
    from freqssl.engine import load_model, seed_everything, state_hash, write_json
    from freqssl.evaluation import prepare_validation, evaluate_volumes
    assert Path(sys.modules['freqssl.engine'].__file__).resolve().is_relative_to(source)
    for name, digest in read(run/'launch_source_hashes.json').items():
        assert sha(source/name) == digest, name
    for name in ('freqssl/evaluation.py', 'freqssl/data.py', 'spider_volume.py'):
        assert sha(source/name) == sha(args.baseline_code/name), name
    arms = ('hard_peer', 'hierarchy', 'ordinal')
    networks = ('teacher1', 'student1', 'teacher2', 'student2', 'teachers_mean', 'students_mean')
    modes = {'hard_peer': (False, False), 'hierarchy': (True, False), 'ordinal': (True, True)}
    manifest = {'training_run': str(run), 'frozen_source': str(source),
                'frozen_sources_verified': True, 'evaluation_script_sha256': sha(__file__),
                'primary_network': 'teacher1', 'split': 'test', 'checkpoint_selection': 'complete valid teacher1 Dice only',
                'test_used_for_selection': False, 'secondary_networks_same_checkpoint': True,
                'model_updates': 0, 'postprocessing': 'none', 'device': args.device,
                'baseline_test': str(args.baseline_test.resolve()),
                'baseline_results_sha256': sha(args.baseline_test/'results.json'),
                'checkpoints': {}, 'diagnostics': {}}
    common = None
    for arm in arms:
        folder = run/arm
        completed = read(folder/'result.json')
        assert completed['status'] == 'completed' and completed['step'] == 6000
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        cfg = payload['config']
        assert cfg['method'] == 'dual_ordinal' and cfg['max_steps'] == 6000
        assert (cfg['ordinal_use_routing'], cfg['ordinal_use_chain']) == modes[arm]
        shared = {k:v for k,v in cfg.items() if k not in ('output_dir', 'ordinal_use_routing', 'ordinal_use_chain')}
        if common is None:
            common = shared
        else:
            assert common == shared
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line]
        train = [r for r in rows if r['kind'] == 'train']
        validation = [r for r in rows if r['kind'] == 'validation']
        assert [r['step'] for r in train] == list(range(1, 6001))
        assert [r['step'] for r in validation] == list(range(500, 6001, 500))
        selected = max(validation, key=lambda r:r['dice'])
        assert payload['step'] == selected['step']
        assert payload['best_score'] == selected['dice'] == completed['best_valid_dice']
        assert read(folder/f"valid_step_{payload['step']}.json")['teacher1']['overall']['dice'] == selected['dice']
        manifest['checkpoints'][arm] = {'path': str(folder/'best.pt'), 'sha256': sha(folder/'best.pt'),
                                       'step': payload['step'], 'best_valid_dice': payload['best_score']}
        tail = train[-500:]
        metrics = ('loss', 'supervised_s1', 'supervised_s2', 'unsupervised_s1', 'unsupervised_s2',
                   'coverage_s1', 'foreground_coverage_s1', 'fine_coverage_s1', 'interval_coverage_s1',
                   'family_only_coverage_s1', 'teacher_disagreement', 'foreground_disagreement')
        manifest['diagnostics'][arm] = {'validation_curve': validation,
            'seconds': completed['seconds_this_invocation'],
            'last500_mean': {k:sum(r[k] for r in tail)/len(tail) for k in metrics if all(k in r for r in tail)}}
        del payload, rows, train
    # Record all choices before reading test images or labels.
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out/'evaluation_script.py')
    write_json(out/'manifest.json', manifest)
    baseline = read(args.baseline_test/'results.json')
    results = {'status': 'running', 'arms': {}, 'split': 'test', 'primary_network': 'teacher1',
               'historical_baselines': {k:baseline['arms'][k] for k in ('mean_teacher', 'dual_cps', 'dual_frequency')},
               'multi_seed_verified': False, 'model_updates': 0}
    write_json(out/'results.json', results)
    dataset = records = None
    start = time.monotonic()
    try:
        for arm in arms:
            print('START', arm, flush=True)
            meta = manifest['checkpoints'][arm]
            model, cfg, payload = load_model(meta['path'], torch.device(args.device))
            seed_everything(cfg)
            before = state_hash(model)
            if dataset is None:
                dataset, records = prepare_validation(cfg, 'test')
                assert records == read(args.baseline_test/'test_reconstruction_audit.json')
                write_json(out/'test_reconstruction_audit.json', records)
                manifest.update(test_cases=len(records), test_slices=len(dataset), baseline_case_audit_exact_match=True)
                write_json(out/'manifest.json', manifest)
            values = evaluate_volumes(model, dataset, records, torch.device(args.device), networks,
                                      batch_size=cfg.eval_batch_size)
            assert state_hash(model) == before
            assert sha(meta['path']) == meta['sha256']
            write_json(out/f'{arm}.json', {'metadata': meta, 'test': values, 'model_unchanged': True})
            results['arms'][arm] = {'checkpoint_step': payload['step'], 'best_valid_dice': payload['best_score'],
                'test_primary': values['teacher1']['overall'],
                'test_networks': {name:v['overall'] for name,v in values.items()}}
            write_json(out/'results.json', results)
            print('DONE', arm, 'test_dice', values['teacher1']['overall']['dice'], flush=True)
            del model, payload, values
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        results.update(status='completed', seconds=time.monotonic()-start)
        write_json(out/'results.json', results)
    except BaseException as error:
        results.update(status='failed', error=repr(error))
        write_json(out/'results.json', results)
        raise


if __name__ == '__main__':
    main()
