"""Build, or verify, the frozen-A probability cache that one reference config uses.

Reads only image pixels through the frozen A teacher named by the config and
writes ``<reference_cache>/{train,valid}`` (FP16, about 7.5 GB for the full
3732-slice pool at 224x224). An existing complete cache is verified instead of
rebuilt; an incomplete one is reported and left for the user to inspect.

    python experiments/build_reference_cache.py --config configs/v3/reference_joint/seed42/JOINT-pos.yaml
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import seed_everything, write_json
from freqssl.evaluation import prepare_validation
from freqssl.reference_data import ReferencePredictionCache, build_prediction_cache, file_hash


def source_base(cfg):
    """The same data protocol with conditioning removed, as A was trained."""
    return replace(cfg, reference_condition='none', reference_sources='case', reference_sampling='random',
                   reference_gt=False, reference_cycle_weight=0., reference_teacher='',
                   reference_teacher_sha256='', reference_cache='').validate()


def check_teacher(cfg, training_split):
    teacher = Path(cfg.reference_teacher)
    if not teacher.is_file():
        raise FileNotFoundError(f'Frozen A teacher not found: {teacher} (sync it from the original machine)')
    digest = file_hash(teacher)
    if cfg.reference_teacher_sha256 and digest != cfg.reference_teacher_sha256:
        raise ValueError(f'Teacher hash {digest} differs from the config ({cfg.reference_teacher_sha256})')
    payload = torch.load(teacher, map_location='cpu', weights_only=True)
    saved = payload['config']
    if (saved.get('reference_condition', 'none') != 'none' or saved.get('slice_context') != 'adaptive'
            or saved.get('seed') != cfg.seed):
        raise ValueError(f'Teacher must be the unconditioned adaptive A of seed {cfg.seed}; got seed {saved.get("seed")}')
    receipt = dict(teacher=str(teacher.resolve()), teacher_sha256=digest, teacher_step=payload.get('step'),
                   teacher_seed=saved.get('seed'), split_checked=False)
    split = teacher.parent / 'split.json'
    if split.exists():
        saved_split = json.loads(split.read_text())
        for key in ('labeled_files', 'unlabeled_files', 'validation_cases'):
            if saved_split[key] != training_split[key]:
                raise ValueError(f'{key} differ from the teacher run; check SPIDER_PNG_ROOT and the data copy')
        receipt['split_checked'] = True
    else:
        print(f'WARNING: {split} is missing; the data inventory was not compared with the teacher run', flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.reference_condition == 'none' or not cfg.reference_cache:
        raise ValueError('The config does not use a reference cache')
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if not str(device).startswith('cuda'):
        print('WARNING: building on CPU; formal runs built their caches on CUDA', flush=True)
    base = source_base(cfg)
    seed_everything(base)
    training, _, split = build_training_data(base)
    validation, _ = prepare_validation(base)
    receipt = check_teacher(cfg, split)
    root = Path(cfg.reference_cache)
    root.mkdir(parents=True, exist_ok=True)
    result = dict(config=str(args.config), cache=str(root.resolve()), device=str(device), **receipt, splits={})
    for name, dataset in (('train', training), ('valid', validation)):
        target = root / name
        if (target / 'manifest.json').exists():
            ReferencePredictionCache(target, dataset, receipt['teacher_sha256'])
            result['splits'][name] = dict(action='verified', slices=len(dataset.names))
            print(f'{name}: verified existing cache ({len(dataset.names)} slices)', flush=True)
            continue
        if target.exists():
            raise FileExistsError(f'{target} exists without a complete manifest; inspect or delete it, then rerun')
        required = len(dataset.names) * 20 * base.image_size ** 2 * 2 + 3 * 1024 ** 3
        if shutil.disk_usage(root).free < required:
            raise OSError(f'Need about {required / 1024 ** 3:.1f} GiB free under {root}')
        print(f'{name}: building {len(dataset.names)} slices on {device}', flush=True)
        manifest = build_prediction_cache(cfg.reference_teacher, dataset, target, device, base.eval_batch_size)
        result['splits'][name] = dict(action='built', slices=len(manifest['slices']), seconds=manifest['seconds'])
    write_json(root / 'receipt.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
