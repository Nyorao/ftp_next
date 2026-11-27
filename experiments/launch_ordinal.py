"""Freeze and launch three matched OAR-Spine mechanism ablations, no test run."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqssl.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/v3/dual_ordinal.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', nargs=3, type=int, default=[0, 1, 2])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.method != 'dual_ordinal' or len(set(args.gpus)) != 3:
        raise ValueError('Require dual_ordinal and three distinct GPUs')
    modes = {'hard_peer': (False, False), 'hierarchy': (True, False), 'ordinal': (True, True)}
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    gpu_records = {}
    for gpu in args.gpus:
        line = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, free, util = map(int, line.strip().split(','))
        if used > 1024 or free < 5000 or util > 50:
            raise RuntimeError(f'GPU {gpu} occupied: {line.strip()}')
        gpu_records[gpu] = dict(used_mib=used, free_mib=free, utilization=util)
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 6*1024**3:
        raise RuntimeError('Need at least 6 GiB for the three runs')
    output.mkdir(parents=True, exist_ok=True)
    source_root = output/'code'
    files = [*sorted((ROOT/'freqssl').glob('*.py')), ROOT/'spider_volume.py', ROOT/'train.py', ROOT/'evaluate.py',
             ROOT/'experiments/compare_v3.py', Path(__file__).resolve(), ROOT/'configs/v3/dual_ordinal.yaml']
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = source_root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output/'launch_source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    receipt = {'status': 'launching', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'gpu_preflight': gpu_records, 'arms': {}, 'test_access': False,
               'selection': 'teacher1 complete native validation Dice only', 'primary_hypothesis':
               'Ordinal chain evidence improves over hierarchy-only routing; compare identical heads and schedule.',
               'new_single_student_runs': 0, 'max_steps_per_arm': cfg.max_steps,
               'historical_controls': 'Existing seed42 6000-step mean_teacher/dual_cps/dual_frequency; no rerun',
               'known_data_limits': ['Legacy direction normalization used GT', 'Patient-level split mapping unverified']}
    for (name, (routing, chain)), gpu in zip(modes.items(), args.gpus):
        resolved = replace(cfg, output_dir=str(output/name), ordinal_use_routing=routing,
                           ordinal_use_chain=chain).validate()
        config_path = output/f'{name}.yaml'
        config_path.write_text(yaml.safe_dump(resolved.to_dict(), sort_keys=False))
        command = [sys.executable, '-u', str(source_root/'train.py'), '--config', str(config_path), '--device', f'cuda:{gpu}']
        with (output/f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source_root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        receipt['arms'][name] = {'pid': process.pid, 'gpu': gpu, 'command': command,
                                'config': str(config_path), 'log': str(output/f'{name}.log')}
        (output/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    receipt['status'] = 'launched'
    (output/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
