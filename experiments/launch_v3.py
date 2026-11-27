"""Launch the v3 paired comparison on one free GPU, preserving process receipts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqssl.config import METHODS, load_config
from experiments.compare_v3 import AUXILIARY_ARMS, SUITE_ARMS, load_references, resolve_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/screening.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--suite", choices=SUITE_ARMS, default="baseline")
    parser.add_argument("--reference-comparison", type=Path)
    parser.add_argument("--arms", nargs="+", choices=(*METHODS, *AUXILIARY_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    args.arms = args.arms or list(SUITE_ARMS[args.suite])
    cfg = load_config(args.config)
    if len(args.arms) != len(set(args.arms)) or len(args.seeds) != len(set(args.seeds)):
        parser.error("Arms and seeds must be unique")
    for arm in args.arms:
        resolve_arm(cfg, arm, args.suite)
    # Fail before launching if the historical baselines do not match this run.
    load_references(cfg, args.seeds, args.reference_comparison)
    gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                                  "--query-gpu=memory.used,memory.free,utilization.gpu",
                                  "--format=csv,noheader,nounits"], text=True).strip()
    used, free, utilization = [int(value.strip()) for value in gpu.split(",")]
    if used > 1024 or free < 4096 or utilization > 50:
        raise RuntimeError(f"GPU {args.gpu} is occupied: {gpu}; no process was started")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh launch directory: {output}")
    # Freeze the resolved launch config so editing a template cannot change a
    # queued arm's training protocol. All arms still share one immutable config.
    import yaml
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    # Execute the snapshot itself: later workspace edits must not alter queued
    # arms or make their recorded source hashes disagree with loaded Python.
    source_root = output / "code"
    paths = [*sorted((ROOT / "freqssl").glob("*.py")),
             *sorted((ROOT / "configs/v3").glob("*.yaml")),
             ROOT / "spider_volume.py", ROOT / "train.py", ROOT / "evaluate.py",
             ROOT / "experiments/compare_v3.py", Path(__file__).resolve()]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "launch_source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    command = [sys.executable, "-u", str(source_root / "experiments/compare_v3.py"),
               "--config", str(config_path), "--output-dir", str(output / "comparison"),
               "--device", f"cuda:{args.gpu}", "--suite", args.suite, "--arms", *args.arms,
               "--seeds", *[str(seed) for seed in args.seeds]]
    if args.reference_comparison is not None:
        command += ["--reference-comparison", str(args.reference_comparison.resolve())]
    with (output / "console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=source_root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    receipt = {"pid": process.pid, "command": command, "cwd": str(source_root), "gpu_index": args.gpu,
               "gpu_before_launch": {"used_mib": used, "free_mib": free, "utilization_percent": utilization},
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "console": str(output / "console.log"), "status": "launched",
               "results": str(output / "comparison/results.json"),
               "per_arm_timeout_seconds": cfg.max_seconds,
               "suite": args.suite, "test_evaluated": False}
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
