"""Launch independent dual-only long-training arms with one frozen code snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from experiments.compare_v3 import resolve_arm
from experiments.train_long_v3 import resolve_long_config, sha256
from freqssl.config import load_config
from freqssl.engine import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/long_200epochs.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=("dual_base", "dual_dice", "dual_region", "dual_aux"),
                        default=["dual_base", "dual_dice", "dual_region"])
    parser.add_argument("--epochs", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--eval-every-epochs", type=int, default=5)
    args = parser.parse_args()
    if len(args.gpus) != len(args.arms) or len(set(args.gpus)) != len(args.gpus) or len(set(args.arms)) != len(args.arms):
        parser.error("Provide one distinct GPU for each distinct arm")
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.cpu_threads)
    output = args.output_dir.resolve()
    settings, steps_per_epoch, _ = resolve_long_config(cfg, output, tuple(args.epochs), args.eval_every_epochs)
    resolved = {arm: resolve_arm(settings, arm, "auxiliary", output / arm / "training") for arm in args.arms}
    storage_root = output
    while not storage_root.exists():
        storage_root = storage_root.parent
    free_bytes = shutil.disk_usage(storage_root).free
    minimum_bytes = len(args.arms) * 2 * 1024**3
    if free_bytes < minimum_bytes:
        raise RuntimeError(f"Need at least {minimum_bytes/1024**3:.0f} GiB for checkpoints/logs; "
                           f"only {free_bytes/1024**3:.1f} GiB free on {storage_root}")
    gpu_before = {}
    for gpu in args.gpus:
        raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                                       "--query-gpu=memory.used,memory.free,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True).strip()
        used, free, utilization = [int(value.strip()) for value in raw.split(",")]
        if used > 1024 or free < 4096 or utilization > 50:
            raise RuntimeError(f"GPU {gpu} is occupied ({raw}); no workers started")
        gpu_before[str(gpu)] = {"used_mib": used, "free_mib": free, "utilization_percent": utilization}
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh group directory: {output}")
    source_root = output / "code"
    paths = [*sorted((ROOT / "freqssl").glob("*.py")), *sorted((ROOT / "configs/v3").glob("*.yaml")),
             ROOT / "spider_volume.py", ROOT / "train.py", ROOT / "evaluate.py",
             ROOT / "experiments/compare_v3.py", ROOT / "experiments/launch_v3.py",
             ROOT / "experiments/train_long_v3.py", Path(__file__).resolve()]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256(source)
    write_json(output / "launch_source_hashes.json", hashes)
    # Read prior summary metrics only. Different budgets are explicit and are
    # never passed through the strict matched-budget reference comparator.
    history = {}
    historical_path = ROOT / "outputs/freqssl_v3_screening_20261021/comparison/results.json"
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        for method in ("mean_teacher", "dual_frequency"):
            key = f"seed{cfg.seed}/{method}"
            if key in historical.get("runs", {}):
                record = historical["runs"][key]
                history[method] = {"source": str(historical_path), "source_sha256": sha256(historical_path),
                                   "steps": record["step"], "best_valid_dice": record["best_valid_dice"],
                                   "matched_budget": False, "purpose": "historical context only"}
    receipt = {"status": "launching", "started_utc": datetime.now(timezone.utc).isoformat(),
               "arms": args.arms, "seed": cfg.seed, "epochs": args.epochs,
               "steps_per_epoch": steps_per_epoch,
               "milestone_steps": {str(e): e*steps_per_epoch for e in args.epochs},
               "gpus_before_launch": gpu_before, "runs": {}, "historical_references": history,
               "no_new_single_student_training": True, "test_evaluated": False,
               "storage_free_bytes_at_launch": free_bytes,
               "intermediate_checkpoint_uses_final_epoch_lr_schedule": True,
               "initialization": "fresh paired seed", "monitoring": "none"}
    write_json(output / "launch.json", receipt)
    try:
        for arm, gpu in zip(args.arms, args.gpus):
            config_path = output / f"{arm}.yaml"
            config_path.write_text(yaml.safe_dump(resolved[arm].to_dict(), sort_keys=False))
            command = [sys.executable, "-u", str(source_root / "experiments/train_long_v3.py"),
                       "--config", str(config_path), "--output-dir", str(output / arm),
                       "--device", f"cuda:{gpu}", "--epochs", *[str(e) for e in args.epochs],
                       "--eval-every-epochs", str(args.eval_every_epochs)]
            console = output / f"{arm}.log"
            with console.open("w") as log:
                process = subprocess.Popen(command, cwd=source_root, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            receipt["runs"][arm] = {"pid": process.pid, "gpu": gpu, "command": command,
                                     "cwd": str(source_root), "console": str(console),
                                     "results": str(output / arm / "results.json"),
                                     "training_status": str(output / arm / "training/status.json")}
            write_json(output / "launch.json", receipt)
        receipt["status"] = "launched"
    except BaseException as error:
        receipt.update(status="partially_launched", error=f"{type(error).__name__}: {error}")
        write_json(output / "launch.json", receipt)
        raise
    write_json(output / "launch.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
