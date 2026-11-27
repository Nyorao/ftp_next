"""Wait for an unoccupied GPU, then run the fixed prompt-free comparison once."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def gpu_state():
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"
    ], text=True)
    occupied = set(processes.splitlines())
    return [{"uuid": p[0], "used_mib": int(p[1]), "free_mib": int(p[2]), "utilization": int(p[3]),
             "has_compute_process": p[0] in occupied}
            for row in rows.splitlines() if (p := [v.strip() for v in row.split(",")])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps", type=int, help="Optional explicit short-test cap")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=48)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "queue.json"
    if status_path.exists() or (output / "manifest.json").exists():
        raise FileExistsError(output)
    snapshot_root = output / "queued_source"
    snapshot_root.mkdir()
    # Pin the runnable files now: later code edits cannot change queued work.
    import hashlib
    hashes = {}
    for pattern in ("*.py", "model/*.py", "experiments/*.py", "configs/*.yaml", "tests/*.py", "data/*.json"):
        for source in ROOT.glob(pattern):
            relative = source.relative_to(ROOT)
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            destination.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    (output / "queued_source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    status = {"status": "waiting_for_idle_gpu", "queue_pid": os.getpid(), "epochs_per_arm": args.epochs,
              "explicit_step_cap": args.steps,
              "policy": "no compute processes, <=256 MiB used, >=8000 MiB free, <=5% utilization; two consecutive polls",
              "source_snapshot": str(snapshot_root)}

    def record():
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    record()
    started = time.monotonic()
    previous = set()
    try:
        while time.monotonic() - started < args.max_wait_hours * 3600:
            devices = gpu_state()
            status["last_gpu_state"] = devices
            eligible = {d["uuid"] for d in devices if not d["has_compute_process"]
                        and d["used_mib"] <= 256 and d["free_mib"] >= 8000 and d["utilization"] <= 5}
            stable = sorted(eligible & previous)
            if stable:
                selected = stable[0]
                command = [sys.executable, "-u", str(snapshot_root / "experiments/compare_frequency.py"),
                           "--output-dir", str(output), "--device", "cuda:0", "--epochs", str(args.epochs)]
                if args.steps is not None:
                    command += ["--steps", str(args.steps)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=selected, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                with (output / "console.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=snapshot_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    status.update(status="training", training_pid=process.pid, gpu_uuid=selected, command=command)
                    record()
                    code = process.wait()
                status.update(status="completed" if code == 0 else "failed", exit_code=code)
                record()
                return
            previous = eligible
            record()
            time.sleep(args.poll_seconds)
        status["status"] = "expired_waiting_for_gpu"
        record()
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        record()
        raise


if __name__ == "__main__":
    main()
