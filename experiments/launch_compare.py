"""Start one detached, sequential pilot after checking available GPU memory."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--arms", nargs="+", choices=("anchor", "dual_no_feedback", "dual_feedback"),
                        default=["anchor", "dual_no_feedback", "dual_feedback"])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    if min(args.steps, args.timeout_seconds) <= 0 or args.gpu < 0:
        parser.error("steps and timeout-seconds must be positive; gpu must be non-negative")
    info = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, free_mib = info[0].strip(), int(info[1])
    # The measured full two-student batch=2 peak is about 2 GiB. Leave headroom.
    if free_mib < 3072:
        raise RuntimeError(f"GPU {args.gpu} has only {free_mib} MiB free; nothing was launched")
    output.mkdir(parents=True)
    command = [sys.executable, "-u", str(ROOT / "experiments/compare_dual.py"),
               "--device", "cuda:0", "--output-dir", str(output),
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--timeout-seconds", str(args.timeout_seconds), "--arms", *args.arms]
    environment = os.environ.copy()
    # UUID avoids ambiguity between physical and process-local GPU numbering.
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    log = output / "console.log"
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    metadata = {
        "pid": process.pid, "launched_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.gpu, "gpu_uuid": uuid, "free_mib_before_launch": free_mib,
        "command": command, "console_log": str(log), "output_dir": str(output),
        "mode": "one background process, sequential arms; no automatic result interpretation",
    }
    (output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
