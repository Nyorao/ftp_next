"""One continuous long LR schedule, with preserved epoch-budget checkpoints."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import run_experiment, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_long_config(cfg, output, epochs, eval_every_epochs):
    epochs = tuple(epochs)
    if not epochs or any(type(e) is not int or e <= 0 for e in epochs) or list(epochs) != sorted(set(epochs)):
        raise ValueError("Epoch milestones must be positive, unique and increasing")
    if type(eval_every_epochs) is not int or eval_every_epochs < 1:
        raise ValueError("eval_every_epochs must be a positive integer")
    if any(e % eval_every_epochs for e in epochs):
        raise ValueError("Every epoch milestone must fall on a validation epoch")
    _, sampler, split = build_training_data(cfg)
    steps_per_epoch = len(sampler)
    settings = replace(cfg, output_dir=str(Path(output).resolve() / "training"),
                       max_steps=epochs[-1]*steps_per_epoch,
                       eval_every=eval_every_epochs*steps_per_epoch).validate()
    return settings, steps_per_epoch, split


def preserve_epoch(output, cfg, epoch, step, result):
    """Freeze both the exact-epoch state and the best validation state so far."""
    output, training = Path(output), Path(cfg.output_dir)
    destination = output / "epochs" / f"{epoch:04d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an epoch snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        recovery = output / "recovery"
        recovery.mkdir(exist_ok=True)
        temporary.replace(recovery / f"incomplete_epoch_{epoch}_{time.time_ns()}")
    temporary.mkdir()
    saved = torch.load(training / "best.pt", map_location="cpu", weights_only=True)
    best_step, best_score = saved["step"], saved["best_score"]
    del saved
    assert best_step <= step and best_score == result["best_valid_dice"]
    sources = {"at_epoch.pt": training / "last.pt", "best_through_epoch.pt": training / "best.pt",
               "at_epoch_valid.json": training / f"valid_step_{step}.json",
               "best_valid.json": training / f"valid_step_{best_step}.json"}
    for name, source in sources.items():
        shutil.copy2(source, temporary / name)
    manifest = {
        "epoch": epoch, "step": step, "planned_final_steps": cfg.max_steps,
        "primary_network": cfg.primary_network,
        "at_epoch_valid_dice": result["final_validation"][cfg.primary_network]["overall"]["dice"],
        "best_valid_dice_through_epoch": best_score, "best_checkpoint_step": best_step,
        "best_selection": "highest complete valid primary-network Dice at or before this epoch",
        "schedule": "milestone within one full-duration polynomial LR schedule",
        "checkpoints": {name: {"path": str(destination / name), "sha256": sha256(temporary / name)}
                        for name in ("at_epoch.pt", "best_through_epoch.pt")},
        "test_evaluated": False,
    }
    write_json(temporary / "summary.json", manifest)
    temporary.replace(destination)
    return manifest


def run_long(cfg, device, output, epochs=(100, 200), eval_every_epochs=5, resume=False,
             on_milestone=None):
    output = Path(output).resolve()
    cfg, steps_per_epoch, split = resolve_long_config(cfg, output, epochs, eval_every_epochs)
    protocol = {
        "config": cfg.to_dict(), "epochs": list(epochs), "steps_per_epoch": steps_per_epoch,
        "epoch_definition": "one pass through labeled slices; unlabeled pool sampled independently",
        "milestone_steps": {str(e): e*steps_per_epoch for e in epochs},
        "eval_every_epochs": eval_every_epochs, "planned_epochs": epochs[-1],
        "initialization": "fresh paired seed; not a continuation of the 6000-step LR schedule",
        "selection": "complete valid teacher1 Dice only; preserve exact and best-so-far at each milestone",
        "intermediate_is_independent_short_schedule": False,
        "historical_baseline_budget_matched": False, "test_evaluated": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        if any(output.iterdir()):
            raise FileExistsError(f"Choose a fresh long-run directory: {output}")
        write_json(output / "protocol.json", protocol)
        write_json(output / "split.json", split)
        results = {"status": "running", "milestones": {}, "planned_epochs": epochs[-1],
                   "steps_per_epoch": steps_per_epoch, "test_evaluated": False}
    else:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume requires exactly the saved long-training protocol")
        results = json.loads((output / "results.json").read_text())
        if results["status"] == "completed":
            return results
        results.pop("error", None)
    write_json(output / "results.json", results)
    training = Path(cfg.output_dir)
    checkpoint = training / "last.pt"
    current = 0
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != cfg.to_dict():
            raise ValueError("Training checkpoint protocol differs from long-run protocol")
        current = int(saved["step"])
        del saved
    try:
        for epoch in epochs:
            target = epoch*steps_per_epoch
            summary_path = output / "epochs" / f"{epoch:04d}" / "summary.json"
            if str(epoch) in results["milestones"]:
                if current < target or not summary_path.exists():
                    raise ValueError("Saved milestone is newer than available training state")
                continue
            if current > target:
                raise ValueError("Cannot reconstruct a missing earlier epoch snapshot from a later checkpoint")
            results.update(status="running", active_target_epoch=epoch)
            write_json(output / "results.json", results)
            if current < target:
                invocation_start = time.monotonic()
                invocation_id = time.time_ns()
                result = None
                try:
                    result = run_experiment(cfg, torch.device(device),
                                            resume=checkpoint if current else None, stop_after=target-current)
                finally:
                    # Retain both halves' cost, including interrupted invocations.
                    (output / "invocations").mkdir(exist_ok=True)
                    write_json(output / "invocations" / f"{invocation_id}.json", dict(
                        start_step=current, target_step=target,
                        completed_step=result["step"] if result is not None else None,
                        status=result["status"] if result is not None else "failed",
                        seconds=time.monotonic()-invocation_start,
                        peak_allocated_bytes=result.get("peak_allocated_bytes") if result is not None else None,
                        peak_reserved_bytes=result.get("peak_reserved_bytes") if result is not None else None))
                assert result["step"] == target and result["final_validation"] is not None
                current = target
            else:
                # Recover interruption between the checkpoint and its epoch receipt.
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                result = {"best_valid_dice": saved["best_score"],
                          "final_validation": json.loads((training / f"valid_step_{target}.json").read_text())}
                del saved
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["step"] != target or any(sha256(v["path"]) != v["sha256"] for v in summary["checkpoints"].values()):
                    raise ValueError("Epoch snapshot changed since it was saved")
            else:
                summary = preserve_epoch(output, cfg, epoch, target, result)
            results["milestones"][str(epoch)] = summary
            write_json(output / "results.json", results)
            if on_milestone is not None:
                on_milestone(epoch)
            print(f"EPOCH {epoch} saved: step={target}, best_valid={summary['best_valid_dice_through_epoch']}", flush=True)
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        results.update(status="completed", active_target_epoch=None)
        write_json(output / "results.json", results)
        return results
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "results.json", results)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--eval-every-epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_long(load_config(args.config), args.device, args.output_dir, tuple(args.epochs),
             args.eval_every_epochs, args.resume)


if __name__ == "__main__":
    main()
