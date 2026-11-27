"""Resume only the pre-authorized confirmation phase from unchanged frozen code.

Completed arms are verified and reused. An interrupted arm retains its total
budget, optimizer, EMA and sampling state; the frozen engine archives its crash
tail. This driver never modifies the frozen training source or protocol.
"""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def load_runner(output):
    code = output / "code"
    protocol = json.loads((output / "protocol.json").read_text())
    for relative, expected in protocol["frozen_sha256"].items():
        if hashlib.sha256((code / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("frozen_context_runner", code / "experiments/run_slice_context.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.ROOT == code
    return runner, protocol


def prepare(output, runner, protocol):
    results = runner.read_json(output / "results.json")
    selected = runner.choose_module(results["baseline"]["overall"], results["screening"]["mean"]["overall"],
                                     results["screening"]["adaptive"]["overall"])
    if not selected or selected != results["selected_module"] or not protocol["confirm_on_pass"]:
        raise ValueError("Confirmation is not authorized by the frozen screening gate")
    if results["status"] == "completed" or results.get("test_evaluated"):
        raise ValueError("Suite already completed or test accessed; inspect rather than rerun")
    for row in [results["baseline"], *results["screening"].values()]:
        if runner.sha256(Path(row["directory"]) / "best.pt") != row["checkpoint_sha256"]:
            raise ValueError("Screening checkpoint changed")
    rules = protocol["confirmation_rules"]
    if rules != {"seeds": [42, 43, 44], "steps": 30000, "min_mean_gain": .003,
                 "min_positive_seeds": 2, "mean_union_present_non_decreasing": True}:
        raise ValueError("This recovery driver requires the original confirmation protocol")
    cfg = runner.ExperimentConfig(**protocol["config"]).validate()
    actions = []
    for seed in rules["seeds"]:
        for mode in ("none", selected):
            key = f"seed{seed}/{mode}"
            directory = output / "confirmation" / key
            full = replace(cfg, output_dir=str(directory), seed=seed, slice_context=mode,
                           max_steps=30000, eval_every=1000, warmup_steps=1500, ramp_steps=3000,
                           max_seconds=86400).validate()
            action, step = "new", 0
            if directory.exists() and any(directory.iterdir()):
                if runner.read_json(directory / "config.json") != full.to_dict():
                    raise ValueError(f"Saved protocol differs: {key}")
                state = runner.torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
                if state["config"] != full.to_dict():
                    raise ValueError(f"Checkpoint protocol differs: {key}")
                step = state["step"]
                if (directory / "result.json").exists() and runner.read_json(directory / "result.json")["status"] == "completed":
                    if step != full.max_steps:
                        raise ValueError("Completed run has incomplete checkpoint")
                    runner.compact_run(directory)
                    action = "reuse_completed"
                else:
                    if not 0 < step < full.max_steps:
                        raise ValueError("Recovery requires an unfinished checkpoint")
                    action = "resume"
            actions.append((key, full, action, step))
    return results, selected, actions


def assert_no_live_run(output, runner):
    for path in output.glob("**/status.json"):
        pid = runner.read_json(path).get("pid")
        if not pid or pid == os.getpid():
            continue
        proc = Path("/proc") / str(pid) / "cmdline"
        try:
            command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            continue
        if str(output) in command:
            raise RuntimeError(f"Another training process is still alive: {pid}")


def run(output, runner, protocol, results, selected, actions):
    assert_no_live_run(output, runner)
    recovery = output / "recovery" / str(time.time_ns())
    recovery.mkdir(parents=True)
    shutil.copy2(output / "results.json", recovery / "previous_results.json")
    shutil.copy2(__file__, recovery / "resume_driver.py")
    for key, cfg, action, step in actions:
        directory = Path(cfg.output_dir)
        if action == "resume":
            destination = recovery / key
            destination.mkdir(parents=True)
            shutil.copy2(directory / "status.json", destination / "previous_status.json")
            runner.write_json(destination / "checkpoint_receipt.json", {
                "resume_step": step, "checkpoint_sha256": runner.sha256(directory / "last.pt"),
                "prior_uncheckpointed_updates_will_be_replayed": True})
    receipt = {"pid": os.getpid(), "status": "running", "frozen_code": str(runner.ROOT),
               "driver_sha256": runner.sha256(recovery / "resume_driver.py"),
               "actions": [{"run": k, "action": a, "step": s} for k, _, a, s in actions],
               "reason": "Previous host process absent; status file was stale. Exit cause not recorded."}
    runner.write_json(recovery / "receipt.json", receipt)
    results.update(status="running", recovery_receipt=str(recovery / "receipt.json"))
    results.pop("error", None)
    runner.write_json(output / "results.json", results)
    device = runner.torch.device(protocol["device"])
    runs, deltas = {}, []
    try:
        for key, cfg, action, step in actions:
            results["active_run"] = "confirmation/" + key
            runner.write_json(output / "results.json", results)
            print(f"CONFIRM {key}: {action}, step={step}/{cfg.max_steps}", flush=True)
            directory = Path(cfg.output_dir)
            if action != "reuse_completed":
                runner.run_experiment(cfg, device, resume=directory / "last.pt" if action == "resume" else None)
            runs[key] = runner.compact_run(directory)
            runner.release()
            seed, mode = key.split("/")
            if mode == selected:
                _, base = runner.selected_validation(output / "confirmation" / seed / "none")
                _, candidate = runner.selected_validation(directory)
                pair = runner.paired_cases(candidate, base)
                runs[key]["paired_cases"] = pair
                deltas.append(pair["mean_delta_pp"])
            runner.write_json(output / "confirmation_progress.json", {"runs": runs, "deltas_pp": deltas})
        union = [runs[f"seed{s}/{selected}"]["overall"]["dice_union_present"] -
                 runs[f"seed{s}/none"]["overall"]["dice_union_present"] for s in (42, 43, 44)]
        confirmed = bool(runner.np.mean(deltas) >= .3 and sum(x > 0 for x in deltas) >= 2 and runner.np.mean(union) >= 0)
        results["confirmation"] = {"runs": runs, "deltas_pp": deltas,
            "mean_delta_pp": float(runner.np.mean(deltas)), "positive_seeds": sum(x > 0 for x in deltas), "passed": confirmed}
        runner.write_json(output / "results.json", results)
        if confirmed:
            runner.write_json(output / "fixed_before_test.json", results["confirmation"])
            test_results = {}
            for key, value in runs.items():
                model, cfg, _ = runner.load_model(Path(value["directory"]) / "best.pt", device)
                runner.seed_everything(cfg)
                data, records = runner.prepare_validation(cfg, split="test")
                test_results[key] = runner.evaluate_volumes(model, data, records, device,
                    networks=("teacher1",), batch_size=cfg.eval_batch_size)["teacher1"]
                del model
                runner.release()
            runner.write_json(output / "test_descriptive.json", test_results)
            results["test_evaluated"] = True
        results.update(status="completed", active_run=None)
        runner.write_json(output / "results.json", results)
        runner.render_report(output, results)
        receipt["status"] = "completed"
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        runner.write_json(output / "results.json", results)
        receipt.update(status="failed", error=results["error"])
        raise
    finally:
        runner.write_json(recovery / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    runner, protocol = load_runner(output)
    if args.inspect_only:
        _, _, actions = prepare(output, runner, protocol)
        print(json.dumps([{"run": k, "action": a, "step": s} for k, _, a, s in actions], indent=2))
        return
    with (output / "confirmation_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results, selected, actions = prepare(output, runner, protocol)
        run(output, runner, protocol, results, selected, actions)


if __name__ == "__main__":
    main()
