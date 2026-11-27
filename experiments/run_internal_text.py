"""Prepare/freeze and execute the three-arm internal-text screening protocol.

--prepare-only creates a reviewable snapshot without starting training.
Runs use that snapshot, identical budgets, and the existing native valid metric.
Explicit --resume resumes a stopped screen; failures are never auto-retried.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from freqssl.config import load_config
from freqssl.engine import run_experiment, write_json
from freqssl.internal_text import load_text_bank
from freqssl.prompt_supervision import file_sha256


ARMS = ("context", "neutral", "text")


def screening_decision(base, neutral, text):
    passes_base = (text["dice"] - base["dice"] >= .003
                   and text["dice_union_present"] >= base["dice_union_present"])
    passes_neutral = (text["dice"] - neutral["dice"] >= .001
                      and text["dice_union_present"] >= neutral["dice_union_present"])
    return {"passed_single_seed_screen": bool(passes_base and passes_neutral),
        "text_minus_context_pp": 100 * (text["dice"] - base["dice"]),
        "text_minus_neutral_pp": 100 * (text["dice"] - neutral["dice"]),
        "thresholds_pp": {"vs_context": .3, "vs_neutral": .1},
        "union_must_not_decrease_for_either_comparison": True,
        "confirmatory_efficacy_claim": False}


def prepare(config, banks, output):
    cfg = load_config(config)
    if (cfg.method != "mean_teacher" or cfg.slice_context != "adaptive"
            or cfg.internal_text != "none" or cfg.prompt_source != "none"
            or cfg.region_weight or cfg.pseudo_dice_weight):
        raise ValueError("Use the unmodified context MT control config")
    metadata = {arm: load_text_bank(banks / f"{arm}.pt", arm)[1] for arm in ("text", "neutral")}
    if metadata["text"]["provenance"] != metadata["neutral"]["provenance"]:
        raise ValueError("Both banks must use the identical frozen text encoder")
    output.mkdir(parents=True, exist_ok=False)
    (output / "banks").mkdir()
    (output / "configs").mkdir()
    for arm in ("text", "neutral"):
        shutil.copy2(banks / f"{arm}.pt", output / "banks" / f"{arm}.pt")
    for arm in ARMS:
        current = replace(cfg, output_dir=str(output / "runs" / arm),
            internal_text="none" if arm == "context" else arm,
            text_bank="" if arm == "context" else str(output / "banks" / f"{arm}.pt")).validate()
        (output / "configs" / f"{arm}.yaml").write_text(yaml.safe_dump(current.to_dict(), sort_keys=False))
    code = output / "code"
    for folder in ("freqssl", "experiments", "tests"):
        shutil.copytree(ROOT / folder, code / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "spider_volume.py"):
        shutil.copy2(ROOT / name, code / name)
    sources = {str(p.relative_to(output)): file_sha256(p) for parent in (code, output / "configs", output / "banks")
               for p in sorted(parent.rglob("*")) if p.is_file()}
    write_json(output / "frozen_sources.json", sources)
    write_json(output / "protocol.json", {
        "arms": list(ARMS), "steps_per_arm": cfg.max_steps, "seed": cfg.seed,
        "fusion": "CRF4 -> single-head text attention, hidden64 -> D3",
        "prompt_available": "fixed at training and inference",
        "loss": "unchanged supervised CE/Dice and MT confidence-masked CE",
        "selection": "best EMA teacher, full native valid, earlier checkpoint on ties",
        "decision": "text-context >=0.3pp; text-neutral >=0.1pp; union nondecreasing for both",
        "test_evaluated": False, "automatic_confirmation": False,
        "text_bank_provenance": metadata})
    write_json(output / "status.json", {"status": "prepared", "training_started": False})


def summarize(output):
    from experiments.run_slice_context import compact_run, selected_validation, paired_cases
    arms, validations = {}, {}
    for arm in ARMS:
        directory = output / "runs" / arm
        arms[arm] = compact_run(directory)
        _, validations[arm] = selected_validation(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        arms[arm]["trainable_parameters"] = manifest["trainable_parameters"]
        payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if arm != "context":
            arms[arm]["teacher_text_gate"] = float(payload["model"]["teachers.0.text_adapter.gate"].tanh())
    decision = screening_decision(*(arms[arm]["overall"] for arm in ARMS))
    result = {"status": "completed", "arms": arms, "decision": decision,
        "paired_text_vs_context": paired_cases(validations["text"], validations["context"]),
        "paired_text_vs_neutral": paired_cases(validations["text"], validations["neutral"]),
        "test_evaluated": False, "multiple_seeds": False,
        "inference_latency_benchmarked": False}
    write_json(output / "results.json", result)
    lines = ["# 内部组织文本融合：三组筛查", "", "单种子完整 valid 筛查，不是独立确认。", "",
        "| Arm | Best step | Dice % | Union-present % | Parameters |",
        "|---|---:|---:|---:|---:|"]
    for arm, values in arms.items():
        lines.append(f"| {arm} | {values['best_step']} | {100*values['overall']['dice']:.4f} | "
                     f"{100*values['overall']['dice_union_present']:.4f} | {values['trainable_parameters']} |")
    lines += ["", f"预声明筛查通过：{decision['passed_single_seed_screen']}。",
        "未评估 test；训练耗时来自共享机器，不代表隔离推理性能。",
        "固定向量表经过可学习投影；优势若存在仍需多种子与更严格向量几何对照。"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", {"status": "completed", "decision": decision})


def execute(output, device):
    if ROOT != output / "code":
        raise RuntimeError("Formal runs must execute their frozen code")
    expected = json.loads((output / "frozen_sources.json").read_text())
    if any(file_sha256(output / path) != digest for path, digest in expected.items()):
        raise ValueError("Frozen source, config or bank changed")
    # One supervisor, including across explicit resume commands.
    with (output / "screen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in ARMS:
            cfg = load_config(output / "configs" / f"{arm}.yaml")
            directory = Path(cfg.output_dir)
            result_path = directory / "result.json"
            if result_path.exists() and json.loads(result_path.read_text())["status"] == "completed":
                continue
            checkpoints = [p for p in (directory / "last.pt", directory / "best.pt") if p.exists()]
            checkpoint = max(checkpoints, key=lambda p: torch.load(p, map_location="cpu", weights_only=True)["step"]) if checkpoints else None
            write_json(output / "status.json", {"status": "running", "arm": arm, "pid": os.getpid()})
            try:
                run_experiment(cfg, device, resume=checkpoint)
            except BaseException as error:
                write_json(output / "status.json", {"status": "failed", "arm": arm, "error": repr(error)})
                raise
        summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/internal_text_context.yaml")
    parser.add_argument("--banks", type=Path, default=ROOT / "outputs/internal_text_banks_20261106")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.execute_frozen:
        execute(output, args.device)
        return
    if not args.prepare_only and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --prepare-only to prepare, not CPU fallback training")
    if not args.resume:
        prepare(args.config, args.banks, output)
    elif not (output / "frozen_sources.json").exists():
        raise ValueError("No prepared screen at this output")
    if args.prepare_only:
        print(f"Prepared; training not started: {output}", flush=True)
        return
    subprocess.run([sys.executable, str(output / "code/experiments/run_internal_text.py"),
        "--output", str(output), "--device", args.device, "--execute-frozen"], check=True,
        cwd=output / "code")


if __name__ == "__main__":
    main()
