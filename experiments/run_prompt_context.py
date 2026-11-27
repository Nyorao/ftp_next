"""Frozen three-arm screening with offline teachers, durable logs and fixed gates."""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from freqssl.config import load_config
from freqssl.engine import write_json
from freqssl.prompt_supervision import file_sha256
from experiments.run_slice_context import compact_run, selected_validation, paired_cases, benchmark_checkpoint


def screening_decision(base, text, neutral):
    improvement = text["dice"] - base["dice"] >= .003 and text["dice_union_present"] >= base["dice_union_present"]
    semantics = text["dice"] - neutral["dice"] >= .001 and text["dice_union_present"] >= neutral["dice_union_present"]
    both = (text["dice"] > base["dice"] and neutral["dice"] > base["dice"]
            and text["dice_union_present"] >= base["dice_union_present"]
            and neutral["dice_union_present"] >= base["dice_union_present"])
    return {"passed_screen": bool(improvement and semantics),
        "improves_context_control": bool(improvement), "anatomical_text_advantage": bool(semantics),
        "interpretation": ("text_guidance_passed_single_seed_screen" if improvement and semantics else
            "extra_pretrained_teacher_may_help_text_semantics_unproven" if both and not semantics else "screen_failed"),
        "thresholds_pp": {"vs_context": .3, "vs_neutral": .1}, "union_must_not_decrease": True,
        "confirmatory_efficacy_claim": False}


def summarize(output, device):
    arms = {}
    valid = {}
    for name in ("context", "text", "neutral"):
        directory = output / "students" / name
        arms[name] = compact_run(directory)
        _, valid[name] = selected_validation(directory)
        rows = valid[name]["cases"]
        anatomy = {}
        for tissue in ("vertebra", "disc", "canal"):
            anatomy[tissue] = float(np.mean([r["anatomy"]["tissue_dice"][tissue] for r in rows
                if r["anatomy"]["tissue_dice"][tissue] is not None]))
        denominator = sum(r["anatomy"]["numbered_gt_pixels"] for r in rows)
        anatomy["same_tissue_wrong_level_fraction"] = sum(r["anatomy"]["same_tissue_wrong_level_pixels"] for r in rows) / denominator
        arms[name]["anatomy"] = anatomy
        arms[name]["absent_class_false_predictions"] = [{"case": row["case"], "class_id": c["class_id"],
            "pred_voxels": c["pred_voxels"]} for row in rows for c in row["classes"] if c["false_positive_absent_class"]]
        values = []
        for line in (directory / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") == "train" and record["step"] > 1800:
                values.append(record)
        arms[name]["supervision_after_ramp"] = {key: float(np.mean([r[key] for r in values if key in r]))
            for key in ("coverage_s1", "fine_coverage_s1", "tissue_coverage_s1", "prompt_reliable_coverage_s1", "prompt_conflict_coverage_s1")
            if any(key in r for r in values)}
        benchmark_path = directory / "volume_benchmark.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            arms[name]["inference"] = {k: v for k, v in benchmark.items() if k != "metrics"}
        else:
            arms[name]["inference"] = benchmark_checkpoint(directory / "best.pt", device, benchmark_path)
    decision = screening_decision(*(arms[n]["overall"] for n in ("context", "text", "neutral")))
    result = {"status": "completed", "arms": arms, "decision": decision,
        "paired_text_vs_context": paired_cases(valid["text"], valid["context"]),
        "paired_text_vs_neutral": paired_cases(valid["text"], valid["neutral"]),
        "teacher_preparation": {source: json.loads((output / "teachers" / source / "result.json").read_text()) for source in ("text", "neutral")},
        "cache_generation": {source: {k: v for k, v in json.loads((output / "caches" / source / "manifest.json").read_text()).items()
            if k in ("seconds", "peak_allocated_bytes", "coverage", "accepted_pixels_per_tissue", "predicted_pixels_per_tissue", "teacher_step")}
            for source in ("text", "neutral")},
        "test_evaluated": False, "multi_seed_confirmation_performed": False}
    write_json(output / "results.json", result)
    lines = ["# 邻片融合 + 文本组织提示：首轮三组筛查", "", "仅使用完整 valid；单种子结果不构成确认性结论。", "",
        "| 组别 | 最佳步 | 主 Dice% | union-present Dice% |", "|---|---:|---:|---:|"]
    for name, arm in arms.items():
        lines.append(f"| {name} | {arm['best_step']} | {arm['overall']['dice']*100:.4f} | {arm['overall']['dice_union_present']*100:.4f} |")
    lines += ["", f"预声明筛查结果：**{decision['interpretation']}**。", "",
        f"文本组相对邻片对照：{100*(arms['text']['overall']['dice']-arms['context']['overall']['dice']):+.4f}pp；"
        f"相对中性提示：{100*(arms['text']['overall']['dice']-arms['neutral']['overall']['dice']):+.4f}pp。", "",
        "完整逐病例差值、组织 Dice、编号混淆、缺失类别误检、监督覆盖和分阶段耗时见 results.json。",
        "教师训练和学生训练使用不同环境；两种教师仅提示内容不同，使用相同基础权重与适配器初始化。",
        "耗时来自共享 GPU，不作为隔离性能基准。未访问 test，未自动延长预算。"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", {"status": "completed", "decision": decision, "test_evaluated": False})


class Process:
    def __init__(self, command, log_path, gpu, stage, root):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONUNBUFFERED="1", MPLCONFIGDIR="/tmp/ftp_next_matplotlib")
        self.handle = log_path.open("a", buffering=1)
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.handle, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
        self.stage, self.root = stage, root
        write_json(root / f"process_{stage}.json", {"pid": self.process.pid, "stage": stage,
            "command": command, "gpu": str(gpu), "started_unix": time.time(), "log": str(log_path)})

    def wait(self):
        code = self.process.wait()
        self.handle.close()
        write_json(self.root / f"exit_{self.stage}.json", {"exit_code": code, "finished_unix": time.time()})
        if code:
            raise RuntimeError(f"Stage {self.stage} failed with exit code {code}; inspect its log")


def done(directory, filename="result.json"):
    path = directory / filename
    return path.exists() and json.loads(path.read_text()).get("status") == "completed"


def resume_checkpoint(directory):
    candidates = [p for p in (directory / "last.pt", directory / "best.pt") if p.exists()]
    return max(candidates, key=lambda p: torch.load(p, map_location="cpu", weights_only=True)["step"]) if candidates else None


def run_pipeline(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if ROOT.resolve() != (output / "code").resolve():
        raise RuntimeError("Run the formal pipeline from its frozen output/code directory")
    hashes = json.loads((output / "frozen_sources.json").read_text())
    if any(file_sha256(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError("Frozen project source changed")
    # A supervisor may have exited while its detached child continued training.
    for receipt in output.glob("process_*.json"):
        previous = json.loads(receipt.read_text())
        command_line = Path(f"/proc/{previous['pid']}/cmdline")
        if command_line.exists() and str(output).encode() in command_line.read_bytes():
            raise RuntimeError(f"Existing stage is still running: {previous['stage']} PID {previous['pid']}")
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    config = load_config(args.config)
    pending = None

    def student(source):
        directory = output / "students" / source
        if done(directory):
            return None
        cfg = replace(config, output_dir=str(directory), slice_context="adaptive",
            prompt_source="none" if source == "context" else source,
            prompt_cache="" if source == "context" else str(output / "caches" / source))
        config_path = output / f"student_{source}.yaml"
        serialized = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
        if config_path.exists() and config_path.read_text() != serialized:
            raise ValueError("Saved student protocol changed")
        config_path.write_text(serialized)
        command = [args.student_python, "train.py", "--config", str(config_path), "--device", "cuda:0"]
        checkpoint = resume_checkpoint(directory)
        if checkpoint:
            command += ["--resume", str(checkpoint)]
        return Process(command, logs / f"student_{source}.log", args.student_gpu, f"student_{source}", output)

    try:
        pending = student("context")
        for source in ("text", "neutral"):
            write_json(output / "status.json", {"status": "running", "stage": f"teacher_{source}", "pid": os.getpid(), "test_evaluated": False})
            directory = output / "teachers" / source
            if not done(directory):
                command = [args.teacher_python, "-m", "prompt_teacher.run", "fit", "--config", str(args.config.resolve()),
                           "--source", source, "--output", str(directory), "--device", "cuda:0"]
                checkpoint = resume_checkpoint(directory)
                if checkpoint:
                    command += ["--resume", str(checkpoint)]
                Process(command, logs / f"teacher_{source}.log", args.teacher_gpu, f"teacher_{source}", output).wait()
            cache = output / "caches" / source
            if source == "neutral":
                text_provenance = json.loads((output / "teachers/text/provenance.json").read_text())
                neutral_provenance = json.loads((directory / "provenance.json").read_text())
                for key in ("foundation_sha256", "initial_frozen_state_sha256", "initial_adaptation_sha256",
                            "labeled_annotation_sha256", "labeled_image_sha256"):
                    if text_provenance[key] != neutral_provenance[key]:
                        raise ValueError(f"Teacher pairing differs: {key}")
            if not done(cache, "manifest.json"):
                write_json(output / "status.json", {"status": "running", "stage": f"cache_{source}", "pid": os.getpid()})
                if cache.exists() and any(cache.iterdir()):
                    recovery = output / "recovery"
                    recovery.mkdir(exist_ok=True)
                    cache.rename(recovery / f"cache_{source}_interrupted_{time.time_ns()}")
                command = [args.teacher_python, "-m", "prompt_teacher.run", "cache", "--config", str(args.config.resolve()),
                           "--source", source, "--output", str(cache), "--checkpoint", str(directory / "best.pt"), "--device", "cuda:0"]
                Process(command, logs / f"cache_{source}.log", args.teacher_gpu, f"cache_{source}", output).wait()
            if pending:
                pending.wait()
            pending = student(source)
        if pending:
            write_json(output / "status.json", {"status": "running", "stage": "student_neutral", "pid": os.getpid()})
            pending.wait()
        write_json(output / "status.json", {"status": "running", "stage": "summarize", "pid": os.getpid()})
        command = [args.student_python, "experiments/run_prompt_context.py", "--output", str(output), "--summarize"]
        Process(command, logs / "summary.log", args.student_gpu, "summary", output).wait()
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error), "pid": os.getpid(),
            "other_student_pid": pending.process.pid if pending and pending.process.poll() is None else None,
            "test_evaluated": False})
        raise


def freeze(output, config):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "code"
    if destination.exists():
        raise FileExistsError(destination)
    for folder in ("freqssl", "prompt_teacher", "experiments", "configs", "tests"):
        shutil.copytree(ROOT / folder, destination / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "spider_volume.py"):
        shutil.copy2(ROOT / name, destination / name)
    (destination / "docs").mkdir()
    shutil.copy2(ROOT / "docs/prompt_context_20261103.md", destination / "docs/prompt_context_20261103.md")
    hashes = {str(p.relative_to(destination)): file_sha256(p) for p in destination.rglob("*") if p.is_file()}
    write_json(output / "frozen_sources.json", hashes)
    write_json(output / "protocol.json", {"config": load_config(config).to_dict(), "arms": ["context", "text", "neutral"],
        "teacher_steps": 6000, "teacher_batch": 4, "seed": 42,
        "decision": "text-context >=0.3pp; text-neutral >=0.1pp; union nondecreasing for both",
        "test_evaluated": False, "automatic_confirmation": False})
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/prompt_context.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-python", default="/home/zhp/.conda/envs/medclipseg/bin/python")
    parser.add_argument("--student-python", default="/home/zhp/.conda/envs/mamba/bin/python")
    parser.add_argument("--teacher-gpu", default="0")
    parser.add_argument("--student-gpu", default="1")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        print(freeze(args.output.resolve(), args.config), flush=True)
    elif args.summarize:
        summarize(args.output.resolve(), torch.device("cuda:0"))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_pipeline(args)


if __name__ == "__main__":
    main()
