"""Frozen, gated CFF-inspired screening and optional paired confirmation.

No test images/labels are opened during development. Output directories cannot
be overwritten; failed runs stop instead of silently retrying.
"""

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from freqssl.config import ExperimentConfig, load_config
from freqssl.data import build_training_data
from freqssl.engine import load_model, make_optimizer, run_experiment, seed_everything, state_hash, train_step, write_json
from freqssl.evaluation import evaluate_volumes, prepare_validation
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_reference(reference, cfg, device):
    """Compare frozen 2026-10-21 code with today's disabled path, not itself."""
    saved = read_json(reference / "config.json")
    prior = ExperimentConfig(**saved)
    differences = [key for key, value in prior.to_dict().items()
                   if key != "output_dir" and value != cfg.to_dict()[key]]
    if differences:
        raise ValueError(f"Unmatched baseline protocol: {differences}")
    result = read_json(reference / "result.json")
    if result["status"] != "completed" or result["step"] != cfg.max_steps:
        raise ValueError("Baseline has not completed the matched budget")
    data, sampler, split = build_training_data(cfg)
    if split != read_json(reference / "split.json"):
        raise ValueError("Baseline split mismatch")
    recorded = read_json(reference / "source_hashes.json")
    for relative, expected in recorded.items():
        if sha256(reference / "sources" / relative) != expected:
            raise ValueError(f"Baseline frozen source changed: {relative}")
    package = "_slice_context_reference"
    spec = importlib.util.spec_from_file_location(package, reference / "sources/freqssl/__init__.py",
                      submodule_search_locations=[str(reference / "sources/freqssl")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    old_engine = importlib.import_module(package + ".engine")
    old_data = importlib.import_module(package + ".data")
    old_config = importlib.import_module(package + ".config").ExperimentConfig(**saved)
    old_dataset, _, old_split = old_data.build_training_data(old_config)
    if old_split != split:
        raise AssertionError("Frozen data protocol mismatch")
    seed_everything(cfg)
    new = StudentTeacherSystem(cfg).to(device)
    old = old_engine.StudentTeacherSystem(old_config).to(device)
    initial = state_hash(new.students[0])
    if initial != read_json(reference / "manifest.json")["initial_student1_hash"]:
        raise AssertionError("Baseline initialization mismatch")
    new_opt, old_opt = make_optimizer(new, cfg), old_engine.make_optimizer(old, old_config)
    indices = next(iter(sampler))
    samples = [data[index] for index in indices]
    for index, sample in zip(indices, samples):
        other = old_dataset[index]
        for key in ("image", "label"):
            if not torch.equal(sample[key], other[key]):
                raise AssertionError(f"Center data differs from frozen baseline: {key}")
    batch = {"image": torch.stack([s["image"] for s in samples]),
             "label": torch.stack([s["label"] for s in samples]),
             "is_labeled": torch.tensor([s["is_labeled"] for s in samples])}
    # Include a fully active pseudo-label step as well as the warmup path.
    for step in (0, cfg.warmup_steps + 1, cfg.warmup_steps + cfg.ramp_steps + 1):
        train_step(new, new_opt, batch, cfg, device, step)
        old_engine.train_step(old, old_opt, batch, old_config, device, step)
        for index in (0,):
            if (state_hash(new.students[index]) != state_hash(old.students[index]) or
                    state_hash(new.teachers[index]) != state_hash(old.teachers[index])):
                raise AssertionError("Disabled context does not exactly reproduce frozen baseline updates")
    artifacts = ("best.pt", "last.pt", "result.json", "config.json", "split.json", "manifest.json", "source_hashes.json")
    return {"source": str(reference), "protocol_and_split_equal": True,
            "center_images_labels_equal": True, "three_updates_student_teacher_exact": True,
            "initial_backbone_sha256": initial,
            "hashes": {name: sha256(reference / name) for name in artifacts}}


def selected_validation(directory):
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    step = checkpoint["step"]
    metrics = read_json(directory / f"valid_step_{step}.json")["teacher1"]
    return step, metrics


def paired_cases(candidate, baseline):
    def case_score(case):
        values = [row["dice"] for row in case["classes"] if row["class_id"] and row["dice"] is not None]
        return float(np.mean(values)) if values else None
    old = {case["case"]: case_score(case) for case in baseline["cases"]}
    new = {case["case"]: case_score(case) for case in candidate["cases"]}
    if old.keys() != new.keys() or any(v is None for v in (*old.values(), *new.values())):
        raise ValueError("Paired validation case mismatch")
    deltas = np.array([new[key] - old[key] for key in sorted(old)])
    rng = np.random.default_rng(20261029)
    means = deltas[rng.integers(len(deltas), size=(10000, len(deltas)))].mean(1)
    return {"delta_pp": {key: 100 * (new[key] - old[key]) for key in sorted(old)},
            "mean_delta_pp": float(deltas.mean() * 100),
            "case_bootstrap_95_percent_interval_pp": (np.quantile(means, [.025, .975]) * 100).tolist(),
            "interval_scope": "validation-case variability, not seed variability or independent confirmation",
            "improved_cases": int((deltas > 0).sum()), "worse_cases": int((deltas < 0).sum())}


def choose_module(baseline, mean, adaptive):
    def passes(value):
        return (value["dice"] - baseline["dice"] >= .003 and
                value["dice_union_present"] >= baseline["dice_union_present"])
    if passes(adaptive) and adaptive["dice"] - mean["dice"] >= .001:
        return "adaptive"
    if passes(mean):
        return "mean"
    return None


def compact_run(directory):
    result = read_json(directory / "result.json")
    if result["status"] != "completed":
        raise ValueError(f"Cannot summarize an incomplete run: {directory}")
    step, valid = selected_validation(directory)
    if abs(result["best_valid_dice"] - valid["overall"]["dice"]) > 1e-12:
        raise ValueError("Selected validation does not match recorded best score")
    gammas = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)["model"]
    config = read_json(directory / "config.json")
    active = config["warmup_steps"] + config["ramp_steps"]
    coverage = []
    with (directory / "metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "train" and row["step"] > active and "coverage_s1" in row:
                coverage.append(row["coverage_s1"])
    return {"directory": str(directory), "best_step": step,
            "checkpoint_sha256": sha256(directory / "best.pt"),
            "overall": valid["overall"],
            "gamma_teacher": {key: float(value) for key, value in gammas.items()
                              if key.startswith("teachers.0.context_fusion") and key.endswith("gamma")},
            "pseudo_coverage_after_ramp": {"steps": len(coverage),
                "mean": float(np.mean(coverage)) if coverage else None,
                "last_500_mean": float(np.mean(coverage[-500:])) if coverage else None},
            **{key: result.get(key) for key in ("status", "step", "seconds_this_invocation", "peak_allocated_bytes", "peak_reserved_bytes")}}


class _SelectedTeacher(torch.nn.Module):
    """The evaluation API with exactly one backbone loaded on the device."""
    def __init__(self, selected):
        super().__init__()
        self.selected = selected
        self.teachers = (selected.backbone,)
        self.students = (selected.backbone,)

    def predict_proba(self, images, network):
        if network != "teacher1":
            raise ValueError("Benchmark selects only teacher1")
        return self.selected(images)


def benchmark_checkpoint(checkpoint, device, output):
    selected, cfg, _ = load_single_network(checkpoint, device, "teacher1")
    model = _SelectedTeacher(selected).eval()
    seed_everything(cfg)
    dataset, records = prepare_validation(cfg)
    # Equal warmup size before timing; includes both CPU preprocessing and GPU work.
    with torch.inference_mode():
        model.predict_proba(torch.stack([dataset[i]["image"] for i in range(2)]).to(device), "teacher1")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = evaluate_volumes(model, dataset, records, device, networks=("teacher1",), batch_size=cfg.eval_batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    receipt = {"seconds_all_valid_volumes": time.perf_counter() - started,
               "volumes": len(records), "includes_preprocessing_reconstruction_metrics": True,
               "cached_encoder": cfg.slice_context != "none", "network": "teacher1",
               "backbones_loaded": 1,
               "timing_limit": "one pass on a shared GPU; not an isolated latency benchmark",
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "metrics": metrics["teacher1"], "test_evaluated": False}
    write_json(output, receipt)
    return {key: value for key, value in receipt.items() if key != "metrics"}


def render_report(output, results):
    rows = ["# 邻切片融合：预设消融结果", "", "## Material Passport", "",
            "来源：冻结源码、原训练预算及原生3D验证。AI辅助实现与核查；文献适配不代表官方复现。",
            "主指标：单个EMA教师、按完整valid选模。历史test多次使用，本轮筛查不读取test。", "",
            "| 组别 | 最佳步数 | valid Dice% | union-present Dice% | 相对baseline/pp |", "|---|---:|---:|---:|---:|"]
    base = results["baseline"]["overall"]["dice"]
    for name, value in [("baseline", results["baseline"]), *results["screening"].items()]:
        metric = value["overall"]
        rows.append(f"| {name} | {value['best_step']} | {100*metric['dice']:.4f} | {100*metric['dice_union_present']:.4f} | {100*(metric['dice']-base):+.4f} |")
    selected = results.get("selected_module")
    rows += ["", f"预设选择结果：{selected or '两组均未通过，不启动确认训练'}。",
             "门槛：Dice至少+0.3pp且union-present不下降；保留adaptive还需超过mean至少0.1pp。",
             "这不是统计显著性门槛。两组均增加邻片输入曝光与编码计算；不能称等计算预算对照。",
             "", "逐病例差值与区间见 results.json；组织Dice与编号混淆见各组valid_step文件及benchmark JSON。",
             "区间只反映验证病例差异，验证集已用于开发；单种子结果不支持稳健性结论。"]
    if "confirmation" in results:
        confirmation = results["confirmation"]
        rows += ["", "## 配对多种子确认", "",
                 f"平均Dice变化：{confirmation['mean_delta_pp']:+.4f}pp；正向种子数：{confirmation['positive_seeds']}/3。",
                 f"确认门槛通过：{confirmation['passed']}。"]
        rows += ["", "| seed | baseline Dice% | 模块 Dice% | 差值/pp |", "|---|---:|---:|---:|"]
        for seed, delta in zip((42, 43, 44), confirmation["deltas_pp"]):
            first = confirmation["runs"][f"seed{seed}/none"]["overall"]["dice"]
            second = confirmation["runs"][f"seed{seed}/{selected}"]["overall"]["dice"]
            rows.append(f"| {seed} | {100*first:.4f} | {100*second:.4f} | {delta:+.4f} |")
        for mode in ("none", selected):
            values = [100*confirmation["runs"][f"seed{s}/{mode}"]["overall"]["dice"] for s in (42, 43, 44)]
            rows.append(f"\n{mode}：均值 {np.mean(values):.4f}%，种子样本标准差 {np.std(values, ddof=1):.4f}pp。")
        rows.append("仅3个种子；标准差和逐病例区间不构成独立测试集上的统计显著性证据。")
    rows += ["", "## 开销与伪标签", "",
             "| 组别 | 训练及验证耗时/s | 训练峰值显存/GiB | 完整21例valid推理/s | ramp后伪标签覆盖% |",
             "|---|---:|---:|---:|---:|"]
    def number(value, scale=1):
        return "未记录" if value is None else f"{value*scale:.3f}"
    for name, value in [("baseline", results["baseline"]), *results["screening"].items()]:
        bench = results.get("benchmarks", {}).get(name, {})
        rows.append(f"| {name} | {number(value.get('seconds_this_invocation'))} | "
                    f"{number(value.get('peak_allocated_bytes'), 1/2**30)} | "
                    f"{number(bench.get('seconds_all_valid_volumes'))} | "
                    f"{number(value.get('pseudo_coverage_after_ramp', {}).get('mean'), 100)} |")
    rows += ["", "耗时来自共享GPU，训练耗时包含完整验证；推理含影像预处理、原生重建及指标，非隔离速度基准。",
             "上下文eval固定微批次8，尾批填零的额外计算已包含在计时中；baseline历史峰值显存未记录。",
             "", "## 组织与编号诊断", "",
             "| 组别 | 椎体Dice% | 椎间盘Dice% | 椎管Dice% | 同组织错误编号/编号GT体素% |",
             "|---|---:|---:|---:|---:|"]
    for name in ("baseline", *results["screening"]):
        path = output / f"benchmark_{name}.json"
        if not path.exists():
            continue
        anatomy = [case["anatomy"] for case in read_json(path)["metrics"]["cases"]]
        tissue = {key: np.mean([case["tissue_dice"][key] for case in anatomy
                               if case["tissue_dice"][key] is not None]) for key in ("vertebra", "disc", "canal")}
        numbered = sum(case["numbered_gt_pixels"] for case in anatomy)
        wrong = sum(case["same_tissue_wrong_level_pixels"] for case in anatomy)
        rows.append(f"| {name} | {100*tissue['vertebra']:.4f} | {100*tissue['disc']:.4f} | "
                    f"{100*tissue['canal']:.4f} | {100*wrong/numbered:.4f} |")
    rows.append("\n组织Dice先逐病例计算再平均；编号项按GT编号体素加权，漏检及组织错分不计入该项。完整20×20混淆矩阵见benchmark JSON。")
    (output / "report.md").write_text("\n".join(rows) + "\n")


def run_prepared(output):
    protocol = read_json(output / "protocol.json")
    for relative, expected in protocol["frozen_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("Run must use its unchanged frozen source")
    if (output / "results.json").exists():
        raise FileExistsError("Suite already started; inspect its status rather than overwrite")
    cfg = ExperimentConfig(**protocol["config"]).validate()
    reference = Path(protocol["reference"])
    device = torch.device(protocol["device"])
    results = {"status": "running", "screening": {}, "test_evaluated": False}
    write_json(output / "results.json", results)
    try:
        results["reference_verification"] = verify_reference(reference, cfg, device)
        release()
        results["baseline"] = compact_run(reference)
        results["benchmarks"] = {"baseline": benchmark_checkpoint(reference / "best.pt", device, output / "benchmark_baseline.json")}
        _, baseline_valid = selected_validation(reference)
        measured = read_json(output / "benchmark_baseline.json")["metrics"]["overall"]
        for metric in ("dice", "dice_union_present"):
            if abs(measured[metric] - baseline_valid["overall"][metric]) > 1e-12:
                raise AssertionError("Historical baseline does not reproduce on current native validation")
        release()
        for mode in ("mean", "adaptive"):
            results["active_run"] = mode
            write_json(output / "results.json", results)
            directory = output / mode
            run_experiment(replace(cfg, output_dir=str(directory), slice_context=mode), device)
            results["screening"][mode] = compact_run(directory)
            _, valid = selected_validation(directory)
            results["screening"][mode]["paired_cases"] = paired_cases(valid, baseline_valid)
            release()
            results["benchmarks"][mode] = benchmark_checkpoint(directory / "best.pt", device, output / f"benchmark_{mode}.json")
            release()
            write_json(output / "results.json", results)
        selected = choose_module(results["baseline"]["overall"], results["screening"]["mean"]["overall"],
                                 results["screening"]["adaptive"]["overall"])
        results["selected_module"] = selected
        write_json(output / "results.json", results)
        render_report(output, results)
        if selected and protocol["confirm_on_pass"]:
            runs, deltas = {}, []
            for seed in (42, 43, 44):
                for mode in ("none", selected):
                    key = f"seed{seed}/{mode}"
                    results["active_run"] = "confirmation/" + key
                    write_json(output / "results.json", results)
                    directory = output / "confirmation" / key
                    full = replace(cfg, output_dir=str(directory), seed=seed, slice_context=mode,
                                   max_steps=30000, eval_every=1000, warmup_steps=1500, ramp_steps=3000,
                                   max_seconds=86400)
                    run_experiment(full, device)
                    runs[key] = compact_run(directory)
                    release()
                _, first = selected_validation(output / "confirmation" / f"seed{seed}/none")
                _, second = selected_validation(output / "confirmation" / f"seed{seed}/{selected}")
                pair = paired_cases(second, first)
                runs[f"seed{seed}/{selected}"]["paired_cases"] = pair
                deltas.append(pair["mean_delta_pp"])
                write_json(output / "confirmation_progress.json", {"runs": runs, "deltas_pp": deltas})
            union_deltas = [runs[f"seed{s}/{selected}"]["overall"]["dice_union_present"] -
                            runs[f"seed{s}/none"]["overall"]["dice_union_present"] for s in (42, 43, 44)]
            confirmed = bool(np.mean(deltas) >= .3 and sum(x > 0 for x in deltas) >= 2 and np.mean(union_deltas) >= 0)
            results["confirmation"] = {"runs": runs, "deltas_pp": deltas,
                  "mean_delta_pp": float(np.mean(deltas)), "positive_seeds": sum(x > 0 for x in deltas), "passed": confirmed}
            if confirmed:
                # All models/checkpoint hashes are frozen in runs before any test access.
                write_json(output / "fixed_before_test.json", results["confirmation"])
                test_results = {}
                for key, receipt in runs.items():
                    model, settings, _ = load_model(Path(receipt["directory"]) / "best.pt", device)
                    seed_everything(settings)
                    data, records = prepare_validation(settings, split="test")
                    test_results[key] = evaluate_volumes(model, data, records, device, networks=("teacher1",),
                                                        batch_size=settings.eval_batch_size)["teacher1"]
                    del model
                    release()
                write_json(output / "test_descriptive.json", test_results)
                results["test_evaluated"] = True
        results.update(status="completed", active_run=None)
        write_json(output / "results.json", results)
        render_report(output, results)
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "results.json", results)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/freqssl_v3_screening_20261021/comparison/seed42/mean_teacher")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/slice_context.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-on-pass", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.run_prepared:
        return run_prepared(output)
    cfg = replace(load_config(args.config), slice_context="none", method="mean_teacher")
    output.mkdir(parents=True, exist_ok=False)
    paths = [*sorted((ROOT / "freqssl").glob("*.py")), *sorted((ROOT / "experiments").glob("*.py")),
             ROOT / "spider_volume.py", ROOT / "train.py", ROOT / "evaluate.py",
             ROOT / "docs/slice_context_20261030.md"]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256(destination)
    shutil.copy2(args.config, output / "config.yaml")
    write_json(output / "protocol.json", {"config": cfg.to_dict(), "device": args.device,
        "reference": str(args.reference.resolve()), "confirm_on_pass": args.confirm_on_pass,
        "frozen_sha256": hashes, "arms": ["mean", "adaptive"],
        "screening_rules": {"min_dice_gain": .003, "min_adaptive_over_mean": .001, "union_present_non_decreasing": True},
        "confirmation_rules": {"seeds": [42, 43, 44], "steps": 30000, "min_mean_gain": .003,
                               "min_positive_seeds": 2, "mean_union_present_non_decreasing": True},
        "data_policy": "same central samples/labels; neighbor images increase exposure and compute",
        "test_policy": "only after fixed multi-seed confirmation passes; historical test is descriptive"})
    command = [sys.executable, str(output / "code/experiments/run_slice_context.py"),
               "--output-dir", str(output), "--run-prepared"]
    write_json(output / "command.json", {"argv": command})
    if args.prepare_only:
        print(json.dumps(command))
    else:
        subprocess.run(command, check=True, cwd=output / "code")


if __name__ == "__main__":
    main()
