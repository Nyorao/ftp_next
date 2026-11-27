"""Frozen eight-arm, 6000-step encoder token/CLS placement screen."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from freqssl.config import load_config
from freqssl.engine import run_experiment, write_json
from freqssl.hierarchical_text import load_hierarchical_bank
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import file_sha256
from experiments.run_hierarchical_text import initialize_device, class_name
from experiments.run_slice_context import selected_validation
from spider_volume import aggregate_cases

ARMS = ("A0", "A1", "G2", "T2", "G3", "T3", "G4", "T4")
LABELS = {"A0": "视觉基线", "A1": "仅共同对齐", **{
    f"{kind}{stage}": f"{'Token' if kind == 'T' else 'CLS'} → E{stage}"
    for kind in ("G", "T") for stage in (2, 3, 4)}}


def read(path):
    return json.loads(Path(path).read_text())


def arm_config(base, arm, bank, output):
    return replace(base, output_dir=str(output), internal_text="none" if arm == "A0" else "text",
        text_bank="" if arm == "A0" else str(bank), text_fusion_layout="e4" if arm == "A0" else "encoder",
        text_layer_mode="last", text_feature_kind="none" if arm in ("A0", "A1") else
            "tokens" if arm.startswith("T") else "global_cls",
        encoder_text_stage=None if arm in ("A0", "A1") else int(arm[1]),
        text_alignment_weight=0. if arm == "A0" else .1, text_alignment_balance="class").validate()


def prepare(config, banks, output, *, arms=ARMS, make_config=arm_config, protocol_extra=None):
    base = load_config(config)
    if (base.method != "mean_teacher" or base.slice_context != "adaptive" or base.internal_text != "none"
            or base.prompt_source != "none" or base.region_weight or base.pseudo_dice_weight
            or base.text_alignment_weight):
        raise ValueError("Start from the original adaptive-context MT protocol")
    torch.set_num_threads(base.cpu_threads)
    banks, output = Path(banks).resolve(), Path(output).resolve()
    _, provenance = load_hierarchical_bank(banks / "text.pt", "text")
    output.mkdir(parents=True, exist_ok=False)
    (output / "banks").mkdir()
    (output / "configs").mkdir()
    shutil.copy2(banks / "text.pt", output / "banks/text.pt")
    visual_reference = text_reference = None
    parameters = {}
    probe = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    for arm in arms:
        cfg = make_config(base, arm, output / "banks/text.pt", output / "runs" / arm)
        (output / "configs" / f"{arm}.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        before = torch.get_rng_state().clone()
        net = StudentTeacherSystem(cfg).students[0]
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError("Model construction changed caller RNG")
        named = {k:v.detach().clone() for k,v in net.named_parameters()}
        visual = {k:v for k,v in named.items() if not k.startswith("encoder_text.")}
        if visual_reference is None:
            visual_reference = visual
            with torch.no_grad():
                reference_logits = net(probe)
        if visual.keys() != visual_reference.keys() or not all(torch.equal(v, visual_reference[k]) for k,v in visual.items()):
            raise AssertionError("Paired visual initialization mismatch")
        if cfg.internal_text != "none":
            if text_reference is None:
                text_reference = named
            if named.keys() != text_reference.keys() or not all(torch.equal(v, text_reference[k]) for k,v in named.items()):
                raise AssertionError("Text-arm initialization mismatch")
        with torch.no_grad():
            torch.testing.assert_close(net(probe), reference_logits, atol=0, rtol=0)
        parameters[arm] = dict(total=sum(p.numel() for p in net.parameters()),
            participating_training=sum(p.numel() for p in net.parameters() if p.requires_grad))
    code = output / "code"
    for folder in ("freqssl", "experiments", "tests"):
        shutil.copytree(ROOT / folder, code / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "spider_volume.py"):
        shutil.copy2(ROOT / name, code / name)
    protocol = dict(arms=list(arms), steps_per_arm=base.max_steps,
        total_steps=base.max_steps * len(arms), seed=base.seed, eval_every=base.eval_every,
        initialization_and_zero_gate_checked=True, parameters=parameters, text_bank=provenance,
        input="three ordered sagittal slices; shared encoder; original adaptive context fusion",
        placement="after encoder block, before skip storage and next downsampling; E2/E3/E4",
        representation="layer12 valid word tokens vs per-class CLS, both 768D; fixed four tissue prompts",
        adapter_dim=base.text_adapter_dim, alignment="all E2/E3/E4 pre-injection centers; frozen 512D anchors",
        alignment_temperature=.07, alignment_weight=.1, alignment_balance="class",
        alignment_stage_reduction="mean, not sum", global_image_distillation=False,
        selection="best EMA teacher1 case-equal GT-present foreground Mean Dice; earlier step on ties",
        fusion_threshold_vs_A0_pp=.3, fusion_threshold_vs_A1_pp=.1, preference_threshold_pp=.1,
        union_non_decreasing_required=True, checkpoint_rank_reversals="flag unstable",
        diagnostics="labeled center forward values before update; gates logged after update",
        test_evaluated=False, multiple_seeds=False, automatic_protocol_changes=False)
    protocol.update(protocol_extra or {})
    write_json(output / "protocol.json", protocol)
    paths = [output / "protocol.json", *[p for parent in (code, output / "configs", output / "banks")
              for p in sorted(parent.rglob("*")) if p.is_file()]]
    write_json(output / "frozen_sources.json", {str(p.relative_to(output)):file_sha256(p) for p in paths})
    write_json(output / "status.json", dict(status="prepared", training_started=False, arms=list(arms)))


def compare(arms, candidate, control, threshold):
    a, b = arms[candidate], arms[control]
    delta = 100 * (a["best"]["dice"] - b["best"]["dice"])
    final = 100 * (a["last"]["dice"] - b["last"]["dice"])
    union_delta = 100 * (a["best"]["dice_union_present"] - b["best"]["dice_union_present"])
    return dict(candidate=candidate, control=control, delta_dice_pp=delta,
        final_delta_dice_pp=final, delta_union_pp=union_delta, threshold_pp=threshold,
        final_delta_union_pp=100 * (a["last"]["dice_union_present"] - b["last"]["dice_union_present"]),
        passed=delta >= threshold and union_delta >= 0, rank_reversal=delta * final < 0)


def decision(arms):
    fusion, placement, granularity = {}, {}, {}
    for arm in ARMS[2:]:
        rows = [compare(arms, arm, c, t) for c,t in (("A0", .3), ("A1", .1))]
        fusion[arm] = dict(comparisons=rows, passed=all(r["passed"] for r in rows),
                           unstable=any(r["rank_reversal"] for r in rows))
    for kind in ("G", "T"):
        choices = [f"{kind}{s}" for s in (2, 3, 4)]
        candidate = max(choices, key=lambda a: (arms[a]["best"]["dice"], int(a[1])))
        rows = [compare(arms, candidate, control, .1) for control in choices if control != candidate]
        unstable = any(r["rank_reversal"] for r in rows)
        placement[kind] = dict(numerical_leader=candidate, comparisons=rows, unstable=unstable,
            preferred=candidate if all(r["passed"] for r in rows) and not unstable else None)
    for stage in (2, 3, 4):
        rows = [compare(arms, f"{kind}{stage}", f"{other}{stage}", .1)
                for kind,other in (("T", "G"), ("G", "T"))]
        preferred = next((r["candidate"] for r in rows if r["passed"] and not r["rank_reversal"]), None)
        granularity[f"E{stage}"] = dict(comparisons=rows, preferred=preferred,
                                       unstable=any(r["rank_reversal"] for r in rows))
    return dict(fusion=fusion, placement=placement, granularity=granularity,
                confirmatory_efficacy_claim=False)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return None if value is None else 100 * value


def summarize(output, *, arm_order=ARMS, labels=LABELS, decide=decision,
              title="编码器 Token／CLS 位置筛查", render_decision=None):
    output = Path(output)
    protocol = read(output / "protocol.json")
    arms, scores = {}, {}
    tables = {k:[] for k in ("overall", "per_class", "per_case", "merged_tissue", "validation_history",
                             "training_diagnostics", "training_window_summary", "paired_case_deltas")}
    split_reference = case_reference = None
    for arm in arm_order:
        directory = output / "runs" / arm
        if not (directory / "result.json").exists() or read(directory / "result.json")["status"] != "completed":
            continue
        result = read(directory / "result.json")
        step, best = selected_validation(directory)
        last = read(directory / f"valid_step_{result['step']}.json")["teacher1"]
        split, cases = read(directory / "split.json"), [c["case"] for c in best["cases"]]
        if split_reference is None:
            split_reference, case_reference = split, cases
        if split != split_reference or cases != case_reference:
            raise ValueError("Training/validation case mismatch across arms")
        arms[arm] = dict(best_step=step, last_step=result["step"], best=best["overall"], last=last["overall"],
            parameters=protocol["parameters"][arm], seconds=result["seconds_this_invocation"],
            peak_allocated_bytes=result["peak_allocated_bytes"], peak_reserved_bytes=result["peak_reserved_bytes"])
        scores[arm] = dict(best=best, last=last)
        for checkpoint, score, checkpoint_step in (("best", best, step), ("last", last, result["step"])):
            if aggregate_cases(score["cases"]) != {k:score[k] for k in ("overall", "per_class", "per_case")}:
                raise ValueError("Saved metrics do not match confusion counts")
            common = dict(arm=arm, checkpoint=checkpoint, step=checkpoint_step)
            tables["overall"].append(dict(**common, mean_dice_pct=percent(score["overall"]["dice"]),
                mean_iou_pct=percent(score["overall"]["iou"]), union_dice_pct=percent(score["overall"]["dice_union_present"]),
                valid_cases=score["overall"]["cases"], **protocol["parameters"][arm],
                seconds=result["seconds_this_invocation"], peak_allocated_bytes=result["peak_allocated_bytes"]))
            for row in score["per_class"]:
                tables["per_class"].append(dict(**common, class_id=row["class_id"], label=class_name(row["class_id"]),
                    dice_pct=percent(row["dice"]), iou_pct=percent(row["iou"]), union_dice_pct=percent(row["dice_union_present"]),
                    gt_present_cases=row["gt_present_cases"], missed_gt_cases=row["missed_gt_cases"],
                    fp_absent_gt_cases=row["fp_absent_gt_cases"]))
            background = []
            for case in score["cases"]:
                cm = np.asarray(case["anatomy"]["confusion_gt_rows_pred_columns"], dtype=np.int64)
                gt, pred, tp = int(cm[0].sum()), int(cm[:,0].sum()), int(cm[0,0])
                if gt:
                    background.append((2 * tp / (gt + pred), tp / (gt + pred - tp)))
            tables["per_class"].append(dict(**common, class_id=0, label="背景",
                dice_pct=100 * float(np.mean([x[0] for x in background])) if background else None,
                iou_pct=100 * float(np.mean([x[1] for x in background])) if background else None,
                gt_present_cases=len(background)))
            for row in score["per_case"]:
                tables["per_case"].append(dict(**common, case=row["case"], dice_pct=percent(row["dice"]),
                    iou_pct=percent(row["iou"]), union_dice_pct=percent(row["dice_union_present"])))
            for tissue in ("vertebra", "canal", "disc"):
                values = [c["anatomy"]["tissue_dice"][tissue] for c in score["cases"]]
                values = [v for v in values if v is not None]
                tables["merged_tissue"].append(dict(**common, tissue=tissue,
                    dice_pct=100 * float(np.mean(values)) if values else None, gt_present_cases=len(values)))
        for path in sorted(directory.glob("valid_step_*.json"), key=lambda p:int(p.stem.split("_")[-1])):
            row = read(path)["teacher1"]["overall"]
            tables["validation_history"].append(dict(arm=arm, step=int(path.stem.split("_")[-1]),
                mean_dice_pct=percent(row["dice"]), mean_iou_pct=percent(row["iou"]), union_dice_pct=percent(row["dice_union_present"])))
        cfg = read(directory / "config.json")
        with (directory / "metrics.jsonl").open() as handle:
            training = [row for row in map(json.loads, handle) if row.get("kind") == "train"]
            for window, records in (("all_steps", training), ("last_500_steps", training[-500:])):
                if records:
                    entry = dict(arm=arm, scope="every_training_step", window=window,
                        samples=len(records), first_step=records[0]["step"], last_step=records[-1]["step"])
                    for key in ("loss", "text_alignment_loss", "coverage_s1", "unsupervised_weight"):
                        values = [r[key] for r in records if key in r]
                        entry[key] = float(np.mean(values)) if values else None
                    tables["training_window_summary"].append(entry)
            for row in training:
                if not (row["step"] == 1 or row["step"] % cfg["log_every"] == 0):
                    continue
                entry = dict(arm=arm, scope="sampled_at_log_interval", step=row["step"], loss=row["loss"], text_alignment_loss=row.get("text_alignment_loss"),
                    pseudo_coverage=row.get("coverage_s1"), unsupervised_weight=row["unsupervised_weight"])
                for i, stage in enumerate((2, 3, 4)):
                    entry[f"E{stage}_gate"] = row.get("text_gates", [None] * 3)[i]
                    entry[f"E{stage}_alignment_loss"] = row.get("text_alignment_loss_per_stage", [None] * 3)[i]
                    for c in range(4):
                        entry[f"E{stage}_tissue_fraction_{c}"] = row.get("text_tissue_prediction_fractions", [[None]*4]*3)[i][c]
                entry.update(row.get("encoder_text_diagnostics", {}))
                tables["training_diagnostics"].append(entry)
    # All ordered pairs support baseline, same-stage and same-representation comparisons.
    for candidate in arms:
        for control in arms:
            if candidate == control:
                continue
            for checkpoint in ("best", "last"):
                reference = {r["case"]:r for r in scores[control][checkpoint]["per_case"]}
                for row in scores[candidate][checkpoint]["per_case"]:
                    other = reference[row["case"]]
                    tables["paired_case_deltas"].append(dict(candidate=candidate, control=control,
                        checkpoint=checkpoint, case=row["case"],
                        dice_delta_pp=100 * (row["dice"] - other["dice"]),
                        iou_delta_pp=100 * (row["iou"] - other["iou"]),
                        union_delta_pp=100 * (row["dice_union_present"] - other["dice_union_present"])))
    complete = len(arms) == len(arm_order)
    result = dict(status="completed" if complete else "partial", arms=arms, test_evaluated=False,
                  multiple_seeds=False, decision=decide(arms) if complete else None)
    write_json(output / "results.json", result)
    report = output / "metrics_report"
    report.mkdir(exist_ok=True)
    for name, rows in tables.items():
        write_csv(report / f"{name}.csv", rows)
    lines = [f"# {title}", "", f"状态：{result['status']}；完成 {len(arms)}/{len(arm_order)} 组。",
        "病例等权、GT存在前景 Mean Dice；单种子，完整验证集选EMA教师，不评估测试集。", "",
        "| 组别 | 最佳步 | 最佳 Dice % | 最佳 IoU % | 最佳 Union % | 6000步 Dice % | 6000步 Union % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in arms.items():
        a, b = row["best"], row["last"]
        lines.append(f"| {arm} {labels[arm]} | {row['best_step']} | {100*a['dice']:.4f} | {100*a['iou']:.4f} | "
                     f"{100*a['dice_union_present']:.4f} | {100*b['dice']:.4f} | {100*b['dice_union_present']:.4f} |")
    if complete and render_decision is not None:
        lines += render_decision(result["decision"])
    elif complete:
        lines += ["", "## 预声明判定", ""]
        for arm, row in result["decision"]["fusion"].items():
            deltas = "; ".join(f"相对{c['control']} {c['delta_dice_pp']:+.4f}pp" for c in row["comparisons"])
            lines.append(f"- {arm}：筛查通过={row['passed']}；排序反转={row['unstable']}；{deltas}。")
        for group, row in result["decision"]["placement"].items():
            lines.append(f"- {group}位置偏好：{row['preferred'] or '尚不能确定'}；排序反转={row['unstable']}。")
        for group, row in result["decision"]["granularity"].items():
            lines.append(f"- {group}粒度偏好：{row['preferred'] or '尚不能确定'}；排序反转={row['unstable']}。")
    lines += ["", "两套检查点的逐类、背景、逐病例、合并组织、配对差值及曲线见CSV；组织编号不等于绝对解剖名称。",
        "training_window_summary.csv使用每步记录；training_diagnostics.csv为首步及日志间隔采样，两者均值不可混用。",
        "总参数与实际训练参数单列；不同通道位置不宣称等有效容量。未计算表面距离指标。",
        "0.3/0.1个百分点是工程筛查门槛；本轮不能证明统计显著性或医学语义利用。"]
    (report / "report.md").write_text("\n".join(lines) + "\n")
    return result


def execute(output, device):
    if ROOT != output / "code":
        raise RuntimeError("Execute the frozen runner")
    if any(file_sha256(output / p) != h for p,h in read(output / "frozen_sources.json").items()):
        raise ValueError("Frozen source/config/bank/protocol changed")
    started = datetime.now(timezone.utc).isoformat()
    arm = None
    with (output / "screen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for arm in ARMS:
                cfg = load_config(output / "configs" / f"{arm}.yaml")
                directory = Path(cfg.output_dir)
                if (directory / "result.json").exists() and read(directory / "result.json")["status"] == "completed":
                    continue
                checkpoint = directory / "last.pt"
                write_json(output / "status.json", dict(status="running", arm=arm, pid=os.getpid(), device=str(device), started_utc=started))
                initialize_device(device)
                run_experiment(cfg, device, resume=checkpoint if checkpoint.exists() else None)
                summarize(output)
                gc.collect()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            result = summarize(output)
            write_json(output / "status.json", dict(status="completed", decision=result["decision"]))
            write_json(output / "exit.json", dict(exit_code=0, started_utc=started,
                       finished_utc=datetime.now(timezone.utc).isoformat()))
        except BaseException as error:
            write_json(output / "status.json", dict(status="failed", arm=arm, error=repr(error)))
            write_json(output / "exit.json", dict(exit_code=1, error=repr(error), finished_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/context_efficiency_reference.yaml")
    parser.add_argument("--banks", type=Path, default=ROOT / "outputs/hierarchical_text_assets_20261108")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-frozen", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        summarize(output)
    elif args.execute_frozen:
        execute(output, args.device)
    else:
        if not args.resume:
            prepare(args.config, args.banks, output)
        if not args.prepare_only:
            subprocess.run([sys.executable, "-u", str(output / "code/experiments/run_encoder_text.py"),
                "--output", str(output), "--device", args.device, "--execute-frozen"], check=True, cwd=output / "code")
        else:
            print(f"Prepared: {output}", flush=True)


if __name__ == "__main__":
    main()
