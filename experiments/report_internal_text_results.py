"""Audit completed internal-text runs and export case/class metric tables.

Uses saved validation counts only; does not train or access test data.
Background is derived separately from stored 20x20 confusion matrices.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_name(label):
    return "背景" if label == 0 else "椎管" if label == 10 else (
        f"椎体{label}" if label < 10 else f"椎间盘{label - 10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("outputs/internal_text_screen_20261106"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.screen.resolve()
    output = args.output or root / "metrics_report_20261108"
    summary = read(root / "results.json")
    exit_info = read(root / "exit_20261107.json")
    assert summary["status"] == "completed" and exit_info["exit_code"] == 0
    assert not summary["test_evaluated"]
    frozen = read(root / "frozen_sources.json")
    assert all(sha(root / name) == value for name, value in frozen.items())
    sys.path.insert(0, str(root / "code"))
    from spider_volume import aggregate_cases
    from experiments.run_internal_text import screening_decision

    arm_names = {"context": "原模型", "neutral": "中性文本", "text": "组织文本"}
    overall, per_class, per_case, tissues, curves, audits = [], [], [], [], [], []
    reference_split, reference_cases = None, None
    score_sets = {}
    for arm in arm_names:
        directory = root / "runs" / arm
        result = read(directory / "result.json")
        cfg = read(directory / "config.json")
        split = read(directory / "split.json")
        assert result["status"] == "completed" and result["step"] == cfg["max_steps"] == 6000
        assert not result["test_evaluated"]
        if reference_split is None:
            reference_split = split
        assert split == reference_split
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        assert last["step"] == 6000 and checkpoint["config"] == cfg and last["config"] == cfg
        step = checkpoint["step"]
        assert step == summary["arms"][arm]["best_step"]
        digest = sha(directory / "best.pt")
        assert digest == summary["arms"][arm]["checkpoint_sha256"]
        path = directory / f"valid_step_{step}.json"
        scores = read(path)["teacher1"]
        score_sets[arm] = scores
        assert aggregate_cases(scores["cases"]) == {k: scores[k] for k in ("overall", "per_class", "per_case")}
        assert scores["overall"]["dice"] == checkpoint["best_score"] == result["best_valid_dice"]
        assert scores["overall"]["cases"] == 21
        cases = [c["case"] for c in scores["cases"]]
        if reference_cases is None:
            reference_cases = cases
        assert cases == reference_cases
        assert scores["metric_policy"]["background_excluded"]
        assert scores["metric_policy"]["surface_metrics_requested"] is False

        background = []
        for case in scores["cases"]:
            matrix = np.asarray(case["anatomy"]["confusion_gt_rows_pred_columns"], dtype=np.int64)
            assert matrix.shape == (20, 20)
            gt, pred = matrix.sum(1), matrix.sum(0)
            for item in case["classes"]:
                label = item["class_id"]
                assert (item["gt_voxels"], item["pred_voxels"], item["intersection_voxels"]) == (
                    int(gt[label]), int(pred[label]), int(matrix[label, label]))
            tp, g, p = int(matrix[0, 0]), int(gt[0]), int(pred[0])
            assert g > 0
            background.append((2 * tp / (g+p), tp / (g+p-tp)))
        bg_dice, bg_iou = np.mean(background, axis=0)
        background_row = dict(class_id=0, class_name="background", dice=float(bg_dice), iou=float(bg_iou),
            dice_union_present=float(bg_dice), iou_union_present=float(bg_iou),
            gt_present_cases=21, pred_present_cases=21, missed_gt_cases=0, fp_absent_gt_cases=0)
        for row in [background_row, *scores["per_class"]]:
            per_class.append(dict(arm=arm, class_id=row["class_id"], label=class_name(row["class_id"]),
                legacy_display_name=row["class_name"], included_in_primary=row["class_id"] != 0,
                dice_pct=100*row["dice"], iou_pct=100*row["iou"],
                union_dice_pct=100*row["dice_union_present"], union_iou_pct=100*row["iou_union_present"],
                gt_present_cases=row["gt_present_cases"], pred_present_cases=row["pred_present_cases"],
                missed_gt_cases=row["missed_gt_cases"], fp_absent_gt_cases=row["fp_absent_gt_cases"]))
        o = scores["overall"]
        overall.append(dict(arm=arm, completed_steps=6000, selected_step=step, valid_cases=21,
            mean_dice_pct=100*o["dice"], mean_iou_pct=100*o["iou"],
            union_dice_pct=100*o["dice_union_present"], union_iou_pct=100*o["iou_union_present"],
            class_macro_dice_pct=100*float(np.mean([x["dice"] for x in scores["per_class"]])),
            class_macro_iou_pct=100*float(np.mean([x["iou"] for x in scores["per_class"]])),
            final_step_mean_dice_pct=100*result["final_validation"]["teacher1"]["overall"]["dice"],
            fp_absent_gt_class_case_pairs=o["fp_absent_gt_class_case_pairs"],
            training_minutes=result["seconds_this_invocation"]/60,
            parameters=summary["arms"][arm]["trainable_parameters"],
            teacher_text_gate=summary["arms"][arm].get("teacher_text_gate")))
        for row in scores["per_case"]:
            per_case.append(dict(arm=arm, case=row["case"], dice_pct=100*row["dice"], iou_pct=100*row["iou"],
                union_dice_pct=100*row["dice_union_present"], union_iou_pct=100*row["iou_union_present"]))
        for tissue in ("vertebra", "canal", "disc"):
            values = [c["anatomy"]["tissue_dice"][tissue] for c in scores["cases"]]
            values = [x for x in values if x is not None]
            tissues.append(dict(arm=arm, tissue=tissue, dice_pct=100*float(np.mean(values)), valid_cases=len(values)))
        for validation in sorted(directory.glob("valid_step_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
            step_scores = read(validation)["teacher1"]["overall"]
            curves.append(dict(arm=arm, step=int(validation.stem.split("_")[-1]),
                dice_pct=100*step_scores["dice"], iou_pct=100*step_scores["iou"],
                union_dice_pct=100*step_scores["dice_union_present"]))
        audits.append(dict(arm=arm, step=step, best_checkpoint_sha256=digest,
            metric_source=str(path), metric_source_sha256=sha(path), aggregation_recomputed_exactly=True,
            confusion_counts_crosschecked=True))
        del checkpoint, last

    decision = screening_decision(*(score_sets[a]["overall"] for a in arm_names))
    assert decision == summary["decision"]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("overall", overall), ("per_class", per_class), ("per_case", per_case),
                       ("merged_tissue", tissues), ("validation_history", curves)):
        write_csv(output / f"{name}.csv", rows)
    audit = dict(status="verified", exit=exit_info, frozen_files_verified=len(frozen),
        arms=audits, decision=decision, same_training_split=True, same_validation_cases=True,
        background_metrics_derived_from_saved_confusion=True, test_evaluated=False,
        accuracy_inference_rerun=False, exporter_sha256=sha(Path(__file__)),
        label_policy="Relative IDs; legacy names retained in CSV only; no anatomical renaming assumed")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 内部文本融合训练结果：2026-11-08", "",
        f"三组均完成6000步；正常退出，完成时间 {exit_info['finished']}。",
        "相同seed42、相同训练划分、完整21例原生valid。下表均为各组按主mean Dice选出的最佳EMA教师；未评估test。", "",
        "主指标：每病例先平均GT存在的前景类别，再对病例等权平均。背景排除。",
        "union-present补充指标也计入GT缺失但被预测出的类别（该类别得0分）。",
        "类别等权macro指标先计算每类有效病例均值，再平均19类；这是补充口径，不用于改变选模或筛查规则。", "",
        "| 组别 | 最佳步 | mean Dice % | mean IoU % | Union Dice % | 19类macro Dice % | 最后6000步Dice % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {arm_names[row['arm']]} | {row['selected_step']} | {row['mean_dice_pct']:.4f} | "
            f"{row['mean_iou_pct']:.4f} | {row['union_dice_pct']:.4f} | {row['class_macro_dice_pct']:.4f} | {row['final_step_mean_dice_pct']:.4f} |")
    lines += ["", "## 逐类别结果", "",
        "每格为 Dice / IoU（%）；只在GT存在该类别的病例上平均。N为这类有效病例数，三组一致。",
        "使用相对类别编号；CSV另保留原输出中的历史显示名，不将其提升为已核对的绝对解剖标签。",
        "背景单独由保存的混淆矩阵计算，不参与上述前景均值。", "",
        "| ID | 类别 | N | 原模型 | 中性文本 | 组织文本 | 文本−原模型 Dice pp |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    lookup = {(r['arm'], r['class_id']): r for r in per_class}
    for label in range(20):
        entries = [lookup[a, label] for a in arm_names]
        assert len({e['gt_present_cases'] for e in entries}) == 1
        cells = " | ".join(f"{e['dice_pct']:.2f} / {e['iou_pct']:.2f}" for e in entries)
        lines.append(f"| {label} | {class_name(label)} | {entries[0]['gt_present_cases']} | {cells} | "
            f"{entries[2]['dice_pct']-entries[0]['dice_pct']:+.2f} |")
    lines += ["", "ID9、19仅有1例，ID8、18只有5例；这些类别的均值不能视为稳定的类别性能估计。", "",
        "## 合并组织后的Dice", "", "先把同组织所有相对编号合并为二值掩码，再逐病例计算Dice；不是对编号类别Dice取均值。", "",
        "| 组织 | 原模型 | 中性文本 | 组织文本 |", "|---|---:|---:|---:|"]
    for t, name in (("vertebra", "椎体"), ("canal", "椎管"), ("disc", "椎间盘")):
        values = [next(r['dice_pct'] for r in tissues if r['arm'] == a and r['tissue'] == t) for a in arm_names]
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in values) + " |")
    lines += ["", "## 本轮判定", "",
        f"组织文本相对原模型主Dice {decision['text_minus_context_pp']:+.4f}pp，相对中性文本 {decision['text_minus_neutral_pp']:+.4f}pp。",
        "未通过预设的+0.3pp/+0.1pp且union不下降的筛查规则。本轮没有证明当前E4文本融合有效；不是所有文本融合方案均无效的结论。",
        "类别等权macro Dice虽有局部改善，但权重不同，不能替换预先规定的主指标。单种子valid结果不构成独立确认。",
        "HD95/ASD在本轮未计算，不能将null记作0；无需重新推理即可从保存结果导出本报告。", "",
        "文件：overall.csv、per_class.csv、per_case.csv、merged_tissue.csv、validation_history.csv；audit.json记录核验与来源哈希。"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
