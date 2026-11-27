"""Synthetic checks for the zero-training union-present decomposition script."""
import json

import numpy as np
import pytest

from experiments import diagnose_union_drop as d


def make_case(name, gt, pred):
    """Build a case row like evaluation.evaluate_volumes from GT/pred voxel label lists."""
    gt, pred = np.asarray(gt), np.asarray(pred)
    confusion = np.bincount(gt * 20 + pred, minlength=400).reshape(20, 20)
    rows = []
    for c in range(1, 20):
        g, p, i = int((gt == c).sum()), int((pred == c).sum()), int(confusion[c, c])
        dice = 2 * i / (g + p) if g + p else None
        rows.append({"class_id": c, "class_name": str(c), "gt_voxels": g, "pred_voxels": p,
                     "intersection_voxels": i, "dice": dice if g else None,
                     "dice_union_present": dice, "false_positive_absent_class": bool(p and not g)})
    return {"case": name, "classes": rows, "anatomy": {"confusion_gt_rows_pred_columns": confusion.tolist()}}


def test_primary_union_and_spurious_classification():
    # GT: vertebrae 1,2 and discs 11,12 (10 voxels each) plus background.
    gt = [1] * 10 + [2] * 10 + [11] * 10 + [12] * 10 + [0] * 20
    pred = list(gt)
    pred[10:14] = [3] * 4        # 4 voxels of GT vertebra 2 called vertebra 3: upper shift
    pred[40:42] = [9] * 2        # 2 background voxels called vertebra 9: far island
    case = d.case_summary(make_case("c1", gt, pred), small_thresholds=[3])
    assert case["K"] == 4 and case["M"] == 2
    dice2 = 2 * 6 / (10 + 6)
    assert case["S"] == pytest.approx(3 + dice2)
    assert case["primary"] == pytest.approx((3 + dice2) / 4)
    assert case["union"] == pytest.approx((3 + dice2) / 6)
    pairs = {p["class_id"]: p for p in case["spurious"]}
    assert pairs[3]["position"] == "above_top+1"
    assert pairs[3]["from_gt_same_tissue_other_level"] == 4 and pairs[3]["dominant_gt_source_class"] == 2
    assert pairs[9]["position"] == "above_top+7" and pairs[9]["from_gt_background"] == 2
    assert case["gt_levels_contiguous"] == {"vertebra": True, "disc": True}
    gap = d.case_summary(make_case("c2", [1] * 3 + [3] * 3 + [11] * 3, [1] * 3 + [3] * 3 + [11] * 3), [])
    assert gap["gt_levels_contiguous"] == {"vertebra": False, "disc": True}
    # Dropping spurious classes with < 3 voxels removes only class 9.
    assert case["union_drop_small_lt3"] == pytest.approx((3 + dice2) / 5)


def test_comparison_effects_sum_to_union_change():
    gt = [1] * 10 + [2] * 10 + [11] * 10 + [0] * 10
    pred_a = list(gt)
    pred_b = list(gt)
    pred_b[0:2] = [2, 2]           # worse overlap
    pred_b[30:33] = [13] * 3       # extra spurious disc
    a = d.summarize_run([make_case("c", gt, pred_a)], [])
    b = d.summarize_run([make_case("c", gt, pred_b)], [])
    row = d.compare(a, b)
    assert row["paired_cases"] == 1
    assert row["mean_S_effect_pp"] + row["mean_M_effect_pp"] == pytest.approx(row["mean_d_union_pp"])
    assert row["total_d_M"] == 1 and row["mean_M_effect_pp"] < 0


def test_cli_writes_report(tmp_path):
    gt = [1] * 5 + [11] * 5 + [0] * 5
    pred = list(gt)
    pred[10] = 2
    payload = {"teacher1": {"cases": [make_case("c", gt, pred)]}}
    path = tmp_path / "valid_step_6000.json"
    path.write_text(json.dumps(payload))
    out = tmp_path / "out"
    d.main(["--run", f"A={path}", "--run", f"B={path}", "--output-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["runs"]["A"]["overall"]["spurious_class_case_pairs"] == 1
    assert summary["comparisons"]["B - A"]["mean_d_union_pp"] == pytest.approx(0)
    assert (out / "report.md").exists() and (out / "spurious_pairs.csv").exists()


def test_rejects_missing_case_rows(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"teacher1": {"overall": {}}}))
    with pytest.raises(ValueError):
        d.load_network(path, "teacher1")
