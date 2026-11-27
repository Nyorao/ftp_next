"""Where does union-present Dice go? Zero-training decomposition of saved validation JSON.

Reads one or more ``valid_step_<step>.json`` files written by ``engine.py`` (or
``evaluate.py`` JSON with the same layout) and explains each run's gap between
the primary GT-present Dice and the union-present supplement.

Per case, with K = foreground classes present in the GT, M = foreground classes
predicted but absent from the GT ("spurious"), and S = sum of Dice over the K
GT-present classes:

    primary = S / K,    union_present = S / (K + M)

For every spurious (case, class) pair the script records how many voxels were
predicted, which GT classes those voxels came from (the confusion column), and
where the class sits relative to the GT's own numbering range of that tissue
(e.g. one level above the highest GT vertebra = an upper field-of-view shift).

Comparing run B against reference run A, the per-case change splits exactly into

    union_B - union_A = [S_B - S_A] / (K + M_B)            ("S effect", overlap quality)
                      + S_A * [1/(K + M_B) - 1/(K + M_A)]  ("M effect", spurious classes)

so a run whose primary Dice rises while union-present falls shows whether the
loss comes from extra spurious classes and of which kind.

The "drop small" columns remove spurious classes with fewer than T predicted
voxels from M while keeping S unchanged. This is an optimistic bound on what
removing small islands could recover, not a simulated post-processing result:
reassigned voxels might also change S.

Only validation-development data is read; nothing is trained or re-evaluated.

    python experiments/diagnose_union_drop.py \
        --run A=outputs/.../A/valid_step_6000.json \
        --run KV-pos=outputs/.../KV-pos/valid_step_6000.json \
        --output-dir outputs/union_drop_diagnosis
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TISSUE = np.array([0] + [1] * 9 + [2] + [3] * 9)          # bg, vertebra, canal, disc
TISSUE_NAMES = {0: "background", 1: "vertebra", 2: "canal", 3: "disc"}
LEVEL = np.array([0] + list(range(1, 10)) + [0] + list(range(1, 10)))  # 1 = most caudal


def load_network(path, network):
    data = json.loads(Path(path).read_text())
    if network not in data:
        raise ValueError(f"{path} has no network {network!r}; available: {sorted(data)}")
    block = data[network]
    if "cases" not in block:
        raise ValueError(f"{path} lacks per-case class rows (cases); re-run evaluation with this repository")
    return block["cases"]


def position_relative_to_gt(class_id, gt_present):
    """Where a spurious numbered class sits relative to the GT levels of its tissue."""
    tissue = int(TISSUE[class_id])
    if tissue not in (1, 3):
        return "canal" if tissue == 2 else "background"
    levels = [int(LEVEL[c]) for c in gt_present if TISSUE[c] == tissue]
    level = int(LEVEL[class_id])
    if not levels:
        return "tissue_absent_in_gt"
    if level > max(levels):
        return f"above_top+{level - max(levels)}"
    if level < min(levels):
        return f"below_bottom-{min(levels) - level}"
    return "inside_gt_range_gap"


def case_summary(case, small_thresholds):
    rows = case["classes"]
    if [row["class_id"] for row in rows] != list(range(1, 20)):
        raise ValueError(f"Unexpected class rows for case {case.get('case')}")
    confusion = np.asarray(case["anatomy"]["confusion_gt_rows_pred_columns"], dtype=np.int64)
    if confusion.shape != (20, 20):
        raise ValueError("Confusion matrix must be 20x20")
    gt_present = [row["class_id"] for row in rows if row["gt_voxels"] > 0]
    spurious = [row for row in rows if row["pred_voxels"] > 0 and row["gt_voxels"] == 0]
    K, M = len(gt_present), len(spurious)
    S = float(sum(row["dice"] for row in rows if row["gt_voxels"] > 0))
    pairs = []
    for row in spurious:
        c = row["class_id"]
        column = confusion[:, c]
        tissue = TISSUE[c]
        same_tissue = int(column[(TISSUE == tissue) & (np.arange(20) != c)].sum())
        from_bg = int(column[0])
        from_canal = int(column[10]) if tissue != 2 else 0
        other = int(column.sum()) - same_tissue - from_bg - from_canal
        source_gt = int(np.argmax(column))
        pairs.append({
            "case": case["case"], "class_id": c, "class_name": row.get("class_name", str(c)),
            "pred_voxels": int(row["pred_voxels"]),
            "position": position_relative_to_gt(c, gt_present),
            "from_gt_background": from_bg, "from_gt_same_tissue_other_level": same_tissue,
            "from_gt_canal": from_canal, "from_gt_other_tissue": other,
            "dominant_gt_source_class": source_gt,
        })
    contiguous = {}
    for tissue, name in ((1, "vertebra"), (3, "disc")):
        levels = sorted(int(LEVEL[c]) for c in gt_present if TISSUE[c] == tissue)
        contiguous[name] = (levels == list(range(levels[0], levels[-1] + 1))) if levels else None
    result = {"case": case["case"], "K": K, "M": M, "S": S,
              "primary": S / K if K else None, "union": S / (K + M) if K + M else None,
              "gt_levels_contiguous": contiguous, "spurious": pairs}
    for threshold in small_thresholds:
        kept = sum(pair["pred_voxels"] >= threshold for pair in pairs)
        result[f"union_drop_small_lt{threshold}"] = S / (K + kept) if K + kept else None
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize_run(cases, small_thresholds):
    per_case = [case_summary(case, small_thresholds) for case in cases]
    pairs = [pair for case in per_case for pair in case["spurious"]]
    positions, sources = {}, {}
    for pair in pairs:
        positions[pair["position"]] = positions.get(pair["position"], 0) + 1
        key = max(("from_gt_background", "from_gt_same_tissue_other_level", "from_gt_canal",
                   "from_gt_other_tissue"), key=lambda name: pair[name])
        sources[key] = sources.get(key, 0) + 1
    sizes = sorted(pair["pred_voxels"] for pair in pairs)
    overall = {"cases": len(per_case), "primary": mean(c["primary"] for c in per_case),
               "union": mean(c["union"] for c in per_case),
               "spurious_class_case_pairs": len(pairs),
               "cases_gt_vertebra_levels_noncontiguous": sum(
                   c["gt_levels_contiguous"]["vertebra"] is False for c in per_case),
               "cases_gt_disc_levels_noncontiguous": sum(
                   c["gt_levels_contiguous"]["disc"] is False for c in per_case),
               "cases_with_spurious": sum(c["M"] > 0 for c in per_case),
               "spurious_by_position": dict(sorted(positions.items())),
               "spurious_by_dominant_source": dict(sorted(sources.items())),
               "spurious_pred_voxels_quantiles": (np.quantile(sizes, [0, .25, .5, .75, 1]).tolist()
                                                  if sizes else None)}
    for threshold in small_thresholds:
        overall[f"union_drop_small_lt{threshold}"] = mean(c[f"union_drop_small_lt{threshold}"] for c in per_case)
    return {"overall": overall, "per_case": per_case}


def compare(reference, other):
    ref = {c["case"]: c for c in reference["per_case"]}
    rows = []
    for case in other["per_case"]:
        a = ref.get(case["case"])
        if a is None or a["K"] != case["K"] or not case["K"]:
            continue  # same GT => same K; mismatches mean different case sets
        s_effect = (case["S"] - a["S"]) / (case["K"] + case["M"])
        m_effect = a["S"] * (1 / (case["K"] + case["M"]) - 1 / (a["K"] + a["M"]))
        rows.append({"case": case["case"], "d_primary": case["primary"] - a["primary"],
                     "d_union": case["union"] - a["union"], "S_effect": s_effect, "M_effect": m_effect,
                     "d_M": case["M"] - a["M"]})
    if not rows:
        return None
    return {"paired_cases": len(rows),
            **{f"mean_{k}_pp": 100 * float(np.mean([r[k] for r in rows]))
               for k in ("d_primary", "d_union", "S_effect", "M_effect")},
            "total_d_M": int(sum(r["d_M"] for r in rows)),
            "cases_more_spurious": sum(r["d_M"] > 0 for r in rows),
            "cases_fewer_spurious": sum(r["d_M"] < 0 for r in rows), "per_case": rows}


def pct(value):
    return "n/a" if value is None else f"{100 * value:.4f}"


def write_report(runs, comparisons, small_thresholds, output):
    lines = ["# Union-present decomposition (development valid, no training)", "",
             "primary = S/K, union = S/(K+M); M = predicted-but-GT-absent foreground classes per case.", "",
             "| run | primary % | union % | spurious pairs | cases with spurious | "
             + " | ".join(f"union if M drops <{t} vox %" for t in small_thresholds) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(small_thresholds)]
    for label, run in runs.items():
        o = run["overall"]
        lines.append(f"| {label} | {pct(o['primary'])} | {pct(o['union'])} | {o['spurious_class_case_pairs']} | "
                     f"{o['cases_with_spurious']} | "
                     + " | ".join(pct(o[f'union_drop_small_lt{t}']) for t in small_thresholds) + " |")
    lines += ["", "The drop-small columns keep S fixed: an optimistic bound, not a post-processing result.", ""]
    for label, run in runs.items():
        o = run["overall"]
        lines += [f"## {label}", "", f"- by position vs GT range: {o['spurious_by_position']}",
                  f"- by dominant GT source of the spurious voxels: {o['spurious_by_dominant_source']}",
                  f"- predicted voxels per spurious pair (min/q1/median/q3/max): {o['spurious_pred_voxels_quantiles']}",
                  f"- cases whose GT levels are not contiguous (vertebra / disc): "
                  f"{o['cases_gt_vertebra_levels_noncontiguous']} / {o['cases_gt_disc_levels_noncontiguous']}", ""]
    if comparisons:
        lines += ["## Paired against the first run", "",
                  "| run - reference | cases | d primary pp | d union pp | S effect pp | M effect pp | total dM | cases +M / -M |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for label, row in comparisons.items():
            if row is None:
                lines.append(f"| {label} | 0 | | | | | | |")
                continue
            lines.append(f"| {label} | {row['paired_cases']} | {row['mean_d_primary_pp']:+.4f} | "
                         f"{row['mean_d_union_pp']:+.4f} | {row['mean_S_effect_pp']:+.4f} | "
                         f"{row['mean_M_effect_pp']:+.4f} | {row['total_d_M']:+d} | "
                         f"{row['cases_more_spurious']} / {row['cases_fewer_spurious']} |")
        lines += ["", "S effect + M effect = d union exactly (per case, then averaged).", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=VALID_JSON",
                        help="Repeatable; the first run is the paired reference.")
    parser.add_argument("--network", default="teacher1")
    parser.add_argument("--small-voxels", type=int, nargs="*", default=[100, 1000])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in args.run:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--run expects LABEL=PATH, got {spec!r}")
        if label in runs:
            raise SystemExit(f"Duplicate run label {label!r}")
        runs[label] = summarize_run(load_network(path, args.network), args.small_voxels)
    labels = list(runs)
    comparisons = {f"{label} - {labels[0]}": compare(runs[labels[0]], runs[label]) for label in labels[1:]}
    (output / "summary.json").write_text(json.dumps({"runs": runs, "comparisons": comparisons}, indent=2) + "\n")
    with (output / "spurious_pairs.csv").open("w", newline="") as handle:
        fields = ["run", "case", "class_id", "class_name", "pred_voxels", "position", "from_gt_background",
                  "from_gt_same_tissue_other_level", "from_gt_canal", "from_gt_other_tissue",
                  "dominant_gt_source_class"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, run in runs.items():
            for case in run["per_case"]:
                for pair in case["spurious"]:
                    writer.writerow({"run": label, **pair})
    write_report(runs, comparisons, args.small_voxels, output)
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
