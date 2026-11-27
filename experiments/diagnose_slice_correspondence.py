"""How much do adjacent sagittal slices agree at the same coordinate? Labels only, no model.

Two roadmap candidates rest on assumptions about neighbouring slices that the
ground truth can check before any training:

* correspondence-aware fusion (CRF that searches neighbour positions) only has
  label-level headroom where the centre label is missing at the same coordinate
  of the neighbour but present within a small window;
* cross-slice level consistency assumes that, where two adjacent slices show the
  same tissue (vertebra or disc) at the same coordinate, they carry the same
  relative level number.

For every pair of adjacent slices of a case (both directions) and every
foreground pixel u of the centre slice, the script counts whether the neighbour
has: the same class at u; the same tissue at u; the same tissue but a different
level at u; the centre class somewhere within a (2r+1)^2 window around u; or no
pixel of that class anywhere. Labels are resized with nearest neighbour to the
training grid first (default 224, PIL nearest, close to but not bit-identical
with the training F.interpolate), so a radius of r grid pixels at stride s of the
network corresponds to r*s here.

Only label PNGs are read. For the train split only the first ``--labelnum``
lexicographically sorted files are used, i.e. exactly the labeled budget of the
training protocol; unlabeled annotations are never opened.

    python experiments/diagnose_slice_correspondence.py --split train --labelnum 1496 \
        --output-dir outputs/slice_correspondence_diagnosis
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

NAME = re.compile(r"^(SPIDER_\d+)_(\d+)_(\d+)\.png$")
TISSUE = np.array([0] + [1] * 9 + [2] + [3] * 9)
LEVEL = np.array([0] + list(range(1, 10)) + [0] + list(range(1, 10)))
TISSUE_NAMES = ("vertebra", "canal", "disc")
COUNTERS = ("fg", "same_class", "same_tissue", "same_tissue_other_level", "class_absent_in_neighbor")


def read_label(path, size):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2 or array.min() < 0 or array.max() >= 20:
        raise ValueError(f"Expected a 2D label PNG with class IDs 0..19: {path}")
    if size:
        array = np.array(Image.fromarray(array.astype(np.uint8)).resize((size, size), Image.NEAREST))
    return array.astype(np.int64)


def pair_counts(center, neighbor, radii):
    """Counters over the centre's foreground pixels, split by centre tissue."""
    out = {name: {key: 0 for key in (*COUNTERS, *(f"within_r{r}" for r in radii))} for name in TISSUE_NAMES}
    present = set(np.unique(neighbor).tolist())
    within = {r: np.zeros(center.shape, bool) for r in radii}
    for c in np.unique(center):
        if c == 0:
            continue
        here = center == c
        target = neighbor == c
        for r in radii:
            reach = binary_dilation(target, structure=np.ones((2 * r + 1, 2 * r + 1), bool)) if target.any() else target
            within[r] |= here & reach
    same_tissue = TISSUE[center] == TISSUE[neighbor]
    numbered = np.isin(TISSUE[center], (1, 3))
    absent = np.isin(center, [c for c in np.unique(center) if c not in present])
    for index, name in enumerate(TISSUE_NAMES, start=1):
        mask = TISSUE[center] == index
        row = out[name]
        row["fg"] += int(mask.sum())
        row["same_class"] += int((mask & (center == neighbor)).sum())
        row["same_tissue"] += int((mask & same_tissue).sum())
        row["same_tissue_other_level"] += int((mask & same_tissue & numbered & (center != neighbor)).sum())
        row["class_absent_in_neighbor"] += int((mask & absent).sum())
        for r in radii:
            row[f"within_r{r}"] += int((mask & within[r]).sum())
    return out


def add(total, part):
    for name, row in part.items():
        for key, value in row.items():
            total[name][key] += value


def empty(radii):
    return {name: {key: 0 for key in (*COUNTERS, *(f"within_r{r}" for r in radii))} for name in TISSUE_NAMES}


def rates(table):
    result = {}
    for name, row in table.items():
        fg = row["fg"]
        result[name] = {"fg_pixels": fg, **{key: (row[key] / fg if fg else None) for key in row if key != "fg"}}
    fg = sum(row["fg"] for row in table.values())
    merged = {key: sum(row[key] for row in table.values()) for key in next(iter(table.values()))}
    result["all_foreground"] = {"fg_pixels": fg, **{key: (merged[key] / fg if fg else None)
                                                    for key in merged if key != "fg"}}
    return result


def run(root, split, labelnum, size, radii, bins=5):
    label_dir = Path(root) / split / "label"
    # The budget is defined on sorted IMAGE names, exactly like freqssl.data._slice_names.
    names = sorted(path.name for path in (Path(root) / split / "image").glob("*.png"))
    if not names:
        raise ValueError(f"No image PNGs under {Path(root) / split / 'image'}")
    if split == "train":
        if labelnum is None or not 0 < labelnum <= len(names):
            raise ValueError("--labelnum must select the labeled budget for the train split")
        names = names[:labelnum]
    cases = {}
    for name in names:
        match = NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid SPIDER slice name: {name}")
        cases.setdefault(match.group(1), {})[int(match.group(2))] = (name, int(match.group(3)))
    overall, by_bin, pairs = empty(radii), [empty(radii) for _ in range(bins)], 0
    for case, slices in sorted(cases.items()):
        labels = {index: read_label(label_dir / name, size) for index, (name, _) in slices.items()}
        for index, (_, count) in slices.items():
            for neighbor in (index - 1, index + 1):
                if neighbor not in labels:
                    continue
                part = pair_counts(labels[index], labels[neighbor], radii)
                add(overall, part)
                position = index / (count - 1) if count > 1 else 0.5
                add(by_bin[min(bins - 1, int(position * bins))], part)
                pairs += 1
    return {"split": split, "labeled_files_used": len(names), "cases": len(cases), "directed_pairs": pairs,
            "grid": size, "radii": list(radii), "overall": rates(overall),
            "by_relative_slice_position": [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", **rates(t)}
                                            for b, t in enumerate(by_bin)]}


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}"


def write_report(result, output):
    radii = result["radii"]
    lines = ["# Adjacent-slice label agreement (labels only, no model)", "",
             f"split={result['split']}, files={result['labeled_files_used']}, cases={result['cases']}, "
             f"directed pairs={result['directed_pairs']}, grid={result['grid']}", "",
             "Rates are over the centre slice's foreground pixels of each tissue.", "",
             "| tissue | fg px | same class @u | same tissue @u | same tissue, other level @u | "
             + " | ".join(f"class within r={r}" for r in radii) + " | class absent in neighbour |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(radii) + "---:|"]
    for name, row in result["overall"].items():
        lines.append(f"| {name} | {row['fg_pixels']} | {pct(row['same_class'])} | {pct(row['same_tissue'])} | "
                     f"{pct(row['same_tissue_other_level'])} | "
                     + " | ".join(pct(row[f'within_r{r}']) for r in radii)
                     + f" | {pct(row['class_absent_in_neighbor'])} |")
    lines += ["", "## By relative slice position (0 and 1 are the lateral ends)", "",
              "| bin | fg px | same class @u | same tissue, other level @u | "
              + " | ".join(f"within r={r}" for r in radii) + " | absent in neighbour |",
              "|---|---:|---:|---:|" + "---:|" * len(radii) + "---:|"]
    for row in result["by_relative_slice_position"]:
        a = row["all_foreground"]
        lines.append(f"| {row['bin']} | {a['fg_pixels']} | {pct(a['same_class'])} | "
                     f"{pct(a['same_tissue_other_level'])} | "
                     + " | ".join(pct(a[f'within_r{r}']) for r in radii)
                     + f" | {pct(a['class_absent_in_neighbor'])} |")
    lines += ["", "Reading guide: within-r minus same-class is the label-level headroom of searching",
              "correspondences inside that radius; 'absent in neighbour' cannot be fixed by any search;",
              "'same tissue, other level' should be near zero for cross-slice level consistency to hold.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=os.environ.get("SPIDER_PNG_ROOT"),
                        help="SPIDER PNG root (default: $SPIDER_PNG_ROOT)")
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--labelnum", type=int, default=1496)
    parser.add_argument("--image-size", type=int, default=224, help="0 keeps native resolution")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if not args.root:
        raise SystemExit("Set --root or SPIDER_PNG_ROOT")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = run(args.root, args.split, args.labelnum if args.split == "train" else None,
                 args.image_size, tuple(args.radii))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, output)
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
