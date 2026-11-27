"""Synthetic checks for the label-only adjacent-slice agreement diagnostic."""
import json

import numpy as np
from PIL import Image
import pytest

from experiments import diagnose_slice_correspondence as d


def test_pair_counts_shift_level_change_and_absent_class():
    center = np.zeros((20, 20), np.int64)
    neighbor = np.zeros((20, 20), np.int64)
    center[2:6, 2:6] = 1          # vertebra 1, neighbour has it shifted by 2 px
    neighbor[4:8, 2:6] = 1
    center[10:12, 10:12] = 2      # vertebra 2 where the neighbour shows vertebra 3
    neighbor[10:12, 10:12] = 3
    center[15:17, 15:17] = 10     # canal absent from the neighbour
    out = d.pair_counts(center, neighbor, radii=(1, 2))
    v = out["vertebra"]
    assert v["fg"] == 16 + 4
    assert v["same_class"] == 8                 # rows 4,5 of the shifted block overlap
    assert v["same_tissue_other_level"] == 4    # vertebra 2 vs 3 at the same coordinate
    assert v["within_r2"] == 16                 # whole vertebra-1 block reachable within 2 px
    assert 8 < v["within_r1"] < 16
    assert v["class_absent_in_neighbor"] == 4   # class 2 nowhere in the neighbour
    assert out["canal"]["class_absent_in_neighbor"] == 4 and out["canal"]["same_class"] == 0


def test_run_reads_only_labeled_budget(tmp_path):
    label_dir = tmp_path / "train" / "label"
    image_dir = tmp_path / "train" / "image"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    for index in range(3):
        array = np.zeros((8, 8), np.uint8)
        array[2:5, 2:5] = 1
        Image.fromarray(array).save(label_dir / f"SPIDER_001_{index}_3.png")
        Image.fromarray(array).save(image_dir / f"SPIDER_001_{index}_3.png")
    # A later, "unlabeled" file whose label must never be read: corrupt content.
    (label_dir / "SPIDER_002_0_1.png").write_bytes(b"not a png")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(image_dir / "SPIDER_002_0_1.png")
    result = d.run(tmp_path, "train", labelnum=3, size=0, radii=(1,))
    assert result["cases"] == 1 and result["directed_pairs"] == 4
    assert result["overall"]["vertebra"]["same_class"] == pytest.approx(1.0)
    out = tmp_path / "out"
    d.main(["--root", str(tmp_path), "--labelnum", "3", "--image-size", "0", "--radii", "1",
            "--output-dir", str(out)])
    assert json.loads((out / "summary.json").read_text())["directed_pairs"] == 4
    assert (out / "report.md").exists()
