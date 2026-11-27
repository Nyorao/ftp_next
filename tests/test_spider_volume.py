import unittest
from pathlib import Path
import tempfile

import numpy as np
from medpy import metric
import SimpleITK as sitk

from correct_volume_geometry import verified_geometry
from spider_volume import aggregate_cases, group_complete_slices, reconstruct, volume_metrics


class SpiderVolumeTests(unittest.TestCase):
    def test_verified_geometry_updates_spacing_for_axis_permutation(self):
        raw = np.zeros((3, 7, 11), dtype=np.uint8)
        raw[1, 2:5, 3:6] = 201
        target = raw.transpose(1, 2, 0).copy()
        target[target == 201] = 11
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.mha"
            image = sitk.GetImageFromArray(raw)
            image.SetSpacing((0.4, 0.7, 3.2))
            sitk.WriteImage(image, str(path))
            record = {"matches": [{"raw_mask": str(path), "transpose": [1, 2, 0], "flip_axes": []}]}
            corrected, evidence = verified_geometry(record, target)
            self.assertEqual(corrected.GetSpacing(), (3.2, 0.4, 0.7))
            self.assertEqual(tuple(reversed(corrected.GetSize())), target.shape)
            self.assertTrue(evidence[0]["exact_label_match"])
            with self.assertRaises(ValueError):
                verified_geometry(record, np.zeros_like(target))

    def test_numeric_order_and_incomplete_volume_rejection(self):
        names = [f"SPIDER_001_{i}_12.png" for i in range(12)]
        self.assertEqual(group_complete_slices(reversed(names))["SPIDER_001"], names)
        with self.assertRaises(ValueError):
            group_complete_slices(names[:-1])
        with self.assertRaises(ValueError):
            group_complete_slices(names + names[:1])

    def test_reconstruction_restores_all_axes_and_flips(self):
        target = np.arange(3 * 5 * 7).reshape(3, 5, 7)
        for axis in range(3):
            for flip in (False, True):
                slices = [np.take(target, i, axis=axis) for i in range(target.shape[axis])]
                if flip:
                    slices = [np.flip(value, axis=0) for value in slices]
                np.testing.assert_array_equal(reconstruct(slices, axis, flip, target.shape), target)

    def test_physical_distances_match_medpy_including_crop(self):
        target = np.zeros((15, 17, 19), dtype=np.uint8)
        prediction = np.zeros_like(target)
        target[3:8, 4:10, 5:11] = 1
        prediction[4:9, 6:12, 5:11] = 1
        spacing = (3.1, 0.7, 0.4)
        row = volume_metrics(prediction, target, spacing)[0]
        self.assertAlmostEqual(row["dice"], metric.binary.dc(prediction == 1, target == 1))
        self.assertAlmostEqual(row["iou"], metric.binary.jc(prediction == 1, target == 1))
        self.assertAlmostEqual(row["hd95_mm"], metric.binary.hd95(prediction == 1, target == 1, voxelspacing=spacing))
        self.assertAlmostEqual(row["asd_mm"], metric.binary.asd(prediction == 1, target == 1, voxelspacing=spacing))

    def test_empty_predictions_and_absent_class_false_positives_are_recorded(self):
        target = np.zeros((4, 5, 6), dtype=np.uint8)
        target[0] = 1
        prediction = np.zeros_like(target)
        prediction[1] = 2
        rows = volume_metrics(prediction, target, (1, 1, 1))
        self.assertEqual(rows[0]["dice"], 0)
        self.assertTrue(rows[0]["missed_gt_class"])
        self.assertIsNone(rows[0]["hd95_mm"])
        self.assertIsNone(rows[1]["dice"])
        self.assertEqual(rows[1]["dice_union_present"], 0)
        self.assertTrue(rows[1]["false_positive_absent_class"])
        summary = aggregate_cases([{"case": "one", "classes": rows}])
        self.assertEqual(summary["overall"]["missed_gt_class_case_pairs"], 1)
        self.assertEqual(summary["overall"]["fp_absent_gt_class_case_pairs"], 1)
        self.assertEqual(summary["overall"]["dice"], 0)
        self.assertIsNone(summary["overall"]["hd95_mm"])


if __name__ == "__main__":
    unittest.main()
