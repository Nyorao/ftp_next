"""No-label deployment, preprocessing parity, and physical-space invariants."""

import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch

from freqssl.config import ExperimentConfig
from freqssl.geometry import sagittal_plan
from freqssl.inference import (image_geometry, legacy_gray8, load_single_network,
                               predict_volume, prediction_image, read_volume)
from freqssl.models import StudentTeacherSystem
from predict_spine import main as predict_main


class IntensityModel(torch.nn.Module):
    def forward(self, images):
        logits = images.new_full((len(images), 20, *images.shape[-2:]), -20)
        logits[:, 1] = images[:, 0]
        logits[:, 11] = -images[:, 0]
        return logits.softmax(1)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gray_conversion_matches_original_file_roundtrip(self):
        rng = np.random.default_rng(41)
        values = [rng.integers(-100, 3000, size=(29, 31), dtype=np.int16),
                  rng.integers(0, 60000, size=(29, 31), dtype=np.uint16),
                  rng.normal(size=(29, 31)).astype(np.float32),
                  np.zeros((29, 31), dtype=np.int16),
                  np.full((29, 31), -27.3)]
        for array in values:
            with self.subTest(dtype=array.dtype, constant=bool(array.min() == array.max())):
                buffer = io.BytesIO()
                plt.imsave(buffer, Image.fromarray(array), cmap="gray", format="png")
                buffer.seek(0)
                with Image.open(buffer) as image:
                    expected = np.array(image.convert("L"))
                np.testing.assert_array_equal(legacy_gray8(array), expected)
        with self.assertRaises(ValueError):
            legacy_gray8(np.full((3, 4), np.nan))

    def checkpoint(self, directory, method="dual_ordinal"):
        cfg = ExperimentConfig(root_path="/this/dataset/does/not/exist", method=method,
                               image_size=32, base_channels=2, max_channels=8,
                               region_grids=(4, 8), cpu_threads=1)
        system = StudentTeacherSystem(cfg).eval()
        path = directory / (method + ".pt")
        torch.save({"format_version": 3, "config": cfg.to_dict(),
                    "model": system.state_dict(), "step": 7}, path)
        return path, system

    def test_one_backbone_exactly_matches_full_system_for_all_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = torch.randn(2, 1, 32, 32, generator=torch.Generator().manual_seed(42))
            for method in ("mean_teacher", "dual_frequency", "dual_ordinal"):
                path, system = self.checkpoint(Path(tmp), method)
                names = ["student1", "teacher1"]
                if method.startswith("dual_"):
                    names += ["student2", "teacher2"]
                for name in names:
                    with self.subTest(method=method, network=name):
                        model, _, info = load_single_network(path, "cpu", name)
                        with torch.inference_mode():
                            torch.testing.assert_close(model(x), system.predict_proba(x, name), atol=0, rtol=0)
                        self.assertEqual(info["backbones_loaded"], 1)
                        self.assertFalse(any(p.requires_grad for p in model.parameters()))
                if method == "mean_teacher":
                    with self.assertRaises(ValueError):
                        load_single_network(path, "cpu", "teacher2")

    def test_prediction_commutes_with_all_physical_axis_permutations_and_flips(self):
        array = np.arange(17 * 19 * 3, dtype=np.int16).reshape(17, 19, 3)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((3.3, 0.8, 1.2))
        image.SetOrigin((17, -5, 42))
        angle = 0.07
        image.SetDirection((np.cos(angle), 0, np.sin(angle), 0, 1, 0,
                            -np.sin(angle), 0, np.cos(angle)))
        model = IntensityModel().eval()
        g = image_geometry(image)
        labels = predict_volume(model, array, sagittal_plan(g), 32, "cpu", batch_size=2)
        reference = prediction_image(labels, g)
        for order in itertools.permutations(range(3)):
            for flags in itertools.product((False, True), repeat=3):
                transformed = sitk.Flip(sitk.PermuteAxes(image, order), flags)
                geom = image_geometry(transformed)
                prediction = predict_volume(model, sitk.GetArrayFromImage(transformed),
                                            sagittal_plan(geom), 32, "cpu", batch_size=2)
                expected = sitk.Flip(sitk.PermuteAxes(reference, order), flags)
                np.testing.assert_array_equal(prediction, sitk.GetArrayFromImage(expected))
                actual = prediction_image(prediction, geom)
                np.testing.assert_allclose(actual.GetDirection(), expected.GetDirection())
                np.testing.assert_allclose(actual.GetOrigin(), expected.GetOrigin())

    def test_geometry_repair_is_bound_to_exact_image_bytes_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.mha"
            array = np.arange(9 * 7 * 3, dtype=np.int16).reshape(9, 7, 3)
            image = sitk.GetImageFromArray(array)
            good = image_geometry(image)
            image.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, -1))
            sitk.WriteImage(image, str(path))
            record = {"converted_image": str(path), "converted_image_array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                      "verified_geometry": good}
            _, plan, receipt = read_volume(path, record)
            self.assertEqual(receipt["output_geometry"], good)
            self.assertEqual(plan["flip_canonical_axes"], [2])
            with self.assertRaises(ValueError):
                read_volume(path, dict(record, converted_image=str(path.with_name("wrong.mha"))))
            with self.assertRaises(ValueError):
                read_volume(path, dict(record, converted_image_array_sha256="wrong"))

    def test_cli_needs_no_dataset_or_labels_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint, _ = self.checkpoint(directory)
            source = directory / "unannotated.mha"
            image = sitk.GetImageFromArray(np.arange(17 * 19 * 3, dtype=np.int16).reshape(17, 19, 3))
            image.SetSpacing((3.1, .8, 1.2))
            image.SetOrigin((-18, 14, 72))
            sitk.WriteImage(image, str(source))
            target = directory / "segmentation.mha"
            argv = ["--checkpoint", str(checkpoint), "--input", str(source),
                    "--output", str(target), "--device", "cpu", "--batch-size", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                receipt = predict_main(argv)
            output = sitk.ReadImage(str(target))
            self.assertEqual(image_geometry(output), image_geometry(image))
            self.assertEqual(receipt["annotation_files_read"], 0)
            self.assertEqual(receipt["network"], "teacher1")
            self.assertEqual(receipt["backbones_loaded"], 1)
            self.assertFalse(receipt["numeric_settings"]["cudnn_allow_tf32"])
            self.assertFalse(receipt["numeric_settings"]["cuda_matmul_allow_tf32"])
            saved = json.loads(target.with_suffix(".json").read_text())
            self.assertEqual(saved["output_sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                predict_main(argv)


if __name__ == "__main__":
    unittest.main()
