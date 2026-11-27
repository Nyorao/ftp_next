import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
import yaml

from config import Config
from experiments.compare_frequency import arm_config, main as comparison_main
from evaluate_volumes import main as evaluation_main
from model.dual_student import SingleTeacherStudents
from trainer import model_features, prediction_scores, student_parameters, train_batch
from test_mean_teacher import tiny_batch, tiny_config


class FrequencyAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prompt_free_features_never_load_bert(self):
        cfg = replace(tiny_config(), use_apgm=False, use_psgm=False)
        with patch("trainer.load_text_features", side_effect=AssertionError("BERT accessed")):
            text, classes = model_features(cfg)
        self.assertEqual(classes.shape, (4, 768))
        self.assertEqual(float(text.abs().sum() + classes.abs().sum()), 0)

    def test_prompt_free_training_has_one_decoder_bn_update_and_zero_prompt_loss(self):
        cfg = replace(tiny_config(), architecture="dual", use_apgm=False, use_psgm=False)
        net = SingleTeacherStudents(torch.zeros(1, 1, 1), torch.zeros(4, 32),
                                    base_channels=4, use_apgm=False, use_psgm=False)
        optimizer = torch.optim.AdamW(student_parameters(net), lr=1e-3)
        result = train_batch(net, optimizer, tiny_batch(), cfg, torch.device("cpu"), epoch=10)
        self.assertEqual(result["prompt"], 0)
        self.assertEqual(result["prompt_weight"], 0)
        for student in (net.student, net.student2):
            self.assertIsNone(student.apgm)
            self.assertIsNone(student.psgm)
            self.assertTrue(any(p.grad is not None for p in student.parameters()))
            for layer in student.unet.up.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    self.assertEqual(int(layer.num_batches_tracked), 1)
        self.assertTrue(all(p.grad is None for p in net.teacher.parameters()))

    def test_same_view_control_routes_lf_to_secondary_at_inference(self):
        cfg = arm_config(tiny_config(), "dual_same_view")
        net = SingleTeacherStudents(torch.zeros(1, 1, 1), torch.zeros(4, 32),
                                    base_channels=4, use_apgm=False, use_psgm=False).eval()
        low, high = torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            prediction, _ = prediction_scores(net, low, high, cfg, "student2")
            expected = net.student2(low, return_plain=False)["hf_prompt"]
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
        for arm in ("anchor_lf", "anchor_raw", "dual_frequency_no_feedback", "dual_frequency"):
            settings = arm_config(cfg, arm)
            self.assertFalse(settings.use_apgm or settings.use_psgm)
        with self.assertRaises(ValueError):
            replace(cfg, secondary_view="invalid").validate()

    def test_complete_runner_native_3d_and_negative_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = root / "spider/slice_png_order"
            mha = root / "spider/mha"
            for split, cases in (("train", (1, 2)), ("valid", (3,)), ("test", (4,))):
                for folder in ("image", "label", "low_frequency", "high_frequency"):
                    (png / split / folder).mkdir(parents=True)
                for case in cases:
                    target = np.zeros((64, 64, 2), dtype=np.uint8)
                    target[45:55, 20:35, :] = 1
                    target[10:20, 20:35, :] = 3
                    image = np.indices(target.shape)[0].astype(np.uint8) * 3
                    for index in range(2):
                        name = f"SPIDER_{case:03d}_{index}_2.png"
                        for folder in ("image", "low_frequency", "high_frequency", "label"):
                            data = target if folder == "label" else image
                            Image.fromarray(data[:, :, index]).save(png / split / folder / name)
                    if split == "valid":
                        for folder, data in (("image", image), ("label", target)):
                            (mha / split / folder).mkdir(parents=True)
                            sitk.WriteImage(sitk.GetImageFromArray(data), str(mha / split / folder / f"SPIDER_{case:03d}.mha"))
            cfg = Config(root_path=str(png), bert_type="missing", text_emb_path="missing",
                         labelnum=2, max_samples=4, image_size=64, base_channels=4,
                         batch_size=2, labeled_bs=1, cpu_threads=2, num_workers=0)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg.to_dict()))
            output = root / "run"
            args = ["compare_frequency", "--config", str(config), "--output-dir", str(output),
                    "--steps", "2", "--validate-every", "2", "--arms", "anchor_lf", "dual_frequency_no_feedback"]
            # Small visual width for a synthetic integration test only.
            with patch("sys.argv", args), patch("trainer.model_features", return_value=(torch.zeros(1, 1, 1), torch.zeros(20, 32))):
                comparison_main()
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(results["status"], "completed")
            self.assertTrue(results["arms"]["dual_frequency_no_feedback"]["negative_control_exact_match"])
            self.assertEqual(results["arms"]["anchor_lf"]["final_validation"]["teacher"]["overall"]["cases"], 1)
            self.assertFalse(results["test_evaluated"])
            evaluation = root / "evaluation"
            arguments = ["evaluate_volumes", "--run-dir", str(output), "--output-dir", str(evaluation),
                         "--splits", "valid", "--networks", "teacher", "--device", "cpu"]
            with patch("sys.argv", arguments):
                evaluation_main()
            evaluated = json.loads((evaluation / "results.json").read_text())
            self.assertEqual(evaluated["status"], "completed")
            entries = evaluated["splits"]["valid"]
            self.assertEqual(entries["anchor_lf/teacher"]["overall"],
                             entries["dual_frequency_no_feedback/teacher"]["overall"])


if __name__ == "__main__":
    unittest.main()
