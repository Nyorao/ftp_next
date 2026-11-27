import os
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from config import Config, load_config
from dataset import SpiderDataset, TwoStreamBatchSampler
from losses import mean_teacher_loss
from model.mean_teacher import FTPStudent, MeanTeacher
from model.psgm import PSGM
from trainer import binary_metrics, load_checkpoint_model, save_checkpoint, train_batch


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get("FTP_NET_ROOT", "/home/zhp/ftp_net"))


def tiny_config():
    return Config(root_path="unused", bert_type="missing-bert", text_emb_path="missing-embeddings",
                  num_classes=4, text_length=8, base_channels=4, image_size=64,
                  labelnum=1, max_samples=2, batch_size=2, labeled_bs=1, num_workers=0)


def tiny_model():
    return MeanTeacher(FTPStudent(torch.randn(1, 8, 16), torch.randn(4, 32), base_channels=4))


def tiny_batch():
    return {"image": torch.randn(2, 1, 64, 64), "lf": torch.randn(2, 1, 64, 64),
            "hf": torch.randn(2, 1, 64, 64), "label": torch.cat((
                torch.randint(4, (1, 64, 64)), torch.full((1, 64, 64), -100))),
            "is_labeled": torch.tensor([True, False])}


class MeanTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_ema_weights_buffers_and_teacher_mode(self):
        student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout())
        model = MeanTeacher(student, ema_decay=0.9)
        with torch.no_grad():
            for parameter in model.student.parameters():
                parameter.fill_(3)
        self.assertEqual(model.update_teacher(), 0)
        with torch.no_grad():
            for parameter in model.student.parameters():
                parameter.fill_(5)
            model.student[1].running_mean.fill_(8)
            model.student[1].num_batches_tracked.fill_(7)
        self.assertEqual(model.update_teacher(), 0.5)
        for parameter in model.teacher.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 4))
            self.assertFalse(parameter.requires_grad)
        torch.testing.assert_close(model.teacher[1].running_mean, torch.full((2,), 8.0))
        self.assertEqual(int(model.teacher[1].num_batches_tracked), 7)
        model.train()
        self.assertTrue(model.student.training)
        self.assertTrue(all(not module.training for module in model.teacher.modules()))

    def test_real_step_trains_both_prompt_modules_but_never_teacher(self):
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-3)
        before = model.student.apgm.scale.detach().clone()
        metrics = train_batch(model, optimizer, tiny_batch(), tiny_config(), torch.device("cpu"), epoch=10)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreater(metrics["consistency"], 0)
        self.assertGreater(metrics["prompt_weight"], 0)
        self.assertFalse(torch.equal(before, model.student.apgm.scale))
        for module in (model.student.unet, model.student.apgm, model.student.psgm):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertFalse(model.teacher.training)
        self.assertEqual(int(model.ema_updates), 1)

    def test_unlabeled_ground_truth_cannot_change_loss(self):
        cfg = tiny_config()
        student = {key: torch.randn(2, 4, 8, 8, requires_grad=True)
                   for key in ("lf_prompt", "hf_prompt", "lf_plain", "hf_plain")}
        teacher = {key: torch.randn(1, 4, 8, 8, requires_grad=True)
                   for key in ("lf_prompt", "hf_prompt")}
        labels = torch.randint(4, (2, 8, 8))
        mask = torch.tensor([True, False])
        first, _ = mean_teacher_loss(student, teacher, labels, mask, 10, cfg)
        labels[1].fill_(9999)
        second, _ = mean_teacher_loss(student, teacher, labels, mask, 10, cfg)
        torch.testing.assert_close(first, second)
        second.backward()
        self.assertTrue(all(value.grad is None for value in teacher.values()))

    def test_unlabeled_dataset_never_opens_a_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("image", "low_frequency", "high_frequency", "label"):
                (root / "train" / kind).mkdir(parents=True)
            for name in ("a.png", "b.png"):
                for kind in ("image", "low_frequency", "high_frequency"):
                    Image.fromarray(np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)).save(root / "train" / kind / name)
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(root / "train/label/a.png")
            dataset = SpiderDataset(root, "train", 64, labeled_count=1)
            sample = dataset[1]  # b.png intentionally has no label file.
            self.assertFalse(sample["is_labeled"])
            self.assertTrue((sample["label"] == -100).all())
            self.assertEqual(tuple(sample["image"].shape), (1, 64, 64))

    def test_checkpoint_is_self_contained_and_restores_ema(self):
        cfg = tiny_config()
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.student.parameters())
        train_batch(model, optimizer, tiny_batch(), cfg, torch.device("cpu"), 10)
        model.eval()
        low, high = torch.randn(1, 2, 65, 67), torch.randn(1, 2, 65, 67)
        expected = model.teacher_predictions(low, high)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, cfg, 1, 1, 0.2)
            with patch("trainer.load_text_features", side_effect=AssertionError("BERT must not reload")):
                restored, restored_cfg = load_checkpoint_model(path, torch.device("cpu"))
            restored.eval()
            actual = restored.teacher_predictions(low, high)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key])
                self.assertEqual(tuple(actual[key].shape), (1, 4, 65, 67))
            self.assertEqual(int(restored.ema_updates), 1)
            self.assertEqual(restored_cfg.bert_type, "missing-bert")

    def test_sampler_labeled_first_reproducible_and_validates_counts(self):
        sampler = TwoStreamBatchSampler(range(6), range(6, 10), 4, 2, seed=2)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), 3)
        for batch in first:
            self.assertTrue(all(i < 6 for i in batch[:2]))
            self.assertTrue(all(i >= 6 for i in batch[2:]))
        sampler.epoch = 1
        self.assertNotEqual(first, list(sampler))
        with self.assertRaises(ValueError):
            TwoStreamBatchSampler([0], [1], 4, 2)

    def test_config_and_foreground_metric_conventions(self):
        load_config(ROOT / "configs/ftp_mt.yaml").validate()
        empty = np.zeros((8, 8), dtype=bool)
        mask = empty.copy()
        mask[2:4, 2:4] = True
        self.assertEqual(binary_metrics(mask, mask), (1.0, 0.0))
        self.assertTrue(np.isnan(binary_metrics(mask, empty)[0]))
        self.assertEqual(binary_metrics(empty, mask)[0], 0)

    @unittest.skipUnless((ORIGINAL / "model/fusion.py").is_file(), "Original source unavailable")
    def test_apgm_class_bodies_are_unchanged(self):
        def classes(path):
            return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.ClassDef)}
        self.assertEqual(classes(ORIGINAL / "model/fusion.py"), classes(ROOT / "model/fusion.py"))

    @unittest.skipUnless((ORIGINAL / "model/convnext_pixel.py").is_file(), "Original source unavailable")
    def test_psgm_matches_original_outputs_and_input_gradients(self):
        spec = importlib.util.spec_from_file_location("original_psgm", ORIGINAL / "model/convnext_pixel.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        # Extract the original method without constructing/loading ConvNeXt.
        legacy = original.ConvNeXt.__new__(original.ConvNeXt)
        nn.Module.__init__(legacy)
        current = PSGM(torch.randn(20, 768)).eval()
        legacy.en_fea_conv = copy.deepcopy(current.en_fea_conv)
        legacy.fusion_mlp = copy.deepcopy(current.fusion_mlp)
        legacy.fusion_conv = copy.deepcopy(current.fusion_conv)
        legacy.txt_emb = current.txt_emb.clone()
        legacy.eval()
        left = torch.randn(1, 768, 2, 2, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        expected = legacy.psgm(left)
        actual = current(right)
        torch.testing.assert_close(expected, actual)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(left.grad, right.grad)


if __name__ == "__main__":
    unittest.main()
