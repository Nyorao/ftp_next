"""Scientific controls for image-only, center-residual context segmentation."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np
from PIL import Image
import torch

from freqssl.augment import aligned_cutmix, strong_view
from freqssl.data import build_training_data, SpiderSliceDataset
from freqssl.engine import make_optimizer, run_experiment, train_step, seed_everything
from freqssl.inference import load_single_network, predict_volume, prepare_slice, SingleNetwork
from freqssl.geometry import sagittal_plan, canonicalize, restore_native
from freqssl.models import StudentTeacherSystem
from freqssl.slice_context import ContextUNet, iter_cached_logits
from test_freqssl import CPUInvariantTest, make_spider_fixture, small_config, synthetic_batch


class ContextTests(CPUInvariantTest):
    def test_predeclared_selection_rejects_union_regression_and_unproven_gate(self):
        from experiments.run_slice_context import choose_module
        base = {"dice": .79, "dice_union_present": .75}
        self.assertIsNone(choose_module(base, {"dice": .792, "dice_union_present": .76},
                                       {"dice": .794, "dice_union_present": .749}))
        self.assertEqual(choose_module(base, {"dice": .795, "dice_union_present": .75},
                                       {"dice": .7955, "dice_union_present": .76}), "mean")
        self.assertEqual(choose_module(base, {"dice": .794, "dice_union_present": .75},
                                       {"dice": .796, "dice_union_present": .76}), "adaptive")

    def test_zero_initialization_preserves_backbone_and_logits(self):
        cfg = small_config(method="mean_teacher")
        base = StudentTeacherSystem(cfg)
        x = torch.randn(2, 3, 32, 32)
        for mode in ("mean", "adaptive"):
            context = StudentTeacherSystem(replace(cfg, slice_context=mode))
            for name, value in base.students[0].state_dict().items():
                self.assertTrue(torch.equal(value, context.students[0].state_dict()[name]))
            torch.testing.assert_close(context.student_logits(x, 0), base.student_logits(x[:, 1:2], 0),
                                       rtol=1e-5, atol=2e-6)
        large = ContextUNet(20, 32, 256, "adaptive")
        self.assertEqual(sum(p.numel() for p in large.parameters()), 5682868 + 5772)
        with self.assertRaises(ValueError):
            replace(cfg, method="dual_cps", slice_context="mean").validate()

    def test_center_random_stream_and_all_channel_cutmix_alignment(self):
        x = torch.randn(4, 3, 32, 32)
        torch.testing.assert_close(strong_view(x, 44, .08, .15)[:, 1:2],
                                   strong_view(x[:, 1:2], 44, .08, .15), rtol=0, atol=0)
        ids = torch.arange(4.).reshape(4, 1, 1, 1).expand(4, 3, 16, 16).clone()
        target = ids[:, :1].clone().requires_grad_()
        mixed, y = aligned_cutmix(ids, target, 9, 1.)
        for channel in range(3):
            self.assertTrue(torch.equal(mixed[:, channel:channel+1], y))
        self.assertFalse(y.requires_grad)

    def test_neighbor_access_center_equivalence_and_missing_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = make_spider_fixture(Path(tmp))
            cfg = small_config(png, method="mean_teacher", labelnum=3, max_samples=8)
            base, _, split = build_training_data(cfg)
            data, _, new_split = build_training_data(replace(cfg, slice_context="adaptive"))
            self.assertEqual(split, new_split)
            # Make neighbors visually different; geometry of all channels still matches.
            for i, name in enumerate(data.names):
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (40, 48), dtype=np.uint8)).save(png / "train/image" / name)
            for name in data.names[3:]:
                (png / "train/label" / name).unlink()
            for index in (0, 2, 3, 4, 7):
                with mock.patch("PIL.Image.open", wraps=Image.open) as opened:
                    item = data[index]
                labels_read = [Path(c.args[0]).name for c in opened.call_args_list if Path(c.args[0]).parent.name == "label"]
                self.assertEqual(labels_read, [data.names[index]] if index < 3 else [])
                torch.testing.assert_close(item["image"][1:2], base[index]["image"], rtol=0, atol=0)
                self.assertTrue(torch.equal(item["label"], base[index]["label"]))
            self.assertEqual(data.neighbors[data.names[3]][2], data.names[3])
            self.assertEqual(data.neighbors[data.names[4]][0], data.names[4])
            (png / "train/image" / data.names[1]).unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                build_training_data(replace(cfg, slice_context="mean", max_samples=7))

    def test_context_gradients_and_ema_on_cpu(self):
        self.check_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_context_gradients_and_ema_on_cuda(self):
        self.check_gradients("cuda:0")

    def check_gradients(self, device):
        cfg = small_config(method="mean_teacher", slice_context="adaptive")
        seed_everything(cfg)
        model = StudentTeacherSystem(cfg).to(device)
        optimizer = make_optimizer(model, cfg)
        batch = synthetic_batch()
        batch["image"] = torch.randn(4, 3, 32, 32)
        train_step(model, optimizer, batch, cfg, torch.device(device), 0)
        self.assertTrue(any(float(f.gamma.detach().abs()) > 0 for f in model.students[0].context_fusion))
        self.assertSameState(model.students[0], model.teachers[0])
        x = batch["image"].to(device).requires_grad_()
        model.students[0](x).square().mean().backward()
        self.assertGreater(float(x.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(x.grad[:, 2].abs().sum()), 0)
        self.assertTrue(any(float(f.weights.weight.grad.abs().sum()) > 0 for f in model.students[0].context_fusion))
        self.assertTrue(all(p.grad is None for p in model.teachers.parameters()))

    def test_cached_features_match_triplets_and_each_slice_encoded_once(self):
        model = ContextUNet(20, 4, 32, "adaptive").eval()
        with torch.no_grad():
            for layer in model.context_fusion:
                layer.gamma.fill_(.4)
                layer.weights.weight.normal_(0, .05)
        for count in (1, 2, 7):
            images = torch.randn(count, 1, 32, 48)
            reads = []
            def read(i):
                reads.append(i)
                return images[i]
            cached = torch.cat([x for _, x in iter_cached_logits(model, count, read, "cpu", 2)])
            triplets = torch.cat([images[[max(0, i-1), i, min(count-1, i+1)]].transpose(0, 1)
                                  for i in range(count)])
            with torch.no_grad():
                ordinary = model(triplets)
            torch.testing.assert_close(cached, ordinary, rtol=1e-4, atol=1e-5)
            self.assertEqual(reads, list(range(count)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_cache_is_independent_of_output_batch_and_partial_encoder_batch(self):
        # Nonzero gates exercise all neighbors. Batch 9 forces an incomplete
        # cache fill, while ordinary triplets encode 27 images.
        for mode in ("mean", "adaptive"):
            model = ContextUNet(20, 32, 256, mode).eval().cuda()
            with torch.inference_mode():
                for layer in model.context_fusion:
                    layer.gamma.fill_(.4)
                    if layer.weights is not None:
                        layer.weights.weight.normal_(0, .05)
                images = torch.randn(9, 1, 64, 80)
                triplets = torch.stack([images[[max(0, i-1), i, min(8, i+1)], 0] for i in range(9)])
                expected = model(triplets.cuda()).cpu()
                for batch in (1, 8):
                    actual = torch.cat([x.cpu() for _, x in iter_cached_logits(
                        model, len(images), lambda i: images[i], "cuda:0", batch)])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
                    self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))

    def test_exact_resume_and_native_evaluation_cpu(self):
        self.check_resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_exact_resume_and_native_evaluation_cuda(self):
        self.check_resume("cuda:0")

    def check_resume(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, root / "full", method="mean_teacher", slice_context="adaptive",
                               labelnum=4, max_samples=8, max_steps=3, eval_every=3)
            full = run_experiment(cfg, device)
            staged = replace(cfg, output_dir=str(root / "staged"))
            run_experiment(staged, device, stop_after=1)
            resumed = run_experiment(staged, device, resume=root / "staged/last.pt")
            for key in ("final_student1_hash", "final_teacher1_hash", "final_validation"):
                self.assertEqual(full[key], resumed[key])
            single, _, receipt = load_single_network(root / "staged/best.pt", device)
            self.assertEqual(receipt["backbones_loaded"], 1)
            self.assertEqual(receipt["slice_context"], "adaptive")
            self.assertIsInstance(single.backbone, ContextUNet)

    def test_label_free_volume_matches_uncached_prediction(self):
        backbone = ContextUNet(20, 4, 32, "adaptive").eval()
        with torch.no_grad():
            for layer in backbone.context_fusion:
                layer.gamma.fill_(.7)
                # Deliberately asymmetric: exchanging left/right must matter.
                layer.weights.bias.copy_(torch.tensor([3., 0., -3.]))
        model = SingleNetwork(backbone).eval()
        array = np.random.default_rng(3).integers(0, 100, (29, 27, 5), dtype=np.int16)
        for lr_sign in (1., -1.):
            geometry = {"size_xyz": [5, 27, 29], "spacing_xyz_mm": [3., 1., 1.],
                        "direction": [lr_sign, 0., 0., 0., 1., 0., 0., 0., lr_sign]}
            plan = sagittal_plan(geometry)
            actual = predict_volume(model, array, plan, 32, "cpu", 2)
            # Independent legacy PNG construction: acquisition slice index
            # increases; only the SI direction is normalized within each plane.
            planes = [array[:, :, i].T[:, ::-1] if lr_sign > 0 else array[:, :, i].T
                      for i in range(array.shape[2])]
            images = torch.stack([prepare_slice(value, 32) for value in planes])
            with torch.no_grad():
                outputs = []
                for i in range(len(images)):
                    x = images[[max(0, i-1), i, min(len(images)-1, i+1)]].transpose(0, 1)
                    prediction = torch.nn.functional.interpolate(model(x).argmax(1).float()[:, None],
                                  size=(27, 29), mode="nearest")[0, 0].numpy().astype(np.uint8)
                    outputs.append((prediction[:, ::-1] if lr_sign > 0 else prediction).T)
            np.testing.assert_array_equal(actual, np.stack(outputs, axis=2))


if __name__ == "__main__":
    unittest.main()
