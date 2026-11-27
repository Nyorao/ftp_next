"""Scientific invariants for the controlled freqssl comparison protocol.

These tests use synthetic data only. Passing them verifies implementation and
comparison controls; it does not demonstrate an improvement on SPIDER.
"""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch

from freqssl.augment import aligned_cutmix, strong_view
from freqssl.config import ExperimentConfig
from freqssl.models import StudentTeacherSystem, haar_components
from freqssl.objectives import masked_pseudo_loss


def small_config(root="/unused", output="/unused-run", method="dual_frequency", **changes):
    values = dict(
        root_path=str(root), output_dir=str(output), method=method,
        num_classes=20, image_size=32, base_channels=4, max_channels=32,
        labelnum=2, max_samples=4, batch_size=4, labeled_bs=2,
        num_workers=0, seed=19, max_steps=2, eval_every=2, log_every=2,
        warmup_steps=0, ramp_steps=0, confidence=0.0,
        background_confidence=0.0, cpu_threads=2, eval_batch_size=2,
        amp=False, max_seconds=120.0,
    )
    values.update(changes)
    return ExperimentConfig(**values).validate()


def synthetic_batch():
    generator = torch.Generator().manual_seed(71)
    return {
        "image": torch.randn((4, 1, 32, 32), generator=generator),
        "label": torch.cat((
            torch.randint(0, 20, (2, 32, 32), generator=generator),
            torch.full((2, 32, 32), -1, dtype=torch.long),
        )),
        "is_labeled": torch.tensor([True, True, False, False]),
    }


class CPUInvariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assertSameState(self, first, second):
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(set(first_state), set(second_state))
        for name in first_state:
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(first_state[name], second_state[name]), name)


class HaarTests(CPUInvariantTest):
    def test_constant_has_no_detail_and_preserves_odd_shape(self):
        values = torch.full((2, 1, 33, 35), 0.375)
        low, high = haar_components(values)
        self.assertEqual(low.shape, values.shape)
        self.assertEqual(high.shape, values.shape)
        self.assertTrue(torch.equal(low, values))
        self.assertEqual(torch.count_nonzero(high).item(), 0)

    def test_signed_detail_reconstructs_input_including_odd_edges(self):
        for shape in ((2, 3, 32, 34), (1, 1, 33, 35), (1, 1, 1, 3)):
            with self.subTest(shape=shape):
                values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
                low, high = haar_components(values)
                self.assertTrue(torch.equal(low + high, values))
        patch = torch.tensor([[[[0.0, 2.0], [4.0, 6.0]]]])
        low, high = haar_components(patch)
        self.assertTrue(torch.equal(low, torch.full_like(patch, 3.0)))
        self.assertLess(high.min().item(), 0)
        self.assertGreater(high.max().item(), 0)

    def test_random_reconstruction_preserves_gradient_and_precision(self):
        values = torch.randn((1, 2, 35, 33), dtype=torch.float64, requires_grad=True)
        low, high = haar_components(values)
        self.assertEqual(low.dtype, torch.float64)
        torch.testing.assert_close(low + high, values, rtol=0, atol=1e-15)
        (low + high).sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))


class StudentTeacherTests(CPUInvariantTest):
    def test_students_independent_teachers_frozen_and_creation_preserves_rng(self):
        before = torch.get_rng_state().clone()
        system = StudentTeacherSystem(small_config())
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        first = tuple(system.students[0].parameters())
        second = tuple(system.students[1].parameters())
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first, second)))
        for student, teacher in zip(system.students, system.teachers):
            self.assertSameState(student, teacher)
            self.assertTrue(all(a.data_ptr() != b.data_ptr()
                                for a, b in zip(student.parameters(), teacher.parameters())))
        system.train()
        self.assertTrue(all(student.training for student in system.students))
        self.assertTrue(all(not module.training for module in system.teachers.modules()))
        self.assertTrue(all(not p.requires_grad for p in system.teachers.parameters()))
        students = {id(p) for p in system.student_parameters()}
        self.assertFalse(students & {id(p) for p in system.teachers.parameters()})
        result = system.student_logits(torch.randn(1, 1, 32, 32), 0)
        self.assertEqual(result.shape, (1, 20, 32, 32))

    def test_each_ema_tracks_only_its_own_student(self):
        system = StudentTeacherSystem(small_config(ema_decay=0.9))
        with torch.no_grad():
            for index, student in enumerate(system.students):
                for parameter in student.parameters():
                    parameter.fill_(2.0 + 4.0 * index)
        self.assertEqual(system.update_teachers(), 0.0)
        for student, teacher in zip(system.students, system.teachers):
            self.assertSameState(student, teacher)
        second_before = copy.deepcopy(system.teachers[1])
        with torch.no_grad():
            for parameter in system.students[0].parameters():
                parameter.fill_(4.0)
        self.assertEqual(system.update_teachers(), 0.5)
        self.assertTrue(all(torch.equal(p, torch.full_like(p, 3.0))
                            for p in system.teachers[0].parameters()))
        self.assertSameState(system.teachers[1], second_before)
        self.assertEqual(system.ema_updates.item(), 2)
        self.assertTrue(all(not p.requires_grad for p in system.teachers.parameters()))

    def test_frequency_view_keeps_raw_context_and_only_changes_second_branch(self):
        cfg = small_config(frequency_strength=0.5)
        system = StudentTeacherSystem(cfg)
        images = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
        low, _ = haar_components(images)
        self.assertTrue(torch.equal(system.make_view(images, 0), images))
        frequency_view = system.make_view(images, 1)
        self.assertFalse(torch.equal(frequency_view, images))
        mixed_low, _ = haar_components(frequency_view)
        self.assertTrue(torch.equal(mixed_low, low))
        plain = StudentTeacherSystem(replace(cfg, method="dual_cps"))
        self.assertTrue(torch.equal(plain.make_view(images, 1), images))


class PseudoLossTests(CPUInvariantTest):
    def test_background_gate_weight_and_all_pixel_denominator(self):
        logits = torch.zeros((1, 3, 1, 4), requires_grad=True)
        teacher = torch.tensor([
            [0.05, 0.90, 0.97, 0.15],
            [0.90, 0.05, 0.02, 0.15],
            [0.05, 0.05, 0.01, 0.70],
        ]).reshape(1, 3, 1, 4).requires_grad_()
        loss, report = masked_pseudo_loss(logits, teacher, 0.8, 0.95, 0.25)
        self.assertAlmostEqual(loss.item(), math.log(3) * 1.25 / 4, places=6)
        self.assertEqual(report["selected_foreground_pixels"], 1)
        self.assertEqual(report["selected_background_pixels"], 1)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["foreground_coverage"], 0.25)
        self.assertEqual(report["background_coverage"], 0.25)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(torch.count_nonzero(logits.grad[..., 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(logits.grad[..., 3]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[..., 0]).item(), 0)

    def test_empty_mask_returns_finite_differentiable_zero(self):
        logits = torch.randn((2, 20, 4, 4), requires_grad=True)
        teacher = torch.full_like(logits, 1 / 20, requires_grad=True)
        loss, report = masked_pseudo_loss(logits, teacher, 0.8, 0.95, 0.25)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(report["selected_pixels"], 0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)


class AugmentationTests(CPUInvariantTest):
    def test_cutmix_uses_same_rectangle_and_donor_for_soft_target(self):
        images = torch.arange(1, 4, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 1, 8, 10).clone()
        probabilities = (0.1 + 0.7 * torch.eye(3)).reshape(3, 3, 1, 1).expand(3, 3, 8, 10).clone()
        probabilities.requires_grad_()
        old_images, old_probabilities = images.clone(), probabilities.detach().clone()
        mixed, target = aligned_cutmix(images, probabilities, seed=37, probability=1.0)
        masks = (mixed != images)[:, 0]
        self.assertTrue(torch.equal(masks[0], masks[1]))
        self.assertTrue(torch.equal(masks[0], masks[2]))
        ys, xs = masks[0].nonzero(as_tuple=True)
        self.assertGreater(len(ys), 0)
        self.assertLess(len(ys), 80)
        expected_mask = torch.zeros_like(masks[0])
        expected_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        self.assertTrue(torch.equal(expected_mask, masks[0]))
        for index in range(3):
            donor = int(mixed[index, 0, ys[0], xs[0]].item()) - 1
            self.assertNotEqual(index, donor)
            self.assertTrue(torch.equal(target[index, :, masks[index]],
                                        old_probabilities[donor, :, masks[index]]))
            self.assertTrue(torch.equal(target[index, :, ~masks[index]],
                                        old_probabilities[index, :, ~masks[index]]))
        self.assertTrue(torch.equal(images, old_images))
        self.assertTrue(torch.equal(probabilities.detach(), old_probabilities))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target.sum(1), torch.ones_like(target[:, 0]))

    def test_strong_view_is_seeded_without_global_rng_side_effects(self):
        images = torch.ones((2, 1, 8, 8))
        before = torch.get_rng_state().clone()
        first = strong_view(images, 22, noise_std=0.08, gain_std=0.15)
        second = strong_view(images, 22, noise_std=0.08, gain_std=0.15)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.equal(first, strong_view(images, 23, 0.08, 0.15)))


class TrainingControlTests(CPUInvariantTest):
    def test_independent_second_student_cannot_change_first_student_or_ema(self):
        from freqssl.engine import make_optimizer, train_step

        cfg = small_config(method="mean_teacher", grad_clip=0.05, cutmix_probability=1.0)
        dual_cfg = replace(cfg, method="dual_independent")
        single, dual = StudentTeacherSystem(cfg), StudentTeacherSystem(dual_cfg)
        self.assertSameState(single.students[0], dual.students[0])
        single_optimizer = make_optimizer(single, cfg)
        dual_optimizer = make_optimizer(dual, dual_cfg)
        batch = synthetic_batch()
        for step in range(3):
            single_report = train_step(single, single_optimizer, batch, cfg, torch.device("cpu"), step)
            dual_report = train_step(dual, dual_optimizer, batch, dual_cfg, torch.device("cpu"), step)
            self.assertGreater(single_report["grad_norm_s1"], cfg.grad_clip)
            self.assertEqual(single_report["supervised_s1"], dual_report["supervised_s1"])
            self.assertEqual(single_report["unsupervised_s1"], dual_report["unsupervised_s1"])
            self.assertSameState(single.students[0], dual.students[0])
            self.assertSameState(single.teachers[0], dual.teachers[0])

    def test_hidden_unlabeled_targets_do_not_change_losses_or_updates(self):
        from freqssl.engine import make_optimizer, train_step

        cfg = small_config(cutmix_probability=1.0)
        first, second = StudentTeacherSystem(cfg), StudentTeacherSystem(cfg)
        original = synthetic_batch()
        changed = {key: value.clone() for key, value in original.items()}
        changed["label"][~changed["is_labeled"]] = 999
        first_report = train_step(first, make_optimizer(first, cfg), original, cfg, torch.device("cpu"), 0)
        second_report = train_step(second, make_optimizer(second, cfg), changed, cfg, torch.device("cpu"), 0)
        self.assertEqual(first_report, second_report)
        self.assertSameState(first, second)


def make_spider_fixture(root):
    png, mha = root / "slice_png_order", root / "mha"
    for split, cases, count in (("train", (1, 2), 4), ("valid", (3,), 2), ("test", (4,), 2)):
        for folder in ("image", "label"):
            (png / split / folder).mkdir(parents=True)
            (mha / split / folder).mkdir(parents=True)
        for case in cases:
            labels = np.zeros((40, 48, count), dtype=np.uint8)
            labels[27:35, 15:32] = 1
            labels[5:13, 15:32] = 3
            pixels = np.broadcast_to(np.arange(40, dtype=np.uint8)[:, None, None] * 3,
                                     labels.shape).copy() + labels * 20
            for folder, array in (("image", pixels), ("label", labels)):
                sitk.WriteImage(sitk.GetImageFromArray(array), str(mha / split / folder / f"SPIDER_{case:03d}.mha"))
                for index in range(count):
                    stored = array[:, :, index].astype(np.uint16) if folder == "label" else array[:, :, index]
                    Image.fromarray(stored).save(png / split / folder / f"SPIDER_{case:03d}_{index}_{count}.png")
    return png


class DataIntegrationTests(CPUInvariantTest):
    def test_unlabeled_files_never_open_and_geometry_is_repeatable(self):
        from freqssl.data import build_training_data
        with tempfile.TemporaryDirectory() as temporary:
            png = make_spider_fixture(Path(temporary))
            cfg = small_config(png, labelnum=4, max_samples=8)
            dataset, sampler, manifest = build_training_data(cfg)
            forbidden = png / "train/label" / dataset.names[4]
            forbidden.unlink()
            with mock.patch("PIL.Image.open", wraps=Image.open) as opened:
                sample = dataset[4]
            self.assertFalse(any(Path(call.args[0]) == forbidden for call in opened.call_args_list))
            self.assertTrue((sample["label"] == -100).all())
            first = dataset[0]
            torch.manual_seed(999)
            self.assertTrue(torch.equal(first["image"], dataset[0]["image"]))
            dataset.set_epoch(1)
            self.assertFalse(torch.equal(first["image"], dataset[0]["image"]))
            self.assertFalse(manifest["offline_frequency_files_used"])
            self.assertFalse(manifest["test_labels_or_images_used"])
            self.assertEqual(len(sampler), 2)

    def test_complete_volume_training_reload_and_exact_mid_epoch_resume(self):
        from freqssl.engine import load_model, run_experiment
        from freqssl.evaluation import evaluate_volumes, prepare_validation
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, root / "uninterrupted", labelnum=4, max_samples=8,
                               max_steps=3, eval_every=3)
            complete = run_experiment(cfg, torch.device("cpu"))
            paused_cfg = replace(cfg, output_dir=str(root / "resumed"))
            paused = run_experiment(paused_cfg, torch.device("cpu"), stop_after=1)
            self.assertEqual(paused["status"], "paused")
            resumed = run_experiment(paused_cfg, torch.device("cpu"), resume=root / "resumed/last.pt")
            self.assertEqual(complete["final_student1_hash"], resumed["final_student1_hash"])
            self.assertEqual(complete["final_teacher1_hash"], resumed["final_teacher1_hash"])
            self.assertEqual(complete["final_validation"], resumed["final_validation"])
            model, loaded_cfg, payload = load_model(root / "resumed/last.pt", torch.device("cpu"))
            data, records = prepare_validation(loaded_cfg)
            result = evaluate_volumes(model, data, records, torch.device("cpu"))
            self.assertEqual(payload["step"], 3)
            self.assertEqual(result["teacher1"], complete["final_validation"]["teacher1"])
            self.assertEqual(result["teacher1"]["overall"]["cases"], 1)
            self.assertFalse(complete["test_evaluated"])
            self.assertTrue((root / "resumed/best.pt").exists())

    def test_crash_tail_archived_before_resume(self):
        from freqssl.engine import recover_log
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "metrics.jsonl"
            log.write_text('{"step":1}\n{"step":2}\n{"ste')
            recover_log(directory, 1)
            self.assertEqual(log.read_text(), '{"step":1}\n')
            tail = next((directory / "recovery").glob("*/metrics_tail.jsonl"))
            self.assertEqual(tail.read_text(), '{"step":2}\n{"ste')


class GPUControlTests(CPUInvariantTest):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for strict paired GPU control")
    def test_gpu_negative_control_and_cross_supervision(self):
        from freqssl.engine import make_optimizer, seed_everything, train_step
        cfg = small_config(method="mean_teacher", grad_clip=0.05)
        seed_everything(cfg)
        device = torch.device("cuda:0")
        single = StudentTeacherSystem(cfg).to(device)
        independent_cfg = replace(cfg, method="dual_independent")
        independent = StudentTeacherSystem(independent_cfg).to(device)
        cross_cfg = replace(cfg, method="dual_frequency")
        cross = StudentTeacherSystem(cross_cfg).to(device)
        single_opt, independent_opt, cross_opt = [make_optimizer(m, c) for m, c in
                                                ((single, cfg), (independent, independent_cfg), (cross, cross_cfg))]
        for step in range(3):
            batch = synthetic_batch()
            train_step(single, single_opt, batch, cfg, device, step)
            train_step(independent, independent_opt, batch, independent_cfg, device, step)
            report = train_step(cross, cross_opt, batch, cross_cfg, device, step)
            self.assertSameState(single.students[0], independent.students[0])
            self.assertSameState(single.teachers[0], independent.teachers[0])
            self.assertEqual(report["target_source_s1"], "teacher2")
            self.assertEqual(report["target_source_s2"], "teacher1")
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(single.students[0].parameters(), cross.students[0].parameters())))


if __name__ == "__main__":
    unittest.main()
