"""Auxiliary-loss gradients, alignment and paired experimental controls."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from freqssl.augment import strong_view
from freqssl.config import ExperimentConfig
from freqssl.engine import make_optimizer, run_experiment, seed_everything, train_step
from freqssl.models import StudentTeacherSystem
from freqssl.objectives import masked_pseudo_dice_loss, multiscale_region_loss
from test_freqssl import CPUInvariantTest, make_spider_fixture, small_config, synthetic_batch


class AuxiliaryObjectiveTests(CPUInvariantTest):
    def test_dice_gates_both_denominator_and_numerator_and_detaches_teacher(self):
        # Confident foreground, rejected pixel, confident background.
        logits = torch.zeros((1, 3, 1, 3), requires_grad=True)
        teacher = torch.tensor([[[[0.05, 0.2, 0.97]],
                                 [[0.90, 0.6, 0.02]],
                                 [[0.05, 0.2, 0.01]]]], requires_grad=True)
        loss, stats = masked_pseudo_dice_loss(logits, teacher)
        # Class 1: one of its two predicted target pixels is accepted.
        expected = (1 - (2 / 3 + 1e-5) / (1 + 1.25 / 9 + 1e-5)) * 0.5
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(stats["pseudo_dice_classes"], 1)
        self.assertEqual(stats["pseudo_dice_reliability"], 0.5)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(torch.count_nonzero(logits.grad[..., 1]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[..., 0]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[..., 2]).item(), 0)
        changed = logits.detach().clone()
        changed[..., 1] = torch.tensor([10., -10., 0.]).view(1, 3, 1)
        self.assertEqual(masked_pseudo_dice_loss(changed, teacher)[0].item(), loss.item())

    def test_dice_empty_or_background_only_returns_differentiable_zero(self):
        for kind in ("uncertain", "background"):
            logits = torch.randn((2, 3, 4, 4), requires_grad=True)
            teacher = torch.full_like(logits, 1 / 3)
            if kind == "background":
                teacher.zero_()
                teacher[:, 0] = 1
            teacher.requires_grad_()
            loss, stats = masked_pseudo_dice_loss(logits, teacher)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats["pseudo_dice_classes"], 0)
            loss.backward()
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)
            self.assertIsNone(teacher.grad)

    def test_dice_rewards_matching_foreground_without_rewarding_background_only(self):
        labels = torch.tensor([[[0, 1, 1, 2], [0, 0, 2, 2]]])
        target = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
        correct = 12 * target - 6
        wrong = correct.roll(1, dims=1)
        correct_loss, _ = masked_pseudo_dice_loss(correct, target)
        wrong_loss, _ = masked_pseudo_dice_loss(wrong, target)
        self.assertLess(correct_loss.item(), 1e-5)
        self.assertGreater(wrong_loss.item(), 0.9)

    def test_region_relations_ignore_channel_permutation_and_scale(self):
        generator = torch.Generator().manual_seed(101)
        feature = torch.randn((2, 6, 32, 32), generator=generator)
        permuted = 7 * feature[:, [2, 5, 1, 0, 4, 3]]
        loss, _ = multiscale_region_loss(feature, permuted)
        self.assertLess(loss.item(), 1e-12)
        self.assertGreater((feature - permuted).square().mean().item(), 1)

    def test_region_detects_spatial_mismatch_and_only_updates_student(self):
        generator = torch.Generator().manual_seed(102)
        student = torch.randn((2, 6, 32, 32), generator=generator, requires_grad=True)
        teacher = student.detach().roll(8, dims=-1).clone().requires_grad_()
        loss, stats = multiscale_region_loss(student, teacher)
        self.assertGreater(loss.item(), 0)
        self.assertEqual(set(stats), {"region_grid4", "region_grid8", "region_grid16"})
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.norm().item(), 0)

    def test_zero_features_are_finite_and_invalid_geometry_is_rejected(self):
        feature = torch.zeros((1, 4, 32, 32), requires_grad=True)
        loss, _ = multiscale_region_loss(feature, feature.detach())
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(feature.grad).all())
        with self.assertRaises(ValueError):
            multiscale_region_loss(feature, feature, (3,))
        with self.assertRaises(ValueError):
            multiscale_region_loss(feature, feature[..., :16, :16])


class AuxiliaryTrainingTests(CPUInvariantTest):
    def test_logits_feature_interface_and_checkpoint_keys_unchanged(self):
        model = StudentTeacherSystem(small_config())
        images = synthetic_batch()["image"]
        keys = set(model.state_dict())
        for index in (0, 1):
            prediction, features = model.student_outputs(images, index)
            self.assertTrue(torch.equal(prediction, model.student_logits(images, index)))
            self.assertEqual(features.shape, (4, 4, 32, 32))
            target, target_features = model.teacher_outputs(images, index)
            self.assertTrue(torch.equal(target, model.teacher_logits(images, index)))
            self.assertFalse(target.requires_grad or target_features.requires_grad)
        self.assertEqual(keys, set(model.state_dict()))

    def test_regions_use_unmixed_unlabeled_views_and_correct_peer_teacher(self):
        cfg = small_config(pseudo_dice_weight=0.5, region_weight=0.1, cutmix_probability=1.0)
        model = StudentTeacherSystem(cfg)
        batch = synthetic_batch()
        teacher_outputs = model.teacher_outputs
        saved = {}

        def record_teacher(images, index):
            value = teacher_outputs(images, index)
            saved[index] = value[1].clone()
            return value

        with mock.patch.object(model, "teacher_outputs", side_effect=record_teacher), \
                mock.patch.object(model, "student_outputs", wraps=model.student_outputs) as clean, \
                mock.patch.object(model, "student_logits", wraps=model.student_logits) as mixed, \
                mock.patch("freqssl.engine.multiscale_region_loss", wraps=multiscale_region_loss) as loss:
            report = train_step(model, make_optimizer(model, cfg), batch, cfg, torch.device("cpu"), 0)
        for index in (0, 1):
            expected = strong_view(batch["image"], cfg.seed + index * 10000019,
                                   cfg.noise_std, cfg.gain_std)[~batch["is_labeled"]]
            self.assertTrue(torch.equal(clean.call_args_list[index].args[0], expected))
            self.assertFalse(torch.equal(mixed.call_args_list[index].args[0][~batch["is_labeled"]], expected))
            self.assertTrue(torch.equal(loss.call_args_list[index].args[1], saved[1-index]))
            self.assertEqual(report[f"region_target_source_s{index+1}"], f"teacher{2-index}")
            self.assertGreater(report[f"weighted_pseudo_dice_s{index+1}"], 0)
            self.assertGreater(report[f"weighted_region_s{index+1}"], 0)
        self.assertTrue(all(p.grad is None for p in model.teachers.parameters()))

    def test_disabled_and_warmup_losses_do_not_forward_auxiliary_features(self):
        for cfg in (small_config(), small_config(pseudo_dice_weight=0.5, region_weight=0.1,
                                                 warmup_steps=10)):
            model = StudentTeacherSystem(cfg)
            with mock.patch.object(model, "student_outputs", side_effect=AssertionError("extra forward")), \
                    mock.patch.object(model, "teacher_outputs", side_effect=AssertionError("extra forward")):
                report = train_step(model, make_optimizer(model, cfg), synthetic_batch(), cfg,
                                    torch.device("cpu"), 0)
            self.assertEqual(report["weighted_region_s1"], 0)
            self.assertEqual(report["weighted_pseudo_dice_s1"], 0)

    def test_independent_control_still_matches_single_with_both_auxiliaries(self):
        cfg = small_config(method="mean_teacher", pseudo_dice_weight=0.5, region_weight=0.1)
        other = replace(cfg, method="dual_independent")
        single, dual = StudentTeacherSystem(cfg), StudentTeacherSystem(other)
        opt1, opt2 = make_optimizer(single, cfg), make_optimizer(dual, other)
        for step in range(2):
            train_step(single, opt1, synthetic_batch(), cfg, torch.device("cpu"), step)
            train_step(dual, opt2, synthetic_batch(), other, torch.device("cpu"), step)
            self.assertSameState(single.students[0], dual.students[0])
            self.assertSameState(single.teachers[0], dual.teachers[0])

    def test_each_auxiliary_changes_updates_and_hidden_labels_do_not(self):
        cfg = small_config()
        states = []
        for changes in ({}, {"pseudo_dice_weight": 0.5}, {"region_weight": 0.1}):
            settings = replace(cfg, **changes)
            model = StudentTeacherSystem(settings)
            train_step(model, make_optimizer(model, settings), synthetic_batch(), settings,
                       torch.device("cpu"), 0)
            states.append({k: v.clone() for k, v in model.students[0].state_dict().items()})
        for state in states[1:]:
            self.assertTrue(any(not torch.equal(v, state[k]) for k, v in states[0].items()))
        cfg = replace(cfg, pseudo_dice_weight=0.5, region_weight=0.1)
        first, second = StudentTeacherSystem(cfg), StudentTeacherSystem(cfg)
        batch, hidden = synthetic_batch(), synthetic_batch()
        hidden["label"][~hidden["is_labeled"]] = 999
        a = train_step(first, make_optimizer(first, cfg), batch, cfg, torch.device("cpu"), 0)
        b = train_step(second, make_optimizer(second, cfg), hidden, cfg, torch.device("cpu"), 0)
        self.assertEqual(a, b)
        self.assertSameState(first, second)

    def test_auxiliary_resume_reproduces_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, root / "complete", labelnum=4, max_samples=8,
                               max_steps=3, eval_every=3, pseudo_dice_weight=0.5, region_weight=0.1)
            complete = run_experiment(cfg, torch.device("cpu"))
            other = replace(cfg, output_dir=str(root / "resumed"))
            run_experiment(other, torch.device("cpu"), stop_after=1)
            resumed = run_experiment(other, torch.device("cpu"), resume=root / "resumed/last.pt")
            for field in ("final_student1_hash", "final_teacher1_hash", "final_validation"):
                self.assertEqual(complete[field], resumed[field])
            saved = json.loads((root / "resumed/config.json").read_text())
            self.assertEqual(ExperimentConfig(**saved).to_dict(), other.to_dict())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for deterministic auxiliary backward")
    def test_cuda_auxiliaries_are_deterministic_and_reach_both_students(self):
        cfg = small_config(pseudo_dice_weight=0.5, region_weight=0.1, cutmix_probability=1.0)
        seed_everything(cfg)
        device = torch.device("cuda:0")
        first, second = StudentTeacherSystem(cfg).to(device), StudentTeacherSystem(cfg).to(device)
        a = train_step(first, make_optimizer(first, cfg), synthetic_batch(), cfg, device, 0)
        b = train_step(second, make_optimizer(second, cfg), synthetic_batch(), cfg, device, 0)
        self.assertEqual(a, b)
        self.assertSameState(first, second)
        self.assertGreater(a["region_s1"], 0)
        self.assertGreater(a["region_s2"], 0)
        self.assertGreater(a["pseudo_dice_s1"], 0)
        self.assertGreater(a["pseudo_dice_s2"], 0)


class AuxiliaryProtocolTests(CPUInvariantTest):
    def test_auxiliary_arms_only_change_method_and_weights_and_default_to_dual_only(self):
        from experiments.compare_v3 import AUXILIARY_ARMS, resolve_arm, SUITE_ARMS
        cfg = small_config(pseudo_dice_weight=0.5, region_weight=0.1)
        records = {name: resolve_arm(cfg, name, "auxiliary").to_dict()
                   for name in AUXILIARY_ARMS}
        self.assertEqual(set(SUITE_ARMS["auxiliary"]), {"dual_aux", "dual_dice", "dual_region"})
        for name, record in records.items():
            common = {k: v for k, v in record.items()
                      if k not in ("method", "pseudo_dice_weight", "region_weight")}
            self.assertEqual(common, {k: v for k, v in cfg.to_dict().items()
                                     if k not in ("method", "pseudo_dice_weight", "region_weight")})
        self.assertEqual(records["single_aux"]["pseudo_dice_weight"], records["dual_aux"]["pseudo_dice_weight"])
        self.assertEqual(records["single_aux"]["region_weight"], records["dual_aux"]["region_weight"])
        self.assertEqual(records["dual_base"]["region_weight"], 0)
        self.assertEqual(records["dual_base"]["pseudo_dice_weight"], 0)
        with self.assertRaises(ValueError):
            resolve_arm(cfg, "dual_aux", "baseline")
        with self.assertRaises(ValueError):
            resolve_arm(small_config(), "dual_aux", "auxiliary")

    def test_reused_references_require_matching_budget_split_and_initialization(self):
        from experiments.compare_v3 import load_references, update_deltas
        from freqssl.data import build_training_data
        from freqssl.engine import state_hash, write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, labelnum=4, max_samples=8,
                               pseudo_dice_weight=0.5, region_weight=0.1)
            reference = root / "reference"
            reference.mkdir()
            runs = {}
            for method, dice in (("mean_teacher", 0.7), ("dual_frequency", 0.69)):
                directory = reference / f"seed{cfg.seed}/{method}"
                directory.mkdir(parents=True)
                settings = replace(cfg, method=method, pseudo_dice_weight=0, region_weight=0)
                _, _, split = build_training_data(settings)
                saved = settings.to_dict()
                # An old v3 checkpoint/config has no new auxiliary fields.
                for name in ("pseudo_dice_weight", "region_weight", "region_grids"):
                    saved.pop(name)
                write_json(directory / "config.json", saved)
                write_json(directory / "split.json", split)
                write_json(directory / "manifest.json", {
                    "initial_student1_hash": state_hash(StudentTeacherSystem(settings).students[0])})
                runs[f"seed{cfg.seed}/{method}"] = {"status": "completed", "step": cfg.max_steps,
                                                   "best_valid_dice": dice, "primary_network": "teacher1"}
            write_json(reference / "results.json", {"status": "completed", "runs": runs})
            reused = load_references(cfg, [cfg.seed], reference)
            self.assertEqual(reused[str(cfg.seed)]["single_base"]["best_valid_dice"], 0.7)
            paired = {"dual_aux": {"best_valid_dice": 0.72}}
            result = {"runs": {f"seed{cfg.seed}/dual_aux": {}}}
            update_deltas(result, paired, reused, cfg.seed)
            self.assertAlmostEqual(result["runs"][f"seed{cfg.seed}/dual_aux"]["valid_dice_delta_vs_single"], 0.02)
            self.assertAlmostEqual(result["runs"][f"seed{cfg.seed}/dual_aux"]["valid_dice_delta_vs_dual_base"], 0.03)
            with self.assertRaisesRegex(ValueError, "different protocol"):
                load_references(replace(cfg, max_steps=3), [cfg.seed], reference)
            file = reference / f"seed{cfg.seed}/mean_teacher/split.json"
            split = json.loads(file.read_text())
            split["labeled_files"] = []
            write_json(file, split)
            with self.assertRaisesRegex(ValueError, "different dataset split"):
                load_references(cfg, [cfg.seed], reference)


if __name__ == "__main__":
    unittest.main()
