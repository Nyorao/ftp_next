"""Leakage, disjoint supervision, cache geometry and compatibility invariants."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch

from freqssl.anatomy import FAMILY_IDS
from freqssl.augment import aligned_cutmix
from freqssl.data import build_training_data
from freqssl.engine import make_optimizer, train_step, run_experiment
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from freqssl.objectives import masked_pseudo_loss
from freqssl.prompt_supervision import (TEXT_PROMPTS, PromptCache, file_sha256,
    mc_tissue_targets, prompt_routed_loss)
from test_freqssl import CPUInvariantTest, make_spider_fixture, small_config, synthetic_batch


def fake_cache(root, split, size=32):
    root.mkdir()
    entries = {}
    for name in split["unlabeled_files"]:
        tissue = np.ones((size, size), dtype=np.uint8)
        tissue[:, size // 2:] = 3
        q = np.eye(4, dtype=np.float16)[tissue].transpose(2, 0, 1)
        file = Path(name).stem + ".npz"
        np.savez(root / file, probabilities=q, tissue=tissue, trusted=np.ones_like(tissue, bool), valid=np.ones_like(tissue, bool))
        entries[name] = {"file": file, "sha256": file_sha256(root / file),
            "image_sha256": file_sha256(Path(split["root_path"]) / "train/image" / name)}
    manifest = {"format_version": 1, "status": "completed", "source": "text", "prompts": list(TEXT_PROMPTS),
        "image_size": size, "family_ids": list(FAMILY_IDS), "mc_samples": 8,
        "unlabeled_annotations_opened": False, "entries": entries,
        **{k: split[k] for k in ("root_path", "labeled_files", "unlabeled_files")}}
    for key, directory in (("labeled_annotation_sha256", "label"), ("labeled_image_sha256", "image")):
        manifest[key] = {n: file_sha256(Path(split["root_path"]) / "train" / directory / n) for n in split["labeled_files"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class PromptTests(CPUInvariantTest):
    def test_predeclared_gates_preserve_union_and_separate_text_effect(self):
        from experiments.run_prompt_context import screening_decision
        base = {"dice": .8, "dice_union_present": .77}
        text = {"dice": .804, "dice_union_present": .772}
        neutral = {"dice": .802, "dice_union_present": .771}
        self.assertTrue(screening_decision(base, text, neutral)["passed_screen"])
        self.assertFalse(screening_decision(base, dict(text, dice_union_present=.769), neutral)["passed_screen"])
        neutral = dict(neutral, dice=.8035)
        decision = screening_decision(base, text, neutral)
        self.assertFalse(decision["passed_screen"])
        self.assertEqual(decision["interpretation"], "extra_pretrained_teacher_may_help_text_semantics_unproven")

    def test_unreliable_fallback_value_and_gradient_are_exact(self):
        logits = torch.randn(2, 20, 7, 9, requires_grad=True)
        ema = torch.rand_like(logits).softmax(1).requires_grad_()
        t = torch.randint(0, 4, (2, 7, 9))
        a, _ = masked_pseudo_loss(logits, ema, .05, .05, .25)
        b, report = prompt_routed_loss(logits, ema, t, torch.zeros_like(t), torch.ones_like(t), .05, .05)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(torch.autograd.grad(a, logits, retain_graph=True)[0], torch.autograd.grad(b, logits)[0]))
        self.assertEqual(report["tissue_coverage"], 0)
        self.assertIsNone(ema.grad)

    def test_conflict_routes_to_group_not_uniform_number_targets(self):
        logits = torch.randn(1, 20, 1, 4, requires_grad=True)
        ema = torch.zeros_like(logits)
        ema[:, 1] = .99
        ema[:, 0] = .01
        tissue = torch.tensor([[[1, 3, 3, 0]]])
        trusted = torch.tensor([[[True, True, False, True]]])
        valid = torch.ones_like(trusted)
        result, report = prompt_routed_loss(logits, ema, tissue, trusted, valid)
        logp = logits.log_softmax(1)
        expected = (-logp[0, 1, 0, 0] - logp[0, 1, 0, 2]
            - .5 * torch.logsumexp(logp[0, 11:20, 0, 1], 0)
            - .5 * .25 * logp[0, 0, 0, 3]) / 4
        torch.testing.assert_close(result, expected)
        self.assertEqual(report["fine_tissue_overlap"], 0)
        self.assertEqual(report["fine_coverage"], .5)
        self.assertEqual(report["tissue_coverage"], .5)
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_low_confidence_ema_uses_reliable_tissue_and_padding_is_ignored(self):
        logits = torch.zeros(1, 20, 1, 2, requires_grad=True)
        t = torch.ones(1, 1, 2, dtype=torch.long)
        loss, info = prompt_routed_loss(logits, logits.detach().softmax(1), t, t.bool(), torch.tensor([[[True, False]]]))
        self.assertAlmostEqual(loss.item(), .5 * -np.log(9 / 20) / 2, places=6)
        self.assertEqual(info["coverage"], .5)
        empty, _ = prompt_routed_loss(logits, logits.detach().softmax(1), t, torch.zeros_like(t), t)
        empty.backward()
        self.assertEqual(empty.item(), 0)

    def test_mc_requires_seven_votes_and_class_specific_confidence(self):
        samples = torch.zeros(8, 1, 4, 1, 3)
        samples[:, :, 1, :, :] = .99
        samples[:, :, 0, :, :] = .01
        samples[-2:, :, 0, :, 1], samples[-2:, :, 1, :, 1] = .51, .49
        samples[:, :, 0, :, 2], samples[:, :, 1, :, 2] = .94, .06
        mean, tissue, trusted, votes = mc_tissue_targets(samples)
        self.assertEqual(votes.tolist(), [[[[8, 6, 8]]]][0])
        self.assertEqual(trusted.tolist(), [[[True, False, False]]])
        self.assertEqual(tissue.tolist(), [[[1, 1, 0]]])

    def test_cutmix_shares_donor_with_auxiliary_targets(self):
        ids = torch.arange(4.).view(4, 1, 1, 1).expand(4, 1, 32, 32)
        packed = torch.cat((ids.expand(-1, 20, -1, -1), ids.expand(-1, 3, -1, -1)), 1)
        images, targets = aligned_cutmix(ids.expand(-1, 3, -1, -1), packed, 46, 1.)
        for channel in range(23):
            self.assertTrue(torch.equal(images[:, 0], targets[:, channel]))

    def test_cache_identity_geometry_and_no_unlabeled_mask_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = make_spider_fixture(Path(tmp))
            cfg = small_config(png, method="mean_teacher")
            original, _, split = build_training_data(cfg)
            cache = fake_cache(Path(tmp) / "cache", split)
            guided, _, _ = build_training_data(replace(cfg, prompt_source="text", prompt_cache=str(cache)))
            for n in split["unlabeled_files"]:
                (png / "train/label" / n).unlink()
            for epoch in (0, 1, 9):
                original.set_epoch(epoch)
                guided.set_epoch(epoch)
                for i in range(len(original)):
                    plain, sample = original[i], guided[i]
                    self.assertTrue(torch.equal(plain["image"], sample["image"]))
                    self.assertTrue(torch.equal(plain["label"], sample["label"]))
                    if i >= cfg.labelnum:
                        raw = guided.prompt_cache.read(guided.names[i])
                        _, transformed_label = guided._geometry(torch.zeros(1, 32, 32), raw[0].long(), i)
                        self.assertTrue(torch.equal(sample["prompt_targets"][0].long(), transformed_label))
                        _, valid = guided._geometry(torch.zeros(1, 32, 32), torch.ones(32, 32).long(), i)
                        self.assertTrue(torch.equal(sample["prompt_targets"][2].bool(), valid.bool()))
            with self.assertRaises(ValueError):
                PromptCache(cache, "neutral", 32, split)
            first = next(cache.glob("*.npz"))
            first.write_bytes(first.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                PromptCache(cache, "text", 32, split)

    def test_active_prompt_updates_student_and_ema_without_changing_model(self):
        cfg = small_config(method="mean_teacher", prompt_source="text", prompt_cache="/not-needed", confidence=.99)
        model = StudentTeacherSystem(cfg)
        plain = StudentTeacherSystem(replace(cfg, prompt_source="none", prompt_cache=""))
        self.assertSameState(model, plain)
        batch = synthetic_batch()
        batch["prompt_targets"] = torch.ones(4, 3, 32, 32)
        before = {k: v.clone() for k, v in model.teachers[0].state_dict().items()}
        metrics = train_step(model, make_optimizer(model, cfg), batch, cfg, torch.device("cpu"), 0)
        self.assertGreater(metrics["tissue_coverage_s1"], 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.teachers[0].state_dict().items()))
        self.assertEqual(len(model.students), 1)

    def test_resume_and_inference_do_not_require_foundation_model_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = make_spider_fixture(Path(tmp))
            base = small_config(png, method="mean_teacher", max_steps=3, eval_every=3)
            _, _, split = build_training_data(base)
            cache = fake_cache(Path(tmp) / "cache", split)
            cfg = replace(base, prompt_source="text", prompt_cache=str(cache), output_dir=str(Path(tmp) / "whole"))
            run_experiment(cfg, "cpu")
            second = replace(cfg, output_dir=str(Path(tmp) / "resumed"))
            run_experiment(second, "cpu", stop_after=1)
            run_experiment(second, "cpu", resume=Path(second.output_dir) / "last.pt")
            a = torch.load(Path(cfg.output_dir) / "last.pt", weights_only=True)
            b = torch.load(Path(second.output_dir) / "last.pt", weights_only=True)
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
            cache.rename(Path(tmp) / "removed-cache")
            with mock.patch("freqssl.prompt_supervision.PromptCache", side_effect=AssertionError("Inference touched cache")):
                loaded = load_single_network(Path(cfg.output_dir) / "last.pt", torch.device("cpu"))
            self.assertIsNotNone(loaded)

    def test_cache_export_uses_only_authorized_annotations(self):
        from prompt_teacher.run import export_cache, protocol
        from prompt_teacher.data import supervision_fingerprint
        class FakeTeacher:
            prompts = TEXT_PROMPTS
            def load_adaptation(self, state):
                pass
            def eval(self):
                return self
            def predict_mc(self, images, seed):
                q = torch.zeros(len(images), 4, 224, 224)
                q[:, 1] = 1
                labels = torch.ones(len(images), 224, 224, dtype=torch.long)
                return q, labels, labels.bool(), labels * 8
        with tempfile.TemporaryDirectory() as tmp:
            png = make_spider_fixture(Path(tmp))
            cfg = small_config(png, method="mean_teacher", image_size=224)
            _, _, split = build_training_data(cfg)
            teacher_dir = Path(tmp) / "teacher"
            teacher_dir.mkdir()
            foundation = teacher_dir / "foundation.bin"
            foundation.write_bytes(b"unit-test-only")
            metadata = {"checkpoint": str(foundation), "mean": [0., 0., 0.], "std": [1., 1., 1.]}
            (teacher_dir / "split.json").write_text(json.dumps(split))
            checkpoint = teacher_dir / "best.pt"
            torch.save({"recipe": protocol(cfg, "text"), "adaptation": {}, "step": 500,
                "provenance": {"foundation_sha256": file_sha256(foundation), **supervision_fingerprint(png, split)}}, checkpoint)
            for name in split["unlabeled_files"]:
                (png / "train/label" / name).unlink()
            with mock.patch("prompt_teacher.run.build_teacher", return_value=(FakeTeacher(), metadata)), \
                 mock.patch("torch.cuda.reset_peak_memory_stats"), mock.patch("torch.cuda.max_memory_allocated", return_value=0):
                export_cache(cfg, "text", checkpoint, Path(tmp) / "cache", torch.device("cpu"))
            cache = PromptCache(Path(tmp) / "cache", "text", 224, split)
            self.assertEqual(set(cache.entries), set(split["unlabeled_files"]))
            self.assertTrue(cache.read(split["unlabeled_files"][0])[1:].bool().all())
