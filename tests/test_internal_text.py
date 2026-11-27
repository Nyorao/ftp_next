"""Internal-text controls, EMA, portable checkpoints and cached inference."""
from dataclasses import replace
from pathlib import Path
import tempfile
import json
import subprocess
import sys
import unittest

import torch
from torch.nn import functional as F
import yaml

from freqssl.context_cache import iter_cached_logits
from freqssl.engine import make_optimizer, run_experiment, train_step, seed_everything, load_model
from freqssl.inference import load_single_network
from freqssl.internal_text import TISSUES, load_text_bank, TissueTextAdapter
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture


def write_bank(directory, source="text"):
    generator = torch.Generator().manual_seed(81 if source == "text" else 82)
    path = Path(directory) / f"{source}.pt"
    torch.save({"format_version": 1, "source": source, "tissues": list(TISSUES),
        "prompts": list(TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS),
        "embeddings": F.normalize(torch.randn(4, 512, generator=generator), dim=-1),
        "provenance": {"source": "synthetic test vectors; no semantic evidence"}}, path)
    return path


class InternalTextTests(CPUInvariantTest):
    def test_config_rejects_confounded_modes_and_bank_mismatch(self):
        cfg = small_config(method="mean_teacher", slice_context="adaptive")
        for changes in (dict(internal_text="bad"), dict(text_bank="unused"),
                        dict(internal_text="text"), dict(text_adapter_dim=0),
                        dict(internal_text="text", text_bank="x", slice_context="none"),
                        dict(internal_text="text", text_bank="x", region_weight=.1)):
            with self.assertRaises(ValueError):
                replace(cfg, **changes).validate()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bank(tmp)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_text_bank(path, "neutral")
            payload = torch.load(path, weights_only=True)
            payload["embeddings"].zero_()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "unit-normalized"):
                load_text_bank(path, "text")

    def test_initial_backbone_rng_and_logits_match_and_pair_has_same_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_config(method="mean_teacher", slice_context="adaptive")
            base = StudentTeacherSystem(cfg)
            states = []
            images = torch.randn(2, 3, 32, 48)
            for source in ("text", "neutral"):
                path = write_bank(tmp, source)
                before = torch.get_rng_state().clone()
                model = StudentTeacherSystem(replace(cfg, internal_text=source, text_bank=str(path)))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                for key, value in base.students[0].state_dict().items():
                    self.assertTrue(torch.equal(value, model.students[0].state_dict()[key]), key)
                torch.testing.assert_close(model.student_logits(images, 0), base.student_logits(images, 0), rtol=0, atol=0)
                self.assertFalse(model.students[0].text_adapter.embeddings.requires_grad)
                states.append(dict(model.students[0].named_parameters()))
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_real_gradients_ema_and_fixed_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_config(method="mean_teacher", slice_context="adaptive",
                internal_text="text", text_bank=str(write_bank(tmp)), text_adapter_dim=8)
            seed_everything(cfg)
            model = StudentTeacherSystem(cfg)
            before = model.students[0].text_adapter.embeddings.clone()
            optimizer = make_optimizer(model, cfg)
            batch = synthetic_batch()
            batch["image"] = torch.randn(4, 3, 32, 32)
            for step in range(3):
                metrics = train_step(model, optimizer, batch, cfg, torch.device("cpu"), step)
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
                if step == 0:
                    self.assertSameState(model.students[0], model.teachers[0])
            adapter = model.students[0].text_adapter
            self.assertGreater(abs(float(adapter.gate)), 0)
            for layer in (adapter.query, adapter.key, adapter.value, adapter.output):
                self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.teachers.parameters()))
            self.assertTrue(torch.equal(before, adapter.embeddings))
            self.assertTrue(torch.equal(before, model.teachers[0].text_adapter.embeddings))

    def test_token_set_permutation_invariance_and_text_content_sensitivity(self):
        embeddings = F.normalize(torch.randn(4, 512), dim=-1)
        layer = TissueTextAdapter(16, 8, embeddings)
        with torch.no_grad():
            layer.gate.fill_(.5)
            x = torch.randn(2, 16, 3, 5)
            expected = layer(x)
            layer.embeddings.copy_(embeddings[[3, 2, 0, 1]])
            torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)
            layer.embeddings.copy_(F.normalize(torch.randn(4, 512), dim=-1))
            self.assertGreater(float((layer(x) - expected).abs().max()), 1e-4)
            weights, _ = layer.attention(x)
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, 15))

    def test_cache_nonzero_text_and_context_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_config(method="mean_teacher", slice_context="adaptive",
                internal_text="text", text_bank=str(write_bank(tmp)), text_adapter_dim=8)
            model = StudentTeacherSystem(cfg).students[0].eval()
            with torch.no_grad():
                model.text_adapter.gate.fill_(.7)
                for fusion in model.context_fusion:
                    fusion.gamma.fill_(.5)
                    fusion.weights.bias.copy_(torch.tensor([2., 0., -1.]))
                for count in (1, 2, 9):
                    images = torch.randn(count, 1, 32, 48)
                    triplets = torch.stack([images[[max(0, i-1), i, min(count-1, i+1)], 0] for i in range(count)])
                    expected = model(triplets)
                    for batch_size in (1, 3, 8):
                        reads = []
                        def read(i):
                            reads.append(i)
                            return images[i]
                        actual = torch.cat([x for _, x in iter_cached_logits(model, count, read, "cpu", batch_size)])
                        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
                        self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))
                        self.assertEqual(reads, list(range(count)))

    def test_checkpoint_resume_and_inference_do_not_need_external_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = write_bank(root)
            data = make_spider_fixture(root / "data")
            cfg = small_config(data, root / "full", method="mean_teacher", slice_context="adaptive",
                internal_text="text", text_bank=str(bank), text_adapter_dim=8,
                labelnum=4, max_samples=8, max_steps=3, eval_every=3)
            full = run_experiment(cfg, "cpu")
            staged = replace(cfg, output_dir=str(root / "staged"))
            run_experiment(staged, "cpu", stop_after=1)
            bank.unlink()
            resumed = run_experiment(staged, "cpu", resume=root / "staged/last.pt")
            for key in ("final_student1_hash", "final_teacher1_hash", "final_validation"):
                self.assertEqual(full[key], resumed[key])
            single, _, receipt = load_single_network(root / "staged/best.pt", "cpu")
            system, _, _ = load_model(root / "staged/best.pt", "cpu")
            system.eval()
            x = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(single(x), system.predict_proba(x, "teacher1"), rtol=0, atol=0)
            self.assertEqual(receipt["internal_text"], "text")
            self.assertFalse(receipt["text_encoder_loaded"])
            self.assertEqual(receipt["backbones_loaded"], 1)
            self.assertEqual(receipt["text_provenance"], system.text_provenance)

    def test_unrestored_placeholder_cannot_silently_predict(self):
        with self.assertRaisesRegex(ValueError, "not been initialized"):
            TissueTextAdapter(16)(torch.randn(1, 16, 2, 2))

    def test_screen_rejects_capacity_only_gain_and_union_regression(self):
        from experiments.run_internal_text import screening_decision
        base = {"dice": .80, "dice_union_present": .78}
        neutral = {"dice": .804, "dice_union_present": .782}
        self.assertFalse(screening_decision(base, neutral,
            {"dice": .8045, "dice_union_present": .783})["passed_single_seed_screen"])
        self.assertFalse(screening_decision(base, neutral,
            {"dice": .807, "dice_union_present": .781})["passed_single_seed_screen"])
        self.assertTrue(screening_decision(base, neutral,
            {"dice": .807, "dice_union_present": .783})["passed_single_seed_screen"])

    def test_frozen_three_arm_runner_completes_on_synthetic_fixture(self):
        from experiments.run_internal_text import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            banks = root / "banks"
            banks.mkdir()
            write_bank(banks, "text")
            write_bank(banks, "neutral")
            data = make_spider_fixture(root / "data")
            cfg = small_config(data, method="mean_teacher", slice_context="adaptive",
                labelnum=4, max_samples=8, max_steps=3, eval_every=3)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg.to_dict()))
            output = root / "screen"
            prepare(config, banks, output)
            # External bank files are no longer necessary after preparation.
            for path in banks.iterdir():
                path.unlink()
            command = [sys.executable, str(output / "code/experiments/run_internal_text.py"),
                "--output", str(output), "--device", "cpu", "--execute-frozen"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(set(result["arms"]), {"context", "text", "neutral"})
            self.assertFalse(result["test_evaluated"])
            self.assertFalse(result["decision"]["confirmatory_efficacy_claim"])


if __name__ == "__main__":
    unittest.main()
