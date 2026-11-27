"""Long-training epoch boundaries preserve schedule, optimizer and old snapshots."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import torch

from experiments.train_long_v3 import preserve_epoch, resolve_long_config, run_long, sha256
from freqssl.engine import load_model, run_experiment, state_hash
from test_freqssl import CPUInvariantTest, make_spider_fixture, small_config


class LongTrainingTests(CPUInvariantTest):
    def test_epoch_budget_uses_actual_sampler_and_validates_milestones(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, labelnum=4, max_samples=8)
            resolved, steps, _ = resolve_long_config(cfg, root / "long", (100, 200), 5)
            self.assertEqual(steps, 2)
            self.assertEqual(resolved.max_steps, 400)
            self.assertEqual(resolved.eval_every, 10)
            self.assertEqual(resolved.warmup_steps, cfg.warmup_steps)
            self.assertEqual(resolved.base_lr, cfg.base_lr)
            for epochs, interval in (((2, 1), 1), ((1, 1), 1), ((0, 2), 1), ((1, 2), 2)):
                with self.assertRaises(ValueError):
                    resolve_long_config(cfg, root / "invalid", epochs, interval)

    def check_exact_staged_training(self, device):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, labelnum=4, max_samples=8,
                               pseudo_dice_weight=0.1, region_weight=1.0)
            resolved, _, _ = resolve_long_config(cfg, root / "long", (1, 2), 1)
            direct_cfg = replace(resolved, output_dir=str(root / "direct"))
            direct = run_experiment(direct_cfg, torch.device(device))
            staged = run_long(cfg, device, root / "long", (1, 2), 1)
            self.assertEqual(staged["status"], "completed")
            final, _, _ = load_model(root / "long/training/last.pt", torch.device("cpu"))
            self.assertEqual(state_hash(final.students[0]), direct["final_student1_hash"])
            self.assertEqual(state_hash(final.teachers[0]), direct["final_teacher1_hash"])
            direct_log = [json.loads(line) for line in (root / "direct/metrics.jsonl").read_text().splitlines()]
            staged_log = [json.loads(line) for line in (root / "long/training/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(direct_log, staged_log)
            first = torch.load(root / "long/epochs/0001/at_epoch.pt", map_location="cpu", weights_only=True)
            self.assertEqual(first["step"], 2)
            self.assertEqual(first["config"]["max_steps"], 4)
            last_first_half = [row for row in staged_log if row["kind"] == "train" and row["step"] == 2][0]
            first_second_half = [row for row in staged_log if row["kind"] == "train" and row["step"] == 3][0]
            self.assertLess(first_second_half["lr"], last_first_half["lr"])
            self.assertEqual(len(staged["milestones"]), 2)
            for milestone in staged["milestones"].values():
                for record in milestone["checkpoints"].values():
                    self.assertEqual(sha256(record["path"]), record["sha256"])

    def test_cpu_two_stage_training_matches_uninterrupted_updates_and_lr(self):
        self.check_exact_staged_training("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for staged GPU control")
    def test_cuda_two_stage_training_matches_uninterrupted_updates_and_lr(self):
        self.check_exact_staged_training("cuda:0")

    def test_interruption_after_snapshot_recovers_and_later_best_cannot_replace_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = make_spider_fixture(root / "data")
            cfg = small_config(png, labelnum=4, max_samples=8)
            directory = root / "long"

            def save_then_interrupt(*args, **kwargs):
                preserve_epoch(*args, **kwargs)
                raise RuntimeError("simulated interruption after snapshot")

            validation = [{"teacher1": {"overall": {"dice": 0.25}}},
                          {"teacher1": {"overall": {"dice": 0.75}}}]
            with mock.patch("freqssl.engine.evaluate_volumes", side_effect=validation):
                with mock.patch("experiments.train_long_v3.preserve_epoch", side_effect=save_then_interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run_long(cfg, "cpu", directory, (1, 2), 1)
                first_path = directory / "epochs/0001/best_through_epoch.pt"
                digest = sha256(first_path)
                completed = run_long(cfg, "cpu", directory, (1, 2), 1, resume=True)
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("error", completed)
            self.assertEqual(sha256(first_path), digest)
            self.assertEqual(completed["milestones"]["1"]["best_valid_dice_through_epoch"], 0.25)
            self.assertEqual(completed["milestones"]["2"]["best_valid_dice_through_epoch"], 0.75)
            with self.assertRaises(FileExistsError):
                run_long(cfg, "cpu", directory, (1, 2), 1)
            with self.assertRaises(ValueError):
                run_long(replace(cfg, base_lr=cfg.base_lr*2), "cpu", directory, (1, 2), 1, resume=True)


if __name__ == "__main__":
    unittest.main()
