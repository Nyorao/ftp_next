import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from losses import feedback_loss
from model.dual_student import SingleTeacherStudents
from test_mean_teacher import tiny_batch, tiny_config
from trainer import (load_checkpoint_model, prediction_scores, save_checkpoint,
                     seed_everything, student_parameters, train_batch)


def model(dual=True):
    torch.manual_seed(17)
    return SingleTeacherStudents(torch.randn(1, 8, 16), torch.randn(4, 32),
                                 base_channels=4, dual=dual, secondary_seed=43)


class DualStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_independent_parameters_and_identical_anchor_initialization(self):
        anchor = model(False)
        anchor_rng = torch.get_rng_state()
        dual = model(True)
        self.assertTrue(torch.equal(anchor_rng, torch.get_rng_state()))
        first = {p.data_ptr() for p in dual.student.parameters()}
        second = {p.data_ptr() for p in dual.student2.parameters()}
        teacher = {p.data_ptr() for p in dual.teacher.parameters()}
        self.assertFalse(first & second or first & teacher or second & teacher)
        for key, expected in anchor.student.state_dict().items():
            torch.testing.assert_close(expected, dual.student.state_dict()[key], rtol=0, atol=0)
        self.assertIsNotNone(dual.student.apgm)
        self.assertIsNone(dual.student.psgm)
        self.assertIsNotNone(dual.student2.psgm)
        self.assertIsNone(dual.student2.apgm)

    def test_no_feedback_negative_control_matches_after_updates(self):
        anchor, dual = model(False), model(True)
        cfg = replace(tiny_config(), architecture="anchor")
        dual_cfg = replace(cfg, architecture="dual", feedback_weight=0)
        optimizers = [torch.optim.AdamW(student_parameters(net), lr=1e-3) for net in (anchor, dual)]
        batch = tiny_batch()
        for step in range(3):
            for net, optimizer, settings in zip((anchor, dual), optimizers, (cfg, dual_cfg)):
                torch.manual_seed(100 + step)
                train_batch(net, optimizer, batch, settings, torch.device("cpu"), epoch=10)
        for member in ("student", "teacher"):
            for key, expected in getattr(anchor, member).state_dict().items():
                torch.testing.assert_close(expected, getattr(dual, member).state_dict()[key], rtol=0, atol=0)
        self.assertTrue(any(p.grad is not None for p in dual.student2.parameters()))
        self.assertTrue(all(p.grad is None for p in dual.teacher.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not accessible")
    def test_cuda_strict_negative_control_matches_after_updates(self):
        cfg = replace(tiny_config(), architecture="anchor", deterministic=True)
        seed_everything(cfg)
        anchor, dual = model(False).cuda(), model(True).cuda()
        dual_cfg = replace(cfg, architecture="dual", feedback_weight=0)
        optimizers = [torch.optim.AdamW(student_parameters(net), lr=1e-3) for net in (anchor, dual)]
        batch = tiny_batch()
        try:
            for step in range(3):
                for net, optimizer, settings in zip((anchor, dual), optimizers, (cfg, dual_cfg)):
                    torch.manual_seed(100 + step)
                    train_batch(net, optimizer, batch, settings, torch.device("cuda:0"), epoch=10)
            for member in ("student", "teacher"):
                for key, expected in getattr(anchor, member).state_dict().items():
                    torch.testing.assert_close(expected, getattr(dual, member).state_dict()[key], rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(False)

    def test_feedback_reaches_only_primary(self):
        cfg = tiny_config()
        first = torch.zeros(1, 4, 3, 3, requires_grad=True)
        second = torch.zeros_like(first)
        second[:, 1] = 8
        second.requires_grad_(True)
        loss, metrics = feedback_loss(first, second, cfg)
        loss.backward()
        self.assertGreater(float(first.grad.abs().sum()), 0)
        self.assertIsNone(second.grad)
        self.assertEqual(metrics["feedback_pixels"], 9)
        self.assertEqual(metrics["feedback_coverage"], 1)

    def test_background_or_empty_gate_has_finite_zero_loss(self):
        cfg = tiny_config()
        first = torch.zeros(1, 4, 3, 3, requires_grad=True)
        second = torch.zeros_like(first)
        second[:, 0] = 8
        loss, metrics = feedback_loss(first, second, cfg)
        self.assertEqual(metrics["feedback_pixels"], 0)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(first.grad.abs().sum()), 0)

    def test_secondary_parameters_do_not_directly_update_teacher(self):
        dual = model(True)
        before = copy.deepcopy(dual.student.state_dict())
        with torch.no_grad():
            for parameter in dual.student2.parameters():
                parameter.add_(100)
        dual.update_teacher()
        for key, expected in before.items():
            torch.testing.assert_close(expected, dual.teacher.state_dict()[key], rtol=0, atol=0)

    def test_dual_checkpoint_restores_all_deployment_outputs(self):
        net = model(True).eval()
        cfg = replace(tiny_config(), architecture="dual", feedback_weight=0.25)
        optimizer = torch.optim.AdamW(student_parameters(net))
        low, high = torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "dual.pt"
            save_checkpoint(file, net, optimizer, cfg, 0, 0, -1)
            restored, settings = load_checkpoint_model(file, torch.device("cpu"))
            restored.eval()
            with torch.no_grad():
                for name in ("student1", "student2", "teacher", "students_mean"):
                    expected, route = prediction_scores(net, low, high, cfg, name)
                    actual, restored_route = prediction_scores(restored, low, high, settings, name)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(route, restored_route)
            self.assertFalse(restored.teacher.training)
            self.assertTrue(all(not p.requires_grad for p in restored.teacher.parameters()))

    def test_invalid_feedback_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(tiny_config(), architecture="anchor", feedback_weight=0.1).validate()


if __name__ == "__main__":
    unittest.main()
