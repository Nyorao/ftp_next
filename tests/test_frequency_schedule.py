"""Scheduling/loss checks only: no model training or optimizer updates."""

from dataclasses import replace
import unittest

import torch

from config import Config
from experiments.frequency_schedule import FeedbackReadiness, training_budget
from losses import mean_teacher_loss


class FrequencyScheduleTests(unittest.TestCase):
    def test_full_200_epochs_are_not_cut_to_old_step_budget(self):
        self.assertEqual(training_budget(1496, 2, 200), (748, 149600))
        self.assertEqual(training_budget(1496, 4, 200), (374, 74800))
        self.assertEqual(training_budget(1496, 2, 200, 2), (748, 2))

    def test_both_students_must_pass_after_warmup(self):
        gate = FeedbackReadiness(50, 0.7, 2)
        self.assertFalse(gate.observe(45, 0.95, 0.95))
        self.assertFalse(gate.observe(50, 0.80, 0.69))
        self.assertFalse(gate.observe(55, 0.80, 0.71))
        self.assertTrue(gate.observe(60, 0.80, 0.72))
        self.assertEqual(gate.activated_epoch, 60)

    def test_failed_or_missing_score_resets_consecutive_checks(self):
        gate = FeedbackReadiness(50, 0.7, 2)
        self.assertFalse(gate.observe(50, 0.8, 0.8))
        self.assertFalse(gate.observe(55, None, 0.8))
        self.assertFalse(gate.observe(60, 0.8, 0.8))
        self.assertFalse(gate.observe(65, float('nan'), 0.8))
        self.assertFalse(gate.observe(70, 0.8, 0.8))
        self.assertTrue(gate.observe(75, 0.8, 0.8))

    def test_loss_never_enables_feedback_from_epoch_alone(self):
        cfg = Config(root_path='unused', bert_type='unused', text_emb_path='unused',
                     num_classes=4, architecture='dual', use_apgm=False, use_psgm=False,
                     feedback_weight=0.05, feedback_require_ready=True,
                     feedback_start_epoch=50, feedback_rampup_epochs=20)
        first = torch.zeros(2, 4, 3, 3)
        second = first.clone()
        second[:, 1] = 8
        student = {'lf_prompt': first, 'hf_prompt': second}
        teacher = {'lf_prompt': torch.zeros(1, 4, 3, 3)}
        labels = torch.zeros(2, 3, 3, dtype=torch.long)
        mask = torch.tensor([True, False])
        def metrics(epoch, **kwargs):
            return mean_teacher_loss(student, teacher, labels, mask, epoch, cfg, **kwargs)[1]
        self.assertEqual(metrics(199)['feedback_weight'], 0)
        self.assertEqual(metrics(199, feedback_ready=True)['feedback_weight'], 0)
        self.assertEqual(metrics(49, feedback_ready=True, feedback_activated_epoch=50)['feedback_weight'], 0)
        first_ready = metrics(55, feedback_ready=True, feedback_activated_epoch=55)
        self.assertGreater(first_ready['feedback_weight'], 0)
        self.assertLess(first_ready['feedback_weight'], cfg.feedback_weight)
        self.assertEqual(metrics(75, feedback_ready=True, feedback_activated_epoch=55)['feedback_weight'], 0.05)


if __name__ == '__main__':
    unittest.main()
