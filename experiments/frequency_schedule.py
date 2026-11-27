"""Pure scheduling rules; no training, GPU allocation, or data access."""

from dataclasses import asdict, dataclass
import math


def training_budget(labeled_samples, labeled_batch, epochs, step_cap=None):
    if min(labeled_samples, labeled_batch, epochs) < 1 or (step_cap is not None and step_cap < 1):
        raise ValueError("Training budget values must be positive")
    steps_per_epoch = labeled_samples // labeled_batch
    if not steps_per_epoch:
        raise ValueError("No complete labeled batch")
    total = epochs * steps_per_epoch
    return steps_per_epoch, min(total, step_cap) if step_cap is not None else total


@dataclass
class FeedbackReadiness:
    minimum_epoch: float
    minimum_dice: float
    required_checks: int
    consecutive_checks: int = 0
    activated_epoch: float | None = None
    last_student_dice: list | None = None

    def observe(self, completed_epochs, student1_dice, student2_dice):
        self.last_student_dice = [student1_dice, student2_dice]
        if self.activated_epoch is not None:
            return True
        scores = self.last_student_dice
        qualifies = completed_epochs >= self.minimum_epoch and all(
            v is not None and math.isfinite(v) and self.minimum_dice <= v <= 1 for v in scores)
        self.consecutive_checks = self.consecutive_checks + 1 if qualifies else 0
        if self.consecutive_checks >= self.required_checks:
            self.activated_epoch = float(completed_epochs)
        return self.activated_epoch is not None

    def state(self):
        return asdict(self)
