from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    root_path: str
    bert_type: str
    text_emb_path: str
    output_dir: str = "./outputs/ftp_mt"
    num_classes: int = 20
    text_length: int = 112
    image_size: int = 224
    base_channels: int = 32
    architecture: str = "shared"
    use_apgm: bool = True
    use_psgm: bool = True
    use_frequency: bool = True
    secondary_view: str = "hf"
    batch_size: int = 12
    labeled_bs: int = 6
    labelnum: int = 1496
    max_samples: int = 3732
    num_workers: int = 4
    max_epochs: int = 200
    max_iterations: int = 90000
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    ema_decay: float = 0.99
    consistency_weight: float = 1.0
    consistency_rampup_epochs: float = 40
    feedback_weight: float = 0.0
    feedback_start_epoch: float = 5.0
    feedback_rampup_epochs: float = 20.0
    feedback_confidence: float = 0.7
    feedback_margin: float = 0.05
    feedback_foreground_only: bool = True
    feedback_require_ready: bool = False
    feedback_min_dice: float = 0.70
    feedback_ready_checks: int = 2
    student_noise_std: float = 0.1
    teacher_noise_std: float = 0.05
    student_gain_std: float = 0.1
    eval_interval: int = 10
    eval_branch: str = "lf"
    seed: int = 42
    deterministic: bool = True
    cpu_threads: int = 4

    def validate(self):
        for name in ("num_classes", "text_length", "base_channels", "max_epochs",
                     "max_iterations", "eval_interval", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.labeled_bs < self.batch_size:
            raise ValueError("Require 0 < labeled_bs < batch_size for semi-supervised training")
        if self.image_size < 64:
            raise ValueError("image_size must be >=64 for the /32 bottleneck")
        if not 0 < self.labelnum < self.max_samples:
            raise ValueError("Require 0 < labelnum < max_samples")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.architecture not in ("shared", "anchor", "dual"):
            raise ValueError("architecture must be shared, anchor or dual")
        if self.secondary_view not in ("lf", "hf"):
            raise ValueError("secondary_view must be lf or hf")
        if self.feedback_weight and self.architecture != "dual":
            raise ValueError("Student feedback requires architecture=dual")
        if not 0 <= self.feedback_confidence <= 1 or not 0 <= self.feedback_margin <= 1:
            raise ValueError("feedback_confidence and feedback_margin must be in [0,1]")
        if not 0 <= self.feedback_min_dice <= 1 or self.feedback_ready_checks < 1:
            raise ValueError("feedback_min_dice must be in [0,1] and feedback_ready_checks positive")
        if self.eval_branch not in ("lf", "hf", "mean"):
            raise ValueError("eval_branch must be lf, hf or mean")
        for name in ("num_workers", "weight_decay", "consistency_weight",
                     "consistency_rampup_epochs", "student_noise_std",
                     "teacher_noise_std", "student_gain_std", "feedback_weight",
                     "feedback_start_epoch", "feedback_rampup_epochs"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.base_lr <= 0:
            raise ValueError("base_lr must be positive")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open() as handle:
        return Config(**yaml.safe_load(handle))
