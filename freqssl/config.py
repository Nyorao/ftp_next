"""One explicit, step-based protocol for every comparison arm."""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import yaml


METHODS = ("supervised", "mean_teacher", "dual_independent", "dual_cps", "dual_frequency", "dual_ordinal")


@dataclass
class ExperimentConfig:
    root_path: str
    output_dir: str = "outputs/freqssl_v3"
    schema_version: int = 3
    method: str = "dual_frequency"
    slice_context: str = "none"
    case_reference: str = "none"
    context_position: str = "none"
    context_geometry_manifest: str = ""
    reference_condition: str = "none"
    reference_sampling: str = "random"
    reference_gt: bool = False
    reference_cycle_weight: float = 0.0
    reference_teacher: str = ""
    reference_teacher_sha256: str = ""
    reference_cache: str = ""
    reference_windows: tuple[int, int] = (2, 1)
    prompt_source: str = "none"
    prompt_cache: str = ""
    prompt_weight: float = 0.5
    prompt_route: str = "legacy"
    prompt_gate: str = ""
    internal_text: str = "none"
    text_bank: str = ""
    text_adapter_dim: int = 64
    text_fusion_layout: str = "e4"
    text_layer_mode: str = "grouped"
    text_feature_kind: str = "none"
    encoder_text_stage: int | None = None
    encoder_text_stages: tuple[int, ...] = ()
    text_routing_mode: str = "independent"
    text_residual_mode: str = "plain"
    text_alignment_weight: float = 0.0
    text_alignment_balance: str = "pixel"
    training_split: str = ""
    checkpoint_every: int = 0
    num_classes: int = 20
    image_size: int = 224
    base_channels: int = 32
    max_channels: int = 256
    labelnum: int = 1496
    max_samples: int = 3732
    batch_size: int = 8
    labeled_bs: int = 4
    num_workers: int = 0
    seed: int = 42
    max_steps: int = 30000
    eval_every: int = 1000
    log_every: int = 25
    base_lr: float = 0.0003
    weight_decay: float = 0.0001
    ema_decay: float = 0.99
    warmup_steps: int = 1500
    ramp_steps: int = 3000
    unsupervised_weight: float = 1.0
    confidence: float = 0.80
    background_confidence: float = 0.95
    background_weight: float = 0.25
    pseudo_dice_weight: float = 0.0
    region_weight: float = 0.0
    region_grids: tuple[int, ...] = (4, 8, 16)
    ordinal_tissue_weight: float = 0.25
    ordinal_chain_threshold: float = 0.5
    ordinal_interval_mass: float = 0.8
    ordinal_max_interval: int = 3
    ordinal_bins: int = 56
    ordinal_use_chain: bool = True
    ordinal_use_routing: bool = True
    frequency_strength: float = 0.5
    noise_std: float = 0.08
    gain_std: float = 0.15
    cutmix_probability: float = 0.5
    grad_clip: float = 12.0
    amp: bool = False
    deterministic: bool = True
    cpu_threads: int = 4
    eval_batch_size: int = 8
    max_seconds: float = 86400.0

    def __post_init__(self):
        self.region_grids = tuple(self.region_grids)
        self.encoder_text_stages = tuple(self.encoder_text_stages)
        self.reference_windows = tuple(self.reference_windows)

    def validate(self):
        if self.schema_version != 3 or self.method not in METHODS:
            raise ValueError("Expected schema_version=3 and a supported freqssl method")
        if self.slice_context not in ("none", "mean", "adaptive"):
            raise ValueError("slice_context must be none, mean or adaptive")
        if self.slice_context != "none" and self.method not in ("mean_teacher", "supervised"):
            raise ValueError("Slice context currently supports only single-student methods")
        if self.case_reference not in ("none", "middle") or self.context_position not in ("none", "relative"):
            raise ValueError("Unknown case reference or context position")
        if self.case_reference != "none" and self.context_position != "none":
            raise ValueError("Reference and position are separate experimental arms")
        if self.case_reference != "none" or self.context_position != "none":
            if (self.slice_context != "adaptive" or self.method != "mean_teacher" or
                    self.internal_text != "none" or self.prompt_source != "none" or
                    self.pseudo_dice_weight or self.region_weight):
                raise ValueError("Case context requires plain adaptive Mean Teacher")
        if self.context_position != "none" and not self.context_geometry_manifest:
            raise ValueError("Relative position requires a verified image geometry manifest")
        if self.reference_condition not in ("none", "row", "kv", "position"):
            raise ValueError("Unknown reference_condition")
        if self.reference_sampling not in ("random", "fixed") or type(self.reference_gt) is not bool:
            raise ValueError("Invalid reference sampling/GT policy")
        if (len(self.reference_windows) != 2 or
                any(type(w) is not int or w < 1 for w in self.reference_windows)):
            raise ValueError("Reference windows must be two positive integer radii")
        if not math.isfinite(self.reference_cycle_weight) or self.reference_cycle_weight < 0:
            raise ValueError("Invalid reference cycle weight")
        if self.reference_condition != "none":
            if (self.method != "mean_teacher" or self.slice_context != "adaptive" or
                    self.case_reference != "none" or self.context_position != "none" or
                    self.internal_text != "none" or self.prompt_source != "none" or
                    self.region_weight or self.pseudo_dice_weight or self.training_split):
                raise ValueError("Reference conditioning requires plain adaptive Mean Teacher")
            if not self.reference_teacher or not self.context_geometry_manifest:
                raise ValueError("Reference conditioning requires a frozen teacher and verified geometry")
            if self.reference_cycle_weight and (self.reference_condition != "kv" or self.reference_gt):
                raise ValueError("Cycle experiment uses prediction-only image-key KV")
        elif self.reference_gt or self.reference_cycle_weight or self.reference_cache or self.reference_teacher:
            raise ValueError("Reference options require an enabled reference condition")
        if self.internal_text not in ("none", "text", "neutral"):
            raise ValueError("internal_text must be none, text or neutral")
        if type(self.text_adapter_dim) is not int or self.text_adapter_dim < 1:
            raise ValueError("text_adapter_dim must be a positive integer")
        if self.text_fusion_layout not in ("e4", "hierarchical", "encoder", "encoder_multi") or self.text_layer_mode not in ("grouped", "last"):
            raise ValueError("Invalid text fusion layout/layer mode")
        if self.text_routing_mode not in ("independent", "aligned") or self.text_residual_mode not in ("plain", "difference"):
            raise ValueError("Invalid text routing/residual mode")
        if self.text_fusion_layout == "encoder_multi":
            if self.internal_text != "text" or self.text_layer_mode != "last":
                raise ValueError("Multi-point encoder requires fixed layer12 tissue text")
            if (self.encoder_text_stages not in ((), (2,), (3,), (2, 3))
                    or any(type(s) is not int for s in self.encoder_text_stages)):
                raise ValueError("Multi-point encoder supports (), (2,), (3,), (2,3)")
        elif self.encoder_text_stages or self.text_routing_mode != "independent" or self.text_residual_mode != "plain":
            raise ValueError("Multi-point text options require encoder_multi layout")
        if self.text_feature_kind not in ("none", "tokens", "global_cls"):
            raise ValueError("Unknown text feature kind")
        if self.text_fusion_layout == "encoder":
            if self.internal_text != "text" or self.text_layer_mode != "last":
                raise ValueError("Encoder screen requires fixed tissue text and layer 12")
            if ((self.text_feature_kind == "none") != (self.encoder_text_stage is None)
                    or (self.encoder_text_stage is not None and
                        (type(self.encoder_text_stage) is not int or self.encoder_text_stage not in (2, 3, 4)))):
                raise ValueError("Encoder fusion selects E2/E3/E4; alignment-only selects none")
        elif self.text_feature_kind != "none" or self.encoder_text_stage is not None:
            raise ValueError("Encoder text options require encoder fusion layout")
        if not math.isfinite(self.text_alignment_weight) or self.text_alignment_weight < 0:
            raise ValueError("text_alignment_weight must be finite and nonnegative")
        if self.text_alignment_balance not in ("pixel", "class"):
            raise ValueError("Unknown tissue alignment balance")
        if self.text_fusion_layout == "hierarchical" and self.internal_text == "none":
            raise ValueError("Hierarchical fusion requires text inputs")
        if self.text_alignment_weight and self.text_fusion_layout not in ("hierarchical", "encoder", "encoder_multi"):
            raise ValueError("Tissue alignment requires hierarchical or encoder fusion")
        if self.internal_text != "none":
            if self.method != "mean_teacher" or self.slice_context != "adaptive":
                raise ValueError("Internal text screening requires adaptive context Mean Teacher")
            if self.prompt_source != "none" or self.pseudo_dice_weight or self.region_weight:
                raise ValueError("Internal text screening preserves the original MT losses")
            if not isinstance(self.text_bank, str) or not self.text_bank:
                raise ValueError("Internal text requires a text_bank path for initialization")
            # Checkpoint inference/resume restores its own embedded vectors.
        elif self.text_bank:
            raise ValueError("text_bank requires an enabled internal_text arm")
        if self.prompt_source not in ("none", "text", "neutral"):
            raise ValueError("prompt_source must be none, text or neutral")
        if self.prompt_route not in ("legacy", "preserve_ema", "competence"):
            raise ValueError("Unknown prompt_route")
        if self.prompt_source != "none" and self.prompt_route == "competence" and not self.prompt_gate:
            raise ValueError("Competence routing requires a frozen gate")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be nonnegative")
        if not math.isfinite(self.prompt_weight) or self.prompt_weight < 0:
            raise ValueError("prompt_weight must be finite and nonnegative")
        if self.prompt_source != "none":
            if self.method != "mean_teacher" or self.pseudo_dice_weight or self.region_weight:
                raise ValueError("Prompt supervision requires single Mean Teacher without auxiliary losses")
            if not isinstance(self.prompt_cache, str) or not self.prompt_cache:
                raise ValueError("Prompt supervision requires a cache location")
            # Existence is checked by the training data loader, never by inference.
        for name in ("num_classes", "base_channels", "max_channels", "max_steps", "eval_every",
                     "log_every", "cpu_threads", "eval_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_classes != 20:
            raise ValueError("The SPIDER volume metric protocol requires num_classes=20")
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size must be >=32 and divisible by 16")
        if not 0 < self.labeled_bs < self.batch_size:
            raise ValueError("Require 0 < labeled_bs < batch_size")
        if not self.labeled_bs <= self.labelnum < self.max_samples:
            raise ValueError("Invalid labeled/unlabeled sample budget")
        if self.max_samples - self.labelnum < self.batch_size - self.labeled_bs:
            raise ValueError("Unlabeled pool is smaller than one batch")
        if self.base_channels < 2 or self.max_channels < self.base_channels or self.num_workers < 0 or self.seed < 0:
            raise ValueError("Invalid channels or worker count")
        if self.amp:
            raise ValueError("The paired v3 protocol uses float32; amp must be false to preserve exact negative controls")
        for name in ("base_lr", "grad_clip", "max_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "unsupervised_weight", "noise_std", "gain_std",
                     "frequency_strength", "warmup_steps", "ramp_steps",
                     "pseudo_dice_weight", "region_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("confidence", "background_confidence", "background_weight", "cutmix_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.frequency_strength > 1:
            raise ValueError("frequency_strength must be <=1 to bound detail amplification")
        if (not self.region_grids or len(set(self.region_grids)) != len(self.region_grids)
                or any(type(g) is not int or g < 2 or g > 32 or self.image_size % g
                       for g in self.region_grids)):
            raise ValueError("region_grids must be unique divisors of image_size between 2 and 32")
        if self.method == "supervised" and (self.pseudo_dice_weight or self.region_weight):
            raise ValueError("Unlabeled auxiliary losses require a teacher-based method")
        if not math.isfinite(self.ordinal_tissue_weight) or self.ordinal_tissue_weight < 0:
            raise ValueError("ordinal_tissue_weight must be finite and nonnegative")
        for name in ("ordinal_chain_threshold", "ordinal_interval_mass"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if (type(self.ordinal_bins) is not int or self.ordinal_bins < 2
                or type(self.ordinal_max_interval) is not int or not 2 <= self.ordinal_max_interval <= 9
                or type(self.ordinal_use_chain) is not bool or type(self.ordinal_use_routing) is not bool):
            raise ValueError("Invalid ordinal routing configuration")
        if self.method == "dual_ordinal" and (self.pseudo_dice_weight or self.region_weight):
            raise ValueError("Ordinal routing has its own objective; legacy auxiliary weights must be zero")
        return self

    @property
    def primary_network(self):
        return "student1" if self.method == "supervised" else "teacher1"

    def to_dict(self):
        values = asdict(self)
        values["region_grids"] = list(self.region_grids)
        values["encoder_text_stages"] = list(self.encoder_text_stages)
        values["reference_windows"] = list(self.reference_windows)
        return values


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return ExperimentConfig(**values).validate()


def arm_config(cfg, method, output_dir=None, seed=None):
    """No arm-specific learning rates, batch sizes or iteration budgets."""
    return replace(cfg, method=method, output_dir=str(output_dir or cfg.output_dir),
                   seed=cfg.seed if seed is None else seed).validate()
