"""Prompt revision protocol and training-only routing; no VLM dependencies."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .anatomy import FAMILY_IDS
from .prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, file_sha256

TEMPLATES = (
    "{C} in a sagittal spine MRI.",
    "Segmentation of {C} in a sagittal MRI of the spine.",
    "The segmentation target is {C} in this sagittal spine MRI.",
    "A sagittal MRI of the spine, with {C} as the segmentation target.",
    "{C} shown in a sagittal MRI of the spine.",
    "In this sagittal spine magnetic resonance image, segment {C}.",
)
VARIANTS = ("T0", "T1", "T2")


def prompt_bank(source):
    if source not in ("text", "neutral"):
        raise ValueError("Unknown prompt source")
    names = (["background and other tissues", "vertebrae", "the spinal canal", "intervertebral discs"]
             if source == "text" else [f"region {c}" for c in "ABCD"])
    bank = [[template.format(C=name) for name in names] for template in TEMPLATES]
    bank = [[s[0].upper() + s[1:] for s in group] for group in bank]
    assert bank[0] == list(TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS)
    return bank


def case_id(name):
    return "_".join(name.split("_")[:2])


def make_partition(split):
    labeled, unlabeled = split["labeled_files"], split["unlabeled_files"]
    complete = set(map(case_id, labeled)) - set(map(case_id, unlabeled))
    cases = sorted(complete, key=lambda c: hashlib.sha256(("prompt_r2:42:" + c).encode()).hexdigest())[:14]
    calibration = [n for n in labeled if case_id(n) in cases]
    return {"format_version": 1, "seed": 42, "policy": "SHA256(prompt_r2:42:+case), first 14 fully authorized cases",
        "authorized_labeled_files": labeled, "authorized_unlabeled_files": unlabeled,
        "labeled_files": [n for n in labeled if n not in set(calibration)],
        "unlabeled_files": unlabeled, "calibration_cases": sorted(cases), "calibration_files": calibration}


def read_partition(path, authorized_labeled, authorized_unlabeled):
    value = json.loads(Path(path).read_text())
    expected = make_partition({"labeled_files": authorized_labeled, "unlabeled_files": authorized_unlabeled})
    if value != expected:
        raise ValueError("Calibration partition differs from the deterministic authorized split")
    return value


def tissue_probabilities(ema):
    return torch.stack([ema[:, [c for c, t in enumerate(FAMILY_IDS) if t == k]].sum(1) for k in range(4)], 1)


def competence_eligible(ema, tissue, trusted, valid, foreground_threshold=.8, background_threshold=.95):
    certainty, label = ema.max(1)
    accepted = certainty >= torch.where(label == 0, background_threshold, foreground_threshold)
    certainty4, label4 = tissue_probabilities(ema).max(1)
    accepted4 = certainty4 >= torch.where(label4 == 0, background_threshold, foreground_threshold)
    return trusted.bool() & valid.bool() & ~accepted & ~accepted4


def unpack_template_probabilities(packed):
    """B,17,H,W -> B,4,H,W hard target / reliability / valid / predicted edge.

    Called after shared geometry and before shared CutMix. Edges therefore remain
    within their source image instead of including artificial donor seams.
    """
    if packed.ndim != 4 or packed.shape[1] != 17:
        raise ValueError("Expected 16 template probabilities and one validity plane")
    q = packed[:, :16].reshape(len(packed), 4, 4, *packed.shape[-2:]).clamp_min(0)
    sums = q.sum(2, keepdim=True)
    valid = (packed[:, 16] >= .999) & (sums[:, :, 0].amin(1) > 0)
    q = q / sums.clamp_min(1e-8)
    mean = q.mean(1)
    confidence, target = mean.max(1)
    votes = (q.argmax(2) == target[:, None]).sum(1)
    trusted = valid & (votes >= 3) & (confidence >= torch.where(target == 0, .95, .8))
    image = target[:, None].float()
    edge = (F.max_pool2d(image, 5, 1, 2) != -F.max_pool2d(-image, 5, 1, 2))[:, 0]
    return torch.stack((target.float(), trusted.float(), valid.float(), edge.float()), 1)


def gate_stage(step):
    if not 1 <= step <= 6000:
        raise ValueError("Revision gate is frozen for steps 1..6000")
    return max(500, ((step - 1) // 500) * 500)


class CompetenceGate:
    def __init__(self, path, source):
        self.sha256 = file_sha256(path)
        self.data = json.loads(Path(path).read_text())
        if (self.data.get("format_version") != 1 or self.data.get("status") != "completed"
                or self.data.get("source") != source or self.data.get("variant") not in VARIANTS
                or self.data.get("prompt_bank") != prompt_bank(source)[:4]):
            raise ValueError("Invalid competence gate provenance")
        self.variant = self.data["variant"]
        self.tables = {}
        if set(self.data["tables"]) != {str(s) for s in range(500, 6000, 500)}:
            raise ValueError("Incomplete gate stages")
        for stage, values in self.data["tables"].items():
            a = np.asarray(values)
            if a.shape != (4, 2) or not np.isin(a, (0, 1)).all():
                raise ValueError("Invalid gate table")
            self.tables[int(stage)] = torch.tensor(a, dtype=torch.bool)

    def lookup(self, tissue, edge, step):
        if tissue.shape != edge.shape or torch.any((tissue < 0) | (tissue > 3) | (edge < 0) | (edge > 1)):
            raise ValueError("Invalid gate indices")
        if step <= 600:
            return torch.zeros_like(tissue, dtype=torch.bool)
        return self.tables[gate_stage(step)].to(tissue.device)[tissue, edge]


def qualify_cell(case_counts, tissue):
    """Counts per independent case: n, teacher-correct, EMA-correct."""
    active = np.asarray([r for r in case_counts if r[0] > 0], dtype=np.float64).reshape(-1, 3)
    result = {"cases": len(active), "pixels": int(active[:, 0].sum()), "enabled": False}
    if len(active) == 0:
        return result
    teacher, ema = active[:, 1] / active[:, 0], active[:, 2] / active[:, 0]
    delta = teacher - ema
    rng = np.random.default_rng(42)
    bounds = np.quantile(delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1), [.025, .975])
    result.update(teacher_accuracy=float(teacher.mean()), ema_accuracy=float(ema.mean()),
                  advantage=float(delta.mean()), case_bootstrap_interval=bounds.tolist())
    result["enabled"] = bool(len(active) >= 5 and result["pixels"] >= 1000
        and teacher.mean() >= (.99 if tissue == 0 else .95) and delta.mean() >= .02 and bounds[0] > 0)
    return result


def teacher_screen(control, candidate, gate):
    """Predeclared practical gates, not statistical confirmation."""
    reference, scores = control["canonical"], candidate["canonical"]
    checks = {
        "canonical_noninferior": scores["dice"] >= reference["dice"],
        "no_tissue_drop_over_half_pp": all(scores["tissue_dice"][k] >= reference["tissue_dice"][k] - .005
                                           for k in reference["tissue_dice"]),
        "heldout_prompt_stable": all(candidate[f"heldout_{i}"]["dice"] >= scores["dice"] - .01 for i in (4, 5)),
        "foreground_gate_exists": any(any(row) for table in gate["tables"].values() for row in table[1:]),
        "foreground_coverage": gate["mean_foreground_guidance_coverage"] >= .001,
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_teacher(screens, validation_scores):
    eligible = [k for k in ("T1", "T2") if screens[k]["passed"]]
    if len(eligible) < 2:
        return eligible[0] if eligible else None
    return "T2" if validation_scores["T2"] - validation_scores["T1"] >= .001 else "T1"
