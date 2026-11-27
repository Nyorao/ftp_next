"""Training-only tissue guidance. No foundation-model dependency in this module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .anatomy import FAMILY_IDS, family_log_probabilities


TEXT_PROMPTS = (
    "Background and other tissues in a sagittal spine MRI.",
    "Vertebrae in a sagittal spine MRI.",
    "The spinal canal in a sagittal spine MRI.",
    "Intervertebral discs in a sagittal spine MRI.",
)
NEUTRAL_PROMPTS = tuple(f"Region {letter} in a sagittal spine MRI." for letter in "ABCD")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_routed_loss(logits, ema, tissue, trusted, valid, confidence=.8,
                       background_confidence=.95, background_weight=.25, tissue_weight=.5,
                       route="legacy", gate_mask=None):
    """Disjoint fine and set-valued targets, with an all-pixel denominator."""
    if logits.ndim != 4 or logits.shape[1] != 20 or ema.shape != logits.shape:
        raise ValueError("Expected paired (B,20,H,W) logits and EMA probabilities")
    if any(x.shape != logits.shape[:1] + logits.shape[2:] for x in (tissue, trusted, valid)):
        raise ValueError("Prompt targets must share student coordinates")
    if torch.any((tissue < 0) | (tissue > 3)):
        raise ValueError("Prompt tissue IDs must be 0..3")
    certainty, target = ema.detach().float().max(1)
    tissue = tissue.detach().long()
    reliable = trusted.detach().bool() & valid.detach().bool()
    foreground = target != 0
    accepted = torch.where(foreground, certainty >= confidence, certainty >= background_confidence)
    family = torch.tensor(FAMILY_IDS, device=logits.device)[target]
    conflict = reliable & (family != tissue)
    if route == "legacy":
        fine_mask = accepted & ~conflict
        coarse_mask = reliable & (~accepted | conflict)
    elif route in ("preserve_ema", "competence"):
        fine_mask = accepted
        coarse_mask = reliable & ~accepted
        if route == "competence":
            from .prompt_revision import competence_eligible
            if gate_mask is None or gate_mask.shape != tissue.shape:
                raise ValueError("Missing competence mask")
            coarse_mask = competence_eligible(ema, tissue, reliable, valid,
                                               confidence, background_confidence) & gate_mask.bool()
    else:
        raise ValueError("Unknown prompt routing mode")
    fine_weights = fine_mask.float() * torch.where(foreground, 1., background_weight)
    coarse_weights = coarse_mask.float() * torch.where(tissue != 0, 1., background_weight)
    logp = logits.float().log_softmax(1)
    fine = (-logp.gather(1, target[:, None]).squeeze(1) * fine_weights).mean()
    coarse = (-family_log_probabilities(logits).gather(1, tissue[:, None]).squeeze(1)
              * coarse_weights).mean()
    loss = fine + tissue_weight * coarse
    selected = fine_mask | coarse_mask
    report = {
        "coverage": float(selected.float().mean()),
        "fine_coverage": float(fine_mask.float().mean()),
        "tissue_coverage": float(coarse_mask.float().mean()),
        "prompt_reliable_coverage": float(reliable.float().mean()),
        "prompt_conflict_coverage": float(conflict.float().mean()),
        "filtered_fine_ce": float(fine.detach()), "tissue_ce": float(coarse.detach()),
        "weighted_tissue_ce": float((tissue_weight * coarse).detach()),
        "selected_foreground_pixels": int((selected & torch.where(coarse_mask, tissue != 0, foreground)).sum()),
        "selected_background_pixels": int((selected & ~torch.where(coarse_mask, tissue != 0, foreground)).sum()),
        "fine_tissue_overlap": int((fine_mask & coarse_mask).sum()),
    }
    return loss, report


def mc_tissue_targets(samples):
    """Exactly eight normalized categorical predictions, not sigmoid masks."""
    if samples.ndim != 5 or samples.shape[0] != 8 or samples.shape[2] != 4:
        raise ValueError("Expected (8,B,4,H,W) Monte Carlo probabilities")
    if not torch.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Invalid Monte Carlo probabilities")
    if not torch.allclose(samples.sum(2), torch.ones_like(samples[:, :, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Tissue probabilities must sum to one")
    mean = samples.mean(0)
    confidence, tissue = mean.max(1)
    votes = (samples.argmax(2) == tissue[None]).sum(0)
    trusted = (votes >= 7) & (confidence >= torch.where(tissue == 0, .95, .8))
    return mean, tissue, trusted, votes


class PromptCache:
    """Fail closed on wrong supervision budget, source, input bytes or cache bytes."""
    def __init__(self, directory, source, image_size, split_manifest):
        self.root = Path(directory).resolve()
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        self.manifest_sha256 = file_sha256(path)
        expected = TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS
        self.version = manifest.get("format_version")
        if (self.version not in (1, 2) or manifest.get("status") != "completed"
                or manifest.get("source") != source or manifest.get("prompts") != list(expected)
                or manifest.get("image_size") != image_size
                or manifest.get("family_ids") != list(FAMILY_IDS)
                or manifest.get("mc_samples") != 8
                or manifest.get("unlabeled_annotations_opened") is not False):
            raise ValueError("Prompt cache protocol mismatch")
        if self.version == 2:
            from .prompt_revision import prompt_bank
            if (manifest.get("prompt_bank") != prompt_bank(source)[:4]
                    or manifest.get("samples_per_template") != 2
                    or manifest.get("teacher_variant") not in ("T0", "T1", "T2")):
                raise ValueError("Prompt bank or sampling protocol mismatch")
        self.channels = 17 if self.version == 2 else 3
        self.geometry_mode = "bilinear" if self.version == 2 else "nearest"
        self.manifest = manifest
        for key in ("labeled_files", "unlabeled_files", "root_path"):
            if manifest.get(key) != split_manifest[key]:
                raise ValueError(f"Prompt cache split mismatch: {key}")
        self.entries = manifest["entries"]
        if set(self.entries) != set(split_manifest["unlabeled_files"]):
            raise ValueError("Prompt cache must contain every and only authorized unlabeled slice")
        input_root = Path(split_manifest["root_path"]) / "train" / "image"
        expected_labeled = set(split_manifest["labeled_files"])
        for key, folder in (("labeled_annotation_sha256", "label"), ("labeled_image_sha256", "image")):
            hashes = manifest.get(key, {})
            if set(hashes) != expected_labeled:
                raise ValueError("Prompt teacher supervision provenance is incomplete")
            for name, digest in hashes.items():
                if file_sha256(input_root.parent / folder / name) != digest:
                    raise ValueError(f"Prompt teacher supervision changed: {name}")
        for name, entry in self.entries.items():
            if entry["file"] != Path(name).stem + ".npz":
                raise ValueError("Unsafe or mismatched prompt cache filename")
            if file_sha256(self.root / entry["file"]) != entry["sha256"]:
                raise ValueError(f"Prompt cache content changed: {name}")
            if file_sha256(input_root / name) != entry["image_sha256"]:
                raise ValueError(f"Prompt source image changed: {name}")
        self.image_size = image_size

    def read(self, name):
        with np.load(self.root / self.entries[name]["file"], allow_pickle=False) as data:
            if self.version == 2:
                q = data["template_probabilities"].astype(np.float32)
                valid = data["valid"].astype(np.float32)
                shape = (self.image_size, self.image_size)
                if (q.shape != (4, 4, *shape) or valid.shape != shape
                        or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
                        or not np.allclose(q.sum(1), 1, atol=.002)
                        or not np.isin(valid, (0, 1)).all()):
                    raise ValueError("Malformed template probability cache")
                return torch.from_numpy(np.concatenate((q.reshape(16, *shape), valid[None])))
            q = data["probabilities"].astype(np.float32)
            tissue = data["tissue"].astype(np.int64)
            trusted, valid = data["trusted"].astype(bool), data["valid"].astype(bool)
        shape = (self.image_size, self.image_size)
        if q.shape != (4, *shape) or any(x.shape != shape for x in (tissue, trusted, valid)):
            raise ValueError("Prompt cache shape mismatch")
        if (not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
                or not np.allclose(q.sum(0), 1., atol=.002)
                or (tissue < 0).any() or (tissue > 3).any()):
            raise ValueError("Malformed prompt cache")
        # Hard targets and masks travel together with nearest-neighbour geometry.
        return torch.from_numpy(np.stack((tissue, trusted, valid)).astype(np.float32))
