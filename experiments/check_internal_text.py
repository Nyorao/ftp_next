"""Real-data, full-width CPU integration check; not an accuracy experiment.

Reuses the first authorized training batch for three diagnostic updates at
schedule indices 0, 1, 1801. No validation/test labels are read here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import seed_everything, train_step, make_optimizer, state_hash, write_json, save_checkpoint
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import file_sha256


def parameter_hash(module, exclude_adapter=False):
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        if exclude_adapter and name.startswith("text_adapter."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/v3/internal_text_context.yaml")
    p.add_argument("--banks", type=Path, default=ROOT / "outputs/internal_text_banks_20261106")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    device = torch.device(args.device)
    seed_everything(cfg)
    dataset, sampler, split = build_training_data(cfg)
    indices = next(iter(sampler))
    samples = [dataset[i] for i in indices]
    batch = {"image": torch.stack([s["image"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
        "is_labeled": torch.tensor([s["is_labeled"] for s in samples])}
    result = {"status": "running", "scope": "same real training batch, 3 diagnostic updates per arm; not efficacy",
        "device": str(device), "cuda_available": torch.cuda.is_available(),
        "shape": list(batch["image"].shape), "split": split,
        "schedule_indices": [0, 1, cfg.warmup_steps + cfg.ramp_steps + 1],
        "validation_or_test_evaluated": False, "arms": {}}
    write_json(args.output / "status.json", {"status": "running"})
    initial_logits = None
    for arm in ("context", "neutral", "text"):
        current = replace(cfg, internal_text="none" if arm == "context" else arm,
            text_bank="" if arm == "context" else str(args.banks / f"{arm}.pt")).validate()
        seed_everything(current)
        model = StudentTeacherSystem(current).to(device)
        student = model.students[0]
        row = {"parameters": sum(p.numel() for p in student.parameters()),
            "initial_visual_parameter_hash": parameter_hash(student, True),
            "initial_all_parameter_hash": parameter_hash(student), "updates": []}
        with torch.no_grad():
            logits = model.student_logits(batch["image"][:1].to(device), 0).cpu()
        if initial_logits is None:
            initial_logits = logits.clone()
        else:
            torch.testing.assert_close(logits, initial_logits, rtol=0, atol=0)
        row["initial_logits_exact_to_control"] = True
        fixed = student.text_adapter.embeddings.clone() if arm != "context" else None
        optimizer = make_optimizer(model, current)
        start = time.monotonic()
        for step in result["schedule_indices"]:
            metrics = train_step(model, optimizer, batch, current, device, step)
            row["updates"].append({"schedule_index": step, **metrics})
            if step == 0 and state_hash(model.students[0]) != state_hash(model.teachers[0]):
                raise AssertionError("First EMA update did not copy the student")
            print(f"{arm} diagnostic index={step}, loss={metrics['loss']:.6f}", flush=True)
        row["seconds_three_updates"] = time.monotonic() - start
        row["teacher_gradients_absent"] = all(p.grad is None for p in model.teachers.parameters())
        if not row["teacher_gradients_absent"]:
            raise AssertionError("Teacher received gradients")
        if arm != "context":
            adapter = student.text_adapter
            row["projection_gradient_l1"] = {name: float(getattr(adapter, name).weight.grad.abs().sum())
                for name in ("query", "key", "value", "output")}
            if not all(value > 0 for value in row["projection_gradient_l1"].values()):
                raise AssertionError("Adapter projection has no gradient")
            row["fixed_vectors_unchanged"] = torch.equal(fixed, adapter.embeddings)
            if not row["fixed_vectors_unchanged"]:
                raise AssertionError("Text embeddings changed")
            # Test dependency using replacement, not permutation of the set.
            model.eval()
            x = batch["image"][:1].to(device)
            with torch.no_grad():
                expected = model.student_logits(x, 0)
                adapter.embeddings.copy_(fixed.roll(37, dims=1))
                changed = model.student_logits(x, 0)
                adapter.embeddings.copy_(fixed)
            row["changed_text_max_logit_delta"] = float((expected - changed).abs().max())
            if row["changed_text_max_logit_delta"] <= 0:
                raise AssertionError("Trained prediction ignores text content")
            # Exercise an actual, standalone EMA checkpoint on real input.
            checkpoint = args.output / f"{arm}_diagnostic.pt"
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            save_checkpoint(checkpoint, model, optimizer, scaler, current, 3, -1.)
            single, _, receipt = load_single_network(checkpoint, device)
            with torch.no_grad():
                torch.testing.assert_close(single(x), model.predict_proba(x, "teacher1"), rtol=0, atol=0)
            row["single_teacher_matches"] = True
            row["text_provenance"] = receipt["text_provenance"]
            # Keep small receipts, not misleading untrained 'best' weights.
            row["diagnostic_checkpoint_sha256"] = file_sha256(checkpoint)
            checkpoint.unlink()
            del single
        result["arms"][arm] = row
        write_json(args.output / "results.json", result)
        del model, student, optimizer
        gc.collect()
    if len({r["initial_visual_parameter_hash"] for r in result["arms"].values()}) != 1:
        raise AssertionError("Visual initialization mismatch")
    if result["arms"]["text"]["initial_all_parameter_hash"] != result["arms"]["neutral"]["initial_all_parameter_hash"]:
        raise AssertionError("Trainable initialization mismatch between text controls")
    result.update(status="passed", efficacy_verified=False,
        source_hashes={str(p.relative_to(ROOT)): file_sha256(p)
            for p in [*sorted((ROOT / "freqssl").glob("*.py")), Path(__file__)]})
    write_json(args.output / "results.json", result)
    write_json(args.output / "status.json", {"status": "passed", "efficacy_verified": False})
    print("Real-data internal-text integration passed; no accuracy claim.", flush=True)


if __name__ == "__main__":
    main()
