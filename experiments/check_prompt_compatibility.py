"""Verify the new disabled prompt path against the frozen context experiment."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from freqssl.config import ExperimentConfig
from freqssl.data import build_training_data
from freqssl.engine import seed_everything, train_step, make_optimizer, state_hash, write_json
from freqssl.models import StudentTeacherSystem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, default=ROOT / "outputs/spine_slice_context_20261030_r2/adaptive")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    source = args.reference / "sources/freqssl"
    spec = importlib.util.spec_from_file_location("frozen_context", source / "__init__.py", submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["frozen_context"] = package
    spec.loader.exec_module(package)
    old_config = importlib.import_module("frozen_context.config")
    old_data = importlib.import_module("frozen_context.data")
    old_models = importlib.import_module("frozen_context.models")
    old_engine = importlib.import_module("frozen_context.engine")
    values = json.loads((args.reference / "config.json").read_text())
    cfg, previous_cfg = ExperimentConfig(**values), old_config.ExperimentConfig(**values)
    seed_everything(cfg)
    current_data, sampler, current_split = build_training_data(cfg)
    previous_data, _, previous_split = old_data.build_training_data(previous_cfg)
    if current_split != previous_split:
        raise AssertionError("Disabled prompt changed the split")
    ids = next(iter(sampler))
    now, before = [current_data[i] for i in ids], [previous_data[i] for i in ids]
    for a, b in zip(now, before):
        for key in ("image", "label"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    batch = {"image": torch.stack([s["image"] for s in now]), "label": torch.stack([s["label"] for s in now]),
             "is_labeled": torch.tensor([s["is_labeled"] for s in now])}
    device = torch.device(args.device)
    new, old = StudentTeacherSystem(cfg).to(device), old_models.StudentTeacherSystem(previous_cfg).to(device)
    a, b = make_optimizer(new, cfg), old_engine.make_optimizer(old, previous_cfg)
    checks = []
    for step in (0, 601, 1801):
        train_step(new, a, batch, cfg, device, step)
        old_engine.train_step(old, b, batch, previous_cfg, device, step)
        if state_hash(new) != state_hash(old):
            raise AssertionError(f"Disabled prompt changed student or EMA updates at step {step}")
        checks.append(step)
    write_json(args.output, {"reference": str(args.reference), "split_exact": True, "inputs_labels_exact": True,
        "student_ema_updates_exact_at_steps": checks, "device": args.device, "final_state_sha256": state_hash(new)})
    print("Frozen context compatibility: exact at all three checked updates", flush=True)


if __name__ == "__main__":
    main()
