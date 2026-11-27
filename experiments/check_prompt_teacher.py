"""Real foundation-model gradient/provenance check; no experiment score selection."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import state_hash, write_json
from freqssl.objectives import segmentation_loss
from prompt_teacher.data import TissueDataset
from prompt_teacher.model import build_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, metadata = build_teacher("text", args.device)
    frozen_hash = state_hash(model.base)
    adapter_hash = state_hash(model.adapters)
    cfg = load_config("configs/v3/slice_context.yaml")
    _, _, split = build_training_data(cfg)
    data = TissueDataset(cfg.root_path, "train", split["labeled_files"], metadata["mean"], metadata["std"])
    samples = [data[i] for i in (100, 200, 300, 400)]
    image = torch.stack([s["image"] for s in samples]).to(args.device)
    target = torch.stack([s["label"] for s in samples]).to(args.device)
    model.eval()
    with torch.no_grad():
        v, t = model.encode(image[:1], sample=False, use_adapters=False)
        native_t = model.base.encode_text(model.tokens, normalize=False)
        torch.testing.assert_close(t, native_t, atol=1e-5, rtol=1e-4)
        native_v = model.base.visual(image[:1].expand(4, -1, -1, -1))
        # Native CLIP returns the pooled CLS vector; dense adaptation retains patches.
        torch.testing.assert_close(v[:, 0], native_v, atol=1e-5, rtol=1e-4)
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.0003, weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = segmentation_loss(logits, target)
        loss.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError("Frozen foundation encoder received a gradient")
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapters.parameters()):
            raise AssertionError("Adapters receive no gradient")
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 12., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if state_hash(model.base) != frozen_hash:
        raise AssertionError("Frozen weights changed")
    model.eval()
    first = model.predict_mc(image[:1], 123)
    second = model.predict_mc(image[:1], 123)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    del optimizer, model
    torch.cuda.empty_cache()
    neutral, _ = build_teacher("neutral", args.device)
    if state_hash(neutral.adapters) != adapter_hash or state_hash(neutral.base) != frozen_hash:
        raise AssertionError("Prompt arms have different initial weights")
    result = {"frozen_encoders_unchanged": True, "native_encoder_equivalence": True,
        "adapter_gradients_nonzero": True, "paired_initialization": True,
        "mc_reproducible": True, "losses": losses, "two_train_steps_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "model": metadata,
        "teacher_trainable_parameters": sum(p.numel() for p in neutral.trainable_parameters())}
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
