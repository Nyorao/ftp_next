"""Real native-volume and unlabeled image checks with a disposable smoke teacher."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.evaluation import prepare_validation
from freqssl.engine import write_json
from prompt_teacher.data import TissueDataset, _read_transposed
from prompt_teacher.model import build_teacher
from prompt_teacher.run import evaluate_teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg = replace(load_config("configs/v3/prompt_context.yaml"), slice_context="none")
    _, _, split = build_training_data(cfg)
    model, metadata = build_teacher("text", "cuda:0")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_adaptation(saved["adaptation"])
    model.eval()
    dataset = TissueDataset(cfg.root_path, "train", split["unlabeled_files"], metadata["mean"], metadata["std"], image_only=True)
    def image_only(path, dtype):
        if Path(path).parent.name == "label":
            raise AssertionError("Unlabeled cache path attempted to read an annotation")
        return _read_transposed(path, dtype)
    with mock.patch("prompt_teacher.data._read_transposed", side_effect=image_only):
        images = torch.stack([dataset[i]["image"] for i in range(4)]).cuda()
        q, tissue, trusted, votes = model.predict_mc(images, 421)
        torch.testing.assert_close(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5)
    validation, records = prepare_validation(cfg)
    valid = TissueDataset(cfg.root_path, "valid", validation.names, metadata["mean"], metadata["std"], image_only=True)
    started = time.monotonic()
    result = evaluate_teacher(model, valid, records[:2], torch.device("cuda:0"), 42)
    write_json(args.output, {"native_cases_validated": [r["case"] for r in result["cases"]],
        "native_validation_seconds": time.monotonic() - started,
        "four_unlabeled_slices_without_annotation_access": True,
        "teacher_output_shape": list(q.shape), "all_probabilities_finite": bool(torch.isfinite(q).all()),
        "smoke_checkpoint_step": saved["step"], "efficacy_result": False, "test_evaluated": False})
    print("Native volume reconstruction and unlabeled teacher generation passed", flush=True)


if __name__ == "__main__":
    main()
