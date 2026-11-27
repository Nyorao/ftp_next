"""Command line interface for the new training and volume evaluation protocol."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .config import METHODS, load_config
from .engine import load_model, run_experiment, seed_everything, write_json
from .evaluation import evaluate_volumes, prepare_validation


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/v3/dual_frequency.yaml"


def resolve_device(value):
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Independent students with reciprocal EMA pseudo supervision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--slice-context", choices=("none", "mean", "adaptive"))
    parser.add_argument("--resume", type=Path, help="Resume last.pt with the identical saved protocol")
    parser.add_argument("--stop-after", type=int, help="Pause after N more steps without changing the LR schedule")
    parser.add_argument("--smoke-test", action="store_true", help="8 real-data steps including cross supervision, then complete valid 3D")
    for name in ("max-steps", "eval-every", "seed", "batch-size", "labeled-bs", "num-workers", "base-channels", "image-size"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args(argv)
    if args.resume:
        _, cfg, _ = load_model(args.resume, torch.device("cpu"))
    else:
        cfg = load_config(args.config)
    updates = {key: value for key, value in vars(args).items()
               if key in cfg.to_dict() and value is not None}
    cfg = replace(cfg, **updates)
    if args.smoke_test:
        if args.resume:
            parser.error("Smoke testing cannot resume a training run")
        cfg = replace(cfg, max_steps=8, eval_every=8, warmup_steps=2, ramp_steps=3,
                      log_every=1, max_seconds=1200,
                      output_dir=args.output_dir or "outputs/freqssl_v3_smoke")
    result = run_experiment(cfg, resolve_device(args.device), args.resume, args.stop_after)
    print(json.dumps({k: v for k, v in result.items() if k != "final_validation"}, indent=2))


def evaluate_checkpoint(checkpoint, root_path, split, networks, device, output_dir=None):
    model, cfg, payload = load_model(checkpoint, device)
    if root_path:
        cfg.root_path = root_path
    seed_everything(cfg)
    dataset, records = prepare_validation(cfg, split=split)
    networks = networks or [cfg.primary_network]
    metrics = evaluate_volumes(model, dataset, records, device, networks=networks,
                               batch_size=cfg.eval_batch_size)
    result = {"checkpoint": str(Path(checkpoint).resolve()), "step": payload["step"],
              "split": split, "networks": metrics,
              "selection_network": cfg.primary_network,
              "surface_distances_computed": False, "postprocessing": "none"}
    if output_dir:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "results.json").exists():
            raise FileExistsError(directory / "results.json")
        write_json(directory / "results.json", result)
    print(json.dumps({"split": split, "step": payload["step"],
                      "overall": {name: value["overall"] for name, value in metrics.items()}}, indent=2))
    return result
