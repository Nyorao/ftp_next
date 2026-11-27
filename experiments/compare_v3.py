"""Run a fixed paired comparison. Never select an arm or checkpoint on test."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from freqssl.config import METHODS, ExperimentConfig, arm_config, load_config
from freqssl.data import build_training_data
from freqssl.engine import run_experiment, state_hash, write_json
from freqssl.models import StudentTeacherSystem


SUITE_ARMS = {
    "baseline": ("mean_teacher", "dual_independent", "dual_cps", "dual_frequency"),
    "auxiliary": ("dual_aux", "dual_dice", "dual_region"),
}
AUXILIARY_ARMS = {
    "single_base": ("mean_teacher", False, False),
    "single_aux": ("mean_teacher", True, True),
    "dual_base": ("dual_frequency", False, False),
    "dual_dice": ("dual_frequency", True, False),
    "dual_region": ("dual_frequency", False, True),
    "dual_aux": ("dual_frequency", True, True),
}


def resolve_arm(cfg, arm, suite, output_dir=None, seed=None):
    if suite == "baseline":
        if arm not in METHODS:
            raise ValueError(f"Unsupported baseline arm: {arm}")
        return arm_config(cfg, arm, output_dir, seed)
    if suite != "auxiliary" or arm not in AUXILIARY_ARMS:
        raise ValueError(f"Unsupported suite/arm: {suite}/{arm}")
    method, use_dice, use_region = AUXILIARY_ARMS[arm]
    if (use_dice and cfg.pseudo_dice_weight <= 0) or (use_region and cfg.region_weight <= 0):
        raise ValueError("Auxiliary arms require positive preset auxiliary weights")
    return replace(arm_config(cfg, method, output_dir, seed),
                   pseudo_dice_weight=cfg.pseudo_dice_weight if use_dice else 0.0,
                   region_weight=cfg.region_weight if use_region else 0.0).validate()


def load_references(cfg, seeds, directory):
    """Reuse completed baselines only when protocol, split and S1 init match.

    Read validation summaries, never test content or baseline model weights.
    Hash the reference evidence and freeze the resolved scores in protocol.json.
    Source equivalence is checked separately by the frozen-runtime audit.
    """
    if directory is None:
        return {}
    directory = Path(directory).resolve()
    result_file = directory / "results.json"
    original = json.loads(result_file.read_text())
    if original["status"] != "completed":
        raise ValueError("Reference comparison has not completed")
    records = {}
    ignored = {"method", "output_dir", "pseudo_dice_weight", "region_weight", "region_grids"}
    for seed in seeds:
        expected = replace(cfg, seed=seed)
        _, _, current_split = build_training_data(expected)
        initial = state_hash(StudentTeacherSystem(replace(expected, method="mean_teacher")).students[0])
        records[str(seed)] = {}
        for alias, method in (("single_base", "mean_teacher"), ("dual_base", "dual_frequency")):
            key = f"seed{seed}/{method}"
            run = directory / key
            reference = original["runs"][key]
            saved_cfg = ExperimentConfig(**json.loads((run / "config.json").read_text())).validate()
            mismatch = [name for name, value in expected.to_dict().items()
                        if name not in ignored and saved_cfg.to_dict()[name] != value]
            if mismatch or saved_cfg.method != method or saved_cfg.pseudo_dice_weight or saved_cfg.region_weight:
                raise ValueError(f"Reference {key} uses a different protocol: {mismatch}")
            if reference["status"] != "completed" or reference["step"] != cfg.max_steps:
                raise ValueError(f"Reference {key} did not complete the matched budget")
            if json.loads((run / "split.json").read_text()) != current_split:
                raise ValueError(f"Reference {key} has a different dataset split")
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["initial_student1_hash"] != initial or reference["primary_network"] != cfg.primary_network:
                raise ValueError(f"Reference {key} has a different initialization or primary network")
            paths = [result_file, run / "config.json", run / "split.json", run / "manifest.json"]
            records[str(seed)][alias] = {
                "reused": True, "source": str(run), "step": reference["step"],
                "best_valid_dice": reference["best_valid_dice"],
                "primary_network": reference["primary_network"],
                "initial_student1_hash": initial,
                "evidence_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths},
            }
    return records


def update_deltas(results, paired, references, seed):
    baselines = {**references.get(str(seed), {}), **paired}
    for reference, field in (("mean_teacher", "valid_dice_delta_vs_single"),
                             ("single_base", "valid_dice_delta_vs_single"),
                             ("single_aux", "valid_dice_delta_vs_single_aux"),
                             ("dual_base", "valid_dice_delta_vs_dual_base")):
        if reference not in baselines:
            continue
        base = baselines[reference]["best_valid_dice"]
        for name, value in paired.items():
            if base is not None and value["best_valid_dice"] is not None:
                results["runs"][f"seed{seed}/{name}"][field] = value["best_valid_dice"] - base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/dual_frequency.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", choices=SUITE_ARMS, default="baseline")
    parser.add_argument("--reference-comparison", type=Path,
                        help="Reuse completed paired single/dual baselines with exactly matched protocol")
    parser.add_argument("--arms", nargs="+", choices=(*METHODS, *AUXILIARY_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--ramp-steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    args.arms = args.arms or list(SUITE_ARMS[args.suite])
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Arms and seeds must be unique")
    cfg = load_config(args.config)
    cfg = replace(cfg, **{key: getattr(args, key) for key in
                  ("max_steps", "eval_every", "warmup_steps", "ramp_steps", "max_seconds")
                  if getattr(args, key) is not None}).validate()
    for arm in args.arms:
        resolve_arm(cfg, arm, args.suite)
    references = load_references(cfg, args.seeds, args.reference_comparison)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a new comparison directory: {output}")
    write_json(output / "protocol.json", {
        "config": cfg.to_dict(), "arms": args.arms, "seeds": args.seeds, "suite": args.suite,
        "arm_settings": {arm: resolve_arm(cfg, arm, args.suite).to_dict() for arm in args.arms},
        "reused_references": references,
        "comparison": "same data split, initialization of S1, augmentations of S1, batch, updates, schedule and selection",
        "primary": "teacher1 valid native 3D Dice; supervised uses student1",
        "ensemble_is_secondary": True, "test_evaluated": False,
        "interpretation": "single-seed screening is not evidence of robust improvement",
        "compute": "same updates per student, not equal GPU time; region loss adds an unmixed unlabeled forward",
        "auxiliary_primary_contrast": "new dual auxiliary arms vs existing dual_base and single_base",
        "attribution_limit": "without single_aux, gains cannot be attributed exclusively to the dual architecture",
        "weights_selected_without_new_validation_or_test": True})
    results = {"status": "running", "runs": {}, "references": references, "test_evaluated": False}
    write_json(output / "results.json", results)
    try:
        for seed in args.seeds:
            paired = {}
            for method in args.arms:
                key = f"seed{seed}/{method}"
                results["active_run"] = key
                write_json(output / "results.json", results)
                settings = resolve_arm(cfg, method, args.suite, output / key, seed)
                result = run_experiment(settings, torch.device(args.device))
                paired[method] = result
                compact = {name: value for name, value in result.items() if name != "final_validation"}
                results["runs"][key] = compact
                update_deltas(results, paired, references, seed)
                if "mean_teacher" in paired and "dual_independent" in paired:
                    baseline, control = paired["mean_teacher"], paired["dual_independent"]
                    matched = all(baseline[field] == control[field] for field in
                                  ("final_student1_hash", "final_teacher1_hash"))
                    results.setdefault("negative_control", {})[str(seed)] = matched
                    if not matched:
                        raise AssertionError("Negative control failed: S1/EMA do not reproduce mean_teacher")
                write_json(output / "results.json", results)
                gc.collect()
                if torch.device(args.device).type == "cuda":
                    torch.cuda.empty_cache()
        results.update(status="completed", active_run=None)
        write_json(output / "results.json", results)
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "results.json", results)
        raise
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
