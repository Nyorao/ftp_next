"""Prompt-free frequency ablations with full training pools and native 3D validation."""

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from config import load_config
from dataset import SpiderDataset, TwoStreamBatchSampler
from evaluate_volumes import file_hash, predict_case, write_json
from experiments.compare_dual import state_hash
from experiments.frequency_schedule import FeedbackReadiness, training_budget
from spider_volume import aggregate_cases, audit_case, group_complete_slices, volume_metrics
from trainer import build_model, save_checkpoint, seed_everything, student_parameters, train_batch


ARMS = ("anchor_lf", "dual_frequency_no_feedback", "dual_same_view", "dual_frequency", "anchor_raw", "dual_raw")
DEFAULT_ARMS = ("anchor_raw", "dual_raw", "dual_frequency")


def arm_config(cfg, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return replace(cfg, architecture="anchor" if arm.startswith("anchor") else "dual",
                   use_apgm=False, use_psgm=False, use_frequency=arm not in ("anchor_raw", "dual_raw"),
                   secondary_view="lf" if arm == "dual_same_view" else "hf",
                   feedback_weight=cfg.feedback_weight if arm in ("dual_same_view", "dual_frequency", "dual_raw") else 0.0)


def snapshot(output):
    hashes = {}
    paths = [*ROOT.glob("*.py"), *ROOT.glob("model/*.py"), *ROOT.glob("experiments/*.py"),
             *ROOT.glob("configs/*.yaml"), *ROOT.glob("tests/*.py"), *ROOT.glob("data/*.json")]
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = file_hash(source)
    write_json(output / "source_hashes.json", hashes)


@torch.inference_mode()
def validate_volumes(model, cfg, dataset, records, device, all_networks=False):
    was_training = model.training
    model.eval()
    networks = ["teacher"]
    if all_networks:
        networks.append("student1")
        if model.student2 is not None:
            networks += ["student2", "students_mean"]
    indices = {name: i for i, name in enumerate(dataset.names)}
    cases = {name: [] for name in networks}
    try:
        for record in records:
            predictions = predict_case(model, cfg, dataset, [indices[n] for n in record["slice_names"]],
                                       record, networks, device)
            reference = sitk.ReadImage(record["reference"])
            target = sitk.GetArrayFromImage(reference)
            for network, prediction in predictions.items():
                # Checkpoint selection uses overlap only. Corrected physical
                # geometry is retained in data/ for the later distance audit.
                cases[network].append({"case": record["case"], "classes": volume_metrics(
                    prediction, target, tuple(reversed(reference.GetSpacing())), surfaces=False)})
    finally:
        model.train(was_training)
    return {network: dict(aggregate_cases(rows), cases=rows) for network, rows in cases.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/frequency_dual.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps", type=int, help="Explicit short-test cap only; omitted for full epochs")
    parser.add_argument("--validate-every", type=int, help="Optional interval in steps for short tests")
    parser.add_argument("--validate-every-epochs", type=int, default=5)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    args = parser.parse_args()
    if (min(args.epochs, args.validate_every_epochs) < 1
            or any(v is not None and v < 1 for v in (args.steps, args.validate_every))
            or len(set(args.arms)) != len(args.arms)):
        parser.error("positive steps/interval and unique arms required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(output)
    cfg = replace(load_config(args.config), max_epochs=args.epochs, use_apgm=False, use_psgm=False)
    cfg.validate()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    seed_everything(cfg)
    training = SpiderDataset(cfg.root_path, "train", cfg.image_size, cfg.labelnum, cfg.max_samples)
    validation = SpiderDataset(cfg.root_path, "valid", cfg.image_size)
    groups = {split: group_complete_slices(p.name for p in (Path(cfg.root_path) / split / "image").glob("*.png"))
              for split in ("train", "valid", "test")}
    if any(set(groups[a]) & set(groups[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Overlapping SPIDER case identifiers across splits")
    records = [audit_case(cfg.root_path, Path(cfg.root_path).parent / "mha", "valid", case, names)
               for case, names in groups["valid"].items()]
    write_json(output / "validation_reconstruction_audit.json", records)
    requested_step_cap = args.steps
    sampler_length, args.steps = training_budget(cfg.labelnum, cfg.labeled_bs, args.epochs, requested_step_cap)
    args.validate_every = args.validate_every or args.validate_every_epochs * sampler_length
    cfg = replace(cfg, max_iterations=args.steps)
    write_json(output / "manifest.json", {
        "protocol": "prompt_free_frequency_v1", "config": cfg.to_dict(), "arguments": vars(args) | {"output_dir": str(output)},
        "arms": args.arms, "labeled_files": training.names[:cfg.labelnum],
        "steps_per_epoch": sampler_length, "requested_step_cap": requested_step_cap,
        "unlabeled_files": training.names[cfg.labelnum:], "validation_cases": sorted(groups["valid"]),
        "test_labels_or_images_used": False, "text_assets_loaded": False,
        "supervision_budget": "full existing 1496/2236 slice pools; same sampler/seed/updates in every arm",
        "validation": "complete native 3D valid; Dice/IoU; foreground GT-present per-case means",
        "selection": "best.pt by complete valid teacher 3D Dice; last.pt fixed budget; test reserved",
        "feedback_schedule": {"direction": "S2_to_S1", "minimum_epoch": cfg.feedback_start_epoch,
                              "require_validation_readiness": cfg.feedback_require_ready,
                              "both_student_3d_dice_threshold": cfg.feedback_min_dice,
                              "consecutive_checks": cfg.feedback_ready_checks,
                              "ramp_epochs_after_activation": cfg.feedback_rampup_epochs},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scope": "single-seed full-epoch comparison unless an explicit short-test step cap is recorded; efficacy unverified",
    })
    snapshot(output)
    results = {"status": "running", "arms": {}, "test_evaluated": False}
    write_json(output / "results.json", results)
    baseline_hashes = None
    try:
        for arm in args.arms:
            settings = arm_config(cfg, arm)
            directory = output / arm
            directory.mkdir()
            settings.output_dir = str(directory)
            write_json(directory / "config.json", settings.to_dict())
            seed_everything(settings)
            training.transform.set_random_state(seed=settings.seed)
            model = build_model(settings).to(device)
            assert all(m.apgm is None and m.psgm is None for m in (model.student, model.teacher))
            optimizer = torch.optim.AdamW(student_parameters(model), lr=settings.base_lr,
                                          weight_decay=settings.weight_decay, eps=1e-5, betas=(0.9, 0.999))
            sampler = TwoStreamBatchSampler(range(settings.labelnum), range(settings.labelnum, len(training)),
                                           settings.batch_size, settings.labeled_bs, settings.seed)
            if len(sampler) != sampler_length:
                raise AssertionError("Epoch length differs from the recorded training budget")
            loader = DataLoader(training, batch_sampler=sampler, num_workers=0,
                                generator=torch.Generator().manual_seed(settings.seed))
            initial_hash = state_hash(model.student)
            best, step, epoch = -1.0, 0, 0
            readiness = FeedbackReadiness(settings.feedback_start_epoch, settings.feedback_min_dice,
                                          settings.feedback_ready_checks)
            started = time.monotonic()
            results["active_arm"] = arm
            write_json(output / "results.json", results)
            print(f"START {arm} epochs={args.epochs} steps_per_epoch={len(sampler)} steps={args.steps} full_pools={settings.labelnum}/{len(training)-settings.labelnum} prompts=False", flush=True)
            with (directory / "metrics.jsonl").open("w") as log:
                while step < args.steps:
                    sampler.epoch = epoch
                    for batch in loader:
                        for group in optimizer.param_groups:
                            group["lr"] = settings.base_lr * (1 - step / args.steps) ** 0.9
                        metrics = train_batch(model, optimizer, batch, settings, device, step / len(sampler),
                                              feedback_ready=readiness.activated_epoch is not None,
                                              feedback_activated_epoch=readiness.activated_epoch)
                        step += 1
                        log.write(json.dumps(dict(metrics, step=step, epoch=epoch + 1, epoch_index=epoch, kind="train"), allow_nan=False) + "\n")
                        log.flush()
                        if step == 1 or step % 50 == 0:
                            print(f"{arm} step={step}/{args.steps} loss={metrics['loss']:.5f} prompt={metrics['prompt']} feedback_pixels={metrics.get('feedback_pixels', 0)}", flush=True)
                        if step % args.validate_every == 0 or step == args.steps:
                            validation_result = validate_volumes(model, settings, validation, records, device,
                                                                 all_networks=True)
                            write_json(directory / f"valid_3d_step_{step}.json", validation_result)
                            if model.student2 is not None and settings.feedback_weight > 0:
                                readiness.observe(step / len(sampler),
                                    validation_result["student1"]["overall"]["dice"],
                                    validation_result["student2"]["overall"]["dice"])
                            training_state = {"completed_epochs": step / len(sampler),
                                              "feedback_readiness": readiness.state()}
                            write_json(directory / "feedback_readiness.json", training_state)
                            dice = validation_result["teacher"]["overall"]["dice"]
                            log.write(json.dumps(dict(kind="validation_3d", step=step, dice=dice,
                                                      **training_state)) + "\n")
                            log.flush()
                            if dice is not None and dice > best:
                                best = dice
                                save_checkpoint(directory / "best.pt", model, optimizer, settings, epoch, step, best, training_state)
                            save_checkpoint(directory / "last.pt", model, optimizer, settings, epoch, step, best, training_state)
                            print(f"VALID_3D {arm} completed_epochs={step / len(sampler):.2f} step={step} cases={len(records)} dice={dice} feedback_activated_epoch={readiness.activated_epoch}", flush=True)
                        if step >= args.steps:
                            break
                    epoch += 1
            final_hashes = {name: state_hash(getattr(model, name)) for name in ("student", "teacher")}
            if arm == "anchor_lf":
                baseline_hashes = final_hashes
            matched = None
            if arm == "dual_frequency_no_feedback" and baseline_hashes is not None:
                matched = final_hashes == baseline_hashes
                if not matched:
                    raise AssertionError("Prompt-free negative control failed; do not compare subsequent arms")
            result = dict(steps=step, completed_epochs=step / len(sampler), feedback_readiness=readiness.state(),
                          seconds=time.monotonic()-started, initial_s1_hash=initial_hash,
                          final_hashes=final_hashes, negative_control_exact_match=matched,
                          best_valid_teacher_dice=best, final_validation=validation_result,
                          trainable_parameters=sum(p.numel() for p in student_parameters(model)))
            write_json(directory / "result.json", result)
            results["arms"][arm] = result
            write_json(output / "results.json", results)
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.update(status="completed", active_arm=None)
        write_json(output / "results.json", results)
    except Exception as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "results.json", results)
        raise


if __name__ == "__main__":
    main()
