"""Paired training, atomic checkpoints, and complete-volume validation."""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import aligned_cutmix, strong_view
from .config import ExperimentConfig
from .data import build_training_data
from .evaluation import evaluate_volumes, prepare_validation
from .models import StudentTeacherSystem
from .objectives import (linear_ramp, masked_pseudo_loss, segmentation_loss,
                         masked_pseudo_dice_loss, multiscale_region_loss)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = not cfg.deterministic


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot_sources(output):
    hashes = {}
    paths = [*sorted((ROOT / "freqssl").glob("*.py")), ROOT / "spider_volume.py",
             ROOT / "train.py", ROOT / "evaluate.py", ROOT / "experiments/compare_v3.py"]
    paths += [source for source in (ROOT / "experiments/train_long_v3.py",
                                    ROOT / "experiments/launch_long_v3.py",
                                    ROOT / "experiments/run_slice_context.py",
                                    ROOT / "experiments/run_prompt_context.py",
                                    ROOT / "experiments/export_internal_text_bank.py",
                                    ROOT / "experiments/export_hierarchical_text_assets.py",
                                    ROOT / "experiments/probe_hierarchical_text.py",
                                    ROOT / "experiments/run_hierarchical_text.py",
                                    ROOT / "experiments/run_encoder_text.py",
                                    ROOT / "experiments/run_dual_text.py",
                                    ROOT / "experiments/check_dual_text.py",
                                    ROOT / "experiments/run_dual_text_long.py",
                                    ROOT / "experiments/check_dual_text_long.py",
                                    ROOT / "experiments/run_internal_text.py",
                                    ROOT / "experiments/run_reference_conditioning.py",
                                    ROOT / "experiments/analyze_reference_conditioning.py") if source.exists()]
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = Path(output) / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(Path(output) / "source_hashes.json", hashes)
    return hashes


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.student_parameters(), lr=cfg.base_lr,
                             weight_decay=cfg.weight_decay)


def train_step(model, optimizer, batch, cfg, device, step, scaler=None):
    """All arms use identical S1 initialization, data, augmentation and schedule.

    dual_independent is the negative control: its first student and first EMA
    should exactly reproduce mean_teacher, including clipping and perturbations.
    Each pseudo-label producer is detached. No validation metric gates training.
    """
    if cfg.method == "dual_ordinal":
        from .ordinal_training import train_ordinal_step
        return train_ordinal_step(model, optimizer, batch, cfg, device, step)
    if cfg.reference_condition != 'none':
        from .reference_training import train_reference_step
        return train_reference_step(model, optimizer, batch, cfg, device, step)
    model.train()
    images = batch["image"].to(device=device, dtype=torch.float32)
    labels = batch["label"].to(device=device, dtype=torch.long)
    labeled = batch["is_labeled"].to(device=device, dtype=torch.bool)
    position = None
    if cfg.context_position == 'relative':
        if 'position' not in batch:
            raise ValueError('Position arm requires position maps')
        position = batch['position'].to(device=device,dtype=torch.float32)
    if not labeled.any() or not (~labeled).any():
        raise ValueError("Every training batch must contain both labeled and unlabeled images")
    enabled_amp = cfg.amp and device.type == "cuda"
    if enabled_amp and scaler is None:
        raise ValueError("AMP requires a persistent GradScaler")
    weight = cfg.unsupervised_weight * linear_ramp(step, cfg.warmup_steps, cfg.ramp_steps)
    if cfg.method == "supervised":
        weight = 0.0
    teacher_probabilities, teacher_features = [], []
    use_regions = cfg.region_weight > 0 and weight > 0
    if cfg.method != "supervised":
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=enabled_amp):
            for i in range(len(model.students)):
                if use_regions:
                    prediction, feature = model.teacher_outputs(images[~labeled], i)
                    teacher_features.append(feature.detach())
                else:
                    prediction = model.teacher_logits(images[~labeled], i,
                        **({} if position is None else {'position':position[~labeled]}))
                teacher_probabilities.append(prediction.float().softmax(1))
    optimizer.zero_grad(set_to_none=True)
    metrics = {"unsupervised_weight": weight}
    total_loss = 0.0
    for index, student in enumerate(model.students):
        seed = cfg.seed + step * 104729 + index * 10000019
        strong = strong_view(images, seed, cfg.noise_std, cfg.gain_std)
        strong_position = None if position is None else position.clone()
        # Keep the aligned, unmixed view for relations. Pixel pseudo-label losses
        # still use the original image/target CutMix and exactly the same seeds.
        clean_unlabeled = strong[~labeled] if use_regions else None
        source = 1 - index if cfg.method in ("dual_cps", "dual_frequency") else index
        target = None
        prompt_target = None
        if teacher_probabilities:
            packed_target = teacher_probabilities[source]
            if position is not None:
                packed_target = torch.cat((packed_target,position[~labeled]),1)
            if cfg.prompt_source != "none":
                auxiliary = batch["prompt_targets"].to(device=device, dtype=torch.float32)[~labeled]
                packed_target = torch.cat((packed_target, auxiliary), 1)
            mixed, target = aligned_cutmix(strong[~labeled], packed_target,
                                           seed + 17, cfg.cutmix_probability)
            if position is not None:
                target,mixed_position = target[:,:20],target[:,20:]
                strong_position[~labeled] = mixed_position
            if cfg.prompt_source != "none":
                target, prompt_target = target[:, :20], target[:, 20:]
            strong = strong.clone()
            strong[~labeled] = mixed
        with torch.autocast(device_type=device.type, enabled=enabled_amp):
            alignment = None
            if getattr(cfg, "text_alignment_weight", 0) > 0:
                options = {}
                if cfg.text_fusion_layout in ("encoder", "encoder_multi"):
                    options["collect_diagnostics"] = step == 0 or (step + 1) % cfg.log_every == 0
                logits, alignment = student.forward_with_alignment(model.make_view(strong,index), **options)
            else:
                logits = model.student_logits(strong, index,
                    **({} if strong_position is None else {'position':strong_position}))
            supervised = segmentation_loss(logits[labeled].float(), labels[labeled])
            unsupervised = logits.float().sum() * 0.0
            if target is not None:
                if prompt_target is None:
                    unsupervised, coverage = masked_pseudo_loss(
                        logits[~labeled].float(), target, cfg.confidence,
                        cfg.background_confidence, cfg.background_weight)
                else:
                    from .prompt_supervision import prompt_routed_loss
                    gate_mask = None
                    if cfg.prompt_route == "competence":
                        from .prompt_revision import CompetenceGate
                        if not hasattr(model, "_competence_gate"):
                            model._competence_gate = CompetenceGate(cfg.prompt_gate, cfg.prompt_source)
                        if prompt_target.shape[1] != 4:
                            raise ValueError("Competence routing requires version-2 prompt targets")
                        gate_mask = model._competence_gate.lookup(prompt_target[:, 0].long(),
                                                                 prompt_target[:, 3].long(), step + 1)
                    unsupervised, coverage = prompt_routed_loss(
                        logits[~labeled].float(), target, prompt_target[:, 0].long(),
                        prompt_target[:, 1].bool(), prompt_target[:, 2].bool(),
                        cfg.confidence, cfg.background_confidence, cfg.background_weight,
                        cfg.prompt_weight, route=cfg.prompt_route, gate_mask=gate_mask)
                metrics.update({f"{key}_s{index+1}": value for key, value in coverage.items()})
                metrics[f"target_source_s{index+1}"] = (f"teacher{source+1}" if prompt_target is None
                    else f"teacher{source+1}+{cfg.prompt_source}_tissue")
            loss = supervised + weight * unsupervised
            if alignment is not None:
                from .hierarchical_text import tissue_alignment_loss
                aligned_loss = tissue_alignment_loss([p[labeled] for p in alignment["tissue_logits"]],
                    labels[labeled], balance=cfg.text_alignment_balance)
                loss = loss + cfg.text_alignment_weight * aligned_loss
                metrics["text_alignment_loss"] = float(aligned_loss.detach())
                if step == 0 or (step + 1) % cfg.log_every == 0:
                    # Diagnostics do not change the objective or consume RNG.
                    # Only labeled examples contribute to tissue supervision.
                    with torch.no_grad():
                        tissue_logits = [p[labeled].detach() for p in alignment["tissue_logits"]]
                        metrics["text_alignment_loss_per_stage"] = [float(tissue_alignment_loss(
                            [p], labels[labeled], balance=cfg.text_alignment_balance)) for p in tissue_logits]
                        if "layer_mixtures" in alignment:
                            metrics["text_layer_mixtures"] = [m[labeled].detach().mean(0).tolist()
                                                              for m in alignment["layer_mixtures"]]
                        if "encoder_text_diagnostics" in alignment:
                            metrics["encoder_text_diagnostics"] = {k:float(v[labeled].detach().mean())
                                for k,v in alignment["encoder_text_diagnostics"].items()}
                            metrics["text_alignment_stages"] = alignment["alignment_stages"]
                        metrics["text_tissue_prediction_fractions"] = [
                            [(p.argmax(1) == c).float().mean().item() for c in range(4)] for p in tissue_logits]
            pseudo_dice = region = logits.float().sum() * 0.0
            if cfg.pseudo_dice_weight > 0 and weight > 0 and target is not None:
                pseudo_dice, dice_metrics = masked_pseudo_dice_loss(
                    logits[~labeled], target, cfg.confidence,
                    cfg.background_confidence, cfg.background_weight)
                loss = loss + weight * cfg.pseudo_dice_weight * pseudo_dice
                metrics.update({f"{key}_s{index+1}": value for key, value in dice_metrics.items()})
            if use_regions:
                _, features = model.student_outputs(clean_unlabeled, index)
                region, region_metrics = multiscale_region_loss(
                    features, teacher_features[source], cfg.region_grids)
                loss = loss + weight * cfg.region_weight * region
                metrics.update({f"{key}_s{index+1}": value for key, value in region_metrics.items()})
                metrics[f"region_target_source_s{index+1}"] = f"teacher{source+1}"
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}, student {index+1}")
        if enabled_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += float(loss.detach())
        metrics.update({f"supervised_s{index+1}": float(supervised.detach()),
                        f"unsupervised_s{index+1}": float(unsupervised.detach()),
                        f"pseudo_dice_s{index+1}": float(pseudo_dice.detach()),
                        f"region_s{index+1}": float(region.detach()),
                        f"weighted_pseudo_ce_s{index+1}": weight * (coverage["filtered_fine_ce"]
                            if prompt_target is not None else float(unsupervised.detach())),
                        f"weighted_pseudo_dice_s{index+1}": weight * cfg.pseudo_dice_weight * float(pseudo_dice.detach()),
                        f"weighted_region_s{index+1}": weight * cfg.region_weight * float(region.detach())})
    if enabled_amp:
        scaler.unscale_(optimizer)
    # Clipping each independent student separately preserves the negative control.
    for index, student in enumerate(model.students):
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        metrics[f"grad_norm_s{index+1}"] = float(norm)
    if enabled_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    metrics["ema_decay"] = model.update_teachers()
    metrics["loss"] = total_loss
    if getattr(cfg, "slice_context", "none") != "none":
        metrics["context_gamma"] = [float(layer.gamma.detach()) for layer in model.students[0].context_fusion]
    if getattr(cfg, "internal_text", "none") != "none":
        if getattr(cfg, "text_fusion_layout", "e4") == "hierarchical":
            metrics["text_gates"] = model.students[0].text_pyramid.gates.detach().tanh().tolist()
        elif cfg.text_fusion_layout in ("encoder", "encoder_multi"):
            metrics["text_gates"] = model.students[0].encoder_text.gate_values()
        else:
            metrics["text_gate"] = float(model.students[0].text_adapter.gate.detach().tanh())
    if len(teacher_probabilities) == 2:
        first, second = [p.argmax(1) for p in teacher_probabilities]
        different = first != second
        foreground = (first != 0) | (second != 0)
        metrics["teacher_disagreement"] = float(different.float().mean())
        metrics["foreground_disagreement"] = float((different & foreground).sum() / foreground.sum().clamp_min(1))
    return metrics


def save_checkpoint(path, model, optimizer, scaler, cfg, step, best_score):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format_version": 3, "config": cfg.to_dict(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best_score": best_score,
                "torch_rng": torch.get_rng_state(),
                "text_provenance": model.text_provenance}, temporary)
    temporary.replace(path)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("Expected a freqssl format-3 checkpoint")
    cfg = ExperimentConfig(**payload["config"]).validate()
    model = StudentTeacherSystem(cfg, initialize_text=False)
    model.load_state_dict(payload["model"], strict=True)
    model.text_provenance = payload.get("text_provenance")
    return model.to(device), cfg, payload


def recover_log(output, step):
    """Retain crash-tail evidence while replaying from the last complete checkpoint."""
    path = output / "metrics.jsonl"
    if not path.exists():
        return
    lines = path.read_text().splitlines(keepends=True)
    keep, tail = [], []
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt metrics log before the last line")
            tail.append(line)
            continue
        (keep if payload["step"] <= step else tail).append(line)
    if not tail:
        return
    best = output / "best.pt"
    if best.exists() and torch.load(best, map_location="cpu", weights_only=True)["step"] > step:
        raise ValueError("best.pt is newer than the resume checkpoint; resume best.pt to retain its selection history")
    recovery = output / "recovery" / f"after_step_{step}_{time.time_ns()}"
    recovery.mkdir(parents=True)
    (recovery / "metrics_tail.jsonl").write_text("".join(tail))
    for artifact in output.glob("valid_step_*.json"):
        if int(artifact.stem.split("_")[-1]) > step:
            artifact.replace(recovery / artifact.name)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(keep))
    temporary.replace(path)
    write_json(recovery / "receipt.json", {"resume_step": step, "archived_log_lines": len(tail),
               "reason": "Replay uncheckpointed updates after interruption; old evidence retained"})


def run_experiment(cfg, device, resume=None, stop_after=None):
    """stop_after pauses without altering the planned LR or pseudo-label schedule."""
    cfg.validate()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this execution environment")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run: {output}")
    seed_everything(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    training, sampler, split_manifest = build_training_data(cfg)
    validation, records = prepare_validation(cfg)
    model = StudentTeacherSystem(cfg, initialize_text=resume is None).to(device)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    step, best, last_validation = 0, -1.0, None
    initial_hash = state_hash(model.students[0])
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if (checkpoint.get("format_version") != 3
                or ExperimentConfig(**checkpoint["config"]).to_dict() != cfg.to_dict()):
            raise ValueError("Resume requires exactly the saved protocol, including total step budget")
        if Path(resume).resolve().parent != output:
            raise ValueError("Resume in the original run directory")
        saved_split = json.loads((output / "split.json").read_text())
        if saved_split != split_manifest:
            raise ValueError("Dataset split changed since the checkpoint")
        hashes = json.loads((output / "source_hashes.json").read_text())
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError("Source changed since the run; resume from its snapshot or start a new run")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.text_provenance = checkpoint.get("text_provenance")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        step, best = checkpoint["step"], checkpoint["best_score"]
        recover_log(output, step)
    else:
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "split.json", split_manifest)
        write_json(output / "validation_audit.json", records)
        snapshot_sources(output)
        write_json(output / "manifest.json", {
            "protocol": "freqssl_v3", "primary_network": cfg.primary_network,
            "selection": "best primary-network Dice on all complete native valid volumes",
            "steps_per_epoch": len(sampler), "planned_steps": cfg.max_steps,
            "batch_size": cfg.batch_size, "labeled_bs": cfg.labeled_bs,
            "initial_student1_hash": initial_hash,
            "trainable_parameters": sum(p.numel() for p in model.student_parameters() if p.requires_grad),
            "slice_context": cfg.slice_context,
            "prompt_source": cfg.prompt_source,
            "internal_text": cfg.internal_text,
            "text_provenance": model.text_provenance,
            "text_vectors_at_inference": cfg.internal_text != "none",
            "prompt_teacher_at_inference": False,
            "image_exposures_per_center": 1 if cfg.slice_context == "none" else (4 if cfg.case_reference=='middle' else 3),
            "case_reference": cfg.case_reference, "context_position": cfg.context_position,
            "reference_condition": cfg.reference_condition,
            "reference_sources": cfg.reference_sources,
            "reference_teacher": cfg.reference_teacher,
            "reference_sampling": cfg.reference_sampling,
            "reference_gt_curriculum": cfg.reference_gt,
            "reference_cycle_weight": cfg.reference_cycle_weight,
            "parameters_per_student": [sum(p.numel() for p in s.parameters()) for s in model.students],
            "torch_version": str(torch.__version__), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "test_content_accessed": False,
            "frequency_source": ("online signed Haar reconstruction after geometry" if cfg.method == "dual_frequency" else "none"),
            "supervision": ("joint tissue/ordinal candidate sets; weak-image chain before CutMix"
                            if cfg.method == "dual_ordinal" and cfg.ordinal_use_routing
                            else "confidence-masked hard pseudo labels"),
            "not_official_reproduction": True, "efficacy_verified": False})
    end_step = min(cfg.max_steps, step + stop_after) if stop_after else cfg.max_steps
    if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0 and not (output / f"step_{step}.pt").exists():
        save_checkpoint(output / f"step_{step}.pt", model, optimizer, scaler, cfg, step, best)
    write_json(output / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    try:
        with (output / "metrics.jsonl").open("a", buffering=1) as log:
            while step < end_step:
                epoch, offset = divmod(step, len(sampler))
                training.set_epoch(epoch)
                sampler.set_epoch(epoch)
                loader = DataLoader(training, batch_sampler=sampler, num_workers=cfg.num_workers,
                                    pin_memory=device.type == "cuda",
                                    generator=torch.Generator().manual_seed(cfg.seed + epoch))
                for batch_index, batch in enumerate(loader):
                    if batch_index < offset:
                        continue
                    if time.monotonic() - started > cfg.max_seconds:
                        save_checkpoint(output / "last.pt", model, optimizer, scaler, cfg, step, best)
                        raise TimeoutError(f"Run exceeded {cfg.max_seconds} seconds; last checkpoint is resumable")
                    lr = cfg.base_lr * (1 - step / cfg.max_steps) ** 0.9
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    metrics = train_step(model, optimizer, batch, cfg, device, step, scaler)
                    step += 1
                    log.write(json.dumps(dict(metrics, kind="train", step=step, epoch=step/len(sampler), lr=lr), allow_nan=False) + "\n")
                    if step == 1 or step % cfg.log_every == 0:
                        print(f"{cfg.method} step={step}/{cfg.max_steps} loss={metrics['loss']:.4f} "
                              f"u_weight={metrics['unsupervised_weight']:.3f} "
                              f"coverage_s1={metrics.get('coverage_s1', 0):.4f}", flush=True)
                        write_json(output / "status.json", {"status": "running", "step": step,
                                   "pid": os.getpid(), "elapsed_seconds": time.monotonic()-started,
                                   "best_dice": None if best < 0 else best})
                    if step % cfg.eval_every == 0 or step == cfg.max_steps:
                        networks = ["student1", "teacher1"]
                        if len(model.students) == 2:
                            networks += ["student2", "teacher2", "students_mean", "teachers_mean"]
                        last_validation = evaluate_volumes(model, validation, records, device,
                                                           networks=networks, batch_size=cfg.eval_batch_size)
                        write_json(output / f"valid_step_{step}.json", last_validation)
                        score = last_validation[cfg.primary_network]["overall"]["dice"]
                        log.write(json.dumps({"kind": "validation", "step": step,
                                              "primary_network": cfg.primary_network, "dice": score}, allow_nan=False) + "\n")
                        if score is not None and score > best:
                            best = score
                            save_checkpoint(output / "best.pt", model, optimizer, scaler, cfg, step, best)
                        save_checkpoint(output / "last.pt", model, optimizer, scaler, cfg, step, best)
                        print(f"VALID {cfg.method} step={step} {cfg.primary_network} dice={score} best={best}", flush=True)
                        write_json(output / "status.json", {"status": "running", "step": step, "pid": os.getpid(), "best_dice": best})
                    if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                        save_checkpoint(output / f"step_{step}.pt", model, optimizer, scaler, cfg, step, best)
                    if step >= end_step:
                        break
        save_checkpoint(output / "last.pt", model, optimizer, scaler, cfg, step, best)
        result = {"status": "completed" if step == cfg.max_steps else "paused", "step": step,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  "seconds_this_invocation": time.monotonic()-started,
                  "best_valid_dice": None if best < 0 else best, "primary_network": cfg.primary_network,
                  "final_student1_hash": state_hash(model.students[0]),
                  "final_teacher1_hash": state_hash(model.teachers[0]),
                  "final_validation": last_validation, "test_evaluated": False,
                  "efficacy_verified": False}
        write_json(output / "result.json", result)
        write_json(output / "status.json", result)
        return result
    except BaseException as error:
        # A failed step may have invalid gradients; preserve the last successfully
        # written checkpoint rather than labeling partially updated state resumable.
        write_json(output / "status.json", {"status": "failed", "step": step,
                   "error": f"{type(error).__name__}: {error}",
                   "last_checkpoint": str(output / "last.pt") if (output / "last.pt").exists() else None})
        raise
