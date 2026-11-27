import json
import logging
import os
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import SpiderDataset, TwoStreamBatchSampler, seed_worker
from losses import mean_teacher_loss
from model.mean_teacher import FTPStudent, MeanTeacher
from model.dual_student import SingleTeacherStudents
from model.prompts import load_text_features


def seed_everything(cfg):
    # Must be configured before the first CUDA BLAS operation.
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_determinism(seed=cfg.seed if cfg.deterministic else None)
    torch.use_deterministic_algorithms(cfg.deterministic)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = not cfg.deterministic
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.set_num_threads(cfg.cpu_threads)


def build_model(cfg, text_features=None):
    text, classes = text_features if text_features is not None else model_features(cfg)
    if classes.shape[0] != cfg.num_classes:
        raise ValueError(f"Expected {cfg.num_classes} class embeddings, got {classes.shape[0]}")
    return model_from_features(cfg, text, classes)


def model_features(cfg):
    if cfg.use_apgm or cfg.use_psgm:
        return load_text_features(cfg.bert_type, cfg.text_emb_path, cfg.text_length)
    # Shape-only buffers retain checkpoint compatibility and the 768-channel
    # visual backbone. Neither buffer participates in prompt-free predictions.
    return torch.zeros(1, 1, 1), torch.zeros(cfg.num_classes, 768)


def model_from_features(cfg, text, classes):
    if cfg.architecture == "shared":
        student = FTPStudent(text, classes, cfg.base_channels, cfg.use_apgm, cfg.use_psgm)
        return MeanTeacher(student, cfg.ema_decay)
    return SingleTeacherStudents(text, classes, cfg.base_channels, cfg.ema_decay,
                                 dual=cfg.architecture == "dual", secondary_seed=cfg.seed + 1,
                                 use_apgm=cfg.use_apgm, use_psgm=cfg.use_psgm)


def student_parameters(model):
    secondary = getattr(model, "student2", None)
    return chain(model.student.parameters(), secondary.parameters() if secondary is not None else ())


def make_views(batch, device, use_frequency=True, noise_std=0.0, gain_std=0.0):
    tensors = [batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
               for key in ("image", "lf", "hf")]
    if gain_std:
        gain = 1 + (torch.randn(tensors[0].shape[0], 1, 1, 1, device=device) * gain_std).clamp(-0.3, 0.3)
        tensors = [value * gain for value in tensors]
    if noise_std:
        tensors = [value + (torch.randn_like(value) * noise_std).clamp(-2 * noise_std, 2 * noise_std)
                   for value in tensors]
    raw, low, high = tensors
    if not use_frequency:
        low = high = raw
    return torch.cat((raw, low), dim=1), torch.cat((raw, high), dim=1)


def train_batch(model, optimizer, batch, cfg, device, epoch,
                feedback_ready=False, feedback_activated_epoch=None):
    model.train()
    labeled = batch["is_labeled"].to(device=device, dtype=torch.bool)
    labels = batch["label"].to(device=device, dtype=torch.long)
    low, high = make_views(batch, device, cfg.use_frequency, cfg.student_noise_std, cfg.student_gain_std)
    teacher_low, teacher_high = make_views(batch, device, cfg.use_frequency, cfg.teacher_noise_std)
    if cfg.secondary_view == "lf":
        high = low
        teacher_high = teacher_low
    with torch.no_grad():
        targets = model.teacher_predictions(teacher_low[~labeled], teacher_high[~labeled])
    optimizer.zero_grad(set_to_none=True)
    outputs = model(low, high)
    loss, metrics = mean_teacher_loss(outputs, targets, labels, labeled, epoch, cfg,
                                     feedback_ready, feedback_activated_epoch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {metrics}")
    loss.backward()
    # Raises before optimizer/EMA updates if a module produces invalid gradients.
    grad_norm = torch.nn.utils.clip_grad_norm_(student_parameters(model), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    metrics["ema_decay"] = model.update_teacher()
    metrics["grad_norm"] = float(grad_norm)
    return metrics


def binary_metrics(prediction, label):
    # Preserve FTP's convention: ignore a class if it is absent from ground truth.
    if not label.any():
        return np.nan, np.nan
    if not prediction.any():
        return 0.0, np.nan
    dice = 2.0 * (prediction & label).sum() / (prediction.sum() + label.sum())
    pred_surface = prediction ^ binary_erosion(prediction)
    label_surface = label ^ binary_erosion(label)
    distances = np.concatenate((distance_transform_edt(~label_surface)[pred_surface],
                                distance_transform_edt(~pred_surface)[label_surface]))
    return float(dice), float(np.percentile(distances, 95))


def finite_mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    result = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    return result


def prediction_scores(model, low, high, cfg, network):
    """Select one deployment network; never silently average teachers/students."""
    if cfg.secondary_view == "lf":
        high = low
    if cfg.architecture == "shared":
        if network not in ("student", "teacher"):
            raise ValueError("shared supports only student or teacher")
        outputs = getattr(model, network)(low, high, return_plain=False)
        if cfg.eval_branch == "mean":
            return (outputs["lf_prompt"].softmax(1) + outputs["hf_prompt"].softmax(1)) / 2, "mean"
        return outputs[f"{cfg.eval_branch}_prompt"], cfg.eval_branch
    if network == "teacher":
        if cfg.eval_branch != "lf":
            raise ValueError("The single teacher has only an LF/APGM route; use eval_branch=lf")
        return model.teacher(low, return_plain=False)["lf_prompt"], "lf"
    if network in ("student", "student1"):
        return model.student(low, return_plain=False)["lf_prompt"], "lf"
    if model.student2 is None:
        raise ValueError("This checkpoint has no second student")
    if network == "student2":
        return model.student2(high, return_plain=False)["hf_prompt"], "hf"
    if network == "students_mean":
        first = model.student(low, return_plain=False)["lf_prompt"].softmax(1)
        second = model.student2(high, return_plain=False)["hf_prompt"].softmax(1)
        return (first + second) / 2, "mean"
    raise ValueError(f"Unknown network: {network}")


@torch.no_grad()
def evaluate(model, loader, cfg, device, network="teacher", max_cases=None):
    was_training = model.training
    model.eval()
    all_metrics = []
    for index, batch in enumerate(loader):
        if max_cases is not None and index >= max_cases:
            break
        low, high = make_views(batch, device, cfg.use_frequency)
        scores, branch = prediction_scores(model, low, high, cfg, network)
        labels = batch["label"]
        prediction = scores.argmax(1).float().unsqueeze(1)
        prediction = F.interpolate(prediction, size=labels.shape[-2:], mode="nearest")
        prediction = prediction[0, 0].cpu().numpy().astype(np.int64)
        label = labels[0].numpy()
        all_metrics.append([binary_metrics(prediction == k, label == k)
                            for k in range(1, cfg.num_classes)])
    model.train(was_training)
    if not all_metrics:
        raise ValueError("Evaluation split is empty")
    values = np.asarray(all_metrics)
    per_case = finite_mean(values, axis=1)
    means = finite_mean(per_case, axis=0)
    per_class = finite_mean(values, axis=0)

    def number(value):
        return float(value) if np.isfinite(value) else None

    return {
        "network": network, "branch": branch, "cases": len(values),
        "dice": number(means[0]), "hd95_pixels": number(means[1]),
        "per_class_dice": [number(x) for x in per_class[:, 0]],
        "per_class_hd95_pixels": [number(x) for x in per_class[:, 1]],
        "aggregation": "foreground GT-present classes per slice, then mean over slices",
    }


def save_checkpoint(path, model, optimizer, cfg, epoch, global_step, best_dice, training_state=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format_version": 2, "model": model.state_dict(), "training_state": training_state,
        "optimizer": optimizer.state_dict(), "config": cfg.to_dict(),
        "epoch": epoch, "global_step": global_step, "best_dice": best_dice,
    }, temporary)
    temporary.replace(path)


def load_checkpoint_model(path, device):
    """Evaluation is self-contained: cached BERT/class tensors live in the state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (1, 2):
        raise ValueError("Not an FTP Mean Teacher checkpoint")
    cfg = Config(**payload["config"])
    state = payload["model"]
    model = model_from_features(cfg, state["student.text_features"], state["student.class_embeddings"])
    model.load_state_dict(state, strict=True)
    return model.to(device), cfg


def run_training(cfg, device, smoke=False):
    cfg.validate()
    if cfg.feedback_require_ready and cfg.feedback_weight > 0 and not smoke:
        raise ValueError("Use experiments/compare_frequency.py for feedback gated by complete 3D validation")
    seed_everything(cfg)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_file = output / "metrics.jsonl"
    if metrics_file.exists() or (output / "last.pt").exists():
        raise FileExistsError(f"Run directory already contains results; choose a new --output-dir: {output}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "train.log"),
    ], force=True)
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    training = SpiderDataset(cfg.root_path, "train", cfg.image_size, cfg.labelnum, cfg.max_samples)
    validation = SpiderDataset(cfg.root_path, "valid", cfg.image_size)
    training.transform.set_random_state(seed=cfg.seed)
    sampler = TwoStreamBatchSampler(range(cfg.labelnum), range(cfg.labelnum, len(training)),
                                    cfg.batch_size, cfg.labeled_bs, cfg.seed)
    loader = DataLoader(training, batch_sampler=sampler, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    (output / "labeled_files.txt").write_text("\n".join(training.names[:cfg.labelnum]) + "\n")
    (output / "unlabeled_files.txt").write_text("\n".join(training.names[cfg.labelnum:]) + "\n")
    logging.info("Visual weights start from scratch; text assets enabled=%s", cfg.use_apgm or cfg.use_psgm)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(student_parameters(model), lr=cfg.base_lr,
                                  weight_decay=cfg.weight_decay, eps=1e-5, betas=(0.9, 0.999))
    trainable = sum(p.numel() for p in student_parameters(model) if p.requires_grad)
    total_steps = min(cfg.max_iterations, cfg.max_epochs * len(loader))
    logging.info("device=%s student_parameters=%d labeled=%d unlabeled=%d steps=%d",
                 device, trainable, cfg.labelnum, len(training) - cfg.labelnum, total_steps)
    global_step = 0
    best_dice = -1.0
    last_validation = None

    def record(payload):
        with metrics_file.open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    for epoch in range(cfg.max_epochs):
        sampler.epoch = epoch
        for batch in loader:
            lr = cfg.base_lr * (1.0 - global_step / total_steps) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_batch(model, optimizer, batch, cfg, device, epoch)
            global_step += 1
            metrics.update(kind="train", step=global_step, epoch=epoch, lr=lr)
            record(metrics)
            if smoke or global_step == 1 or global_step % 20 == 0:
                logging.info("step=%d loss=%.5f supervised=%.5f consistency=%.5f prompt=%.5f",
                             global_step, metrics["loss"], metrics["supervised"], metrics["consistency"], metrics["prompt"])
            if global_step >= total_steps:
                break
        final_epoch = global_step >= total_steps or epoch + 1 == cfg.max_epochs
        if (epoch + 1) % cfg.eval_interval == 0 or final_epoch:
            last_validation = evaluate(model, val_loader, cfg, device, max_cases=2 if smoke else None)
            record(dict(last_validation, kind="validation", epoch=epoch, step=global_step))
            logging.info("validation=%s", last_validation)
            if last_validation["dice"] is not None and last_validation["dice"] > best_dice:
                best_dice = last_validation["dice"]
                if not smoke:
                    save_checkpoint(output / "best.pt", model, optimizer, cfg, epoch, global_step, best_dice)
        if not smoke:
            save_checkpoint(output / "last.pt", model, optimizer, cfg, epoch, global_step, best_dice)
        if final_epoch:
            break
    summary = {"smoke_test": smoke, "steps": global_step, "student_parameters": trainable,
               "ema_updates": int(model.ema_updates), "validation": last_validation,
               "best_dice": None if best_dice < 0 else best_dice,
               "checkpoints_saved": not smoke, "test_split_evaluated": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    logging.info("Finished: %s", output)
    return summary
