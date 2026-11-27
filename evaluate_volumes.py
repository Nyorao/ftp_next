"""Audit and evaluate fixed FTP-MT checkpoints on complete SPIDER 3D cases."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataset import SpiderDataset
from spider_volume import (CLASS_NAMES, aggregate_cases, audit_case, group_complete_slices,
                           reconstruct, save_volume, volume_metrics)
from trainer import load_checkpoint_model, make_views


ROOT = Path(__file__).resolve().parent
ARMS = ("anchor", "dual_no_feedback", "dual_feedback")
NETWORKS = ("teacher", "student1", "student2", "students_mean")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def predict_case(model, cfg, dataset, indices, record, networks, device):
    slices = {network: [] for network in networks}
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)
    for batch in loader:
        low, high = make_views(batch, device, cfg.use_frequency)
        if cfg.secondary_view == "lf":
            high = low
        scores = {}
        if "teacher" in networks:
            scores["teacher"] = model.teacher(low, return_plain=False)["lf_prompt"]
        if "student1" in networks or "students_mean" in networks:
            scores["student1"] = model.student(low, return_plain=False)["lf_prompt"]
        if "student2" in networks or "students_mean" in networks:
            scores["student2"] = model.student2(high, return_plain=False)["hf_prompt"]
        if "students_mean" in networks:
            scores["students_mean"] = (scores["student1"].softmax(1) + scores["student2"].softmax(1)) / 2
        for network in networks:
            labels = scores[network].argmax(1).float().unsqueeze(1)
            labels = F.interpolate(labels, size=batch["label"].shape[-2:], mode="nearest")
            # Dataset PILReader-compatible axes are W,H; restore native PNG H,W.
            slices[network].append(labels[0, 0].cpu().numpy().astype(np.uint8).T)
    return {network: reconstruct(values, record["slice_axis"], record["undo_vertical_flip"],
                                 record["array_shape_zyx"]) for network, values in slices.items()}


def score_case(prediction, record, path, surfaces):
    reference = sitk.ReadImage(record["reference"])
    target = sitk.GetArrayFromImage(reference)
    save_volume(path, prediction, reference)
    return {"case": record["case"], "prediction": str(path),
            "geometry_and_saved_voxels_verified": True,
            "classes": volume_metrics(prediction, target, tuple(reversed(reference.GetSpacing())), surfaces)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS))
    parser.add_argument("--arms", nargs="+", help="Completed experiment groups; defaults to all groups in the protocol")
    parser.add_argument("--checkpoint-name", choices=("last.pt", "best.pt"), default="last.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.metric_workers < 1:
        parser.error("metric-workers must be positive")
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an evaluation: {output}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "evaluate.log")], force=True)
    training = json.loads((run / "manifest.json").read_text())
    training_results = json.loads((run / "results.json").read_text())
    frequency_protocol = training.get("protocol") == "prompt_free_frequency_v1"
    available_arms = training["arms"] if frequency_protocol else list(ARMS)
    arms = args.arms or available_arms
    if not arms or len(set(arms)) != len(arms) or any(arm not in available_arms for arm in arms):
        raise ValueError("Requested arms must be unique members of the saved protocol")
    if any(arm not in training_results["arms"] for arm in arms):
        raise ValueError("Every requested arm must have a completed result")
    if frequency_protocol:
        if any(training_results["arms"][arm]["steps"] != training["arguments"]["steps"] for arm in arms):
            raise ValueError("Requested frequency arms did not complete the fixed update budget")
        negative = training_results["arms"].get("dual_frequency_no_feedback", {})
        if negative.get("negative_control_exact_match") is False:
            raise ValueError("Frequency negative control failed")
    else:
        if training_results["status"] != "completed":
            raise ValueError("Only completed equal-budget pilot arms may be evaluated here")
        if args.checkpoint_name != "last.pt":
            raise ValueError("The legacy pilot has only fixed last.pt checkpoints")
        if any(v != 0 for v in training_results["arms"]["dual_no_feedback"]["negative_control_max_abs_difference"].values()):
            raise ValueError("The pilot negative control failed")
    png_root = Path(training["config"]["root_path"])
    volume_root = png_root.parent / "mha"
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is not accessible")
    started = time.monotonic()
    manifest = {
        "protocol": "SPIDER_native_3D_v1", "training_run": str(run),
        "png_root": str(png_root), "mha_root": str(volume_root), "splits": args.splits,
        "checkpoint_rule": ("best.pt selected by complete valid teacher 3D Dice" if args.checkpoint_name == "best.pt"
                            else "fixed last.pt at the saved training update budget"),
        "arms": arms, "checkpoint_name": args.checkpoint_name,
        "networks": args.networks, "device": str(device), "metric_workers": args.metric_workers,
        "class_names": list(CLASS_NAMES),
        "inference": "2D slices at 224 -> argmax -> nearest native PNG size -> undo W/H transpose and recorded flip -> original 3D MHA axes",
        "postprocessing": "none; no TTA; no connected-component filtering",
        "aggregation": "per-volume mean over GT-present foreground classes, then equal-weight mean over volumes",
        "absent_policy": "GT absent: excluded from primary means, false positives counted; GT present/prediction empty: Dice/IoU=0, distances undefined; both empty: excluded",
        "supplementary_policy": "dice_union_present/iou_union_present also score GT-absent false positives as zero, excluding only both-empty pairs",
        "surface_metrics": "teacher only; HD95 of pooled bidirectional six-neighbour surface distances; ASD is directed prediction-to-GT mean, matching original evaluator; units mm from reversed MHA spacing",
        "scope": ("complete-volume evaluation of a one-seed prompt-free frequency ablation; convergence unverified"
                  if frequency_protocol else "complete-volume evaluation of a one-seed 240-step subset-trained pilot; not a converged benchmark"),
    }
    write_json(output / "manifest.json", manifest)
    sources = output / "sources"
    sources.mkdir()
    source_files = [Path(__file__), ROOT / "spider_volume.py", ROOT / "dataset.py", ROOT / "trainer.py",
                    ROOT / "config.py", *sorted((ROOT / "model").glob("*.py"))]
    source_hashes = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        destination = sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = file_hash(path)
    write_json(output / "source_hashes.json", source_hashes)
    results = {"status": "running", "splits": {}}
    write_json(output / "results.json", results)
    try:
        groups = {split: group_complete_slices(sorted(p.name for p in (png_root / split / "image").glob("*.png")))
                  for split in ("train", "valid", "test")}
        overlaps = {f"{left}/{right}": sorted(set(groups[left]) & set(groups[right]))
                    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))}
        if any(overlaps.values()):
            raise ValueError(f"Patient overlap across splits: {overlaps}")
        audit = {"patient_overlap": overlaps, "splits": {}}
        for split in args.splits:
            records = []
            for case, names in groups[split].items():
                records.append(audit_case(png_root, volume_root, split, case, names))
            audit["splits"][split] = {"cases": len(records), "slices": sum(r["slice_count"] for r in records),
                                      "case_records": records}
            logging.info("AUDIT %s: %d complete volumes, all PNG-label reconstructions exactly match MHA", split, len(records))
        write_json(output / "data_audit.json", audit)
        if args.audit_only:
            results.update(status="audit_completed", seconds=time.monotonic() - started)
            write_json(output / "results.json", results)
            return
        for split in args.splits:
            dataset = SpiderDataset(png_root, split, training["config"]["image_size"])
            index_by_name = {name: i for i, name in enumerate(dataset.names)}
            records = audit["splits"][split]["case_records"]
            split_results = results["splits"][split] = {}
            reusable = {}
            for arm in arms:
                checkpoint = run / arm / args.checkpoint_name
                checkpoint_hash = file_hash(checkpoint)
                model, cfg = load_checkpoint_model(checkpoint, device)
                if cfg.architecture not in ("anchor", "dual") or cfg.num_classes != 20:
                    raise ValueError("Expected the independent-student 20-class SPIDER checkpoint")
                model.eval()
                hashes = {"teacher": module_hash(model.teacher), "student1": module_hash(model.student)}
                if model.student2 is not None:
                    hashes["student2"] = module_hash(model.student2)
                    hashes["students_mean"] = hashes["student1"] + ":" + hashes["student2"]
                selected = [n for n in args.networks if n in hashes]
                # Reuse only identical network states AND identical preprocessing/metric settings.
                keys = {n: (n, hashes[n], cfg.image_size, cfg.use_frequency, cfg.secondary_view) for n in selected}
                fresh = [n for n in selected if keys[n] not in reusable]
                collected = {n: [] for n in fresh}
                for network in fresh:
                    (output / split / arm / network / "mha").mkdir(parents=True)
                pending = {}

                def collect(futures):
                    for future in futures:
                        network = pending.pop(future)
                        case_result = future.result()
                        collected[network].append(case_result)
                        logging.info("SCORED %s/%s/%s %s (%d/%d)", split, arm, network,
                                     case_result["case"], len(collected[network]), len(records))

                with ThreadPoolExecutor(max_workers=args.metric_workers) as executor:
                    for record in records:
                        if not fresh:
                            break
                        indices = [index_by_name[name] for name in record["slice_names"]]
                        predicted = predict_case(model, cfg, dataset, indices, record, fresh, device)
                        for network, volume in predicted.items():
                            path = output / split / arm / network / "mha" / f"{record['case']}.mha"
                            future = executor.submit(score_case, volume, record, path, network == "teacher")
                            pending[future] = network
                        while len(pending) >= args.metric_workers * 2:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            collect(done)
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        collect(done)
                for network in selected:
                    identity = f"{arm}/{network}"
                    if network in fresh:
                        cases = sorted(collected[network], key=lambda value: value["case"])
                        summary = aggregate_cases(cases)
                        payload = dict(summary, cases=cases, arm=arm, network=network,
                                       checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                                       module_state_hash=hashes[network], reused_from=None)
                        reusable[keys[network]] = identity
                    else:
                        source = reusable[keys[network]]
                        payload = dict(split_results[source], arm=arm, network=network,
                                       checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                                       reused_from=source)
                    directory = output / split / arm / network
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / "result.json", payload)
                    export_csv(directory / "per_class.csv", payload["per_class"])
                    export_csv(directory / "per_case.csv", payload["per_case"])
                    split_results[identity] = payload
                    logging.info("DONE %s/%s: 3D Dice %.6f, reused_from=%s", split, identity,
                                 payload["overall"]["dice"], payload["reused_from"])
                write_json(output / "results.json", results)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        results.update(status="completed", seconds=time.monotonic() - started)
        write_json(output / "results.json", results)
    except Exception as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}", seconds=time.monotonic() - started)
        write_json(output / "results.json", results)
        raise


if __name__ == "__main__":
    main()
