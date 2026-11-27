import argparse
import json

import torch
def main():
    parser = argparse.ArgumentParser(description="Evaluate freqssl v3 on complete native SPIDER volumes; legacy checkpoints remain supported")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root-path")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--network", choices=("student", "teacher", "student1", "student2", "teacher1", "teacher2", "students_mean", "teachers_mean"))
    parser.add_argument("--output-dir")
    parser.add_argument("--branch", choices=("lf", "hf", "mean"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    version = torch.load(args.checkpoint, map_location="cpu", weights_only=True).get("format_version")
    if version == 3:
        if args.branch is not None:
            parser.error("v3 selects networks, not legacy LF/HF branches")
        from freqssl.cli import evaluate_checkpoint
        network = {"student": "student1", "teacher": "teacher1"}.get(args.network, args.network)
        evaluate_checkpoint(args.checkpoint, args.root_path, args.split,
                            [network] if network else None, device, args.output_dir)
        return
    from torch.utils.data import DataLoader
    from dataset import SpiderDataset
    from trainer import evaluate, load_checkpoint_model
    model, cfg = load_checkpoint_model(args.checkpoint, device)
    torch.set_num_threads(cfg.cpu_threads)
    if args.root_path is not None:
        cfg.root_path = args.root_path
    if args.branch is not None:
        cfg.eval_branch = args.branch
    dataset = SpiderDataset(cfg.root_path, args.split, cfg.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(json.dumps(evaluate(model, loader, cfg, device, args.network or "teacher"), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
