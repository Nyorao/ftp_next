import argparse
from pathlib import Path

def legacy_main():
    import torch
    from config import load_config
    from trainer import run_training
    parser = argparse.ArgumentParser(description="FTP modules with a U-Net Mean Teacher")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/ftp_mt.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true", help="Two real-data steps and two validation cases; no checkpoints")
    for option in ("max-iterations", "max-epochs", "num-workers", "batch-size", "labeled-bs",
                   "image-size", "base-channels"):
        parser.add_argument("--" + option, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in ("output_dir", "max_iterations", "max_epochs", "num_workers", "batch_size",
                 "labeled_bs", "image_size", "base_channels"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.smoke_test:
        cfg.max_iterations = 2
        cfg.max_epochs = 1
        cfg.batch_size = 2
        cfg.labeled_bs = 1
        cfg.num_workers = 0
        if args.output_dir is None:
            cfg.output_dir = "./outputs/smoke"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")
    run_training(cfg, device, smoke=args.smoke_test)


def main():
    import yaml
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", default=str(Path(__file__).parent / "configs/v3/dual_frequency.yaml"))
    args, _ = selector.parse_known_args()
    values = yaml.safe_load(Path(args.config).read_text())
    if values.get("schema_version") == 3:
        from freqssl.cli import train_main
        train_main()
    else:
        legacy_main()


if __name__ == "__main__":
    main()
