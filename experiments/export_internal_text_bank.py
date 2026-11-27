"""Cache fixed BiomedCLIP text vectors, using locally pinned weights only.

Run in the existing medclipseg environment; training/inference only need torch.
No image, annotation, task-trained teacher, or network service is used.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from open_clip import create_model_from_pretrained, get_tokenizer
    from freqssl.internal_text import TISSUES, TEXT_DIM
    from freqssl.prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, file_sha256
    from prompt_teacher.model import MODEL_ID, MODEL_REVISION
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--"))
    if (root / "refs/main").read_text().strip() != MODEL_REVISION:
        raise ValueError("Cached BiomedCLIP revision differs from the project pin")
    snapshot = root / "snapshots" / MODEL_REVISION
    tokenizer_root = root.parent / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer_revision = (tokenizer_root / "refs/main").read_text().strip()
    tokenizer_snapshot = tokenizer_root / "snapshots" / tokenizer_revision
    model, _ = create_model_from_pretrained("hf-hub:" + MODEL_ID, device="cpu")
    model.requires_grad_(False).eval()
    tokenizer = get_tokenizer("hf-hub:" + MODEL_ID)
    provenance = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "checkpoint_sha256": file_sha256(snapshot / "open_clip_pytorch_model.bin"),
        "config_sha256": file_sha256(snapshot / "open_clip_config.json"),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_assets_sha256": {name: file_sha256(tokenizer_snapshot / name)
            for name in ("vocab.txt", "config.json", "tokenizer_config.json")},
        "packages": {name: version(name) for name in ("torch", "open_clip_torch", "transformers")},
        "extraction": "eval encode_text(normalize=True), context_length=256, CPU float32",
        "text_encoder_frozen": True, "task_trained_checkpoint_used": False,
        "image_or_annotation_access": False, "network_access": False,
        "exporter_sha256": file_sha256(__file__),
    }
    # Both banks go through the same frozen encoder in one call.
    with torch.inference_mode():
        tokens = tokenizer(list(TEXT_PROMPTS + NEUTRAL_PROMPTS), context_length=256)
        vectors = model.encode_text(tokens, normalize=True).float().cpu()
    if vectors.shape != (8, TEXT_DIM) or not torch.isfinite(vectors).all():
        raise ValueError("Unexpected encoder output")
    args.output.mkdir(parents=True)
    manifest = {"provenance": provenance, "banks": {}}
    for index, (source, prompts) in enumerate((("text", TEXT_PROMPTS), ("neutral", NEUTRAL_PROMPTS))):
        embeddings = vectors[index * 4:(index + 1) * 4].clone()
        payload = {"format_version": 1, "source": source, "tissues": list(TISSUES),
                   "prompts": list(prompts), "embeddings": embeddings, "provenance": provenance}
        path = args.output / f"{source}.pt"
        torch.save(payload, path)
        manifest["banks"][source] = {"path": str(path.resolve()), "sha256": file_sha256(path),
            "prompts": list(prompts), "shape": list(embeddings.shape),
            "pairwise_cosine": (embeddings @ embeddings.T).tolist()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "shape_per_bank": [4, TEXT_DIM],
                      "encoder_revision": MODEL_REVISION, "network_access": False}), flush=True)


if __name__ == "__main__":
    main()
