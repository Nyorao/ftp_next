"""Load the original, frozen text inputs without loading visual checkpoints."""

from pathlib import Path

import torch


# Same fixed prompt used by Spider_dataset.generate_text_encode(mode='fixed').
SPIDER_PROMPT = (
    "An MR image of a spine, containing intervertebral disc L5/S, vertebra L5, "
    "intervertebral disc L4/L5, vertebra L4, intervertebral disc L3/L4, vertebra L3, "
    "intervertebral disc L2/L3, vertebra L2, intervertebral disc L1/L2, vertebra L1, "
    "intervertebral disc T12/L1, vertebra T12, intervertebral disc T11/T12, vertebra T11, "
    "intervertebral disc T10/T11, vertebra T10, intervertebral disc T9/T10, vertebra T9 "
    "and the spinal canal."
)


@torch.no_grad()
def load_text_features(bert_path, class_embedding_path, text_length=112):
    """Cache eval-mode BERT features for the label-independent, fixed prompt.

    The returned tensors are registered as model buffers and saved in checkpoints.
    Only local assets are read; no downloads or remote model code are allowed.
    """
    from transformers import AutoModel, AutoTokenizer

    if not Path(bert_path).is_dir():
        raise FileNotFoundError(f"Local BERT directory does not exist: {bert_path}")
    if not Path(class_embedding_path).is_file():
        raise FileNotFoundError(f"Class embeddings do not exist: {class_embedding_path}")
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    bert = AutoModel.from_pretrained(bert_path, local_files_only=True)
    bert.requires_grad_(False)
    bert.eval()
    tokens = tokenizer(
        SPIDER_PROMPT, padding="max_length", max_length=text_length,
        truncation=True, return_tensors="pt",
    )
    features = bert(**tokens, return_dict=True).last_hidden_state.detach().clone()
    classes = torch.load(class_embedding_path, map_location="cpu", weights_only=True)
    if not isinstance(classes, torch.Tensor) or classes.ndim != 2:
        raise ValueError("Class embedding file must contain a [classes, channels] tensor")
    return features.float(), classes.detach().float()
