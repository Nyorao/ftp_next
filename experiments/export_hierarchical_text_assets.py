"""Offline, pinned BiomedCLIP tokens and paired image targets for a small probe.

Only eight already-labeled training slices are read. Four different cases train
the probe; four further cases diagnose transfer. Neither is a validation set.
"""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/v3/context_efficiency_reference.yaml'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from torch.nn import functional as F
    from open_clip import create_model_from_pretrained, get_tokenizer
    from freqssl.config import load_config
    from freqssl.data import build_training_data, _read_transposed, _case_id
    from freqssl.prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, file_sha256
    from prompt_teacher.model import MODEL_ID, MODEL_REVISION
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cache = Path.home() / '.cache/huggingface/hub'
    model_root = cache / ('models--' + MODEL_ID.replace('/', '--'))
    if (model_root / 'refs/main').read_text().strip() != MODEL_REVISION:
        raise ValueError('Cached model differs from pinned revision')
    snapshot = model_root / 'snapshots' / MODEL_REVISION
    token_root = cache / 'models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract'
    token_revision = (token_root / 'refs/main').read_text().strip()
    token_snapshot = token_root / 'snapshots' / token_revision
    model, transform = create_model_from_pretrained('hf-hub:' + MODEL_ID, device='cpu')
    model.requires_grad_(False).eval()
    tokenizer = get_tokenizer('hf-hub:' + MODEL_ID)
    provenance = dict(model_id=MODEL_ID, revision=MODEL_REVISION,
        checkpoint_sha256=file_sha256(snapshot / 'open_clip_pytorch_model.bin'),
        config_sha256=file_sha256(snapshot / 'open_clip_config.json'),
        tokenizer_revision=token_revision,
        tokenizer_assets_sha256={n:file_sha256(token_snapshot/n) for n in ('vocab.txt','config.json','tokenizer_config.json')},
        packages={n:version(n) for n in ('torch','open_clip_torch','transformers')},
        extraction='eval CPU float32; 256-token context; hidden states 1..12, first 32 positions',
        pretrained_towers_frozen=True, task_trained_checkpoint_used=False, network_access=False,
        exporter_sha256=file_sha256(__file__))
    with torch.inference_mode():
        tokens = tokenizer(list(TEXT_PROMPTS + NEUTRAL_PROMPTS), context_length=256)
        pad = tokenizer.tokenizer.pad_token_id
        if (tokens[:,32:] != pad).any():
            raise ValueError('A fixed prompt exceeds the 32-token cache')
        mask = tokens != pad
        states = model.text.transformer(input_ids=tokens, attention_mask=mask.long(),
                                       output_hidden_states=True, return_dict=True).hidden_states
        hidden = torch.stack(states[1:])[:,:,:32].float().clone()
        anchors = model.encode_text(tokens, normalize=True).float()
        # Remove padding and tokenizer control tokens from dense token attention.
        valid = mask[:,:32].clone()
        for token_id in tokenizer.tokenizer.all_special_ids:
            valid &= tokens[:,:32] != token_id
    cfg = load_config(args.config)
    dataset, sampler, _ = build_training_data(cfg)
    selected, cases = [], set()
    for batch in sampler:
        for index in batch[:cfg.labeled_bs]:
            case = _case_id(dataset.names[index])
            if case not in cases:
                selected.append(index)
                cases.add(case)
            if len(selected) == 8:
                break
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError('Need eight distinct labeled training cases')
    images, labels, rgb = [], [], []
    normalization = transform.transforms[-1]
    mean = torch.tensor(normalization.mean)[:,None,None]
    std = torch.tensor(normalization.std)[:,None,None]
    for index in selected:
        sample = dataset[index]
        raw = _read_transposed(dataset.root / 'image' / dataset.names[index], torch.float32)[None]
        if raw.min() < 0 or raw.max() > 255:
            raise ValueError('Expected 8-bit image intensity for the pinned preprocessing')
        raw = F.interpolate(raw[None], size=dataset.image_size, mode='bilinear', align_corners=False)[0]
        raw, _ = dataset._geometry(raw, None, index)
        standardized = (raw-raw.mean()) / raw.std(unbiased=False).clamp_min(1e-6)
        torch.testing.assert_close(standardized, sample['image'][1:2], atol=1e-6, rtol=1e-6)
        images.append(sample['image'])
        labels.append(sample['label'])
        # Undo the legacy data tensor transpose for the foundation encoder's
        # normal anatomical orientation. Global targets have no pixel axes.
        rgb.append((raw.transpose(-1,-2).expand(3,-1,-1)/255 - mean)/std)
    with torch.inference_mode():
        visual = model.encode_image(torch.stack(rgb), normalize=True).float()
    args.output.mkdir(parents=True)
    manifest = dict(provenance=provenance, banks={}, probe=dict(
        train_files=[dataset.names[i] for i in selected[:4]],
        heldout_training_files=[dataset.names[i] for i in selected[4:]],
        case_disjoint=True, valid_or_test_content_used=False, unlabeled_annotations_used=False,
        image_preprocess='common resized/augmented center; native axis order; RGB /255; checkpoint mean/std; no extra crop',
        image_mean=list(normalization.mean), image_std=list(normalization.std),
        global_targets_shape=list(visual.shape), config=cfg.to_dict()))
    for j,(source,prompts) in enumerate((('text',TEXT_PROMPTS),('neutral',NEUTRAL_PROMPTS))):
        path = args.output / (source+'.pt')
        payload = dict(format_version=2, source=source, prompts=list(prompts), provenance=provenance,
            hidden_tokens=hidden[:,j*4:j*4+4].clone(), token_mask=valid[j*4:j*4+4].clone(),
            anchors=anchors[j*4:j*4+4].clone())
        torch.save(payload,path)
        manifest['banks'][source] = dict(sha256=file_sha256(path), valid_token_counts=payload['token_mask'].sum(-1).tolist(),
                                         hidden_shape=list(payload['hidden_tokens'].shape))
    batches = {}
    for name, indices in (('train',slice(0,4)),('heldout',slice(4,8))):
        batches[name] = dict(image=torch.stack(images)[indices],label=torch.stack(labels)[indices],
                             global_target=visual[indices].clone())
    torch.save(batches,args.output/'probe_batches.pt')
    manifest['probe']['batch_sha256'] = file_sha256(args.output/'probe_batches.pt')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),hidden_shape=list(hidden.shape),
                          cases=len(cases),provenance=provenance)),flush=True)


if __name__ == '__main__':
    main()
