"""Image-only frozen-teacher caches and budget-checked reference sampling."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import _case_id, _read_transposed
from .reference_conditioning import row_prior, si_coordinates


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_probabilities(value):
    """The declared FP16 storage roundtrip is also used at fresh deployment."""
    value = value.detach().half().float()
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError('Invalid frozen teacher probabilities')
    return value / value.sum(-3, keepdim=True).clamp_min(1e-8)


def probability_statistics(value):
    prediction = value.argmax(0)
    return dict(canal_area=int((prediction == 10).sum()), foreground_area=int((prediction != 0).sum()))


def select_reference(statistics):
    if not statistics:
        raise ValueError('A reference requires a nonempty case')
    middle = (len(statistics) - 1) // 2
    if max(row['canal_area'] for row in statistics) == 0:
        return middle
    return max(range(len(statistics)), key=lambda i: (statistics[i]['canal_area'], -abs(i - middle), -i))


def candidate_references(statistics):
    maximum = max(row['foreground_area'] for row in statistics)
    candidates = [i for i, row in enumerate(statistics)
                  if row['foreground_area'] > 0 and row['foreground_area'] >= .1 * maximum]
    return candidates or [select_reference(statistics)]


def group_names(names):
    groups = {}
    for name in names:
        groups.setdefault(_case_id(name), []).append(name)
    for case, values in groups.items():
        values.sort(key=lambda n: int(n.rsplit('_', 2)[1]))
        count = int(values[0].rsplit('_', 1)[1].split('.')[0])
        if len(values) != count or [int(n.rsplit('_', 2)[1]) for n in values] != list(range(count)):
            raise ValueError(f'Incomplete reference image pool: {case}')
    return groups


class ReferencePredictionCache:
    def __init__(self, root, dataset, teacher_sha256, *, verify_images=True):
        self.root = Path(root)
        path = self.root / 'manifest.json'
        self.manifest_hash = file_hash(path)
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        if (m.get('version') != 1 or m.get('complete') is not True or m['split'] != dataset.split
                or m['image_size'] != list(dataset.image_size)
                or m['teacher_sha256'] != teacher_sha256 or set(m['slices']) != set(dataset.names)):
            raise ValueError('Reference cache provenance, geometry or inventory mismatch')
        self.groups = group_names(dataset.names)
        self.selected, self.candidates = {}, {}
        for case, names in self.groups.items():
            stats = [m['slices'][name] for name in names]
            self.selected[case], self.candidates[case] = select_reference(stats), candidate_references(stats)
        if verify_images:
            for name in dataset.names:
                if file_hash(dataset.root / 'image' / name) != m['slices'][name]['image_sha256']:
                    raise ValueError(f'Reference image changed: {name}')
        self.checked = set()

    def read(self, name):
        row = self.manifest['slices'][name]
        path = self.root / 'probabilities' / (name + '.npy')
        if name not in self.checked:
            if file_hash(path) != row['probability_sha256']:
                raise ValueError(f'Reference probability changed: {name}')
            self.checked.add(name)
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float16 or value.shape != (20, *self.manifest['image_size']):
            raise ValueError(f'Invalid reference probability shape/dtype: {name}')
        value = torch.from_numpy(value.astype(np.float32))
        if not torch.isfinite(value).all() or (value < 0).any() or (value.sum(0) <= 0).any():
            raise ValueError(f'Invalid reference probabilities: {name}')
        return value / value.sum(0, keepdim=True)


@torch.inference_mode()
def build_prediction_cache(checkpoint, dataset, destination, device='cpu', batch_size=8):
    """Read image pixels only, even when dataset also has authorized targets."""
    from .engine import write_json
    from .inference import load_single_network
    from .context_cache import iter_cached_logits
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'probabilities').mkdir()
    started = time.monotonic()
    model, cfg, receipt = load_single_network(checkpoint, device, 'teacher1')
    if cfg.slice_context != 'adaptive' or cfg.case_reference != 'none' or cfg.context_position != 'none' or cfg.reference_condition != 'none':
        raise ValueError('Reference source must be the unconditioned A model')
    if tuple(dataset.image_size) != (cfg.image_size, cfg.image_size):
        raise ValueError('Reference source preprocessing resolution differs')
    m = dict(version=1, complete=False, split=dataset.split, image_size=list(dataset.image_size),
             teacher_sha256=file_hash(checkpoint), teacher=receipt, slices={},
             precision='FP16 stored; FP32 class renormalization on read', annotation_files_read=0,
             torch_version=str(torch.__version__), numeric_settings=dict(
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                 cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    write_json(destination / 'status.json', dict(status='running', annotation_files_read=0))
    indices = {name: i for i, name in enumerate(dataset.names)}
    try:
        for case, names in group_names(dataset.names).items():
            for start, logits in iter_cached_logits(model.backbone, len(names),
                    lambda i: dataset.read_image(indices[names[i]]), device, batch_size):
                # Save the original FP16 roundtrip, normalize only after loading.
                stored = logits.float().softmax(1).half().cpu()
                for offset, value in enumerate(stored):
                    name = names[start + offset]
                    path = destination / 'probabilities' / (name + '.npy')
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream, value.numpy(), allow_pickle=False)
                    temporary.replace(path)
                    m['slices'][name] = dict(probability_statistics(canonical_probabilities(value)),
                        image_sha256=file_hash(dataset.root / 'image' / name), probability_sha256=file_hash(path))
            write_json(destination / 'status.json', dict(status='running', case=case, slices=len(m['slices'])))
        m.update(complete=True, seconds=time.monotonic() - started)
        write_json(destination / 'manifest.json', m)
        write_json(destination / 'status.json', dict(status='completed', slices=len(m['slices']), seconds=m['seconds']))
    except BaseException as error:
        write_json(destination / 'status.json', dict(status='failed', error=repr(error)))
        raise
    return m


class ReferenceTrainingDataset(Dataset):
    """Wrap the original target path so target geometry/randomness are unchanged."""
    def __init__(self, base, cfg, cache):
        self.base, self.cfg, self.cache = base, cfg, cache
        self.lookup = {name: i for i, name in enumerate(base.names)}
        self.labeled_cases = {_case_id(name) for name in base.labeled_names}

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def _image(self, name):
        image = _read_transposed(self.base.root / 'image' / name, torch.float32)[None, None]
        return F.interpolate(image, size=self.base.image_size, mode='bilinear', align_corners=False)[0]

    def __getitem__(self, index):
        sample = self.base[index]
        name = self.base.names[index]
        case = _case_id(name)
        names = self.cache.groups[case]
        target_index = names.index(name)
        reference_index = self.cache.selected[case]
        if self.cfg.reference_sampling == 'random':
            candidates = [i for i in self.cache.candidates[case] if i != target_index]
            candidates = candidates or self.cache.candidates[case]
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.base.epoch, index, 71349]))
            reference_index = int(rng.choice(candidates))
        reference_name = names[reference_index]
        probabilities = self.cache.read(reference_name)
        si = si_coordinates(1, self.base.image_size)[0]
        valid = torch.ones_like(si)
        parts = [probabilities, si, valid]
        channels = [('probabilities', 20), ('si', 1), ('valid', 1)]
        if self.cfg.reference_condition == 'row':
            parts.append(row_prior(probabilities[None])[0]); channels.append(('prior', 20))
        eligible = bool(self.cfg.reference_gt and reference_name != name and reference_name in self.base.labeled_names)
        if self.cfg.reference_gt:
            gt = torch.zeros_like(probabilities)
            if eligible:
                label = _read_transposed(self.base.root / 'label' / reference_name, torch.long)
                label = F.interpolate(label[None, None].float(), size=self.base.image_size, mode='nearest')[0, 0].long()
                if label.min() < 0 or label.max() >= 20:
                    raise ValueError('Invalid authorized reference GT')
                gt = F.one_hot(label, 20).permute(2, 0, 1).float()
            parts.append(gt); channels.append(('gt', 20))
            if self.cfg.reference_condition == 'row':
                parts.append(row_prior(gt[None])[0]); channels.append(('gt_prior', 20))
        reference_names = ([names[min(len(names)-1, max(0, reference_index+d))] for d in (-1, 0, 1)]
                           if self.cfg.reference_cycle_weight else [reference_name])
        image = torch.cat([self._image(n) for n in reference_names])
        auxiliary = torch.cat(parts)
        image, _, auxiliary = self.base._geometry(image, None, index, auxiliary, 'bilinear')
        image = torch.cat([(v-v.mean())/v.std(unbiased=False).clamp_min(1e-6) for v in image.split(1)])
        cursor, fields = 0, {}
        for key, count in channels:
            fields[key] = auxiliary[cursor:cursor+count]; cursor += count
        validity = fields['valid']
        for key in fields:
            if key != 'valid':
                fields[key] = fields[key] / validity.clamp_min(1e-8)
        sample.update({'reference_' + key: value for key, value in fields.items()})
        sample.update(reference_image=image[1:2] if len(reference_names) == 3 else image,
            reference_gt_available=eligible, reference_name=reference_name,
            reference_index=reference_index, reference_is_target=reference_name == name,
            reference_unlabeled_case=case not in self.labeled_cases)
        if self.cfg.reference_cycle_weight:
            sample['reference_triplet'] = image
        return sample
