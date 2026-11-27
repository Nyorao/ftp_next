"""Independent middle-reference and position arms; baseline classes stay intact."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .slice_context import ContextUNet, CenterResidualFusion


def reference_index(count, policy='middle', center=None):
    if count < 1:
        raise ValueError('Empty case')
    if policy == 'middle':
        return (count-1)//2
    if policy == 'quarter':
        return (count-1)//4
    if policy == 'target' and center is not None and 0 <= center < count:
        return center
    raise ValueError('Invalid reference policy')


def relative_position(index, count, reverse=False):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid slice index')
    p = 2*index/(count-1)-1 if count > 1 else 0.
    return -p if reverse else p


def verify_dataset_geometry(dataset, manifest_path):
    """Bind acquisition PNG order to verified image bytes; never opens labels."""
    from .geometry import canonicalize
    from .inference import read_volume, prepare_slice
    from spider_volume import group_complete_slices
    path = Path(manifest_path)
    payload = path.read_bytes()
    records = [r for r in json.loads(payload)['records'] if r['split'] == dataset.split]
    lookup = {r['case']:r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Duplicate geometry cases')
    indices = {name:i for i,name in enumerate(dataset.names)}
    positions, receipts = {}, {}
    for case,names in group_complete_slices(dataset.names).items():
        if case not in lookup:
            raise ValueError(f'Missing geometry for {dataset.split}/{case}')
        record = lookup[case]
        array, plan, receipt = read_volume(record['converted_image'], record)
        reverse = 0 in plan['flip_canonical_axes']
        ordered = canonicalize(array,plan)
        if reverse:
            ordered = ordered[::-1]
        if len(ordered) != len(names):
            raise ValueError('Geometry slice count mismatch')
        for i,name in enumerate(names):
            if not torch.equal(prepare_slice(ordered[i],dataset.image_size), dataset.read_image(indices[name])):
                raise ValueError(f'PNG/geometry order mismatch: {name}')
            positions[name] = relative_position(i,len(names),reverse)
        receipts[case] = dict(image_sha256=receipt['input_array_sha256'],
            reverse_physical_order=reverse, spacing_mm=plan['canonical_spacing_mm'][0],
            ordered_png_inputs_exact=True)
    return positions, dict(manifest_sha256=hashlib.sha256(payload).hexdigest(),
                           cases=receipts, annotation_files_read=0)


class CaseFusion(CenterResidualFusion):
    def __init__(self, channels, slices, use_position):
        super().__init__(channels,'adaptive')
        self.slices = slices
        if slices == 4:
            self.weights = nn.Conv2d(4*channels,4,1)
            nn.init.zeros_(self.weights.weight)
            nn.init.zeros_(self.weights.bias)
        self.position_bias = None
        if use_position:
            self.position_bias = nn.Sequential(nn.Conv2d(1,16,1),nn.ReLU(),nn.Conv2d(16,3,1))
            nn.init.zeros_(self.position_bias[-1].weight)
            nn.init.zeros_(self.position_bias[-1].bias)

    def forward(self, features, position=None):
        if features.ndim != 5 or features.shape[1] != self.slices:
            raise ValueError('Wrong number of context slices')
        logits = self.weights(features.flatten(1,2))
        if self.position_bias is not None:
            logits = logits + self.position_bias(F.interpolate(position,size=features.shape[-2:],mode='nearest'))
        weights = logits.softmax(1)
        return features[:,1] + self.gamma*((weights[:,:,None]*features).sum(1)-features[:,1])


class CaseContextUNet(ContextUNet):
    def __init__(self,num_classes,base_channels,max_channels,mode='adaptive',
                 case_reference='none',context_position='none'):
        if mode != 'adaptive' or (case_reference,context_position) not in (
                ('middle','none'),('none','relative')):
            raise ValueError('Choose one case-context arm')
        super().__init__(num_classes,base_channels,max_channels,mode)
        self.case_reference, self.context_position = case_reference, context_position
        self.context_slices = 4 if case_reference == 'middle' else 3
        with torch.random.fork_rng(devices=[]):
            self.context_fusion = nn.ModuleList(CaseFusion(self.widths[i],self.context_slices,
                context_position=='relative') for i in (2,3,4))

    def _check_position(self, position, batch, device):
        if self.context_position == 'none':
            if position is not None:
                raise ValueError('Reference arm must not receive position')
            return
        if (position is None or position.ndim != 4 or position.shape[:2] != (batch,1)
                or min(position.shape[-2:]) < 1 or position.device != device or
                not torch.isfinite(position).all() or (position.abs()>1).any()):
            raise ValueError('Expected a finite B,1,H,W position map in [-1,1] on the input device')

    def _decode_case(self,features,position):
        skips = [features[0][:,1],features[1][:,1]]
        skips.extend(f(v,position) for f,v in zip(self.context_fusion,features[2:]))
        return self.decode_features(skips)

    def context_features(self,triplets,position=None):
        self._check_position(position,len(triplets[0]),triplets[0].device)
        if self.training:
            return self._decode_case(triplets,position)
        chunks=[]
        for start in range(0,len(triplets[0]),self.inference_microbatch):
            values=[v[start:start+self.inference_microbatch] for v in triplets]
            n=len(values[0])
            p=None if position is None else position[start:start+n]
            if n < self.inference_microbatch:
                values=[torch.cat((v,v.new_zeros((self.inference_microbatch-n,*v.shape[1:])))) for v in values]
                if p is not None:
                    p=torch.cat((p,p.new_zeros((self.inference_microbatch-n,*p.shape[1:]))))
            chunks.append(self._decode_case(values,p)[:n])
        return torch.cat(chunks)

    def forward_features(self,images,position=None):
        if images.ndim != 4 or images.shape[1] != self.context_slices:
            raise ValueError('Case-context input channel mismatch')
        encoded=self.encode(images.reshape(-1,1,*images.shape[-2:]))
        return self.context_features([v.reshape(len(images),self.context_slices,*v.shape[1:]) for v in encoded],position)

    def forward(self,images,position=None):
        return self.project(self.forward_features(images,position))

    def forward_with_features(self,images,position=None):
        features=self.forward_features(images,position)
        return self.project(features),features
