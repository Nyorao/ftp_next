"""Parameter/MAC accounting and image-only legacy/optimized cache comparison."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from freqssl.config import load_config
from freqssl.context_cache import iter_cached_logits as optimized_cache
from freqssl.data import SpiderSliceDataset
from freqssl.engine import seed_everything, state_hash, write_json
from freqssl.inference import load_single_network
from freqssl.models import UNet
from freqssl.slice_context import ContextUNet, iter_cached_logits as legacy_cache
from spider_volume import group_complete_slices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConvCounter:
    def __init__(self,model):
        self.macs,self.encoder_batches,self.handles = defaultdict(int),[],[]
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                def hook(layer,inputs,output,name=name):
                    self.macs[name.split('.')[0]] += output.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1]
                    if name=='encoders.0.0':
                        self.encoder_batches.append(len(inputs[0]))
                self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def architecture_profiles():
    result = {}
    for name,channels,cap,context in [('single32',32,256,False),('adaptive32',32,256,True),
                                      ('adaptive24',24,192,True)]:
        model = ContextUNet(20,channels,cap,'adaptive') if context else UNet(20,channels,cap)
        # Train-mode forward avoids eval padding; no backward or parameter update.
        model.train()
        counter = ConvCounter(model)
        try:
            model(torch.zeros(1,3 if context else 1,224,224))
        finally:
            counter.close()
        macs = dict(counter.macs)
        result[name] = dict(widths=list(model.widths),parameters=sum(p.numel() for p in model.parameters()),
            parameters_by_component={n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()},
            student_forward_conv_macs_by_component=macs,
            student_forward_conv_macs_per_center=sum(macs.values()),
            ideal_cached_conv_macs_per_center=sum(macs.values())-macs['encoders']*(2/3 if context else 0))
    return result


@torch.inference_mode()
def infer(function,model,images,device):
    reads = []
    def read(i):
        reads.append(i)
        return images[i]
    values = torch.cat([v.cpu() for _,v in function(model,len(images),read,device,8)])
    assert reads==list(range(len(images)))
    return values


def padded_encodings(count,optimized):
    if optimized:
        return 8*((count+7)//8)
    cached,total = set(),0
    for start in range(0,count,8):
        end = min(count,start+8)
        needed=set(range(max(0,start-1),min(count,end+1)))
        missing=sorted(needed-cached)
        for offset in range(0,len(missing),8):
            total += 8*((len(missing[offset:offset+8])+7)//8)
        cached=(cached|needed)&set(range(end-1,count))
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/spine_slice_context_20261030_r2/confirmation/seed42/adaptive/best.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('repeats must be positive')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='running'))
    cfg=load_config(ROOT/'configs/v3/context_efficiency_reference.yaml')
    seed_everything(cfg)
    try:
        profiles=architecture_profiles()
        write_json(out/'architecture.json',profiles)
        selected,cfg,_=load_single_network(args.checkpoint,torch.device(args.device))
        model=selected.backbone
        seed_everything(cfg)
        initial=state_hash(model)
        dataset=SpiderSliceDataset(cfg.root_path,'valid',cfg.image_size,augment=False,
                                  seed=cfg.seed,slice_context=cfg.slice_context)
        lookup={name:i for i,name in enumerate(dataset.names)}
        groups=list(group_complete_slices(dataset.names).items())
        # Fixed image-only selection: first acquisition case, shortest and longest.
        chosen=dict([groups[0],min(groups,key=lambda x:len(x[1])),max(groups,key=lambda x:len(x[1]))])
        timings=[]
        for case,names in chosen.items():
            images=torch.stack([dataset.read_image(lookup[name]) for name in names])
            pair={}
            for label,function in [('legacy',legacy_cache),('optimized',optimized_cache)]:
                counter=ConvCounter(model)
                try:
                    pair[label]=infer(function,model,images,args.device)
                finally:
                    counter.close()
                pair[label+'_macs']=dict(counter.macs)
                pair[label+'_encoder_slots']=sum(counter.encoder_batches)
                assert sum(counter.encoder_batches)==padded_encodings(len(images),label=='optimized')
            max_error=float((pair['legacy']-pair['optimized']).abs().max())
            changed=int((pair['legacy'].argmax(1)!=pair['optimized'].argmax(1)).sum())
            if changed or max_error>1e-5:
                raise AssertionError((case,max_error,changed))
            del pair['legacy'],pair['optimized']
            samples={'legacy':[],'optimized':[]}
            for repeat in range(args.repeats):
                order=[('legacy',legacy_cache),('optimized',optimized_cache)]
                if repeat%2:
                    order.reverse()
                for label,function in order:
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize()
                    started=time.perf_counter()
                    output=infer(function,model,images,args.device)
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize()
                    samples[label].append(time.perf_counter()-started)
                    del output
            row=dict(case=case,slices=len(names),max_logit_error=max_error,changed_pixels=changed,
                **pair,seconds=samples,median_seconds={k:statistics.median(v) for k,v in samples.items()})
            timings.append(row)
            write_json(out/'cache_cases.json',timings)
            print(json.dumps(row),flush=True)
        assert state_hash(model)==initial
        p=profiles['adaptive32']
        e=p['student_forward_conv_macs_by_component']['encoders']/3
        rest=p['student_forward_conv_macs_per_center']-3*e
        costs={}
        for label,opt in [('legacy',False),('optimized',True)]:
            encoder_slots=sum(padded_encodings(len(names),opt) for _,names in groups)
            output_slots=sum(8*((len(names)+7)//8) for _,names in groups)
            costs[label]=dict(encoder_slots=encoder_slots,output_slots=output_slots,
                             conv_macs=encoder_slots*e+output_slots*rest)
        paths=['freqssl/context_cache.py','freqssl/slice_context.py','freqssl/models.py',
            'freqssl/evaluation.py','freqssl/inference.py','experiments/profile_context_efficiency.py',
            'tests/test_context_efficiency.py','configs/v3/context_efficiency_reference.yaml',
            'configs/v3/context_efficiency_width24.yaml']
        result=dict(status='completed',device=args.device,torch_version=torch.__version__,
            cpu_threads=torch.get_num_threads(),checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),
            model_state_unchanged=True,annotation_files_read=0,train_steps=0,profiles=profiles,
            cache_checks=timings,all_valid_shape_accounting=dict(cases=len(groups),real_slices=sum(len(n) for _,n in groups),costs=costs),
            source_hashes={p:sha(ROOT/p) for p in paths},
            notes=['MAC counts Conv2d multiply-accumulates only; 1 MAC = 2 arithmetic FLOPs.',
                   'Counts exclude normalization, activation, interpolation, fusion elementwise ops and backward.',
                   'Timing uses preloaded normalized images, warmed paths and alternating order; shared host, not isolated hardware.',
                   'Actual all-valid timing was not measured; all-valid MACs are shape accounting.',
                   'Narrow model has not been trained/evaluated for accuracy.'])
        write_json(out/'results.json',result)
        write_json(out/'status.json',dict(status='completed'))
    except Exception as error:
        write_json(out/'status.json',dict(status='failed',error=repr(error)))
        raise


if __name__=='__main__':
    main()
