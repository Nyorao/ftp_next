"""Read-only model diagnostics on eight fixed training images; no optimization.

Measure actual gated residual magnitude rather than inferring it from a scalar
gate. Pooled token cosines are descriptive geometry, not semantic quality scores.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from freqssl.inference import load_single_network
from freqssl.prompt_supervision import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=ROOT/'outputs/hierarchical_text_screen_20261109_r2')
    parser.add_argument('--assets',type=Path,default=ROOT/'outputs/hierarchical_text_assets_20261108')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    manifest=json.loads((args.assets/'manifest.json').read_text())
    assert file_sha256(args.assets/'probe_batches.pt')==manifest['probe']['batch_sha256']
    batches=torch.load(args.assets/'probe_batches.pt',weights_only=True)
    images=torch.cat([batches[n]['image'] for n in ('train','heldout')])
    results=dict(scope='eight previously cached labeled-training images; no validation efficacy estimate',
        images=manifest['probe']['train_files']+manifest['probe']['heldout_training_files'],
        optimization_steps=0,test_content_accessed=False,geometry={},models={},
        script_sha256=file_sha256(__file__),image_asset_sha256=file_sha256(args.assets/'probe_batches.pt'))
    offdiag=~torch.eye(4,dtype=torch.bool)
    for source in ('text','neutral'):
        bank=torch.load(args.screen/'banks'/(source+'.pt'),weights_only=True)
        h,mask=bank['hidden_tokens'],bank['token_mask']
        pooled=(F.layer_norm(h,(768,))*mask[None,:,:,None]).sum(2)/mask.sum(-1)[None,:,None]
        pooled=F.normalize(pooled,dim=-1)
        groups={}
        for name,ids in (('shallow',[0,1,2,3]),('middle',[4,5,6,7]),('deep',[8,9,10,11])):
            within=[float((pooled[ids,c]@pooled[ids,c].T)[offdiag].mean()) for c in range(4)]
            between=[float((pooled[l]@pooled[l].T)[offdiag].mean()) for l in ids]
            groups[name]=dict(same_prompt_between_layers=sum(within)/4,
                              different_prompts_within_layer=sum(between)/4)
        results['geometry'][source]=dict(groups=groups,valid_token_counts=mask.sum(-1).tolist(),
            definition='LayerNorm each token; valid-token mean per prompt; L2 normalize; off-diagonal mean cosine')
    for arm in ('hierarchical','neutral','last_layer'):
        results['models'][arm]={}
        for checkpoint in ('best','last'):
            path=args.screen/'runs'/arm/(checkpoint+'.pt')
            model,cfg,receipt=load_single_network(path,'cpu',network='teacher1')
            stages=[]
            def observe(module,inputs,outputs):
                features,stage=inputs
                fused,logits,mix=outputs
                assert len(features)==len(images)==8
                rms=lambda x:x.square().mean((1,2,3)).sqrt()
                ratio=rms(fused-features)/rms(features).clamp_min(1e-8)
                assert torch.isfinite(ratio).all()
                stages.append(dict(stage=int(stage),rms_residual_ratio_mean=float(ratio.mean()),
                    rms_residual_ratio_per_image=ratio.tolist(),
                    layer_weight_mean=mix.mean(0).tolist(),layer_weight_std=mix.std(0,unbiased=False).tolist()))
            handle=model.backbone.text_pyramid.register_forward_hook(observe)
            with torch.inference_mode():
                model(images)
            handle.remove()
            assert [v['stage'] for v in stages]==[0,1,2]
            results['models'][arm][checkpoint]=dict(step=receipt['checkpoint_step'],stages=stages,
                gates=model.backbone.text_pyramid.gates.detach().tanh().tolist(),checkpoint_sha256=file_sha256(path))
            print(arm,checkpoint,[(v['stage'],round(100*v['rms_residual_ratio_mean'],4)) for v in stages],flush=True)
            del model
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
