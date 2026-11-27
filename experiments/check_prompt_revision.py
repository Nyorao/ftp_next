"""Real frozen-backbone pairing, anchor gradients and multicache contract probe."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import torch
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import seed_everything,state_hash,write_json
from freqssl.prompt_revision import make_partition,unpack_template_probabilities
from prompt_teacher.model import build_teacher
from prompt_teacher.revision import PromptViews,paired_loss,predict_templates
from prompt_teacher.data import TissueDataset


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=load_config(ROOT/'configs/v3/prompt_context.yaml');seed_everything(cfg)
 _,_,split=build_training_data(cfg);partition=make_partition(split)
 assert len(partition['labeled_files'])==1223 and len(partition['calibration_files'])==273
 assert not(set(partition['calibration_cases'])&set(n.rsplit('_',2)[0] for n in partition['labeled_files']+partition['unlabeled_files']))
 model,meta=build_teacher('text','cuda:0');views=PromptViews(model,'text');model.train()
 initial=model.adaptation_state();base_hash=state_hash(model.base)
 ds=TissueDataset(cfg.root_path,'train',partition['labeled_files'][:4],meta['mean'],meta['std'],augment=True)
 xs=[ds[i] for i in range(4)];images=torch.stack([s['image'] for s in xs]).cuda();labels=torch.stack([s['label'] for s in xs]).cuda()
 # Anchor equals the original text encoder, not the adapted text stream.
 with torch.no_grad():
  _,native=model.encode(images,use_adapters=False)
  torch.testing.assert_close(torch.nn.functional.normalize(native,dim=-1).reshape(4,4,-1),views.anchors[0][None].expand(4,-1,-1),rtol=1e-5,atol=1e-5)
 assert not views.anchors.requires_grad
 before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state().clone()
 loss,details=paired_loss(model,views,images,labels,'T2',0);loss.backward()
 assert torch.equal(before,torch.get_rng_state()) and torch.equal(cuda_before,torch.cuda.get_rng_state())
 assert all(p.grad is None for p in model.base.parameters())
 grad=sum(float(p.grad.abs().sum()) for p in model.adapters.parameters() if p.grad is not None);assert grad>0
 opt=torch.optim.AdamW(model.trainable_parameters(),lr=3e-4);opt.step()
 assert state_hash(model.base)==base_hash
 model.load_adaptation(initial);model.eval()
 q=predict_templates(model,views,images,421);q2=predict_templates(model,views,images,421)
 torch.testing.assert_close(q,q2,rtol=0,atol=0)
 packed=torch.cat([q.flatten(1,2),torch.ones(4,1,224,224,device='cuda:0')],1);targets=unpack_template_probabilities(packed)
 assert targets.shape==(4,4,224,224)
 # T0 common-random-number views have zero disagreement, while sampled IDs never use held-out prompts.
 model.train();model.zero_grad(set_to_none=True);_,t0=paired_loss(model,views,images,labels,'T0',0)
 assert abs(t0['js'])<1e-7
 for step in [0,500,5999]:
  choices=views.choices(4,step,42,'T2');assert (choices[:,0]!=choices[:,1]).all() and choices.max()<4
 write_json(a.output,dict(partition_sizes=[1223,273,2236],frozen_backbone_unchanged=True,anchors_frozen=True,
    anchors_match_native_text=True,adapter_gradient_sum=grad,prompt_sampling_preserves_rng=True,
    cached_template_predictions_repeat_exactly=True,template_output_shape=list(q.shape),t0_js=t0['js'],
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),T2_loss_terms=details,test_evaluated=False))
 print('Real revision teacher probe passed',flush=True)

if __name__=='__main__':main()
