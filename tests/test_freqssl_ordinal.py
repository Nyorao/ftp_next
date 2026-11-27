"""Numerical, gradient, geometry, and checkpoint invariants for the new method."""
import itertools
import json
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from freqssl.anatomy import (FAMILIES, family_probabilities, family_log_probabilities,
                             marginal_set_loss, monotone_marginals, ordinal_evidence, route_targets)
from freqssl.augment import aligned_cutmix
from freqssl.engine import (load_model, make_optimizer, save_checkpoint, seed_everything,
                            state_hash, train_step, run_experiment)
from freqssl.models import StudentTeacherSystem
from freqssl.ordinal_model import OrdinalUNet
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture


class OrdinalMathTests(CPUInvariantTest):
    def test_forward_backward_matches_exhaustive_paths_with_missing_levels(self):
        e = torch.randn(2, 4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(51))
        paths = list(itertools.combinations_with_replacement(range(3), 4))
        scores=[]
        for path in paths:
            cost=sum(1.5 + 2*(b-a-1) if b>a else 0 for a,b in zip(path,path[1:]))
            scores.append(sum(e[:,t,s] for t,s in enumerate(path))-cost)
        probability=torch.stack(scores,1).softmax(1)
        expected=torch.zeros_like(e)
        for index,path in enumerate(paths):
            for t,s in enumerate(path): expected[:,t,s]+=probability[:,index]
        torch.testing.assert_close(monotone_marginals(e),expected,rtol=1e-12,atol=1e-12)

    def test_free_endpoints_and_skip_do_not_require_all_levels(self):
        e=torch.full((1,4,9),-40.)
        e[0,:2,3]=0; e[0,2:,6]=0
        self.assertEqual(monotone_marginals(e).argmax(-1).tolist(),[[3,3,6,6]])

    def test_marginal_loss_is_ce_for_singleton_and_zero_for_abstention(self):
        logits=torch.randn(1,20,2,3,requires_grad=True)
        labels=torch.tensor([[[1,2,3],[10,11,0]]])
        allowed=F.one_hot(labels,20).permute(0,3,1,2).float()
        packed=torch.cat((allowed,torch.ones(1,1,2,3)),1).requires_grad_()
        loss=marginal_set_loss(logits,packed)
        torch.testing.assert_close(loss,F.cross_entropy(logits,labels))
        loss.backward(); self.assertIsNone(packed.grad)
        zero=marginal_set_loss(logits,torch.cat((torch.ones_like(allowed),torch.zeros(1,1,2,3)),1))
        self.assertEqual(zero.item(),0.)

    def test_tissue_only_has_no_gradient_for_conditional_level_logits(self):
        tissue=torch.randn(1,4,2,3,requires_grad=True)
        levels=torch.randn(1,18,2,3,requires_grad=True)
        t=tissue.log_softmax(1)
        logits=torch.cat((t[:,:1],t[:,1:2]+levels[:,:9].log_softmax(1),t[:,2:3],t[:,3:]+levels[:,9:].log_softmax(1)),1)
        target=torch.zeros(1,21,2,3);target[:,1:10]=1;target[:,20]=1
        marginal_set_loss(logits,target).backward()
        self.assertLess(levels.grad.abs().max().item(),1e-7)
        self.assertGreater(tissue.grad.abs().max().item(),0)

    def test_adjacent_disagreement_keeps_candidates_without_forcing_an_id(self):
        p=torch.zeros(1,20,3,5);q=p.clone();p[:,3]=1;q[:,4]=1
        target,stats=route_targets(p.requires_grad_(),q.requires_grad_(),use_chain=False)
        self.assertEqual(stats['interval_coverage'],1.)
        self.assertEqual(stats['fine_coverage'],0.)
        self.assertTrue((target[:,:20].sum(1)==2).all())
        self.assertTrue((target[:,3:5]==1).all());self.assertFalse(target.requires_grad)

    def test_tissue_disagreement_abstains_and_diffuse_levels_use_family(self):
        p=torch.zeros(1,20,2,4);q=p.clone();p[:,1:10]=1/9;q[:,11:20]=1/9
        target,stats=route_targets(p,q)
        self.assertEqual(stats['abstention'],1.); self.assertEqual(target[:,20].sum().item(),0)
        target,stats=route_targets(p,p)
        self.assertEqual(stats['family_only_coverage'],1.)
        self.assertTrue((target[:,:20].sum(1)==9).all())

    def test_chain_reroutes_an_inverted_high_confidence_sequence(self):
        p=torch.zeros(1,20,4,32)
        # Correct tensor order should DECREASE with increasing final-axis index.
        p[:,1,:,:16]=1;p[:,9,:,16:]=1
        plain,sp=route_targets(p,p,use_chain=False)
        routed,sr=route_targets(p,p,bins=32)
        self.assertEqual(sp['fine_coverage'],1.)
        self.assertLess(sr['fine_coverage'],1.)
        self.assertGreater(sr['family_only_coverage'],0.)
        # No correction to an invented ID: every retained set contains its source.
        self.assertTrue((routed[:,:20]*p).sum(1).eq(1).all())

    def test_background_and_tiny_width_are_finite(self):
        p=torch.zeros(1,20,2,2);p[:,0]=1
        target,stats=route_targets(p,p)
        self.assertTrue(torch.isfinite(ordinal_evidence(p)).all())
        self.assertEqual(stats['fine_coverage'],1.)
        self.assertTrue(target[:,20].eq(.25).all())

    def test_routing_is_symmetric_in_teacher_order(self):
        generator=torch.Generator().manual_seed(117)
        p=torch.randn(2,20,4,16,generator=generator).mul(5).softmax(1)
        q=torch.randn(2,20,4,16,generator=generator).mul(5).softmax(1)
        first,_=route_targets(p,q,confidence=.5)
        second,_=route_targets(q,p,confidence=.5)
        self.assertTrue(torch.equal(first,second))

    def test_cutmix_moves_set_and_weight_with_source_pixel(self):
        image=torch.zeros(2,1,16,16);image[1]=1
        packed=torch.ones(2,21,16,16);packed[0,20]=0;packed[1,20]=1
        mixed,target=aligned_cutmix(image,packed,17,1.)
        self.assertTrue(torch.equal(mixed[:,0],target[:,20]))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError): monotone_marginals(torch.full((1,2,3),float('nan')))
        with self.assertRaises(ValueError): marginal_set_loss(torch.zeros(1,20,2,2),torch.zeros(1,21,2,2))
        with self.assertRaises(ValueError): small_config(method='dual_ordinal',region_weight=1)


class OrdinalModelTests(CPUInvariantTest):
    def check_resume(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=make_spider_fixture(root/'data')
            cfg=small_config(data,root/'direct',method='dual_ordinal',labelnum=4,max_samples=8,
                             max_steps=4,eval_every=2)
            run_experiment(cfg,device)
            staged=replace(cfg,output_dir=str(root/'staged'))
            run_experiment(staged,device,stop_after=2)
            run_experiment(staged,device,resume=root/'staged/last.pt')
            first,_,_=load_model(root/'direct/last.pt',torch.device('cpu'))
            second,_,_=load_model(root/'staged/last.pt',torch.device('cpu'))
            self.assertSameState(first,second)
            self.assertEqual((root/'direct/metrics.jsonl').read_text(),(root/'staged/metrics.jsonl').read_text())

    def test_cpu_resume_preserves_both_students_teachers_and_metrics(self):
        self.check_resume(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for ordinal resume verification')
    def test_cuda_resume_preserves_both_students_teachers_and_metrics(self):
        self.check_resume(torch.device('cuda:0'))

    def test_hard_peer_ablation_uses_the_recorded_teacher(self):
        cfg=small_config(method='dual_ordinal',ordinal_use_routing=False)
        seed_everything(cfg);model=StudentTeacherSystem(cfg)
        metrics=train_step(model,make_optimizer(model,cfg),synthetic_batch(),cfg,torch.device('cpu'),1)
        self.assertEqual(metrics['target_source_s1'],'teacher2')
        self.assertEqual(metrics['target_source_s2'],'teacher1')
        self.assertGreater(metrics['weighted_pseudo_ce_s1'],0)

    def test_factorization_normalizes_and_preserves_tissue_marginal(self):
        model=OrdinalUNet(20,4,32)
        x=torch.randn(2,1,32,32)
        features=model.forward_features(x)
        tissue=model.head(features)[:,:4].softmax(1)
        logits=model._factorized_logits(features)
        torch.testing.assert_close(logits.exp().sum(1),torch.ones(2,32,32))
        torch.testing.assert_close(family_probabilities(logits.softmax(1)),tissue)
        torch.testing.assert_close(family_log_probabilities(logits).exp(),tissue)

    def test_step_gradients_teacher_detach_initial_pairing_and_checkpoint(self):
        cfg=small_config(method='dual_ordinal');seed_everything(cfg)
        model=StudentTeacherSystem(cfg)
        control=StudentTeacherSystem(small_config(method='mean_teacher'))
        self.assertSameState(model.students[0],control.students[0])
        before=[state_hash(s) for s in model.students]; opt=make_optimizer(model,cfg)
        metrics=train_step(model,opt,synthetic_batch(),cfg,torch.device('cpu'),1)
        self.assertTrue(all(state_hash(s)!=h for s,h in zip(model.students,before)))
        self.assertTrue(all(p.grad is None for t in model.teachers for p in t.parameters()))
        self.assertGreater(metrics['loss'],0)
        self.assertGreater(model.students[1].axial[-1].weight.grad.abs().sum().item(),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            save_checkpoint(path,model,opt,torch.cuda.amp.GradScaler(enabled=False),cfg,1,.3)
            restored,config,payload=load_model(path,torch.device('cpu'))
            self.assertSameState(model,restored);self.assertEqual(config.method,'dual_ordinal')
            self.assertEqual(payload['step'],1)


if __name__=='__main__': unittest.main()
