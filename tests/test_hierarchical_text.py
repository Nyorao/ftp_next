"""Hierarchy, alignment supervision, MT isolation and portable inference."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from freqssl.context_cache import iter_cached_logits
from freqssl.engine import make_optimizer, train_step, run_experiment, load_model
from freqssl.hierarchical_text import (load_hierarchical_bank, HierarchicalTextFusion,
                                      tissue_alignment_loss, global_alignment_loss, centered_global_alignment_loss)
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from freqssl.prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture


def write_bank(directory, source='text'):
    rng = torch.Generator().manual_seed(81 if source == 'text' else 82)
    path = Path(directory) / (source+'.pt')
    torch.save(dict(format_version=2, source=source,
        prompts=list(TEXT_PROMPTS if source == 'text' else NEUTRAL_PROMPTS),
        hidden_tokens=torch.randn(12,4,32,768,generator=rng),
        token_mask=torch.arange(32)[None].expand(4,-1) < torch.tensor([9,10,11,12])[:,None],
        anchors=F.normalize(torch.randn(4,512,generator=rng),dim=-1),
        provenance={'source':'synthetic unit-test fixture, not pretrained'}),path)
    return path


def config(path, **changes):
    return small_config(method='mean_teacher', slice_context='adaptive', internal_text='text',
        text_bank=str(path), text_fusion_layout='hierarchical', text_adapter_dim=8,
        text_alignment_weight=.1, **changes)


class HierarchicalTextTests(CPUInvariantTest):
    def test_config_and_bank_reject_invalid_inputs(self):
        base = small_config(method='mean_teacher',slice_context='adaptive')
        for changes in (dict(text_fusion_layout='hierarchical'),dict(text_alignment_weight=.1),
                        dict(text_alignment_weight=float('nan')),dict(text_layer_mode='bad')):
            with self.assertRaises(ValueError):
                replace(base,**changes).validate()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bank(tmp)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_hierarchical_bank(path,'neutral')
            payload = torch.load(path,weights_only=True)
            payload['token_mask'][0].zero_()
            torch.save(payload,path)
            with self.assertRaises(ValueError):
                load_hierarchical_bank(path,'text')

    def test_zero_gates_preserve_visual_initialization_rng_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = StudentTeacherSystem(small_config(method='mean_teacher',slice_context='adaptive'))
            x = torch.randn(2,3,32,48)
            params = []
            for source in ('text','neutral'):
                cfg = replace(config(write_bank(tmp,source)),internal_text=source)
                before = torch.get_rng_state().clone()
                model = StudentTeacherSystem(cfg)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                for key,value in base.students[0].state_dict().items():
                    self.assertTrue(torch.equal(value,model.students[0].state_dict()[key]),key)
                torch.testing.assert_close(model.student_logits(x,0),base.student_logits(x,0),atol=0,rtol=0)
                out,aux = model.students[0].forward_with_alignment(x)
                torch.testing.assert_close(out,base.student_logits(x,0),atol=0,rtol=0)
                self.assertEqual([list(v.shape[-2:]) for v in aux['tissue_logits']],[[2,3],[4,6],[8,12]])
                params.append(dict(model.students[0].named_parameters()))
            for key in params[0]:
                self.assertTrue(torch.equal(params[0][key],params[1][key]),key)

    def test_padding_cannot_change_attention_and_last_mode_ignores_other_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,_ = load_hierarchical_bank(write_bank(tmp),'text')
            layer = HierarchicalTextFusion([16,16,16],8,bank,'last')
            x = torch.randn(2,16,3,5)
            with torch.no_grad():
                layer.gates.fill_(.5)
                expected = layer(x,0)
                layer.hidden_tokens[:11].normal_(1000,100)
                layer.hidden_tokens.masked_fill_(~layer.token_mask[None,:,:,None],1e6)
                actual = layer(x,0)
                for a,b in zip(expected,actual):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_alignment_gradients_ema_and_unlabeled_mask_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(write_bank(tmp))
            models = [StudentTeacherSystem(cfg) for _ in range(2)]
            opts = [make_optimizer(m,cfg) for m in models]
            batch = synthetic_batch()
            batch['image'] = torch.randn(4,3,32,32)
            modified = dict(batch,label=batch['label'].clone())
            modified['label'][2:] = 99999
            frozen = models[0].students[0].text_pyramid.hidden_tokens.clone()
            for step in range(3):
                for m,opt,b in zip(models,opts,(batch,modified)):
                    metrics = train_step(m,opt,b,cfg,torch.device('cpu'),step)
                    self.assertTrue(torch.isfinite(torch.tensor(metrics['text_alignment_loss'])))
                self.assertSameState(models[0],models[1])
                if step == 0:
                    self.assertSameState(models[0].students[0],models[0].teachers[0])
                    self.assertGreater(float(models[0].students[0].text_pyramid.key.weight.grad.abs().sum()),0)
            layer = models[0].students[0].text_pyramid
            for p in (layer.key.weight,layer.value.weight,*[q.weight for q in layer.queries],
                      *[v.weight for v in layer.outputs],*[m[-1].weight for m in layer.mixers]):
                self.assertGreater(float(p.grad.abs().sum()),0)
            self.assertTrue(torch.equal(layer.hidden_tokens,frozen))
            self.assertTrue(torch.equal(models[0].teachers[0].text_pyramid.hidden_tokens,frozen))
            self.assertTrue(all(p.grad is None for p in models[0].teachers.parameters()))

    def test_area_targets_and_global_target_detachment(self):
        p = torch.zeros(1,4,1,1,requires_grad=True)
        labels = torch.tensor([[[0,1],[10,11]]])
        loss = tissue_alignment_loss([p],labels)
        torch.testing.assert_close(loss,torch.tensor(4.).log())
        loss.backward()
        torch.testing.assert_close(p.grad,torch.zeros_like(p))
        with self.assertRaises(ValueError):
            tissue_alignment_loss([p],torch.full_like(labels,-100))
        target = F.normalize(torch.randn(2,512),dim=-1).requires_grad_()
        student = torch.randn(2,512,requires_grad=True)
        global_alignment_loss(student,target).backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(student.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            global_alignment_loss(student,torch.zeros_like(target))

    def test_class_balance_does_not_reward_background_only_solution(self):
        labels = torch.zeros(1,10,10,dtype=torch.long)
        labels[0,0,:3] = torch.tensor([1,10,11])
        logits = torch.zeros(1,4,10,10,requires_grad=True)
        with torch.no_grad():
            logits[:,0] = 4.
        pixel = tissue_alignment_loss([logits],labels,'pixel')
        balanced = tissue_alignment_loss([logits],labels,'class')
        self.assertLess(float(pixel),.3)
        self.assertGreater(float(balanced),3.)
        balanced.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Uniform logits are stationary for equally weighted four-class bias.
        uniform = torch.zeros(1,4,1,1,requires_grad=True)
        tissue_alignment_loss([uniform],labels,'class').backward()
        torch.testing.assert_close(uniform.grad,torch.zeros_like(uniform),atol=1e-7,rtol=0)

    def test_centered_alignment_distinguishes_paired_from_mean_solution(self):
        targets = F.normalize(torch.randn(4,512),dim=-1).requires_grad_()
        paired = targets.detach().clone().requires_grad_()
        torch.testing.assert_close(centered_global_alignment_loss(paired,targets),torch.tensor(0.),atol=1e-6,rtol=0)
        swapped = paired.roll(1,0)
        self.assertGreater(float(centered_global_alignment_loss(swapped,targets)),.8)
        mean = targets.detach().mean(0,keepdim=True).expand(4,-1).clone().requires_grad_()
        loss = centered_global_alignment_loss(mean,targets)
        self.assertGreater(float(loss),.9)
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertTrue(torch.isfinite(mean.grad).all())
        with self.assertRaises(ValueError):
            centered_global_alignment_loss(paired,torch.ones_like(targets))

    def test_cache_nonzero_gates_and_self_contained_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = write_bank(root)
            data = make_spider_fixture(root/'data')
            cfg = config(bank,root=str(data),output=str(root/'full'),labelnum=4,max_samples=8,max_steps=3,eval_every=3)
            full = run_experiment(cfg,'cpu')
            staged = replace(cfg,output_dir=str(root/'staged'))
            run_experiment(staged,'cpu',stop_after=1)
            bank.unlink()
            resumed = run_experiment(staged,'cpu',resume=root/'staged/last.pt')
            for key in ('final_student1_hash','final_teacher1_hash','final_validation'):
                self.assertEqual(full[key],resumed[key])
            single,_,receipt = load_single_network(root/'staged/best.pt','cpu')
            system,_,_ = load_model(root/'staged/best.pt','cpu')
            system.eval()
            with torch.no_grad():
                x = torch.randn(1,3,32,32)
                torch.testing.assert_close(single(x),system.predict_proba(x,'teacher1'),atol=0,rtol=0)
                net = system.teachers[0]
                net.text_pyramid.gates.fill_(.7)
                for fuse in net.context_fusion:
                    fuse.gamma.fill_(.3)
                images = torch.randn(9,1,32,48)
                triplets = torch.stack([images[[max(0,i-1),i,min(8,i+1)],0] for i in range(9)])
                expected = net(triplets)
                for batch_size in (1,3,8):
                    actual = torch.cat([v for _,v in iter_cached_logits(net,9,lambda i:images[i],'cpu',batch_size)])
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
                    self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))
            self.assertFalse(receipt['text_encoder_loaded'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_cached_hierarchy_with_nonzero_residuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            net = StudentTeacherSystem(config(write_bank(tmp))).students[0].cuda().eval()
            with torch.no_grad():
                net.text_pyramid.gates.fill_(.6)
                for fusion in net.context_fusion:
                    fusion.gamma.fill_(.4)
                images = torch.randn(9,1,32,48,device='cuda')
                triplets = torch.stack([images[[max(0,i-1),i,min(8,i+1)],0] for i in range(9)])
                expected = net(triplets)
                for batch_size in (1,3,8):
                    actual = torch.cat([v for _,v in iter_cached_logits(net,9,lambda i:images[i],'cuda',batch_size)])
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
                    self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))


if __name__ == '__main__':
    unittest.main()
