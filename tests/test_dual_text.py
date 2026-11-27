"""Mechanism invariants for shared routing and differentiable zero-text subtraction."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from experiments.run_dual_text import ARMS, arm_config
from freqssl.context_cache import iter_cached_logits
from freqssl.dual_text import DualSpatialText, INTERVENTIONS
from freqssl.encoder_text import EncoderSpatialText
from freqssl.engine import make_optimizer, train_step, run_experiment, load_model
from freqssl.hierarchical_text import load_hierarchical_bank
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture
from test_hierarchical_text import write_bank


def config(bank, arm="N7", **changes):
    base = small_config(method="mean_teacher", slice_context="adaptive", text_adapter_dim=8, **changes)
    return arm_config(base, arm, bank, base.output_dir)


class DualTextTests(CPUInvariantTest):
    def test_config_rejects_ambiguous_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(write_bank(tmp))
            for changes in (dict(encoder_text_stages=(3,2)), dict(encoder_text_stages=(4,)),
                    dict(encoder_text_stages=(True,)), dict(encoder_text_stage=2),
                    dict(text_feature_kind="tokens"), dict(text_layer_mode="grouped"),
                    dict(text_routing_mode="other"), dict(text_fusion_layout="encoder")):
                with self.assertRaises(ValueError):
                    replace(cfg,**changes).validate()
            self.assertEqual(cfg.to_dict()["encoder_text_stages"],[2,3])

    def test_all_arm_initialization_rng_zero_gates_and_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write_bank(tmp)
            base = StudentTeacherSystem(config(bank,"N0")).students[0]
            images = torch.randn(2,3,32,48)
            reference = None
            for arm in ARMS[1:]:
                before = torch.get_rng_state().clone()
                net = StudentTeacherSystem(config(bank,arm)).students[0]
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                for name,value in base.state_dict().items():
                    self.assertTrue(torch.equal(value,net.state_dict()[name]),name)
                named = {k:v.detach().clone() for k,v in net.named_parameters()}
                if reference is None:
                    reference = named
                self.assertEqual(named.keys(),reference.keys())
                self.assertTrue(all(torch.equal(v,reference[k]) for k,v in named.items()))
                with torch.no_grad():
                    output,aux = net.forward_with_alignment(images,collect_diagnostics=True)
                    torch.testing.assert_close(output,base(images),atol=0,rtol=0)
                    self.assertEqual([list(x.shape[-2:]) for x in aux["tissue_logits"]],[[8,12],[4,6],[2,3]])
            net = StudentTeacherSystem(config(bank)).students[0]
            with torch.no_grad():
                image = torch.randn(3,1,32,48)
                before = net.encode(image)
                net.encoder_text.gates[0].fill_(.6)
                after = net.encode(image)
                for level in range(5):
                    self.assertEqual(torch.equal(before[level],after[level]),level<2)
                net.encoder_text.gates[1].fill_(.4)
                both = net.encode(image)
                for level in range(5):
                    self.assertEqual(torch.equal(after[level],both[level]),level<3)

    def test_zero_text_cancels_visual_and_parameter_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,_ = load_hierarchical_bank(write_bank(tmp),"text")
            module = DualSpatialText([16,16,16],8,bank)
            with torch.no_grad():
                module.gates[0].fill_(.7)
            features = torch.randn(2,16,4,5,requires_grad=True)
            q = module.queries[0](module.norms[0](features.flatten(2).transpose(1,2)))
            q.retain_grad()
            output,stats = module.correction(features,0,q,torch.zeros_like(q),diagnostics=True)
            torch.testing.assert_close(output,features,atol=0,rtol=0)
            self.assertTrue(torch.equal(stats["raw_difference_rms_ratio"],torch.zeros(2)))
            gradient = torch.randn_like(output)
            (output*gradient).sum().backward()
            torch.testing.assert_close(features.grad,gradient,atol=0,rtol=0)
            self.assertEqual(float(q.grad.abs().sum()),0.)
            for collection in (module.queries[0],module.norms[0],module.fusions[0]):
                for parameter in collection.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(float(parameter.grad.abs().sum()),0.)
            self.assertEqual(float(module.gates[0].grad),0.)

    def test_supervised_logits_are_route_and_receive_main_task_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            net = StudentTeacherSystem(config(write_bank(tmp))).students[0]
            with torch.no_grad():
                net.encoder_text.gates[0].fill_(.4)
                net.encoder_text.gates[1].fill_(.5)
            captured = []
            original = net.encoder_text.attend
            def capture(q,text,mask,aligned_logits):
                aligned_logits.retain_grad()
                result = original(q,text,mask,aligned_logits)
                torch.testing.assert_close(result[1],aligned_logits.softmax(1).flatten(2).transpose(1,2),atol=0,rtol=0)
                captured.append(aligned_logits)
                return result
            net.encoder_text.attend = capture
            logits,aux = net.forward_with_alignment(torch.randn(2,3,32,48),collect_diagnostics=True)
            self.assertEqual(len(captured),2)
            for i,value in enumerate(captured):
                self.assertEqual(aux["tissue_logits"][i].untyped_storage().data_ptr(),value.untyped_storage().data_ptr())
            # No auxiliary loss here: only the segmentation objective.
            torch.nn.functional.cross_entropy(logits,torch.randint(0,20,logits.shape[:1]+logits.shape[2:])).backward()
            for i,value in enumerate(captured):
                self.assertGreater(float(value.grad.abs().sum()),0.)
                grad = net.encoder_text.alignment_heads[i].weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()),0.)
            self.assertIsNone(net.encoder_text.alignment_heads[2].weight.grad)

    def test_mask_length_single_token_and_legacy_plain_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp,torch.no_grad():
            bank,_ = load_hierarchical_bank(write_bank(tmp),"text")
            for routing in ("independent","aligned"):
                module = DualSpatialText([16,16,16],8,bank,routing_mode=routing)
                module.gates[0].fill_(.6)
                features = torch.randn(2,16,3,5)
                logits = module.alignment_logits(features,0)
                expected,_ = module(features,2,logits)
                module.hidden_tokens[11].masked_fill_(~module.token_mask[...,None],1e6)
                actual,_ = module(features,2,logits)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                q = module.queries[0](module.norms[0](features.flatten(2).transpose(1,2)))
                cls = module.hidden_tokens[11,:,:1].clone()
                single,prob,_ = module.attend(q,cls,torch.ones(4,1,dtype=torch.bool),logits)
                text,mask = module.stage_text_inputs(3)
                global_text,global_prob,_ = module.attend(q,text,mask,logits)
                torch.testing.assert_close(single,global_text,atol=0,rtol=0)
                torch.testing.assert_close(prob,global_prob,atol=0,rtol=0)
                mask = torch.arange(5)[None] < torch.tensor([1,2,3,5])[:,None]
                repeated,repeated_prob,_ = module.attend(q,cls.expand(-1,5,-1).clone(),mask,logits)
                torch.testing.assert_close(single,repeated,atol=2e-6,rtol=1e-6)
                torch.testing.assert_close(prob,repeated_prob,atol=2e-6,rtol=1e-6)
            dual = DualSpatialText([16,16,16],8,bank,routing_mode="independent",residual_mode="plain")
            dual.gates[0].fill_(.3)
            dual.gates[1].fill_(.4)
            old = EncoderSpatialText([16,16,16],8,bank)
            old.load_state_dict(dual.state_dict())
            for stage,kind in ((2,"tokens"),(3,"global_cls")):
                old.active_stage,old.feature_kind = stage,kind
                torch.testing.assert_close(dual(features,stage)[0],old(features,stage)[0],atol=0,rtol=0)

    def test_gradients_frozen_text_ema_unlabeled_gt_isolation_all_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write_bank(tmp)
            for arm in ARMS[1:]:
                cfg = config(bank,arm)
                models = [StudentTeacherSystem(cfg) for _ in range(2)]
                opts = [make_optimizer(m,cfg) for m in models]
                batch = synthetic_batch()
                batch["image"] = torch.randn(4,3,32,32)
                changed = dict(batch,label=batch["label"].clone())
                changed["label"][2:] = 99999
                frozen = {k:v.clone() for k,v in models[0].students[0].encoder_text.named_buffers()}
                for step in range(3):
                    logs = [train_step(m,o,b,cfg,torch.device("cpu"),step)
                        for m,o,b in zip(models,opts,(batch,changed))]
                    self.assertEqual(logs[0],logs[1])
                    self.assertSameState(*models)
                    if step==0:
                        self.assertSameState(models[0].students[0],models[0].teachers[0])
                for name,p in models[0].students[0].encoder_text.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,f"{arm}:{name}")
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(float(p.grad.abs().sum()),0,f"{arm}:{name}")
                    else:
                        self.assertIsNone(p.grad)
                for net in (models[0].students[0],models[0].teachers[0]):
                    for name,value in net.encoder_text.named_buffers():
                        self.assertTrue(torch.equal(value,frozen[name]))
                self.assertTrue(all(p.grad is None for p in models[0].teachers.parameters()))

    def _cache(self,device):
        with tempfile.TemporaryDirectory() as tmp,torch.no_grad():
            net = StudentTeacherSystem(config(write_bank(tmp))).students[0].to(device).eval()
            module = net.encoder_text
            module.gates[0].fill_(.6)
            module.gates[1].fill_(.5)
            for fusion in net.context_fusion:
                fusion.gamma.fill_(.3)
            bank = {k:v.clone() for k,v in module.named_buffers()}
            for state in INTERVENTIONS:
                module.set_intervention(state)
                for count in (1,9):
                    images = torch.randn(count,1,32,48,device=device)
                    triplets = torch.stack([images[[max(0,i-1),i,min(count-1,i+1)],0] for i in range(count)])
                    expected = net(triplets)
                    for size in (1,3,8):
                        actual = torch.cat([x for _,x in iter_cached_logits(net,count,lambda i:images[i],device,size)])
                        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
                        self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))
            module.set_intervention("zero_both")
            image = torch.randn(3,1,32,48,device=device)
            zero = net.encode(image)
            module.set_intervention("full")
            module.gates[0].zero_()
            module.gates[1].zero_()
            for a,b in zip(zero,net.encode(image)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            module.set_intervention("class_mismatch")
            for stage in (2,3):
                text,mask = module.stage_text_inputs(stage)
                original = bank["hidden_tokens"][11] if stage==2 else bank["hidden_tokens"][11,:,:1]
                torch.testing.assert_close(text,original[[3,0,1,2]],atol=0,rtol=0)
                expected_mask = bank["token_mask"][[3,0,1,2]] if stage==2 else torch.ones_like(mask)
                self.assertTrue(torch.equal(mask,expected_mask))
            for name,value in module.named_buffers():
                self.assertTrue(torch.equal(value,bank[name]))
            module.train()
            with self.assertRaises(ValueError):
                module.set_intervention("zero_e2")

    def test_cpu_cache_nonzero_both_and_interventions(self):
        self._cache("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_cuda_cache_nonzero_both_and_interventions(self):
        self._cache("cuda:0")

    def _resume(self,device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = write_bank(root)
            data = make_spider_fixture(root/"data")
            cfg = config(bank,root=str(data),output=str(root/"full"),labelnum=4,max_samples=8,max_steps=3,eval_every=3)
            full = run_experiment(cfg,device)
            staged = replace(cfg,output_dir=str(root/"staged"))
            run_experiment(staged,device,stop_after=1)
            bank.unlink()
            resumed = run_experiment(staged,device,resume=root/"staged/last.pt")
            for key in ("final_student1_hash","final_teacher1_hash","final_validation"):
                self.assertEqual(full[key],resumed[key])
            single,_,receipt = load_single_network(root/"staged/best.pt",device)
            system,_,_ = load_model(root/"staged/best.pt",device)
            system.eval()
            with torch.no_grad():
                x = torch.randn(1,3,32,32,device=device)
                torch.testing.assert_close(single(x),system.predict_proba(x,"teacher1"),atol=0,rtol=0)
            self.assertFalse(receipt["text_encoder_loaded"])

    def test_portable_exact_resume_cpu(self):
        self._resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_portable_exact_resume_cuda(self):
        self._resume("cuda:0")


if __name__ == "__main__":
    unittest.main()
