"""Encoder placement, fair representation comparison, isolation and portability."""
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import unittest

# Must precede the first CUDA matrix operation, including standalone cache tests.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from freqssl.context_cache import iter_cached_logits
from freqssl.encoder_text import EncoderSpatialText
from freqssl.engine import make_optimizer, train_step, run_experiment, load_model
from freqssl.hierarchical_text import load_hierarchical_bank
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture
from test_hierarchical_text import write_bank


def config(bank, kind="tokens", stage=2, **changes):
    return small_config(method="mean_teacher", slice_context="adaptive", internal_text="text",
        text_bank=str(bank), text_fusion_layout="encoder", text_layer_mode="last",
        text_feature_kind=kind, encoder_text_stage=stage, text_adapter_dim=8,
        text_alignment_weight=.1, text_alignment_balance="class", **changes)


class EncoderTextTests(CPUInvariantTest):
    def test_config_rejects_ambiguous_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(write_bank(tmp))
            for changes in (dict(encoder_text_stage=1), dict(encoder_text_stage=True),
                            dict(text_feature_kind="none"), dict(encoder_text_stage=None),
                            dict(text_layer_mode="grouped"), dict(text_fusion_layout="hierarchical")):
                with self.assertRaises(ValueError):
                    replace(cfg, **changes).validate()

    def test_initialization_rng_zero_gates_and_common_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write_bank(tmp)
            base = StudentTeacherSystem(small_config(method="mean_teacher", slice_context="adaptive"))
            images = torch.randn(2, 3, 32, 48)
            reference = None
            for kind, stage in [("none", None)] + [(k,s) for s in (2,3,4) for k in ("tokens", "global_cls")]:
                before = torch.get_rng_state().clone()
                net = StudentTeacherSystem(config(bank, kind, stage)).students[0]
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                for name, value in base.students[0].state_dict().items():
                    self.assertTrue(torch.equal(value, net.state_dict()[name]))
                current = {n:p.detach().clone() for n,p in net.named_parameters()}
                if reference is None:
                    reference = current
                self.assertTrue(all(torch.equal(v, reference[k]) for k,v in current.items()))
                with torch.no_grad():
                    expected = base.student_logits(images, 0)
                    torch.testing.assert_close(net(images), expected, atol=0, rtol=0)
                    output, aux = net.forward_with_alignment(images, collect_diagnostics=True)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
                    self.assertEqual([list(v.shape[-2:]) for v in aux["tissue_logits"]], [[8,12],[4,6],[2,3]])

    def test_injection_changes_deeper_encoder_features_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            bank = write_bank(tmp)
            for stage in (2,3,4):
                net = StudentTeacherSystem(config(bank, stage=stage)).students[0]
                image = torch.randn(3,1,32,48)
                before = net.encode(image)
                net.encoder_text.gates[stage-2].fill_(.7)
                after = net.encode(image)
                for level in range(5):
                    self.assertEqual(torch.equal(before[level],after[level]), level < stage)

    def test_padding_single_token_equivalence_and_length_normalization(self):
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            bank, _ = load_hierarchical_bank(write_bank(tmp), "text")
            layer = EncoderSpatialText([16,16,16], 8, bank)
            layer.gates[0].fill_(.5)
            features = torch.randn(2,16,3,5)
            expected, _ = layer(features,2)
            layer.hidden_tokens[11].masked_fill_(~layer.token_mask[...,None],1e6)
            actual, _ = layer(features,2)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            cls = layer.hidden_tokens[11,:,:1].clone()
            mask = torch.ones(4,1,dtype=torch.bool)
            single, _ = layer.fuse(features,0,cls,mask)
            layer.feature_kind = "global_cls"
            global_out, _ = layer(features,2)
            torch.testing.assert_close(single,global_out,atol=0,rtol=0)
            repeated = cls.expand(-1,5,-1).clone()
            unequal_mask = torch.arange(5)[None] < torch.tensor([1,2,3,5])[:,None]
            repeated_out, _ = layer.fuse(features,0,repeated,unequal_mask)
            torch.testing.assert_close(single,repeated_out,atol=2e-6,rtol=1e-6)

    def test_gradients_frozen_assets_ema_and_unlabeled_gt_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write_bank(tmp)
            for kind,stage in (("tokens",2),("global_cls",4),("none",None)):
                cfg = config(bank,kind,stage)
                models = [StudentTeacherSystem(cfg) for _ in range(2)]
                opts = [make_optimizer(m,cfg) for m in models]
                batch = synthetic_batch()
                batch["image"] = torch.randn(4,3,32,32)
                modified = dict(batch,label=batch["label"].clone())
                modified["label"][2:] = 99999
                frozen = models[0].students[0].encoder_text.hidden_tokens.clone()
                for step in range(3):
                    for model,opt,data in zip(models,opts,(batch,modified)):
                        metrics = train_step(model,opt,data,cfg,torch.device("cpu"),step)
                        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
                    self.assertSameState(models[0],models[1])
                    if step == 0:
                        self.assertSameState(models[0].students[0],models[0].teachers[0])
                module = models[0].students[0].encoder_text
                for name,p in module.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,name)
                        self.assertGreater(float(p.grad.abs().sum()),0,name)
                    else:
                        self.assertIsNone(p.grad,name)
                self.assertTrue(torch.equal(module.hidden_tokens,frozen))
                self.assertTrue(torch.equal(models[0].teachers[0].encoder_text.hidden_tokens,frozen))
                self.assertTrue(all(p.grad is None for p in models[0].teachers.parameters()))

    def _cached_nonzero(self, device):
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            bank = write_bank(tmp)
            for kind,stage in (("tokens",2),("global_cls",3),("tokens",4),("none",None)):
                net = StudentTeacherSystem(config(bank,kind,stage)).students[0].to(device).eval()
                if stage is not None:
                    net.encoder_text.gates[stage-2].fill_(.6)
                for fusion in net.context_fusion:
                    fusion.gamma.fill_(.3)
                for count in (1,9):
                    images = torch.randn(count,1,32,48,device=device)
                    triplets = torch.stack([images[[max(0,i-1),i,min(count-1,i+1)],0] for i in range(count)])
                    expected = net(triplets)
                    for batch in (1,3,8):
                        actual = torch.cat([x for _,x in iter_cached_logits(net,count,lambda i:images[i],device,batch)])
                        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
                        self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))

    def test_cpu_cache_with_nonzero_text(self):
        self._cached_nonzero("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_cache_with_nonzero_text(self):
        self._cached_nonzero("cuda:0")

    def _resume(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = write_bank(root)
            data = make_spider_fixture(root / "data")
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_portable_exact_resume_cuda(self):
        self._resume("cuda:0")


if __name__ == "__main__":
    unittest.main()
