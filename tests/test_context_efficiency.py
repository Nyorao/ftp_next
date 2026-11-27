"""Equivalence, bounded redundant work, and trainability of the narrower model."""
import unittest
from dataclasses import replace

import torch

from freqssl.config import load_config
from freqssl.context_cache import iter_cached_logits
from freqssl.engine import state_hash, train_step, make_optimizer, seed_everything
from freqssl.models import StudentTeacherSystem
from freqssl.slice_context import ContextUNet, iter_cached_logits as legacy_cache


class EfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def check_cache(self, device):
        for mode in ('mean','adaptive'):
            model = ContextUNet(20,4,32,mode).to(device).eval().requires_grad_(False)
            for layer in model.context_fusion:
                layer.gamma.fill_(.7)
                if layer.weights is not None:
                    layer.weights.weight.normal_(0,.2)
            before = state_hash(model)
            for count in (1,2,8,9,14,17,21,35):
                images = torch.randn(count,1,32,48)
                for batch in (1,8,11):
                    expected = torch.cat([v for _,v in legacy_cache(model,count,lambda i:images[i],device,batch)])
                    reads, encoded_batch_sizes = [], []
                    def read(i):
                        reads.append(i)
                        return images[i]
                    hook = model.encoders[0].register_forward_pre_hook(
                        lambda _,args:encoded_batch_sizes.append(len(args[0])))
                    try:
                        actual = torch.cat([v for _,v in iter_cached_logits(model,count,read,device,batch)])
                    finally:
                        hook.remove()
                    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-5 if device!='cpu' else 0)
                    self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))
                    self.assertEqual(reads,list(range(count)))
                    self.assertEqual(encoded_batch_sizes,[8]*((count+7)//8))
            self.assertEqual(state_hash(model),before)

    def test_cache_exact_and_only_final_encoder_batch_padded(self):
        self.check_cache('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cache_cuda(self):
        self.check_cache('cuda:0')

    def test_narrow_pair_changes_only_capacity_and_updates_teacher(self):
        cfg = load_config('configs/v3/context_efficiency_width24.yaml')
        ref = load_config('configs/v3/context_efficiency_reference.yaml')
        differences = {k for k,v in cfg.to_dict().items() if v != ref.to_dict()[k]}
        self.assertEqual(differences,{'base_channels','max_channels','output_dir'})
        seed_everything(cfg)
        model = StudentTeacherSystem(cfg)
        reference = StudentTeacherSystem(ref)
        small = sum(p.numel() for p in model.students[0].parameters())
        large = sum(p.numel() for p in reference.students[0].parameters())
        self.assertLess(small, .57*large)
        # Shape/gradient smoke only; this does not estimate real-data accuracy.
        x = torch.randn(2,3,32,32)
        batch = {'image':x,'label':torch.randint(0,20,(2,32,32)),
                 'is_labeled':torch.tensor([True,False])}
        before = state_hash(model.students[0])
        metrics = train_step(model,make_optimizer(model,cfg),batch,cfg,torch.device('cpu'),2400)
        self.assertNotEqual(state_hash(model.students[0]),before)
        self.assertEqual(state_hash(model.students[0]),state_hash(model.teachers[0]))
        self.assertGreater(metrics['unsupervised_weight'],0)


if __name__=='__main__':
    unittest.main()
