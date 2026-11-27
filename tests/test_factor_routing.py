"""Mechanism tests: neutral baseline, independent interventions and true gradients."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from freqssl.anatomy import FAMILIES, family_probabilities
from freqssl.augment import aligned_cutmix
from freqssl.factor_routing import (AdvantageRouter, advantage_loss, advantage_targets,
                                   factor_fusion, flat_fusion, pack_soft_targets,
                                   soft_consistency_loss)


class FactorRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def pair(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(141)
        return tuple(torch.randn(2, 20, 4, 6, generator=generator, dtype=dtype).softmax(1)
                     for _ in range(2))

    def test_neutral_is_original_mean_and_swap_negates_gates(self):
        a, b = self.pair()
        zero = torch.zeros(2, 3, 4, 6, dtype=a.dtype)
        torch.testing.assert_close(factor_fusion(a, b, zero), (a+b)/2, rtol=1e-14, atol=1e-15)
        gates = torch.randn_like(zero)
        fused = factor_fusion(a, b, gates)
        torch.testing.assert_close(fused, factor_fusion(b, a, -gates), rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(fused.sum(1), torch.ones_like(a[:, 0]))
        self.assertTrue((fused >= 0).all())

    def test_each_gate_changes_only_its_declared_factor(self):
        a, b = self.pair()
        g = torch.randn(2, 3, 4, 6, dtype=a.dtype)
        baseline = factor_fusion(a, b, g)
        changed = g.clone(); changed[:, 1:] += 2
        q = factor_fusion(a, b, changed)
        torch.testing.assert_close(family_probabilities(baseline), family_probabilities(q))
        changed = g.clone(); changed[:, 0] += 2
        q = factor_fusion(a, b, changed)
        for ids in (FAMILIES[1], FAMILIES[3]):
            torch.testing.assert_close(baseline[:, ids]/baseline[:, ids].sum(1, keepdim=True),
                                       q[:, ids]/q[:, ids].sum(1, keepdim=True))

    def test_coupling_all_three_gates_recovers_flat_fusion_at_any_gate(self):
        a, b = self.pair()
        gate = torch.linspace(-4, 4, 48, dtype=a.dtype).reshape(2, 1, 4, 6)
        torch.testing.assert_close(factor_fusion(a, b, gate.expand(-1, 3, -1, -1)),
                                   flat_fusion(a, b, gate), rtol=1e-13, atol=1e-15)

    def test_flat_gain_is_sum_of_tissue_and_applicable_level_gains(self):
        a, b = self.pair()
        labels = torch.tensor([0, 1, 9, 10, 11, 19]).expand(2, 4, 6)
        factors, _ = advantage_targets(a, b, labels, mode='factor')
        flat, _ = advantage_targets(a, b, labels, mode='flat')
        torch.testing.assert_close(factors.sum(1, keepdim=True), flat, rtol=1e-12, atol=1e-14)

    def test_zero_and_tiny_tissue_support_do_not_invent_mass(self):
        a = torch.zeros(1, 20, 2, 2); b = a.clone()
        a[:, 0] = 1; b[:, 10] = 1
        a[:, 1] = 1e-30; b[:, 1] = 1e-30
        q = factor_fusion(a, b, torch.full((1, 3, 2, 2), 2.))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q[:, 11:] == 0).all())
        self.assertTrue((q[:, 2:10] == 0).all())
        torch.testing.assert_close(q[:, 1], a[:, 1], rtol=1e-6, atol=0)
        torch.testing.assert_close(q.sum(1), torch.ones_like(q[:, 0]))

    def test_advantage_targets_equal_autograd_negative_nll_derivatives(self):
        a, b = self.pair()
        labels = torch.tensor([0, 1, 9, 10, 11, 19]).expand(2, 4, 6)
        for mode, fuse, channels in [('factor', factor_fusion, 3), ('flat', flat_fusion, 1)]:
            gates = torch.zeros(2, channels, 4, 6, dtype=a.dtype, requires_grad=True)
            loss = -fuse(a, b, gates).gather(1, labels[:, None]).log().sum()
            actual = -torch.autograd.grad(loss, gates)[0]
            expected, mask = advantage_targets(a, b, labels, mode=mode)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
            self.assertTrue((expected.abs() <= 1).all())
            self.assertTrue((expected[~mask] == 0).all())
            reverse, _ = advantage_targets(b, a, labels, mode=mode)
            torch.testing.assert_close(expected, -reverse)
            moved = -fuse(a, b, .01*expected).gather(1, labels[:, None]).log().sum()
            self.assertLess(moved.item(), loss.item())

    def test_cross_factor_fusion_can_recover_an_id_both_teachers_miss(self):
        a = torch.zeros(1, 20, 1, 1); b = a.clone()
        a[:, 1] = .1; a[:, 2] = .8; a[:, 0] = .1
        b[:, 1] = .4; b[:, 11] = .6
        self.assertEqual(a.argmax(1).item(), 2)
        self.assertEqual(b.argmax(1).item(), 11)
        self.assertEqual(((a+b)/2).argmax(1).item(), 2)
        gates = torch.tensor([2., -2., 0.]).view(1, 3, 1, 1)
        self.assertEqual(factor_fusion(a, b, gates).argmax(1).item(), 1)

    def test_router_is_neutral_initially_and_antisymmetric_after_learning(self):
        a, b = self.pair(torch.float32)
        for mode in ('flat', 'factor'):
            router = AdvantageRouter(mode, width=4)
            self.assertTrue((router(a, b) == 0).all())
            with torch.no_grad():
                router.head.weight.normal_(0, .1)
            torch.testing.assert_close(router(a, b), -router(b, a), rtol=0, atol=0)
            self.assertGreater(router(a, b).abs().max().item(), 0)
            self.assertTrue((router(a, a) == 0).all())

    def test_training_router_does_not_update_teacher_probabilities(self):
        a, b = [p.requires_grad_() for p in self.pair(torch.float32)]
        labels = torch.ones(2, 4, 6, dtype=torch.long)
        router = AdvantageRouter(width=4)
        targets, masks = advantage_targets(a, b, labels)
        loss = advantage_loss(router(a, b), targets, masks, labels)
        loss.backward()
        self.assertIsNone(a.grad); self.assertIsNone(b.grad)
        self.assertGreater(router.head.weight.grad.abs().sum().item(), 0)
        self.assertFalse(targets.requires_grad)

    def test_soft_targets_keep_level_disagreement_and_common_coverage(self):
        a = torch.zeros(2, 20, 16, 16); b = a.clone()
        a[:, 1] = 1; b[:, 2] = 1
        mean = (a+b)/2
        changed = factor_fusion(a, b, torch.ones(2, 3, 16, 16))
        neutral = pack_soft_targets(a, b, mean)
        routed = pack_soft_targets(a, b, changed)
        self.assertTrue((neutral[:, 20] == 1).all())
        self.assertTrue((neutral[:, 1:3] == .5).all())
        self.assertTrue(torch.equal(neutral[:, 20], routed[:, 20]))

    def test_cutmix_moves_probability_distribution_and_weight_together(self):
        image = torch.zeros(2, 1, 16, 16); image[1] = 1
        a = torch.zeros(2, 20, 16, 16); a[0, 0] = 1; a[1, 1] = 1
        packed = pack_soft_targets(a, a, a)
        mixed, target = aligned_cutmix(image, packed, 17, 1.)
        self.assertTrue(torch.equal(mixed[:, 0], target[:, 1]))
        torch.testing.assert_close(target[:, 20], .25 + .75*mixed[:, 0])

    def test_soft_loss_reduces_to_ce_and_blocks_all_target_gradients(self):
        logits = torch.randn(2, 20, 4, 6, requires_grad=True)
        labels = torch.ones(2, 4, 6, dtype=torch.long)
        target = F.one_hot(labels, 20).permute(0, 3, 1, 2).float()
        packed = torch.cat((target, torch.ones(2, 1, 4, 6)), 1).requires_grad_()
        loss = soft_consistency_loss(logits, packed)
        torch.testing.assert_close(loss, F.cross_entropy(logits, labels))
        loss.backward(); self.assertIsNone(packed.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        with torch.no_grad(): packed[:, 20] = 0
        self.assertEqual(soft_consistency_loss(logits, packed).item(), 0)

    def test_invalid_probabilities_and_unlabeled_masks_fail_closed(self):
        a, b = self.pair()
        with self.assertRaises(ValueError):
            advantage_targets(a, b, torch.full((2, 4, 6), -100, dtype=torch.long))
        with self.assertRaises(ValueError):
            factor_fusion(a*2, b, torch.zeros(2, 3, 4, 6))
        with self.assertRaises(ValueError):
            flat_fusion(a, b, torch.full((2, 1, 4, 6), float('nan')))

    def test_frozen_probe_runs_without_unlabeled_annotations_or_test_data(self):
        from freqssl.models import StudentTeacherSystem
        from test_freqssl import make_spider_fixture, small_config
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            png = make_spider_fixture(folder/'data')
            cfg = small_config(png, method='dual_ordinal', labelnum=4, max_samples=8, seed=42)
            model = StudentTeacherSystem(cfg)
            checkpoint = folder/'best.pt'
            torch.save(dict(format_version=3, config=cfg.to_dict(), model=model.state_dict(), step=4500), checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # A request for any U annotation or any test image/label must fail.
            for path in sorted((png/'train/label').glob('*.png'))[4:]:
                path.unlink()
            shutil.rmtree(png/'test')
            shutil.rmtree(png.parent/'mha/test')
            out = folder/'probe'
            prepare = subprocess.run([sys.executable, str(root/'experiments/probe_factor_routing.py'),
                '--prepare-only', '--smoke', '--checkpoint', str(checkpoint), '--output-dir', str(out)],
                capture_output=True, text=True, timeout=45)
            self.assertEqual(prepare.returncode, 0, prepare.stderr)
            command = json.loads((out/'command.json').read_text())
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            scores = json.loads((out/'results.json').read_text())
            self.assertTrue(scores['teacher_parameters_unchanged'])
            self.assertFalse(scores['test_read'])
            self.assertFalse(scores['screening_passed'])
            self.assertEqual(scores['final_step'], 2)
            self.assertEqual(scores['valid_cases'], 1)
            self.assertEqual(set(scores['dice']), {'teacher1', 'teacher2', 'mean', 'flat',
                                                  'factor_tissue', 'factor_level', 'factor'})
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            again = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
