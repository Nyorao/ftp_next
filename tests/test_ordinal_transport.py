"""Scientific invariants for partial label-axis transport, independent of data."""
import unittest
import torch
from torch.nn import functional as F

from freqssl.augment import aligned_cutmix
from freqssl.objectives import masked_pseudo_loss
from freqssl.ordinal_transport import estimate_offset, transport_peer, cross_targets, anchored_mean


def probabilities(labels):
    return F.one_hot(torch.as_tensor(labels).long(), 20).permute(0, 3, 1, 2).float()


class OrdinalTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def sequence(self):
        labels = torch.tensor([1, 2, 3, 11, 12, 13])[None, None].expand(1, 16, 6)
        return probabilities(labels), probabilities(labels + 3)

    def test_known_shift_and_teacher_exchange(self):
        a, b = self.sequence()
        ta, tb, evidence = cross_targets(a, b)
        reverse = estimate_offset(b, a)
        self.assertEqual(evidence['offset'].item(), 3)
        self.assertEqual(reverse['offset'].item(), -3)
        torch.testing.assert_close(evidence['scores'], reverse['scores'].flip(1))
        self.assertTrue(torch.equal(ta, a)); self.assertTrue(torch.equal(tb, b))

    def test_missing_one_anchor_or_incoherent_offset_abstains(self):
        a = probabilities(torch.tensor([1, 11])[None, None].expand(1, 16, 2))
        b = probabilities(torch.tensor([4, 14])[None, None].expand(1, 16, 2))
        self.assertEqual(estimate_offset(a, b)['offset'].item(), 0)
        a = probabilities(torch.tensor([1, 2, 3, 4])[None, None].expand(1, 16, 4))
        b = probabilities(torch.tensor([2, 3, 6, 7])[None, None].expand(1, 16, 4))
        self.assertEqual(estimate_offset(a, b)['offset'].item(), 0)

    def test_large_structure_does_not_outvote_three_other_levels(self):
        a = probabilities(torch.tensor([1]*160 + [2]*16 + [3]*16 + [4]*16)[None, None])
        b = probabilities(torch.tensor([1]*160 + [4]*16 + [5]*16 + [6]*16)[None, None])
        self.assertEqual(estimate_offset(a, b)['offset'].item(), 2)

    def test_zero_offset_exactly_recovers_targets_loss_and_mean(self):
        torch.manual_seed(43)
        a, b = (torch.randn(2, 20, 4, 5).softmax(1) for _ in range(2))
        shift = torch.zeros(2, dtype=torch.long)
        target = transport_peer(a, b, shift)
        self.assertTrue(torch.equal(target, b))
        self.assertTrue(torch.equal(anchored_mean(a, b, target, shift), (a+b)/2))
        logits = torch.randn_like(a, requires_grad=True)
        actual = masked_pseudo_loss(logits, target)[0]
        expected = masked_pseudo_loss(logits, b)[0]
        self.assertTrue(torch.equal(actual, expected))

    def test_no_wrap_renormalization_or_tissue_crossing(self):
        own = torch.zeros(1, 20, 1, 1); own[:, 0] = 1
        peer = torch.zeros_like(own)
        peer[:, 0] = .1; peer[:, 1] = .3; peer[:, 4] = .4; peer[:, 10] = .1; peer[:, 11] = .1
        moved = transport_peer(own, peer, torch.tensor([3]))
        self.assertAlmostEqual(moved[:, 1].item(), .4)
        self.assertAlmostEqual(moved.sum().item(), .6)
        self.assertTrue(torch.equal(moved[:, (0, 10)], peer[:, (0, 10)]))
        self.assertEqual(moved[:, 9].item(), 0)
        self.assertEqual(moved[:, 11:].sum().item(), 0)
        loss, stats = masked_pseudo_loss(torch.zeros_like(own, requires_grad=True), moved)
        self.assertEqual(stats['selected_pixels'], 0); self.assertEqual(loss.item(), 0)

    def test_confident_absolute_agreements_are_preserved(self):
        a, b = self.sequence()
        b[:, :, :, 0] = a[:, :, :, 0]
        moved = transport_peer(a, b, torch.tensor([3]))
        self.assertTrue(torch.equal(moved[:, :, :, 0], b[:, :, :, 0]))
        self.assertTrue(torch.equal(moved[:, :, :, 1:], a[:, :, :, 1:]))

    def test_all_background_and_low_confidence_abstain(self):
        bg = probabilities(torch.zeros(2, 16, 16, dtype=torch.long))
        self.assertTrue((estimate_offset(bg, bg)['offset'] == 0).all())
        a, b = self.sequence(); a = a * .7 + .3 / 20; b = b * .7 + .3 / 20
        self.assertEqual(estimate_offset(a, b)['offset'].item(), 0)

    def test_stopped_targets_cutmix_and_finite_student_gradient(self):
        a, b = self.sequence(); a.requires_grad_(); b.requires_grad_()
        target, _, _ = cross_targets(a, b)
        self.assertFalse(target.requires_grad)
        target = torch.cat((target, target.flip(-1)))
        images = target.argmax(1, keepdim=True).float()
        mixed, packed = aligned_cutmix(images, target, seed=27, probability=1.)
        self.assertTrue(torch.equal(mixed[:, 0], packed.argmax(1).float()))
        logits = torch.zeros_like(packed, requires_grad=True)
        loss, _ = masked_pseudo_loss(logits, packed); loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all()); self.assertGreater(logits.grad.abs().sum(), 0)
        self.assertIsNone(a.grad); self.assertIsNone(b.grad)


if __name__ == '__main__':
    unittest.main()
