"""Scientific controls for context interventions and patient-paired summaries."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from freqssl.context_diagnostics import (CONDITIONS, bootstrap_cases, context_indices,
    encode_volume, feature_relationship, observed_regions, predict_encoded, region_masks,
    score_logits, slice_metadata, utility_contrasts)
from freqssl.engine import state_hash
from freqssl.slice_context import ContextUNet, iter_cached_logits
from experiments.summarize_slice_context import clustered_summary, direction_candidates, signal_consistent


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_indices_endpoints_singleton_and_interventions(self):
        self.assertEqual(context_indices(3,7,'full'),(2,3,4))
        self.assertEqual(context_indices(3,7,'no_previous'),(3,3,4))
        self.assertEqual(context_indices(3,7,'no_next'),(2,3,3))
        self.assertEqual(context_indices(3,7,'center_only'),(3,3,3))
        self.assertEqual(context_indices(3,7,'swapped'),(4,3,2))
        self.assertEqual(context_indices(3,7,'gap3'),(0,3,6))
        self.assertEqual(context_indices(0,7,'gap3'),(0,0,3))
        self.assertEqual(context_indices(6,7,'gap2'),(4,6,6))
        for condition in CONDITIONS:
            self.assertEqual(context_indices(0,1,condition),(0,0,0))
        with self.assertRaises(ValueError):
            context_indices(7,7,'full')

    def test_geometry_changes_analysis_not_acquisition_selection(self):
        a,b = [slice_metadata(2,11,4.,reverse) for reverse in (False,True)]
        self.assertEqual(a['position'],b['position'])
        self.assertEqual(a['physical_position'],.2)
        self.assertEqual(b['physical_position'],.8)
        self.assertEqual(a['previous_physical_direction'],'right')
        self.assertEqual(b['previous_physical_direction'],'left')
        self.assertEqual(a['signed_mm_per_index'],-b['signed_mm_per_index'])
        self.assertTrue(a['both_gap2_real'])
        self.assertFalse(a['both_gap3_real'])

    def test_decoder_and_observer_match_original_network_every_condition(self):
        torch.manual_seed(3)
        model = ContextUNet(20,4,32,'adaptive').eval().requires_grad_(False)
        for layer in model.context_fusion:
            layer.gamma.fill_(.37)
            layer.weights.weight.normal_(0,.1)
        images = torch.randn(9,1,32,32)
        before = state_hash(model)
        with torch.inference_mode():
            encoded = encode_volume(model,images)
            for condition in CONDITIONS:
                triplets = torch.stack([images[list(context_indices(i,9,condition)),0] for i in range(9)])
                expected = model(triplets)
                for batch in (1,8,9):
                    output = list(predict_encoded(model,encoded,condition,batch_size=batch,observe=True))
                    actual = torch.cat([x[1] for x in output])
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    self.assertEqual(sum(len(x[2][0][0]) for x in output),9)
                    for _,logits,observed in output:
                        for weights,residual in observed:
                            self.assertEqual(len(weights),len(logits))
                            torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]))
                            if condition == 'center_only':
                                self.assertEqual(int(torch.count_nonzero(residual)),0)
            cached = torch.cat([value for _,value in iter_cached_logits(model,9,lambda i:images[i],torch.device('cpu'))])
            actual = torch.cat([value for _,value,_ in predict_encoded(model,encoded,'full')])
            torch.testing.assert_close(actual,cached,rtol=0,atol=0)
        self.assertEqual(state_hash(model),before)
        self.assertTrue(all(not layer._forward_hooks for layer in model.context_fusion))

    def test_hooks_are_removed_on_failure(self):
        model=ContextUNet(20,4,32,'adaptive').eval()
        with torch.inference_mode():
            encoded=encode_volume(model,torch.randn(1,1,32,32))
            with mock.patch.object(model,'context_features',side_effect=RuntimeError('probe')):
                with self.assertRaises(RuntimeError):
                    list(predict_encoded(model,encoded,'full',observe=True))
        self.assertTrue(all(not layer._forward_hooks for layer in model.context_fusion))

    def test_foreground_boundary_and_absent_class_scoring(self):
        target=np.zeros((8,8),dtype=np.int64);target[2:6,2:6]=1
        masks=region_masks(target)
        self.assertEqual(int(masks['vertebra'].sum()),16)
        self.assertEqual(int(masks['disc'].sum()),0)
        logits=torch.full((20,8,8),-10.)
        logits[0]=0;logits[1,2:6,2:6]=10;logits[19,0,0]=10
        full=target.copy()
        _,scores=score_logits(logits,target,full,np.array([0,1]))
        self.assertEqual(scores['all']['absent_class_pixels'],1)
        self.assertEqual(scores['all']['harmed'],1)
        self.assertEqual(scores['foreground']['errors'],0)
        self.assertIsNone(scores['disc']['class_balanced_ce'])
        self.assertEqual(scores['all']['pixels'],64)
        _,scores=score_logits(logits,target,full,np.array([0,1,19]))
        self.assertEqual(scores['all']['absent_class_pixels'],0)
        self.assertEqual(scores['all']['slice_absent_class_pixels'],1)

    def test_interaction_sign_and_nonadditive_utility(self):
        u=utility_contrasts(dict(full=.1,no_previous=.25,no_next=.3,center_only=.4))
        self.assertAlmostEqual(u['joint'],.3)
        self.assertAlmostEqual(u['previous_alone'],.1)
        self.assertAlmostEqual(u['next_alone'],.15)
        self.assertAlmostEqual(u['interaction'],.05)
        self.assertAlmostEqual(u['previous_minus_next'],-.05)

    def test_local_matching_detects_known_shift(self):
        torch.manual_seed(5)
        center=torch.randn(16,12,12)
        aligned=feature_relationship(center,center)
        self.assertLess(float(aligned['cosine_distance'].abs().max()),1e-6)
        self.assertLess(float(aligned['local_match_gain'].abs().max()),1e-6)
        shifted=feature_relationship(center,torch.roll(center,1,dims=2))
        self.assertGreater(float(shifted['local_match_gain'][:,1:-1].mean()),.5)

    def test_region_weight_coefficients_include_gamma(self):
        weights=torch.tensor([.2,.5,.3]).reshape(1,3,1,1)
        norms=torch.ones((1,2,1,1))
        result=observed_regions([(weights,norms)],0,np.ones((8,8),dtype=int),[-.5])
        r=result['foreground'][0]
        np.testing.assert_allclose(r['effective_coefficients'],[-.1,1.25,-.15])
        self.assertIsNone(result['background'][0])


class SummaryTests(unittest.TestCase):
    def test_bootstrap_unit_is_case_not_seed_times_case(self):
        one={str(i):float(i) for i in range(21)}
        result=clustered_summary({'42':one,'43':one,'44':one})
        # The summary sorts case IDs before deterministic resampling.
        reference=bootstrap_cases(one[case] for case in sorted(one))
        self.assertEqual(result['n_cases'],21)
        self.assertEqual(result['ci95'],reference['ci95'])
        self.assertEqual(result['mean'],10.)

    def test_candidate_screen_requires_three_seeds_and_complete_run(self):
        summary=clustered_summary({s:{str(i):.5 for i in range(21)} for s in ('42','43','44')})
        self.assertTrue(signal_consistent(summary,1))
        self.assertFalse(signal_consistent(summary,-1))
        partial=clustered_summary({'42':{str(i):.5 for i in range(21)}})
        self.assertFalse(signal_consistent(partial))
        associations={'foreground':{'position_vs_physical_direction_utility':summary}}
        paired={c:{'dice':summary,'dice_union_present':summary} for c in ('gap2','gap3')}
        candidates=direction_candidates(associations,{},paired,False)
        self.assertFalse(any(r['signal_supported'] for r in candidates))
        candidates=direction_candidates(associations,{},paired,True)
        self.assertEqual(sum(r['selected_for_method_design'] for r in candidates),2)
        self.assertFalse(any(r['training_authorized_by_this_report'] for r in candidates))


if __name__=='__main__':
    unittest.main()
