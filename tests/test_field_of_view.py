import unittest

import torch
import numpy as np

from freqssl.field_of_view import canvas_pair


class FieldOfViewTests(unittest.TestCase):
    def test_intervention_keeps_content_identical_with_no_crop_or_photometric_change(self):
        images = torch.randn(2,1,224,224,generator=torch.Generator().manual_seed(42))
        views, places = canvas_pair(images)
        self.assertTrue(torch.equal(places[0].extract(views[0]),places[1].extract(views[1])))
        self.assertTrue(torch.equal(views[0].flatten(1).sort(1).values,views[1].flatten(1).sort(1).values))
        self.assertEqual(places[1].si_start-places[0].si_start,32)
        self.assertEqual(places[0].ap_start,places[1].ap_start)

    def test_transport_preserves_a_pointwise_predictor_exactly(self):
        images = torch.randn(1,1,224,224,generator=torch.Generator().manual_seed(71))
        views, places = canvas_pair(images)
        # A sign classifier is an exact arithmetic reference. SIMD softmax can
        # differ by a float32 ULP at different canvas addresses even for a 1x1
        # convolution, which must not be mistaken for a coordinate error.
        outputs=[where.extract(torch.cat((view,-view),1)) for view,where in zip(views,places)]
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)

    def test_stress_dice_retains_case_gt_present_foreground_policy(self):
        from experiments.probe_fov_equivariance import case_dice
        confusion=np.zeros((20,20),dtype=np.int64)
        confusion[1,1]=60;confusion[1,0]=40;confusion[0,1]=40
        confusion[2,2]=20;confusion[2,0]=20;confusion[0,2]=20
        confusion[0,3]=99  # Absent-GT class is excluded from the primary mean.
        self.assertAlmostEqual(case_dice(confusion),.55,places=12)


if __name__=='__main__':unittest.main()
