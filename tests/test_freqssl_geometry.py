"""Physical-coordinate and invertibility checks; all arrays synthetic."""
import itertools
import unittest
import numpy as np
import SimpleITK as sitk

from freqssl.geometry import canonicalize, restore_native, sagittal_plan, transformed_geometry


class GeometryTests(unittest.TestCase):
    def test_permutations_and_flips_match_itk_physical_coordinates(self):
        image=sitk.GetImageFromArray(np.arange(3*4*5,dtype=np.int16).reshape(3,4,5))
        image.SetSpacing((.5,1.2,3.0));image.SetOrigin((17,-8,21))
        a=.17;image.SetDirection((np.cos(a),-np.sin(a),0,np.sin(a),np.cos(a),0,0,0,1))
        for order in itertools.permutations(range(3)):
            for flags in itertools.product((False,True),repeat=3):
                flips=[i for i,flag in enumerate(flags) if flag]
                geometry=transformed_geometry(image.GetSize(),image.GetSpacing(),image.GetOrigin(),image.GetDirection(),order,flips)
                expected=sitk.PermuteAxes(image,[2-axis for axis in reversed(order)])
                expected=sitk.Flip(expected,[2-i in flips for i in range(3)])
                self.assertEqual(geometry['size_xyz'],list(expected.GetSize()))
                np.testing.assert_allclose(geometry['spacing_xyz_mm'],expected.GetSpacing(),atol=1e-12)
                np.testing.assert_allclose(geometry['origin_xyz'],expected.GetOrigin(),atol=1e-12)
                np.testing.assert_allclose(geometry['direction'],expected.GetDirection(),atol=1e-12)
                array=sitk.GetArrayFromImage(expected)
                plan=sagittal_plan(geometry)
                np.testing.assert_array_equal(restore_native(canonicalize(array,plan),plan),array)

    def test_canonical_axes_have_known_anatomical_sign(self):
        # Physical z increases toward superior; canonical final axis must reverse.
        a=np.arange(3*4*5).reshape(3,4,5)
        geometry=transformed_geometry((5,4,3),(1,1,1),(0,0,0),np.eye(3).ravel(),(0,1,2))
        plan=sagittal_plan(geometry)
        np.testing.assert_array_equal(canonicalize(a,plan),a.transpose(2,1,0)[:,:,::-1])
        self.assertFalse(plan['uses_labels'])

    def test_invalid_or_strongly_oblique_geometry_is_not_silently_guessed(self):
        g=transformed_geometry((5,4,3),(1,1,1),(0,0,0),np.eye(3).ravel(),(0,1,2))
        a=np.pi/4;g['direction']=[np.cos(a),-np.sin(a),0,np.sin(a),np.cos(a),0,0,0,1]
        with self.assertRaises(ValueError):sagittal_plan(g)
        g['direction']=[1]*9
        with self.assertRaises(ValueError):sagittal_plan(g)


if __name__=='__main__':unittest.main()
