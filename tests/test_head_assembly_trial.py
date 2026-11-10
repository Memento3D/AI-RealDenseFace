import unittest
import numpy as np
from tools.headapi.head_assembly_trial import removal_mask,condition_neck

class HeadVolumeTests(unittest.TestCase):
    def test_triangle_crossing_cut_is_retained(self):
        v=np.array([[0,0,.8],[.1,0,.8],[0,0,.7],[0,.1,.9]])
        result=removal_mask(v,[[0,1,2],[0,1,3]],[-.2,-.2,.75],[.2,.2,1.1])
        np.testing.assert_array_equal(result,[False,True])
    def test_raised_hand_outside_volume_is_retained(self):
        v=np.array([[-.3,-.3,.9],[-.2,-.3,.9],[-.3,-.2,1.]])
        self.assertFalse(removal_mask(v,[[0,1,2]],[-.2,-.14,.745],[.17,.24,1.1])[0])
    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):removal_mask([[0,0,0]],[[0,0,0]],[0,0,0],[0,1,1])
    def test_neck_fit_preserves_all_non_neck_vertices(self):
        v=np.array([[.10,.15,.78],[.10,.15,.78],[.10,.15,.85]])
        fitted=condition_neck(v,[True,False,True])
        np.testing.assert_array_equal(fitted[1:],v[1:])
        self.assertLess(np.linalg.norm(fitted[0,:2]-[-.022,.08]),np.linalg.norm(v[0,:2]-[-.022,.08]))

if __name__=='__main__':unittest.main()
