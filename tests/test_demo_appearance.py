import unittest
import numpy as np
from tools.headapi.demo_appearance import presentation_regions,atlas_positions


class DemoAppearanceTests(unittest.TestCase):
    def test_unknown_neck_filled_but_front_eyes_and_observed_face_excluded(self):
        positions=np.array([[[0,-.5,1],[0,.4,1],[0,.1,1],[0,.9,.8]]],np.float32)
        observed=np.array([[False,False,True,False]]);skin=np.ones_like(observed)
        neck,scalp=presentation_regions(positions,observed,skin,np.array([0,.54,1]),np.array([0,.4,1]),np.array([0,0,1]),2.)
        self.assertTrue(neck[0,0]);self.assertFalse(neck[0,1]);self.assertFalse(scalp[0,1])
        self.assertFalse(neck[0,2]);self.assertFalse(scalp[0,2]);self.assertTrue(scalp[0,3])

    def test_lower_rear_gap_gets_skin_and_observed_scalp_skin_is_preserved(self):
        positions=np.array([[[0,.1,.2],[0,.9,1],[float('nan'),0,0]]],np.float32)
        observed=np.array([[False,True,False]]);skin=np.array([[False,True,False]])
        neck,scalp=presentation_regions(positions,observed,skin,np.array([0,.54,1]),np.array([0,.4,1]),np.array([0,0,1]),2.)
        self.assertTrue(neck[0,0]);self.assertFalse(scalp[0,1]);self.assertFalse(neck[0,2]);self.assertFalse(scalp[0,2])

    def test_uv_position_mapping_and_outside_uv(self):
        face_index=np.full((4,4),-1,np.int32);face_index[2,0]=0
        v=np.array([[0,0,0],[2,0,0],[0,3,0]],float);f=np.array([[0,1,2]]);uv=np.array([[[0,0],[1,0],[0,1]]],float)
        p=atlas_positions(face_index,v,f,uv)
        np.testing.assert_allclose(p[2,0],[.25,1.125,0]);self.assertTrue(np.isnan(p[0,0]).all())
