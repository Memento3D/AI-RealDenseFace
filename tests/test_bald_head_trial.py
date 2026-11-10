import unittest
import numpy as np
from tools.headapi.bald_head_trial import surface_colors, fill_regions, topology_report


class BaldHeadTrialTests(unittest.TestCase):
    def test_constant_skin_continuation_is_constant_even_without_anchors(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[5,0,0],[6,0,0],[5,1,0]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        prior=np.array([150,110,90.])
        result=surface_colors(vertices,faces,vertices[:1],prior[None],np.ones(1),prior)
        np.testing.assert_allclose(result,np.tile(prior,(6,1)),atol=1e-8)

    def test_disconnected_eye_does_not_receive_skin_anchor(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[5,0,0],[6,0,0],[5,1,0]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        result=surface_colors(vertices,faces,vertices[:1],np.array([[220,80,70]]),np.ones(1),[140,120,100])
        np.testing.assert_allclose(result[3:],np.tile([140,120,100],(3,1)),atol=1e-8)
        self.assertGreater(result[0,0],200)

    def test_observed_face_and_protected_eye_unchanged_but_scalp_hair_replaced(self):
        p=np.array([[[0,.8,1],[0,.3,1],[0,.1,1],[0,-.5,1],[0,.9,1]]])
        observed=np.array([[True,False,True,False,False]])
        skin=np.array([[False,False,True,False,False]])
        protected=np.array([[1,1,0,0,0]])
        head=np.array([[True,True,True,True,False]])
        fill,_=fill_regions(p,observed,skin,protected,head,np.array([0,.3,1]),2.)
        np.testing.assert_array_equal(fill,[[True,False,False,True,False]])

    def test_open_head_is_not_reported_manufacturing_ready(self):
        result=topology_report(np.array([[0,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
        self.assertEqual(result['boundary_edges'],3)
        self.assertFalse(result['watertight'])
        self.assertFalse(result['manufacturing_ready'])
