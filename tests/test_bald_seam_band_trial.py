import unittest
import numpy as np
from tools.headapi.bald_seam_band_trial import low_frequency_delta,seam_endpoints,paired_residuals


class BaldSeamBandTests(unittest.TestCase):
    def test_paired_residual_protects_observed_side_more_and_closes_step(self):
        atlas=np.array([[[100,100,100],[160,160,160]]],np.uint8)
        estimated=np.array([[True,False]]);observed=~estimated
        pos=np.array([[[0,0,0],[.01,0,0]]],float)
        e,o=paired_residuals(atlas,estimated,observed,pos,.1,np.array([[0,0,1.]]),np.zeros((1,2),int))
        np.testing.assert_allclose(e[2],[[45,45,45]])
        np.testing.assert_allclose(o[2],[[-15,-15,-15]])
        np.testing.assert_allclose(atlas[e[0],e[1]]+e[2],atlas[o[0],o[1]]+o[2])

    def test_constant_skin_has_no_correction(self):
        a=np.zeros((30,30),bool);a[:,:15]=True
        delta=low_frequency_delta(np.full((30,30,3),120,np.uint8),a,~a,3)
        np.testing.assert_allclose(delta,0,atol=1e-3)

    def test_two_sided_correction_reduces_low_frequency_step(self):
        a=np.zeros((30,30),bool);a[:,:15]=True
        atlas=np.full((30,30,3),160,np.uint8);atlas[a]=100
        delta=low_frequency_delta(atlas,a,~a,3)
        self.assertGreater(delta[15,14,0],0)
        self.assertLess(delta[15,15,0],0)
        self.assertLess(abs((160+delta[15,15,0])-(100+delta[15,14,0])),60)

    def test_uv_adjacency_cannot_bridge_spatially_distant_surfaces(self):
        estimated=np.array([[True,False]]);observed=~estimated
        pos=np.array([[[0,0,0],[10,0,0]]],float)
        seam=seam_endpoints(estimated,observed,pos,.1,np.array([[0,0,1.]]),np.zeros((1,2),int))
        self.assertEqual(seam.sum(),0)
