import unittest
import numpy as np
from tools.headapi.bald_tone_trial import correction_field,smoothstep,bounded_rgb,tone_weights


class BaldToneTests(unittest.TestCase):
    def test_existing_seam_and_protected_hole_receive_zero_weight(self):
        skin=np.zeros((30,30),bool);skin[:,:15]=True
        inferred=~skin;skin[5:8,5:8]=False
        weight=tone_weights(skin,inferred,np.ones((30,30)))
        self.assertTrue((weight[:,14:16]==0).all())
        self.assertTrue((weight[5:8,5:8]==0).all())
        self.assertEqual(weight[20,0],1.)
        self.assertEqual(weight[20,29],1.)

    def test_uniform_reference_has_zero_correction(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
        lab=np.tile([60,12,15.],(3,1))
        np.testing.assert_allclose(correction_field(v,f,v,lab,np.ones(3),lab[0]),0,atol=1e-8)

    def test_field_limits_and_disconnected_component_prior(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[4,0,0],[5,0,0],[4,1,0]],float);f=np.array([[0,1,2],[3,4,5]])
        result=correction_field(v,f,v[:3],np.zeros((3,3)),np.ones(3),np.array([100,30,30]))
        self.assertLessEqual(np.abs(result[:,0]).max(),4)
        self.assertLessEqual(np.abs(result[:,1:]).max(),3)
        np.testing.assert_allclose(result[3:],0,atol=1e-8)

    def test_rgb_delta_and_feature_envelope_are_bounded(self):
        base=np.array([[100,120,130]],np.uint8)
        result=bounded_rgb(base,np.array([[200,0,130.]]))
        np.testing.assert_array_equal(result,[[112,108,130]])
        np.testing.assert_array_equal(smoothstep(np.array([0,.12,.25,.5]),.12,.25),[0,0,1,1])
