import unittest
import numpy as np
from tools.headapi.neck_completion_trial import neck_blend


class NeckCompletionTests(unittest.TestCase):
    def test_face_and_ineligible_pixels_are_never_blended(self):
        result=neck_blend(np.array([.1,.3,.4,.8,.9]),np.array([True,True,True,True,False]),.3,.52)
        np.testing.assert_array_equal(result[[0,1,4]],[0,0,0])
        self.assertEqual(result[3],1)
        self.assertGreater(result[2],0)

    def test_blend_is_bounded_monotone_and_rejects_invalid_range(self):
        values=np.linspace(-1,2,100)
        result=neck_blend(values,np.ones_like(values,bool),.35,.65)
        self.assertTrue(((result>=0)&(result<=1)).all());self.assertTrue((np.diff(result)>=0).all())
        with self.assertRaises(ValueError):neck_blend(values,np.ones_like(values,bool),.65,.35)
