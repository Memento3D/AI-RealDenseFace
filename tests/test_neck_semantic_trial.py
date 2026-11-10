import unittest
import numpy as np
from tools.headapi.neck_semantic_trial import neck_certificate,barycentric_pixels


class NeckSemanticTests(unittest.TestCase):
    def test_only_observed_neck_or_skin_below_face_is_unlocked(self):
        labels=np.array([[14,14,16,1,14]])
        owner=np.array([[1,1,1,0,1]])
        positions=np.array([[[0,-.5,0],[0,.1,0],[0,-.5,0],[0,-.5,0],[0,-.5,0]]])
        head=np.array([[True,True,True,True,False]])
        np.testing.assert_array_equal(neck_certificate(labels,owner,positions,head,0,1),[[True,False,False,False,False]])

    def test_reprojection_uses_pixel_center_barycentric_coordinates(self):
        uv=np.array([[[0,0],[1,0],[0,1]]],float)
        result=barycentric_pixels(np.array([2]),np.array([0]),np.array([0]),uv,4)
        np.testing.assert_allclose(result,[[.5,.125,.375]])
