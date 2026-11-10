import unittest
import numpy as np
from tools.headapi.texel_transfer import raster,source_classes,bilinear

class PixelTransferTests(unittest.TestCase):
    def test_nearest_surface_and_pixel_center_barycentric(self):
        uv=np.array([[0,0],[1,0],[0,1],[0,0],[1,0],[0,1]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        ids,bc=raster(uv,faces,np.array([1,1,1,-1,-1,-1]),4)
        self.assertEqual(ids[2,1],1)
        np.testing.assert_allclose(bc[2,1],[.25,.375,.375])
        self.assertEqual(ids[0,3],-1)

    def test_degenerate_triangle_does_not_claim_coverage(self):
        ids,_=raster(np.array([[0,0],[.5,.5],[1,1]]),np.array([[0,1,2]]),np.zeros(3),4)
        self.assertTrue((ids==-1).all())

    def test_inferred_or_missing_neighbors_never_become_observed(self):
        labels=np.ones((2,2),np.uint8);confidence=np.full((2,2),200,np.uint8)
        classes,scores=source_classes(np.array([[.5,.5]]),labels,confidence)
        self.assertEqual(classes[0],1);self.assertEqual(scores[0],200)
        labels[0,0]=13;confidence[0,0]=13
        classes,scores=source_classes(np.array([[.5,.5]]),labels,confidence)
        self.assertEqual(classes[0],2);self.assertLessEqual(scores[0],13)
        labels[0,0]=0
        self.assertEqual(source_classes(np.array([[.5,.5]]),labels,confidence)[0][0],0)

    def test_uv_image_origin_matches_existing_texture_contract(self):
        image=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],np.uint8)
        sampled=bilinear(image,np.array([[.25,.75],[.25,.25]]))
        np.testing.assert_array_equal(sampled,[[255,0,0],[0,0,255]])

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):raster(np.array([[np.nan,0],[1,0],[0,1]]),np.array([[0,1,2]]),np.zeros(3),4)
