import unittest
try:
    import numpy as np
    from tools.headapi.texture_finish import finish_atlas
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional image trial dependencies unavailable')
class TextureFinishTests(unittest.TestCase):
    def fixture(self):
        atlas=np.full((96,96,3),120,np.uint8);owner=np.ones((96,96),np.uint8)
        return atlas,owner,np.ones((96,96),np.uint8),np.ones((96,96),np.uint8),np.zeros((96,96),np.uint8),np.ones((96,96),np.float32)

    def test_enclosed_small_hole_is_inferred_and_large_missing_stays_unknown(self):
        args=list(self.fixture());args[1][40:42,40:42]=0;args[0][40:42,40:42]=180
        args[1][65:80,65:80]=0;args[0][65:80,65:80]=180
        out,conf,provenance,report=finish_atlas(*args)
        self.assertEqual(report['filled_texels'],4)
        self.assertTrue((provenance[40:42,40:42]==2).all());self.assertTrue((conf[40:42,40:42]<=.15).all())
        self.assertTrue((out[65:80,65:80]==180).all());self.assertTrue((provenance[65:80,65:80]==0).all())

    def test_protected_or_chart_edge_holes_are_not_filled(self):
        args=list(self.fixture());args[1][40:42,40:42]=0;args[4][43,43]=1
        args[1][0:2,10:12]=0
        out,conf,provenance,report=finish_atlas(*args)
        self.assertEqual(report['filled_texels'],0);self.assertTrue(report['protected_unchanged'])

    def test_seam_reduces_jump_without_changing_interior_features(self):
        args=list(self.fixture());args[1][:,48:]=2;args[0][:,48:]=140;args[4][25:30,46:50]=1
        before=args[0].copy();out,conf,provenance,report=finish_atlas(*args)
        self.assertLess(report['seam_after']['median_rgb_jump'],report['seam_before']['median_rgb_jump'])
        self.assertTrue(np.array_equal(out[25:30,46:50],before[25:30,46:50]))
        self.assertLessEqual(report['maximum_observed_channel_change'],3)

    def test_uniform_atlas_and_no_seam_is_unchanged(self):
        args=self.fixture();out,conf,provenance,report=finish_atlas(*args)
        self.assertTrue(np.array_equal(out,args[0]));self.assertEqual(report['changed_observed_texels'],0)
