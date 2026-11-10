import io,pickle,unittest
import numpy as np
from tools.headapi.ear_texture_trial import mask_indices,ear_observation,RestrictedMasks,feather_fill


class EarTextureTests(unittest.TestCase):
    def test_observed_ear_support_keeps_inferred_transition_connected(self):
        original=np.full((40,40,3),100,np.uint8);target=np.full_like(original,180)
        region=np.zeros((40,40),bool);region[3:37,3:37]=True
        supported=region.copy();supported[:,:20]=False
        observed=region&~supported
        _,without=feather_fill(original,target,supported,region)
        result,with_observed=feather_fill(original,target,supported,region,observed)
        self.assertGreater(with_observed[20,20],without[20,20])
        np.testing.assert_array_equal(result[observed],original[observed])

    def test_inferred_fill_feathers_at_edge_and_never_changes_observed(self):
        original=np.full((40,40,3),100,np.uint8);target=np.full_like(original,180)
        supported=np.zeros((40,40),bool);supported[5:35,5:35]=True
        result,alpha=feather_fill(original,target,supported,supported)
        np.testing.assert_array_equal(result[~supported],original[~supported])
        self.assertLess(alpha[5,5],alpha[20,20])
        self.assertLessEqual(float(alpha.max()),.8+1e-6)
        self.assertLessEqual(result.max(),164)

    def test_mask_indices_reject_duplicates_out_of_bounds_and_floats(self):
        for values in [[1,1],[0,5],[-1,2],[1.,2.]]:
            with self.subTest(values=values),self.assertRaises(ValueError):mask_indices(values,5)

    def test_observation_requires_template_ear_semantics_owner_and_confidence(self):
        result=ear_observation(np.array([True,True,True,False,True]),np.array([7,1,8,7,8]),np.array([1,1,0,1,1]),np.ones(5,bool),np.array([.8,.8,.8,.8,.1]))
        np.testing.assert_array_equal(result,[True,False,False,False,False])

    def test_mask_loader_refuses_non_numpy_globals(self):
        with self.assertRaises(ValueError):RestrictedMasks(io.BytesIO(pickle.dumps(ValueError))).load()
