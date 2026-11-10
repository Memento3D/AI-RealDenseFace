import unittest
import numpy as np
from tools.headapi.bald_transition_trial import bounded_transition, render_gutter, boundary_metric, boundary_corrections


class BaldTransitionTests(unittest.TestCase):
    def test_uv_neighbors_far_apart_on_head_do_not_become_anchors(self):
        atlas=np.full((1,2,3),120,np.uint8)
        fill=np.array([[True,False]]);safe=~fill
        positions=np.array([[[0,0,0],[10,0,0]]],float)
        y,_,_=boundary_corrections(atlas,fill,safe,positions,.1)
        self.assertEqual(len(y),0)

    def test_boundary_residual_reduces_jump_without_wrapping_image(self):
        atlas=np.array([[[100,100,100],[120,120,120],[240,240,240]]],np.uint8)
        fill=np.array([[True,False,False]]);safe=np.array([[False,True,False]])
        y,x,correction=boundary_corrections(atlas,fill,safe)
        np.testing.assert_array_equal(x,[0])
        result=atlas.copy()
        result[y,x]=bounded_transition(atlas[y,x],atlas[y,x].astype(float)+correction,np.zeros(len(y)),1.)
        self.assertLess(boundary_metric(result,fill,safe)['median'],boundary_metric(atlas,fill,safe)['median'])
        np.testing.assert_array_equal(result[~fill],atlas[~fill])

    def test_metric_samples_only_estimated_to_safe_skin_edges(self):
        image=np.array([[[100,100,100],[110,100,100],[250,250,250]]],np.uint8)
        fill=np.array([[True,False,False]]);safe=np.array([[False,True,False]])
        metric=boundary_metric(image,fill,safe)
        self.assertEqual(metric['count'],1)
        self.assertEqual(metric['median'],10.)

    def test_far_skin_is_unchanged_and_near_delta_bounded(self):
        base=np.full((3,3),100,np.uint8);target=np.full((3,3),220.)
        result=bounded_transition(base,target,np.array([0.,.5,2.]),1.)
        np.testing.assert_array_equal(result[2],base[2])
        self.assertEqual(int(result[0,0]),130)
        self.assertLess(result[1,0],result[0,0])

    def test_constant_skin_is_fixed_point(self):
        base=np.full((2,3),130,np.uint8)
        np.testing.assert_array_equal(bounded_transition(base,base,np.array([0.,.2]),1.),base)

    def test_gutter_preserves_valid_texels_and_has_finite_radius(self):
        atlas=np.zeros((9,9,3),np.uint8);atlas[4,4]=[150,110,90]
        valid=np.zeros((9,9),bool);valid[4,4]=True
        result,gutter=render_gutter(atlas,valid,2)
        np.testing.assert_array_equal(result[valid],atlas[valid])
        np.testing.assert_array_equal(result[4,5],[150,110,90])
        self.assertFalse(gutter[0,0]);self.assertFalse(gutter[4,4])
