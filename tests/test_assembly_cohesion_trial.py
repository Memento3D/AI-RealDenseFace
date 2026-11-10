import unittest
import numpy as np
from tools.headapi.assembly_cohesion_trial import arm_weights,smooth_seam_normals,boundary_loops,bridge_rings,scalp_envelope_scale

class CohesionTests(unittest.TestCase):
    def test_torso_and_other_hand_are_locked(self):
        np.testing.assert_array_equal(arm_weights([[0,0,.9],[-.2,0,.7],[.2,0,.9]]),[0,0,0])
        self.assertEqual(arm_weights([[-.15,0,.97]])[0],1)
        self.assertEqual(arm_weights([[-.095,0,.97]])[0],1)
    def test_normals_match_across_split_uv_seams(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,1]],float);original=v.copy()
        n=smooth_seam_normals(v,[[0,1,2],[3,4,5]])
        np.testing.assert_allclose(n[0],n[3]);np.testing.assert_array_equal(v,original)
    def test_uv_seams_do_not_become_holes(self):
        v=[[0,0,0],[1,0,0],[0,1,0],[1,0,0],[1,1,0],[0,1,0]]
        loops=boundary_loops(v,[[0,1,2],[3,4,5]])
        self.assertEqual(len(loops),1);self.assertEqual(len(loops[0]),4)
    def test_neck_zipper_keeps_both_boundary_rings(self):
        a=np.array([[np.cos(t),np.sin(t),0] for t in np.linspace(0,2*np.pi,8,endpoint=False)])
        b=np.array([[.7*np.cos(t),.7*np.sin(t),.2] for t in np.linspace(0,2*np.pi,12,endpoint=False)])
        v,f=bridge_rings(a,b[::-1]);loops=boundary_loops(v,f)
        self.assertEqual(len(loops),2);self.assertEqual(sorted(map(len,loops)),[8,12]);self.assertEqual(len(f),20)
    def test_boundary_points_retain_original_precision(self):
        v=np.array([[.0123456789,0,0],[1,0,0],[0,1,0]])
        loop=boundary_loops(v,[[0,1,2]])[0]
        self.assertTrue(any(np.array_equal(p,v[0]) for p in loop))
    def test_cap_envelope_expands_without_moving_scalp(self):
        v=np.array([[.08,0,.94],[0,.08,.94],[0,0,.97]]);copy=v.copy()
        factor,audit=scalp_envelope_scale(v,[0,0],[.06,.06],.93)
        self.assertGreater(factor,1);self.assertEqual(audit['outside_after'],0);np.testing.assert_array_equal(v,copy)

if __name__=='__main__':unittest.main()
