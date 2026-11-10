import unittest
import numpy as np
import trimesh
from tools.customization.fit_math import head_frame,glasses_transform,transform_points,validate_options,profile_for_head
from tools.customization.fit_trial import refine_temples,surface_details


def fixture():
    p=np.zeros((68,3));p[36]=[-.04,0,0];p[39]=[-.02,0,0]
    p[42]=[.02,0,0];p[45]=[.04,0,0];p[8]=[0,0,-.1]
    p[27:31]=[[0,-.012,.005],[0,-.02,-.01],[0,-.026,-.02],[0,-.03,-.03]]
    return p,{'left_lens_center_m':[-.036,-.05,0],'right_lens_center_m':[.036,-.05,0],
              'bridge_height_above_lens_m':.01,'bridge_clearance_m':.002,'lens_spacing_factor':1.18}


class CustomizationFitTests(unittest.TestCase):
    def test_pose_equivariance_and_orthonormal_frame(self):
        p,profile=fixture();origin,basis,_=head_frame(p)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis),1)
        angle=.43;c=np.cos(angle);s=np.sin(angle)
        T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[.3,-.4,.9]
        first,_=glasses_transform(p,profile);second,_=glasses_transform(transform_points(p,T),profile)
        np.testing.assert_allclose(second,T@first,atol=1e-12)

    def test_scale_responds_to_eye_spacing(self):
        p,profile=fixture();_,a=glasses_transform(p,profile);_,b=glasses_transform(p*1.2,profile)
        self.assertAlmostEqual(b['scale']/a['scale'],1.2)

    def test_invalid_or_inverted_landmarks_fail(self):
        p,profile=fixture()
        for bad in [np.zeros((68,3)),p[:67],np.full((68,3),np.nan),p*np.array([1,-1,1])]:
            with self.assertRaises(ValueError):glasses_transform(bad,profile)

    def test_nose_depth_changes_placement_without_changing_scale(self):
        p,profile=fixture();first,a=glasses_transform(p,profile)
        p[27:31,1]-=.008;second,b=glasses_transform(p,profile)
        self.assertEqual(a['scale'],b['scale'])
        self.assertAlmostEqual(second[1,3]-first[1,3],-.008)

    def test_mutually_exclusive_headwear(self):
        for valid in ['cap','hair']:self.assertEqual(validate_options(valid),valid)
        for invalid in ['cap+hair',None,[],['cap','hair']]:
            with self.assertRaises(ValueError):validate_options(invalid)

    def test_invalid_profiles_fail(self):
        p,profile=fixture();profile['bridge_clearance_m']=.5
        with self.assertRaises(ValueError):glasses_transform(p,profile)
        p,profile=fixture();profile['right_lens_center_m']=profile['left_lens_center_m']
        with self.assertRaises(ValueError):glasses_transform(p,profile)

    def test_temple_clearance_preserves_frame_and_head(self):
        p,profile=fixture();head=trimesh.creation.box(extents=[.1,.2,.2]);before=head.vertices.copy()
        source=np.array([[.06,0,0],[.06,.02,.01],[.06,-.05,.01]])
        vertices=np.array([[.049,0,0],[.049,.02,.01],[.049,-.05,.01]])
        mesh=trimesh.Trimesh(vertices=vertices.copy(),faces=[[0,1,2]],process=False)
        audit=refine_temples(mesh,source,head,p,profile)
        self.assertLessEqual(audit['max_displacement_mm'],8.00001)
        np.testing.assert_array_equal(mesh.vertices[2],vertices[2])
        np.testing.assert_array_equal(head.vertices,before)
        _,_,signed=surface_details(head,mesh.vertices[:2]);self.assertTrue((signed>.0014).all())

    def test_ear_support_does_not_move_front_anchors(self):
        p,profile=fixture();head=trimesh.creation.box(extents=[.1,.2,.2])
        source=np.array([[.06,0,0],[.06,.02,.01],[.06,-.05,.01]])
        v=np.array([[.06,0,0],[.06,.02,.01],[.06,-.05,.01]])
        mesh=trimesh.Trimesh(v.copy(),[[0,1,2]],process=False)
        ears=np.array([[-.06,.01,.03],[.06,.01,.03]])
        audit=refine_temples(mesh,source,head,p,profile,ears)
        np.testing.assert_array_equal(mesh.vertices[2],v[2])
        self.assertGreater(mesh.vertices[0,2],v[0,2])
        self.assertLessEqual(max(abs(item['support_rise_mm']) for item in audit['ear_support_adjustment']),12)
        with self.assertRaises(ValueError):refine_temples(mesh,source,head,p,profile,np.full((2,3),np.nan))

    def test_frame_width_adapts_and_rejects_excessive_upscaling(self):
        p,profile=fixture();source=np.array([[-.07,0,0],[.07,0,0]])
        result,audit=profile_for_head(profile,[[-.08,0,0],[.08,0,0]],p,source)
        self.assertGreater(result['lens_spacing_factor'],profile['lens_spacing_factor'])
        self.assertFalse(audit['requires_wider_asset']);self.assertNotIn('applied_uniform_upscale',profile)
        result,audit=profile_for_head(profile,[[-.2,0,0],[.2,0,0]],p,source)
        self.assertTrue(audit['requires_wider_asset']);self.assertEqual(result,profile)
        __import__('json').dumps(audit,allow_nan=False)


if __name__=='__main__':unittest.main()
