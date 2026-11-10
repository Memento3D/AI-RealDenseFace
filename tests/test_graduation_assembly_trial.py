import unittest
import numpy as np
from tools.headapi.graduation_assembly_trial import upright_transform,cap_parts
import trimesh

class UprightPoseTests(unittest.TestCase):
    def test_eye_anchor_and_shape_preserved(self):
        reference=np.eye(4);reference[:3,:3]*=.8
        source=np.array([.01,.02,.03]);target=np.array([0,0,.88]);m=upright_transform(reference,source,target,.94)
        np.testing.assert_allclose(m[:3,:3]@source+m[:3,3],target)
        np.testing.assert_allclose(np.linalg.svd(m[:3,:3])[1],[.752]*3)
    def test_pitch_is_not_carried_from_template(self):
        angle=.2;c,s=np.cos(angle),np.sin(angle);r=np.eye(4);r[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
        result=upright_transform(r,[0,0,0],[0,0,0])
        np.testing.assert_allclose(result[:3,:3],np.eye(3),atol=1e-15)
    def test_reflection_rejected(self):
        m=np.eye(4);m[0,0]=-1
        with self.assertRaises(ValueError):upright_transform(m,[0,0,0],[0,0,0])
    def test_cap_parts_have_closed_topology(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=.1);head.apply_translation([0,0,.88])
        parts,_=cap_parts(head)
        self.assertEqual(set(parts),{'Cap_Cup','Cap_Board','Cap_Cord'})
        for part in parts.values():self.assertTrue(part.is_watertight)
    def test_fabric_band_does_not_form_hemisphere(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=.1);head.apply_translation([0,0,.88])
        parts,meta=cap_parts(head);v=parts['Cap_Cup'].vertices
        bottom=v[:64,:2];top=v[9*64:10*64,:2];center=np.array(meta['center_xy'])
        bottom_radius=np.linalg.norm(bottom-center,axis=1).mean();top_radius=np.linalg.norm(top-center,axis=1).mean()
        self.assertGreater(top_radius/bottom_radius,.9)
        self.assertLess(meta['board_width'],.23)

if __name__=='__main__':unittest.main()
