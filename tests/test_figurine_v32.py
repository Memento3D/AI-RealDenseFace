"""Detected scalp labels must drive the boundary, including narrow sideburns."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
    import trimesh
    import manifold3d
    from figurine_hairline32 import propagate_unknown,clip_surface,thicken_patch,collect_observations,stabilize_labels,refine_source_surface
    from figurine_geometry import union_print_parts
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional figurine CPU dependencies unavailable')
class DetectedHairlineTests(unittest.TestCase):
    def test_synthetic_closure_descendants_are_tracked(self):
        mesh=trimesh.creation.box()
        refined,excluded=refine_source_surface(mesh,len(mesh.faces)-1)
        self.assertEqual(len(refined.faces),len(mesh.faces)*16)
        self.assertGreater(len(excluded),3)
        tri=mesh.triangles[-1];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        self.assertLess(np.max(abs((refined.vertices[excluded]-tri[0])@normal)),1e-10)

    def test_known_labels_remain_fixed_during_unknown_interpolation(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=2)
        known=abs(head.vertices[:,1])>.5
        values=(head.vertices[:,1]>.5).astype(float)
        result=propagate_unknown(head,values,known)
        np.testing.assert_array_equal(result[known],values[known])
        self.assertTrue(((result>=0)&(result<=1)).all())
        self.assertTrue(((result[~known]>0)&(result[~known]<1)).any())

    def test_unknown_half_probability_is_not_invented_as_hair(self):
        a=trimesh.creation.icosphere(subdivisions=1,radius=1);b=a.copy();b.apply_translation([4,0,0])
        mesh=trimesh.util.concatenate([a,b]);n=len(a.vertices)
        values=np.r_[np.ones(n),np.full(n,.5)];known=np.r_[np.ones(n,bool),np.zeros(n,bool)]
        result,removed=stabilize_labels(mesh,values,values,known)
        self.assertTrue((result[:n]>.5).all());self.assertTrue((result[n:]<.5).all())

    def test_unknown_positive_island_requires_observed_hair_support(self):
        a=trimesh.creation.icosphere(subdivisions=1,radius=1);b=a.copy();b.apply_translation([4,0,0])
        mesh=trimesh.util.concatenate([a,b]);n=len(a.vertices)
        values=np.r_[np.ones(n),np.full(n,.6)];known=np.r_[np.ones(n,bool),np.zeros(n,bool)]
        result,removed=stabilize_labels(mesh,values,values,known)
        self.assertEqual(removed,n);self.assertTrue((result[n:]==0).all())

    def test_no_observations_do_not_create_a_default_hairline(self):
        head=trimesh.creation.icosphere(subdivisions=1,radius=2)
        with self.assertRaisesRegex(ValueError,'Not enough observed'):
            propagate_unknown(head,np.zeros(len(head.vertices)),np.zeros(len(head.vertices),bool))

    def test_out_of_frame_is_unknown_not_skin(self):
        head=trimesh.creation.icosphere(subdivisions=1,radius=2)
        camera=np.array([[10,0,-10000,100000],[0,-10,-10000,100000],[0,0,-1,10.]])
        mask=np.zeros((32,32),bool);depth=np.full((32,32),np.inf);label=np.ones((32,32),np.uint8)
        values,known,_,_=collect_observations(head,[camera],[mask],[depth],[label])
        self.assertEqual(int(known.sum()),0)

    def test_a_deep_narrow_observed_sideburn_survives_clipping(self):
        head=trimesh.creation.icosphere(subdivisions=3,radius=2)
        x,y,z=head.vertices.T
        hair=(y>.6)|((x>1.4)&(abs(z)<.45)&(y>-.8))
        patch,normals,known,boundary=clip_surface(head,hair.astype(float),np.ones(len(x),bool))
        self.assertLess(patch.vertices[:,1].min(),-.5)
        mesh,outer,thickness=thicken_patch(patch,normals,boundary,dict(base_mm=.4,anchor_mm=.2,edge_taper_mm=.4))
        self.assertTrue(mesh.is_volume)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(boundary),10)

    def test_sliver_cleanup_does_not_hide_a_detached_real_part(self):
        a=trimesh.creation.box();b=a.copy();b.apply_translation([4,0,0])
        with self.assertRaisesRegex(ValueError,'Detached'):
            union_print_parts([a,b],allow_tiny_surface_slivers=True)

if __name__=='__main__':unittest.main()
