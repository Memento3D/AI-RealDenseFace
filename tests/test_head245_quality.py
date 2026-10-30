import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
from head245_skin import skin_reference
from head245_glasses import tube

class MultiviewSkinTests(unittest.TestCase):
    settings={'minimum_pixels':150,'disagreement_deltaE00':10,'override_rgb':None}
    def test_uses_better_exposed_side_not_blue_front(self):
        colors=[[100,95,115],[103,72,60],[177,131,97]]
        ims=[Image.new('RGB',(64,64),tuple(c)) for c in colors];labels=[np.ones((64,64),np.uint8) for _ in ims]
        base,masks,r=skin_reference(ims,labels,self.settings)
        np.testing.assert_allclose(base,np.array(colors[2])/255)
        self.assertEqual(r['selected_view'],'right');self.assertEqual(r['confidence'],'LOW');self.assertFalse(r['true_skin_albedo_verified'])
    def test_glasses_and_reflections_excluded(self):
        a=np.full((64,64,3),[150,110,90],np.uint8);a[:24]=[240,240,240];labels=np.ones((64,64),np.uint8);labels[:24]=6
        base,_,r=skin_reference([Image.fromarray(a)],[labels],self.settings)
        np.testing.assert_allclose(base,[150/255,110/255,90/255])
    def test_no_skin_refuses(self):
        with self.assertRaisesRegex(ValueError,'No trustworthy'):skin_reference([Image.new('RGB',(64,64))],[np.zeros((64,64),np.uint8)],self.settings)
    def test_explicit_color_override_is_labelled(self):
        im=Image.new('RGB',(64,64),(100,90,80));base,_,r=skin_reference([im],[np.ones((64,64),np.uint8)],dict(self.settings,override_rgb=[200,150,120]))
        np.testing.assert_allclose(base,[200/255,150/255,120/255]);self.assertEqual(r['color_status'],'USER_DESIGN_OVERRIDE')
    def test_nan_override_rejected(self):
        with self.assertRaises(ValueError):skin_reference([Image.new('RGB',(64,64),(100,90,80))],[np.ones((64,64),np.uint8)],dict(self.settings,override_rgb=[np.nan,1,1]))

class FeatureIsolationTests(unittest.TestCase):
    def settings(self):return {'radius_px':8.,'sigma_small_px':2.5,'sigma_large_px':8.,'dark_L_limit':9.,'bright_L_limit':2.,'strength':.85}
    def test_glasses_photo_is_not_painted_on_skin(self):
        from figurine245_appearance import prepared_front
        rgb=np.full((100,100,3),[170,125,95],np.uint8);labels=np.ones((100,100),np.uint8)
        labels[40:55,10:90]=6;rgb[40:55,10:90]=[40,60,160];labels[70:75,20:30]=4
        out,active,r=prepared_front(Image.fromarray(rgb),labels,np.array([170,125,95])/255,self.settings())
        np.testing.assert_allclose(out[47,50],[170,125,95],atol=1)
    def test_missing_brows_have_two_landmark_supported_regions(self):
        from figurine245_appearance import prepared_front
        obs=np.full((478,2),50.)
        for indices,x in [([70,63,105,66,107,55,65,52,53,46],20),([336,296,334,293,300,276,283,282,295,285],65)]:
            polygon=[[x,20],[x+4,18],[x+8,18],[x+12,19],[x+16,21],[x+16,24],[x+12,22],[x+8,21],[x+4,21],[x,23]]
            obs[indices]=polygon
        labels=np.ones((100,100),np.uint8);labels[40:45,20:30]=4
        out,active,r=prepared_front(Image.new('RGB',(100,100),(160,110,90)),labels,np.array([160,110,90])/255,self.settings(),obs)
        self.assertTrue(r['landmark_brow_fallback_used']);self.assertGreater(active[20,27],0);self.assertGreater(active[20,72],0)

class FrameGeometryTests(unittest.TestCase):
    def test_closed_lens_is_volume(self):
        t=np.arange(64)*2*np.pi/64;p=np.c_[2*np.cos(t),np.sin(t),np.zeros(64)];m=tube(p,.18,True)
        self.assertTrue(m.is_volume);self.assertTrue(m.is_winding_consistent)
        self.assertGreaterEqual(m.bounds[1,2]-m.bounds[0,2],.34)
        self.assertLessEqual(m.bounds[1,2]-m.bounds[0,2],.36+1e-8)
    def test_open_arm_has_caps_and_positive_volume(self):
        m=tube([[0,0,0],[1,0,-1],[2,0,-3]],.18);self.assertTrue(m.is_volume);self.assertEqual(len(m.split()),1)
    def test_attach_frame_to_existing_glb_without_overwrite_error(self):
        import tempfile
        import trimesh
        from run_pipeline_v2 import glb
        from run_pipeline_v245 import append_frame
        from add_graduation_cap import load
        from figurine_preview import accessor
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'appearance.glb';head=trimesh.creation.box()
            glb(path,[('Source',head,[.5,.4,.3,1.])]);doc,blob=load(path);original=accessor(doc,blob,0).copy()
            frame=tube([[0,0,0],[1,0,-1],[2,0,-3]],.18)
            append_frame(path,frame,[43,43,39]);doc,blob=load(path)
            np.testing.assert_array_equal(accessor(doc,blob,0),original)
            self.assertEqual(len(doc['scenes'][0]['nodes']),2)
    def test_bad_radius_refuses(self):
        with self.assertRaises(ValueError):tube([[0,0,0],[1,0,0],[2,1,0]],np.nan)

class NumericalUnionTests(unittest.TestCase):
    def test_tiny_local_float_fragment_is_recorded(self):
        import trimesh
        from figurine245_boolean import solid_union
        a=trimesh.creation.box([2,2,2]);b=trimesh.creation.box([.0001,.0001,.0001]);b.apply_translation([3,0,0])
        result=solid_union([a,b]);self.assertEqual(len(result.split()),1)
        self.assertEqual(len(result.metadata['surface_processing']['discarded_numerical_fragments']),1)
    def test_real_detached_piece_is_rejected(self):
        import trimesh
        from figurine245_boolean import solid_union
        a=trimesh.creation.box([2,2,2]);b=trimesh.creation.box([.1,.1,.1]);b.apply_translation([3,0,0])
        with self.assertRaisesRegex(ValueError,'Detached'):solid_union([a,b])
    def test_long_thin_fragment_not_deleted_on_volume_alone(self):
        import trimesh
        from figurine245_boolean import solid_union
        a=trimesh.creation.box([2,2,2]);b=trimesh.creation.box([1,.00001,.00001]);b.apply_translation([3,0,0])
        with self.assertRaisesRegex(ValueError,'Detached'):solid_union([a,b])

class SupportSearchTests(unittest.TestCase):
    def test_fewer_islands_wins_over_lower_area_cost(self):
        from unittest.mock import patch
        import trimesh
        import head245_supports as support
        settings={'lift_mm':3.,'voxel_pitch_mm':.5,'max_supports':64,'protect_front':True,'tip_length_mm':.8}
        poses=[(1.,20,0,np.eye(4)),(9.,50,180,np.eye(4))]
        with patch.object(support,'candidate_orientations',return_value=poses),patch.object(support,'scan_layers',side_effect=[([],[{'layer':1}]),([],[])]) as scanner:
            selected,trials=support.choose_orientation(trimesh.creation.box(),settings)
        self.assertEqual(selected[1:3],(50,180));self.assertEqual(trials[0]['unresolved_islands'],1)
        self.assertTrue(all(call.args[-1] for call in scanner.call_args_list))
if __name__=='__main__':unittest.main()
