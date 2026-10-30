import sys,tempfile,unittest,json,types
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
from PIL import Image
import trimesh
from scalp_cap import clean_mask,infer_boundary,cap_patch,close_cap,volume_basis
from back_view import prepare_back
from run_on_head import run

class ScalpPipelineTests(unittest.TestCase):
    def config(self):return {'theta_samples':32,'phi_samples':35,'min_facing':.5,'front_views_max_theta_deg':115,'unknown_prior_weight':.1,'boundary_smoothness':16,'cap_rings':14}
    def face(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=10)
        c=np.array([[4.5,0,0,64],[0,-4.5,0,64],[0,0,-.001,1.]])
        image=Image.fromarray(np.full((128,128,3),60,np.uint8));label=np.ones((128,128),np.uint8);label[:60]=17
        return {'head':head,'eye_centers':np.array([[-2,0,4],[2,0,4]]),'cameras':[c.copy() for _ in range(3)],'images':[image.copy() for _ in range(3)],'labels':[label.copy() for _ in range(3)]}
    def test_no_posterior_root_observations_are_claimed_from_oblique_front_views(self):
        f=self.face();s=self.config();field=infer_boundary(f,[l==17 for l in f['labels']],s)
        posterior=abs(field['theta'])>np.deg2rad(115)
        np.testing.assert_array_equal(field['weights'][posterior],0)
        self.assertFalse(field['report']['posterior_observed'])
    def test_back_view_probes_below_the_bounded_nape_boundary(self):
        f=self.face();image=f['images'][0].copy()
        f['cameras'].append(np.array([[-4.5,0,0,64],[0,-4.5,0,64],[0,0,.001,1.]]))
        f['images'].append(image);label=np.ones((128,128),np.uint8);label[:95]=17
        f['labels']=[label.copy() for _ in range(4)];masks=[row==17 for row in f['labels']]
        settings=self.config();settings.update(theta_samples=160,phi_samples=90)
        field=infer_boundary(f,masks,settings)
        self.assertEqual(field['report']['posterior_probe_max_deg'],150)
        self.assertGreaterEqual(field['report']['posterior_supported_meridians'],8)
    def test_unreliable_parser_view_is_not_treated_as_negative_hair_evidence(self):
        f=self.face();f['hair_view_valid']=[True,True,False];s=self.config()
        field=infer_boundary(f,[l==17 for l in f['labels']],s)
        self.assertFalse(field['report']['source_views'][2]['used_for_hair_fit'])
        self.assertEqual(field['report']['source_views'][2]['visible_supported_samples'],0)
    def test_scaffold_closes_to_one_positive_manifold_volume_without_moving_head(self):
        f=self.face();before=f['head'].vertices.copy();s=self.config();field=infer_boundary(f,[l==17 for l in f['labels']],s)
        p,d,r,b,t=cap_patch(f,field,s);hair=close_cap(p,d,np.full(len(p.vertices),.7),b,.14)
        self.assertTrue(hair.is_volume);self.assertEqual(hair.euler_number,2);self.assertEqual(len(hair.split()),1)
        np.testing.assert_array_equal(f['head'].vertices,before)
    def test_cap_surrounds_rear_and_sides_instead_of_a_front_plate(self):
        f=self.face();s=self.config();field=infer_boundary(f,[l==17 for l in f['labels']],s)
        p,d,r,b,t=cap_patch(f,field,s);hair=close_cap(p,d,np.full(len(p.vertices),.7),b,.14)
        origin=field['origin'];phi=np.deg2rad(30);theta=np.deg2rad([0,90,180,270]);rays=np.c_[np.sin(phi)*np.sin(theta),np.full(4,np.cos(phi)),np.sin(phi)*np.cos(theta)]
        loc,ids,_=hair.ray.intersects_location(np.tile(origin,(4,1)),rays,multiple_hits=False)
        self.assertEqual(set(ids),{0,1,2,3})
    def test_pole_offset_is_independent_of_azimuth(self):
        epsilon=1e-5;d=np.array([[epsilon,np.sqrt(1-epsilon**2),0],[-epsilon,np.sqrt(1-epsilon**2),0],[0,np.sqrt(1-epsilon**2),epsilon]])
        b=volume_basis(d);np.testing.assert_allclose(b[:,0],1,atol=1e-8);self.assertLess(b[:,1:].max(),1e-8)
        np.testing.assert_allclose(b.sum(1),1,atol=1e-12)
    def test_no_hair_mask_is_rejected(self):
        with self.assertRaises(ValueError):clean_mask(Image.new('RGB',(64,64)),np.zeros((64,64),np.uint8))
    def test_boundary_color_cleanup_does_not_erase_main_hair_mass(self):
        label=np.zeros((64,64),np.uint8);label[10:50,10:50]=17
        rgb=np.full((64,64,3),190,np.uint8);rgb[10:50,10:50]=40;rgb[10,10:50]=200
        mask,r=clean_mask(Image.fromarray(rgb),label)
        self.assertTrue(mask[30,30]);self.assertGreater(mask.sum(),1200);self.assertGreater(r['bright_boundary_pixels_removed'],0)
    def test_wrong_subject_workspace_rejected_before_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'subject').mkdir();(p/'subject/subject.json').write_text(json.dumps({'workspace_sha256':'wrong','face_sha256':'wrong'}));(p/'workspace').write_text('source');(p/'subject/original_face.glb').write_text('not touched')
            with self.assertRaisesRegex(ValueError,'SHA differs'):run(p,p/'workspace',p/'model',p/'output',{})
            self.assertFalse((p/'output').exists())
    def test_non_head_back_image_is_rejected_without_guessing_camera(self):
        class Session:
            def get_inputs(self):return [types.SimpleNamespace(name='image')]
            def run(self,*args):return [np.zeros((1,19,512,512),np.float32)]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'back.png';Image.new('RGB',(128,128),'gray').save(p)
            with patch('onnxruntime.InferenceSession',return_value=Session()):
                with self.assertRaisesRegex(ValueError,'no supported hair/head region'):prepare_back(p,Path('ignored.onnx'),{})
    def test_landscape_back_photo_is_parsed_in_square_crops(self):
        class Session:
            def get_inputs(self):return [types.SimpleNamespace(name='image')]
            def run(self,*args):
                labels=np.zeros((512,512),np.uint8)
                labels[120:390,160:350]=17
                labels[340:420,130:160]=7
                labels[340:420,350:380]=8
                logits=np.full((1,19,512,512),-1,np.float32)
                logits[:,0]=0
                for label in (7,8,17):logits[0,label,labels==label]=1
                return [logits]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'back.png';Image.new('RGB',(1280,720),'gray').save(p)
            crop_sizes=[]
            def preprocess_square(image):
                crop_sizes.append(image.size)
                return np.zeros((1,3,512,512),np.float32)
            anchors=[{'side':1,'center':[1.,0.,0.]},{'side':-1,'center':[-1.,0.,0.]}]
            with patch('onnxruntime.InferenceSession',return_value=Session()), \
                    patch('back_view.preprocess',side_effect=preprocess_square), \
                    patch('back_view.ear_anchors',return_value=anchors):
                _,_,_,report=prepare_back(p,Path('ignored.onnx'),{})
            self.assertEqual(crop_sizes[0],(432,432))
            self.assertTrue(all(width==height for width,height in crop_sizes))
            self.assertLessEqual(report['ear_fit_max_px'],5)

if __name__=='__main__':unittest.main()
