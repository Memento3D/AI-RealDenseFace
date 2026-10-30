import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head24_policy import choose_face,crop_transform,apply_pixels,eye_gate,rigid_map,project,validate_camera_contract,InputRejected

class Head24PolicyTests(unittest.TestCase):
    def test_dominant_subject_beats_small_background_face(self):
        self.assertEqual(choose_face([[800,300,1100,650],[100,500,180,600]],(1920,2560)),0)
    def test_ambiguous_subject_stops(self):
        with self.assertRaises(InputRejected):choose_face([[200,200,450,500],[600,200,850,500]],(1024,1024))
    def test_no_default_whole_image_box(self):
        with self.assertRaises(InputRejected):choose_face([],(1920,2560))
    def test_tiny_face_stops(self):
        with self.assertRaises(InputRejected):choose_face([[0,0,30,40]],(1920,2560))
    def test_crop_roundtrip(self):
        crop,m=crop_transform([800,350,1050,700],(1920,2560));p=np.array([[801,351],[950,650]])
        self.assertTrue(np.allclose(apply_pixels(apply_pixels(p,m),np.linalg.inv(m)),p))
        self.assertEqual(crop[2]-crop[0],crop[3]-crop[1])
    def test_crop_keeps_head_margin(self):
        crop,m=crop_transform([800,350,1050,700],(1920,2560))
        self.assertLess(crop[1],350);self.assertGreater(crop[3],700)
    def test_missing_crop_margin_stops(self):
        with self.assertRaises(InputRejected):crop_transform([0,0,700,700],(800,800))
    def test_eye_order_not_anatomical(self):
        a=np.array([[100,100],[200,100]])
        self.assertLess(eye_gate(a[::-1],a)['max_eye_center_error_in_interocular_units'],1e-8)
    def test_large_false_face_rejected(self):
        with self.assertRaises(InputRejected):eye_gate([[100,300],[300,300]],[[100,100],[200,100]])
    def test_eye_geometry_unchanged(self):
        p=np.array([[100.,100],[200,100]]);original=p.copy();eye_gate(p,p)
        self.assertTrue(np.array_equal(p,original))
    def test_nan_gate_rejected(self):
        with self.assertRaises(InputRejected):eye_gate([[float('nan'),0],[1,2]],[[0,0],[100,0]])
    def test_rigid_pose_maps_all_points(self):
        p=np.random.default_rng(12).normal(size=(100,3));r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        t,error=rigid_map(p,p@r.T+[2,3,4]);self.assertLess(error,1e-12)
        self.assertTrue(np.allclose(t[:3,:3],r))
    def test_nonrigid_map_rejected(self):
        p=np.random.default_rng(12).normal(size=(100,3));q=p.copy();q[0,0]+=.01
        with self.assertRaises(InputRejected):rigid_map(p,q)
    def test_explicit_camera_uses_own_expression(self):
        p=np.random.default_rng(12).normal(size=(100,3));p[:,2]+=10
        P=np.array([[500,0,512,0],[0,500,512,0],[0,0,1,0.]])
        uv,z=project(p,P);q=np.column_stack((uv[:,0]/1024*z,(1-uv[:,1]/1024)*z,z))
        self.assertLess(validate_camera_contract(p,q,P,(1024,1024))['max_px'],1e-8)
        different=p.copy();different[:,0]+=.1
        with self.assertRaises(InputRejected):validate_camera_contract(different,q,P,(1024,1024))
    def test_camera_behind_rejected(self):
        with self.assertRaises(InputRejected):project(np.array([[0,0,-1.]]),np.eye(3,4))

if __name__=='__main__':unittest.main()
