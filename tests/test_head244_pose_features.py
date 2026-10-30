import sys,unittest,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from head24_pose import cranial_camera
from head24_features import front_observation,mouth_region,observed_mouth_anchor
from head24_policy import InputRejected

class PoseTests(unittest.TestCase):
    def fixture(self):
        phi=np.linspace(0,2*np.pi,32,endpoint=False);theta=np.linspace(.04,np.pi-.04,24)
        v=np.array([[.10*np.sin(t)*np.cos(p),.15*np.cos(t),.12*np.sin(t)*np.sin(p)] for t in theta for p in phi])
        camera=np.array([[600,0,512,512],[0,600,512,512],[0,0,1,1.]])
        return v,camera,np.ones(len(v),bool),np.array([[-.03,0,.1],[.03,0,.1]])
    def test_neck_rigid_motion_is_corrected_in_camera_not_geometry(self):
        v,p,head,eyes=self.fixture();original=v.copy();a=np.radians(7);r=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        view=v@r.T+[.002,.004,0];new,report=cranial_camera(v,view,p,head,eyes)
        self.assertLess(report['cranial_holdout_p95_px'],1e-8)
        self.assertGreater(report['raw_upper_all_p95_px'],5)
        self.assertTrue(np.array_equal(v,original));self.assertFalse(np.array_equal(new,p))
    def test_eye_motion_does_not_define_cranial_frame(self):
        v,p,head,eyes=self.fixture();extra=np.array([[-.03,.02,.1],[.03,.02,.1]])
        v=np.vstack((v,extra));head=np.r_[head,False,False];view=v.copy();view[-2:,1]+=.08
        new,report=cranial_camera(v,view,p,head,eyes)
        self.assertTrue(np.allclose(new,p));self.assertLess(report['cranial_holdout_p95_px'],1e-8)
    def test_real_nonrigid_cranium_is_not_accepted(self):
        v,p,head,eyes=self.fixture();view=v.copy();rng=np.random.default_rng(8);view+=rng.normal(0,.035,view.shape)
        with self.assertRaisesRegex(InputRejected,'CRANIAL_POSE_PROXY_MISMATCH'):cranial_camera(v,view,p,head,eyes)
    def test_repeated_corners_do_not_increase_support(self):
        v,p,head,eyes=self.fixture();_,a=cranial_camera(v,v,p,head,eyes)
        _,b=cranial_camera(np.repeat(v,3,axis=0),np.repeat(v,3,axis=0),p,np.repeat(head,3),eyes)
        self.assertEqual(a['train_unique_vertices'],b['train_unique_vertices']);self.assertEqual(a['holdout_unique_vertices'],b['holdout_unique_vertices'])
    def test_insufficient_support_rejected(self):
        v,p,head,eyes=self.fixture()
        with self.assertRaises(InputRejected):cranial_camera(v[:10],v[:10],p,head[:10],eyes)

class FeatureTests(unittest.TestCase):
    def fixture(self):
        im=Image.new('RGB',(64,64),(100,90,95));lm=np.tile([32.,32.],(478,1));lm[61]=[20,32];lm[0]=[32,25];lm[291]=[44,32];lm[17]=[32,40]
        observation={'crop_pixel_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),'landmarks_crop':lm.tolist()}
        return im,lm,observation
    def test_front_landmarks_are_bound_to_exact_pixels(self):
        im,lm,obs=self.fixture();self.assertTrue(np.array_equal(front_observation({'front_observation':obs},im),lm))
        with self.assertRaisesRegex(ValueError,'another crop'):front_observation({'front_observation':obs},Image.new('RGB',im.size))
    def test_glasses_without_eye_semantics_do_not_invent_eyelids(self):
        from figurine24_appearance import prepared_front
        im,lm,_=self.fixture();labels=np.ones((64,64),np.uint8);labels[10:24,12:52]=6
        settings={'radius_px':8,'sigma_small_px':2.5,'sigma_large_px':8,'dark_L_limit':9,'bright_L_limit':2,'strength':.85}
        _,_,report=prepared_front(im,labels,np.array([.5,.45,.4]),settings,lm,True)
        self.assertFalse(report['eye_semantics_available']);self.assertEqual(report['detail_pixels'],0);self.assertFalse(report['hidden_eye_details_recovered'])
    def test_no_semantics_no_observation_rejected(self):
        from figurine24_appearance import prepared_front
        im,_,_=self.fixture();labels=np.ones((64,64),np.uint8)
        with self.assertRaises(ValueError):prepared_front(im,labels,np.array([.5,.45,.4]),{})
    def test_mouth_roi_projects_to_source_without_moving_it(self):
        im,lm,_=self.fixture();mask=mouth_region(lm,im.size)
        pixels=np.array([[x,y] for x in range(16,49,2) for y in range(22,43,2)])
        vertices=np.column_stack(((pixels[:,0]-32)/10,(32-pixels[:,1])/10,np.ones(len(pixels))))
        q=np.column_stack((pixels[:,0]/64,1-pixels[:,1]/64,np.ones(len(pixels))));original=vertices.copy()
        _,width,report=observed_mouth_anchor(vertices,q,mask,np.ones(len(vertices),bool),np.ones(len(vertices)),1,np.zeros(3))
        self.assertGreater(width,.5);self.assertTrue(np.array_equal(vertices,original));self.assertFalse(report['source_geometry_modified'])

if __name__=='__main__':unittest.main()
