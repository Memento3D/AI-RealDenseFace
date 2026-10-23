import unittest
from unittest.mock import patch
import numpy as np
from memento3d_head.adapters.photo_registration import PAIRS,fit_camera
from memento3d_head.adapters.shared_camera import search_shared_focal,huber_cost,heuristic_weights
from memento3d_head.domain.contracts import ValidationError

class SharedCameraTests(unittest.TestCase):
    def test_shared_focal_recovers_synthetic_camera_and_preserves_points(self):
        import cv2
        points=np.random.default_rng(10).normal(size=(68,3))*30
        selected=points[[i for i,_,_ in PAIRS]]
        before=points.copy();views={};sizes={}
        camera=np.array([[880.,0,400],[0,880.,300],[0,0,1]])
        for slot,yaw in [('front',0),('left',-.6),('right',.6)]:
            targets=cv2.projectPoints(selected,np.array([.1,yaw,.02]),np.array([3.,5.,600]),camera,None)[0].reshape(-1,2)
            k,r,t,projected=fit_camera(selected,targets,(800,600))
            views[slot]={'camera_matrix':k.tolist(),'rotation_vector':r.reshape(3).tolist(),
                         'translation_mm':t.reshape(3).tolist(),'detected_yaw_degrees':yaw*180/np.pi,
                         'landmarks':[{'observed_px':a.tolist(),'projected_px':b.tolist()} for a,b in zip(targets,projected)]}
            sizes[slot]=(800,600)
        _,_,result=search_shared_focal(points,views,sizes)
        self.assertAlmostEqual(result['focal_px'],880.,places=5)
        self.assertLess(result['candidate_weighted_huber'],1e-6)
        self.assertTrue(result['candidate_reduces_fit_objective'])
        np.testing.assert_array_equal(before,points)

    def test_different_dimensions_rejected(self):
        with self.assertRaises(ValidationError):
            search_shared_focal(np.zeros((68,3)),{}, {'front':(800,600),'left':(900,600)})

    def test_depth_heuristic_only_downweights_two_eye_points(self):
        points=np.zeros((7,3));points[[1,2],2]=10
        baseline={'rotation_vector':[0.,0.,0.],'translation_mm':[0.,0.,500.]}
        weights=heuristic_weights(points,baseline,40.)
        np.testing.assert_allclose(weights,[1,.35,.35,1,1,1,1])
        np.testing.assert_allclose(heuristic_weights(points,baseline,0.),np.ones(7))

    def test_robust_cost_limits_large_outlier_influence(self):
        residual=np.array([[100.,0.]])
        self.assertLess(huber_cost(residual,np.ones(1)),.5*100**2)

    def test_cli_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.camera_experiment_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['refine-cameras','--registration-run','run','--head-run','head',
                                   '--mica-config','mica.json','--output','out']),0)
        execute.assert_called_once()
