import unittest
import numpy as np
from memento3d_head.adapters.photo_registration import fit_camera,embedded_points
from memento3d_head.domain.contracts import ValidationError

class RegistrationTests(unittest.TestCase):
    def test_known_camera_reprojection(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('vision extra required')
        points=np.random.default_rng(42).normal(size=(20,3))*30
        camera=np.array([[800.,0,400],[0,800,300],[0,0,1]])
        targets=cv2.projectPoints(points,np.array([.1,.05,-.02]),np.array([0.,0,600]),camera,None)[0].reshape(-1,2)
        _,_,_,projected=fit_camera(points,targets,(800,600))
        self.assertLess(np.max(np.linalg.norm(projected-targets,axis=1)),1e-5)

    def test_reject_nonfinite_or_planar(self):
        for points in (np.zeros((7,3)),np.full((7,3),np.nan)):
            with self.assertRaises(ValidationError):
                fit_camera(points,np.zeros((7,2)),(800,600))

    def test_wrong_topology_rejected(self):
        with self.assertRaises(ValidationError):
            embedded_points(np.zeros((4,3)),np.zeros((4,3),dtype=int),np.zeros(68),np.ones((68,3))/3)

    def test_overlay_preserves_geometry_and_reports_held_out_chin(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from memento3d_head.adapters.photo_registration import draw_overlay, PAIRS
        import cv2
        points=np.random.default_rng(8).normal(size=(68,3))*20
        camera=np.array([[800.,0,400],[0,800,300],[0,0,1]])
        xy=cv2.projectPoints(points,np.array([.02,.01,0]),np.array([0.,0,600]),camera,None)[0].reshape(-1,2)
        observed=np.zeros((478,3))
        for i,j,_ in PAIRS+((8,152,'chin'),):
            observed[j,:2]=xy[i]/[800,600]
        before=points.copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(800,600),'white').save(root/'photo.png')
            report=draw_overlay(root/'photo.png',points,np.array([[0,1,2],[2,3,4]]),points,observed,root/'overlay.png')
            self.assertTrue((root/'overlay.png').is_file())
            self.assertLess(report['mean_fit_residual_px'],1e-4)
            self.assertFalse(report['landmarks'][-1]['used_for_fit'])
            self.assertFalse(report['geometry_modified'])
        np.testing.assert_array_equal(before,points)

    def test_registration_dispatches_before_photo_validation(self):
        from unittest.mock import patch
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.registration_commands.execute',return_value=0) as execute:
            result=main(['register-head','--head-run','run','--front','front.jpg',
                         '--model','model.task','--model-sha256','a'*64,
                         '--mica-config','mica.json','--output','out'])
        self.assertEqual(result,0)
        execute.assert_called_once()
        self.assertEqual(str(execute.call_args.args[0].front),'front.jpg')

    def test_embedding_accepts_single_batch_dimension(self):
        vertices=np.random.default_rng(4).normal(size=(5023,3))
        faces=np.tile([0,1,2],(9976,1))
        indices=np.zeros(68,dtype=int)
        weights=np.tile([.2,.3,.5],(68,1))
        flat=embedded_points(vertices,faces,indices,weights)
        batched=embedded_points(vertices,faces,indices[None],weights[None])
        np.testing.assert_allclose(flat,batched)
        self.assertEqual(batched.shape,(68,3))
        with self.assertRaises(ValidationError):
            embedded_points(vertices,faces,indices,np.tile(weights,(2,1,1)))
