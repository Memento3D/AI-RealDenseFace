import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
from memento3d_head.adapters.calibrated_camera import load_calibration,fit_fixed_camera
from memento3d_head.adapters.photo_registration import PAIRS,fit_camera,draw_overlay
from memento3d_head.adapters.head_store import sha256
from memento3d_head.domain.contracts import ValidationError

class CalibrationComparisonTests(unittest.TestCase):
    def test_pose_with_distortion_and_offcenter_principal_point(self):
        import cv2
        points=np.random.default_rng(21).normal(size=(7,3))*45
        original=points.copy()
        k=np.array([[950.,0,720],[0,940.,330],[0,0,1]])
        distortion=np.array([.2,-.3,.01,.02,.1])
        target=cv2.projectPoints(points,np.array([.12,-.4,.03]),np.array([12.,-8.,420]),k,distortion)[0].reshape(-1,2)
        initial={'rotation_vector':[.1,-.3,0.],'translation_mm':[10.,-5.,450.]}
        solution=fit_fixed_camera(points,target,k,distortion,initial,points)
        self.assertLess(np.max(np.linalg.norm(solution[3]-target,axis=1)),1e-5)
        np.testing.assert_array_equal(points,original)

    def test_profile_rejects_wrong_size_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            profile=json.loads(Path('configs/webcam.calibration.experimental.json').read_text())
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValidationError):load_calibration(path,(640,360))
            profile['models']['standard5']['distortion'][0]=float('nan')
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValidationError):load_calibration(path,(1280,720))

    def test_cli_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.calibration_comparison_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['compare-calibration','--registration-run','run','--head-run','head',
                                   '--mica-config','mica.json','--calibration','calibration.json','--output','out']),0)
        execute.assert_called_once()

    def test_comparison_outputs_both_models_and_never_overwrites(self):
        import cv2
        from memento3d_head.calibration_comparison_commands import execute
        points=np.random.default_rng(22).normal(size=(68,3))*25
        faces=np.array([[0,1,2],[2,3,4],[4,5,6],[6,7,8]])
        selected=points[[i for i,_,_ in PAIRS]]
        k=np.array([[950.,0,720],[0,940.,330],[0,0,1]])
        distortion=np.array([.2,-.3,.01,.02,.1])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';head=root/'head';out=root/'out';run.mkdir();head.mkdir()
            mesh=head/'head.obj';mesh.write_text('synthetic fixture; geometry supplied by test boundary')
            profile=json.loads(Path('configs/webcam.calibration.experimental.json').read_text())
            calibration=root/'profile.json';calibration.write_text(json.dumps(profile))
            report={'status':'REGISTRATIONS_EXPORTED','pipeline_type':'three_view_camera_registration',
                    'input':{'mesh':sha256(mesh)},'views':{},'assessment':{'photos':[]},'artifacts':{}}
            for slot,yaw in [('front',0.),('left',-.5),('right',.5)]:
                r=np.array([.1,yaw,.02]);t=np.array([0.,0.,500.])
                xy=cv2.projectPoints(points,r,t,k,distortion)[0].reshape(-1,2)
                observed=np.zeros((478,3))
                for i,j,_ in PAIRS+((8,152,'chin'),):observed[j,:2]=xy[i]/[1280,720]
                photo=run/f'{slot}.normalized.png';Image.new('RGB',(1280,720),'white').save(photo)
                overlay=run/f'{slot}.registration.png'
                baseline=draw_overlay(photo,points,faces,points,observed,overlay)
                report['views'][slot]=baseline
                report['assessment']['photos'].append({'slot':slot,'face':{'landmarks':observed.tolist()}})
                for path in (photo,overlay):report['artifacts'][path.name]={'sha256':sha256(path)}
            (run/'report.json').write_text(json.dumps(report))
            args=SimpleNamespace(registration_run=run,head_run=head,mica_config=root/'mica.json',calibration=calibration,output=out)
            def worker(*args,**kwargs):
                np.savez(out/'embedding.npz',faces=np.zeros(68),barycentric=np.zeros((68,3)))
                return SimpleNamespace(stdout='',stderr='',returncode=0)
            config={'python':'python','assets':{'data/FLAME2020/landmark_embedding.npy':{'path':'fixture','sha256':'fixture'}}}
            before=sha256(mesh)
            with patch('memento3d_head.adapters.mica_assets.verify_config',return_value=config), \
                 patch('memento3d_head.adapters.smooth_preview.read_geometry',return_value=(points,faces)), \
                 patch('memento3d_head.adapters.photo_registration.embedded_points',return_value=points), \
                 patch('memento3d_head.calibration_comparison_commands.subprocess.run',side_effect=worker):
                self.assertEqual(execute(args),0)
                result=json.loads((out/'report.json').read_text())
                self.assertEqual(set(result['views']['front']),{'baseline','standard5','radial2'})
                self.assertEqual(len(result['views']['left']['standard5']['distortion_coefficients']),5)
                self.assertFalse(result['geometry_modified'])
                with Image.open(out/'front.calibration_comparison.png') as image:self.assertEqual(image.width,3840)
                result_hash=sha256(out/'report.json')
                self.assertEqual(execute(args),1)
                self.assertEqual(sha256(out/'report.json'),result_hash)
            self.assertEqual(sha256(mesh),before)
