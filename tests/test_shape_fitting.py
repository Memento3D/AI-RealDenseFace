import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from memento3d_head.adapters.photo_registration import PAIRS
from memento3d_head.adapters.shape_fitting import fit_shape,barycentric_points
from memento3d_head.domain.contracts import ValidationError

class ShapeFitTests(unittest.TestCase):
    def fixture(self, offset=.2):
        import cv2
        base=np.array([[0.,0.,35.],[30.,0.,0.],[0.,30.,0.],[-20.,-20.,-25.]])
        faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
        rng=np.random.default_rng(14)
        indices=rng.integers(0,4,size=68)
        weights=rng.uniform(.1,1.,size=(68,3));weights/=weights.sum(axis=1,keepdims=True)
        basis=rng.normal(size=(4,3,8))*.3
        truth=base+basis[:,:,0]*offset
        points=barycentric_points(truth,faces,indices,weights)[[i for i,_,_ in PAIRS]]
        k=np.array([[1000.,0,640],[0,1000.,360],[0,0,1]])
        views={}
        for slot,yaw in [('front',0.),('left',-.6),('right',.6)]:
            r=np.array([.1,yaw,.02]);t=np.array([0.,0.,400.])
            xy=cv2.projectPoints(points,r,t,k,None)[0].reshape(-1,2)
            views[slot]={'camera_matrix':k.tolist(),'rotation_vector':r.tolist(),'translation_mm':t.tolist(),
                         'landmarks':[{'observed_px':p.tolist()} for p in xy]}
        return base,basis,faces,indices,weights,views

    def test_shared_shape_reduces_objective_without_modifying_source(self):
        data=self.fixture(1.)
        base,basis,*_=data;before=base.copy();before_basis=basis.copy()
        candidate,delta,info=fit_shape(*data)
        self.assertTrue(info['candidate_reduces_regularized_objective'])
        self.assertTrue(all(b<a for a,b in zip(info['objective_history'],info['objective_history'][1:])))
        self.assertLessEqual(np.max(np.abs(delta)),.5+1e-10)
        self.assertLessEqual(info['max_vertex_displacement_mm'],3.+1e-10)
        np.testing.assert_array_equal(base,before);np.testing.assert_array_equal(basis,before_basis)
        self.assertFalse(info['cameras_modified']);self.assertFalse(info['contours_used_for_fit'])
        self.assertGreater(np.linalg.norm(candidate-base),0)

    def test_exact_match_does_not_invent_shape_change(self):
        data=self.fixture(0.)
        candidate,delta,info=fit_shape(*data)
        np.testing.assert_allclose(delta,0,atol=1e-10)
        np.testing.assert_allclose(candidate,data[0],atol=1e-10)
        self.assertFalse(info['candidate_reduces_regularized_objective'])

    def test_large_target_disagreement_stays_bounded(self):
        data=self.fixture(100.)
        _,delta,info=fit_shape(*data)
        self.assertLessEqual(np.max(np.abs(delta)),.5+1e-10)
        self.assertLessEqual(info['max_vertex_displacement_mm'],3.+1e-10)

    def test_invalid_basis_rejected(self):
        data=list(self.fixture())
        data[1][0,0,0]=float('nan')
        with self.assertRaises(ValidationError):fit_shape(*data)

    def test_cli_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.shape_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['fit-head','--registration-run','run','--head-run','head',
                                   '--mica-config','mica.json','--output','out']),0)
        execute.assert_called_once()

    def test_changed_identity_rejected_before_worker(self):
        from memento3d_head.shape_commands import execute
        from memento3d_head.adapters.head_store import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';head=root/'head';run.mkdir();head.mkdir()
            mesh=head/'head.obj';mesh.write_text('fixture')
            identity=head/'identity.npy';np.save(identity,np.zeros(300,dtype=np.float32))
            previous={'status':'GEOMETRY_EXPORTED','artifacts':{
                'head.obj':{'sha256':sha256(mesh)},'identity.npy':{'sha256':sha256(identity)}}}
            (head/'report.json').write_text(json.dumps(previous))
            source={'status':'REGISTRATIONS_EXPORTED','pipeline_type':'three_view_camera_registration',
                    'views':{'front':{},'left':{},'right':{}},
                    'input':{'head_report':sha256(head/'report.json'),'mesh':sha256(mesh)}}
            (run/'report.json').write_text(json.dumps(source))
            np.save(identity,np.ones(300,dtype=np.float32))
            args=SimpleNamespace(registration_run=run,head_run=head,mica_config=root/'mica.json',output=root/'out')
            with patch('memento3d_head.shape_commands.subprocess.run') as worker:
                self.assertEqual(execute(args),1)
            worker.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_nonfinite_observation_rejected(self):
        data=self.fixture()
        data[-1]['left']['landmarks'][0]['observed_px'][0]=float('nan')
        with self.assertRaises(ValidationError):fit_shape(*data)
