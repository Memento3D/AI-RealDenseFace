"""Research mode and rendering contracts; all inference doubles are explicit."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from memento3d_head.adapters.experiment_input import LocalExperimentStore
from memento3d_head.adapters.smooth_preview import read_geometry, render_smooth, rasterize, vertex_normals
from memento3d_head.application.experiment_head import experiment_head
from memento3d_head.bootstrap import main
from memento3d_head.domain.contracts import FaceObservation, Inspection, Photo, Settings, ValidationError
from memento3d_head.pipelines.single_photo_validation_v1 import SinglePhotoValidationV1


class EvidenceInspector:
    """Deterministic inspection evidence, not MediaPipe or MICA inference."""
    def __init__(self, yaw=0, issues=()):
        self.yaw=yaw
        self.issues=issues

    def inspect(self, photo, destination, settings):
        Image.new('RGB',(800,1000),'gray').save(destination/'front.normalized.png')
        return Inspection('front','a'*64,800,1000,100,150,
                          FaceObservation((.2,.2,.8,.8),self.yaw,0,0,()),self.issues)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.mesh=self.root/'tetra.obj'
        self.mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')

    def test_single_gate_retains_frontal_quality_checks(self):
        result=SinglePhotoValidationV1(EvidenceInspector(30)).execute(Photo('front',Path('fixture')),self.root,Settings())
        self.assertIn('NOT_FRONTAL',result.issues)

    def test_bad_photo_does_not_call_backend(self):
        class Never:
            def reconstruct(self,*args):
                raise AssertionError('Inference must not execute')
        report=experiment_head(Photo('front',Path('fixture')),self.root,Settings(),'real_capture',
                               SinglePhotoValidationV1(EvidenceInspector(issues=('LOW_SHARPNESS',))),
                               Never(),LocalExperimentStore(),{'test_double':True})
        self.assertEqual(report['status'],'NEEDS_RETAKE')
        self.assertEqual(report['artifacts'],{})
        self.assertTrue(report['research_only'])
        self.assertFalse(report['input']['photo_set_validated'])

    def test_model_failure_has_input_evidence_and_report(self):
        class Broken:
            def reconstruct(self,front,output):
                self.front=front
                raise ValidationError('MISSING_ASSETS','No inference in unit test')
        backend=Broken()
        report=experiment_head(Photo('front',Path('fixture')),self.root,Settings(),'synthetic',
                               SinglePhotoValidationV1(EvidenceInspector()),backend,LocalExperimentStore(),{})
        self.assertEqual(report['error']['code'],'MISSING_ASSETS')
        self.assertEqual(backend.front,self.root/'front.input.png')
        self.assertFalse(report['multi_view_fitting'])
        self.assertEqual(report['input']['source_kind_declared'],'synthetic')
        self.assertTrue((self.root/'report.json').is_file())

    def test_single_snapshot_preserves_bytes(self):
        Image.new('RGB',(800,1000),'gray').save(self.root/'front.normalized.png')
        evidence=LocalExperimentStore().snapshot_front(self.root)
        self.assertEqual((self.root/'front.input.png').read_bytes(),(self.root/'front.normalized.png').read_bytes())
        self.assertEqual(evidence['normalized_sha256'],hashlib.sha256((self.root/'front.input.png').read_bytes()).hexdigest())

    def test_single_interruption_retained(self):
        class Cancel:
            def execute(self,*args):
                raise KeyboardInterrupt()
        report=experiment_head(Photo('front',Path('fixture')),self.root,Settings(),'unknown',Cancel(),None,LocalExperimentStore(),{})
        self.assertEqual(report['status'],'CANCELLED')

    def test_render_preserves_obj_and_geometry(self):
        before=self.mesh.read_bytes()
        report=render_smooth(self.mesh,self.root/'smooth.png')
        self.assertEqual(self.mesh.read_bytes(),before)
        self.assertEqual(report['angles_degrees'],[0,-45,45,-90,90])
        self.assertFalse(report['geometry_modified'])
        with Image.open(self.root/'smooth.png') as preview:
            self.assertEqual(preview.size,(1200,940))

    def test_depth_selects_near_triangle_regardless_of_order(self):
        vertices=np.array([[-1,-1,0],[1,-1,0],[0,1,0],[-1,-1,2],[1,-1,2],[0,1,2]],dtype=float)
        faces=np.array([[0,1,2],[3,4,5]])
        normals=np.tile([0,0,1.],(6,1))
        for order in (faces,faces[::-1]):
            pixels,depth=rasterize(vertices,order,normals,64,64,20)
            self.assertAlmostEqual(depth[32,32],2.0)
            self.assertTrue(np.isfinite(depth).any())

    def test_normals_do_not_modify_vertices(self):
        vertices,faces=read_geometry(self.mesh)
        before=vertices.copy()
        normals=vertex_normals(vertices,faces)
        np.testing.assert_array_equal(vertices,before)
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1)

    def test_invalid_obj_fails_instead_of_drawing_template(self):
        self.mesh.write_text('v nan 1 0\nf 1 1 1\n')
        with self.assertRaises(ValidationError):
            render_smooth(self.mesh,self.root/'bad.png')
        self.assertFalse((self.root/'bad.png').exists())

    def test_render_cli_no_inference(self):
        output=self.root/'render'
        result=main(['render-head','--mesh',str(self.mesh),'--output',str(output)])
        self.assertEqual(result,0)
        report=json.loads((output/'report.json').read_text())
        self.assertFalse(report['inference_run'])
        self.assertEqual(report['status'],'PREVIEW_EXPORTED')
        self.assertEqual(report['source_mesh_sha256'],hashlib.sha256(self.mesh.read_bytes()).hexdigest())

    def test_research_output_cannot_overwrite(self):
        report=self.root/'report.json'
        report.write_text('original')
        result=main(['render-head','--mesh',str(self.mesh),'--output',str(self.root)])
        self.assertEqual(result,1)
        self.assertEqual(report.read_text(),'original')
