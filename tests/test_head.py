"""CPU integration contracts, never substituted for MICA inference evidence."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from memento3d_head.adapters.head_store import LocalHeadStore
from memento3d_head.adapters.mesh_preview import validate_and_preview
from memento3d_head.adapters.mica_backend import stop_worker
from memento3d_head.application.reconstruct_head import reconstruct_head
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.head import HeadRequest
from memento3d_head.head_commands import execute
from types import SimpleNamespace


class HeadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.validation=self.root/'validation'
        self.validation.mkdir()
        self.front=self.root/'front.jpg'
        Image.new('RGB',(800,1000),'gray').save(self.front)
        digest=hashlib.sha256(self.front.read_bytes()).hexdigest()
        self.report={'schema_version':1,'pipeline_type':'photo_set_validation','pipeline_version':1,
                     'status':'NEEDS_HUMAN_REVIEW','error':None,
                     'provenance':{'source_kind_declared':'real_capture'},
                     'assessment':{'issues':[],'photos':[{'slot':slot,'source_sha256':digest} for slot in ('front','left','right')]}}
        self.write_report()
        self.output=self.root/'output'
        self.output.mkdir()

    def write_report(self):
        (self.validation/'report.json').write_text(json.dumps(self.report))

    def request(self,**changes):
        values={'validation_run':self.validation,'front':self.front,'reviewed_by':'Tester'}
        return HeadRequest(**(values|changes))

    def test_front_snapshot_bound_to_source(self):
        result=LocalHeadStore().prepare(self.request(),self.output)
        self.assertEqual(result['views_used_for_geometry'],['front'])
        self.assertTrue((self.output/'front.input.png').is_file())
        self.assertTrue((self.output/'validation.snapshot.json').is_file())

    def test_front_modified_rejected(self):
        Image.new('RGB',(800,1000),'red').save(self.front)
        with self.assertRaisesRegex(ValidationError,'Front bytes differ'):
            LocalHeadStore().prepare(self.request(),self.output)

    def test_rejected_photos_never_reconstruct(self):
        self.report['status']='NEEDS_RETAKE'
        self.write_report()
        class Never:
            def reconstruct(self,*args):
                raise AssertionError('backend must not execute')
        result=reconstruct_head(self.request(),self.output,Never(),LocalHeadStore())
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['error']['code'],'VALIDATION_NOT_PASSED')

    def test_synthetic_requires_explicit_override(self):
        self.report['provenance']['source_kind_declared']='synthetic'
        self.write_report()
        with self.assertRaises(ValidationError):
            LocalHeadStore().prepare(self.request(),self.output)
        result=LocalHeadStore().prepare(self.request(allow_synthetic=True),self.output)
        self.assertEqual(result['source_kind_declared'],'synthetic')

    def test_empty_reviewer_rejected(self):
        with self.assertRaises(ValidationError):
            LocalHeadStore().prepare(self.request(reviewed_by=' '),self.output)

    def test_backend_failure_retained(self):
        class Broken:
            def reconstruct(self,*args):
                raise ValidationError('MISSING_ASSETS','weights missing')
        result=reconstruct_head(self.request(),self.output,Broken(),LocalHeadStore())
        self.assertEqual(result['error']['code'],'MISSING_ASSETS')
        self.assertTrue((self.output/'report.json').is_file())
        self.assertFalse(result['multi_view_fitting'])

    def test_cancellation_reported(self):
        class Cancel:
            def reconstruct(self,*args):
                raise KeyboardInterrupt()
        result=reconstruct_head(self.request(),self.output,Cancel(),LocalHeadStore())
        self.assertEqual(result['status'],'CANCELLED')

    def test_preview_real_geometry_not_template_fallback(self):
        mesh=self.root/'tetra.obj'
        mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
        result=validate_and_preview(mesh,self.root/'preview.png')
        self.assertEqual(result['faces'],4)
        with Image.open(self.root/'preview.png') as preview:
            self.assertEqual(preview.size,(1200,460))

    def test_invalid_mesh_rejected(self):
        mesh=self.root/'invalid.obj'
        mesh.write_text('v nan 0 0\nf 1 1 999\n')
        with self.assertRaises(ValidationError):
            validate_and_preview(mesh,self.root/'preview.png')

    def test_existing_run_not_overwritten(self):
        report=self.output/'report.json'
        report.write_text('original')
        code=execute(SimpleNamespace(command='mica-doctor',output=self.output,mica_config=self.root/'missing.json'))
        self.assertEqual(code,1)
        self.assertEqual(report.read_text(),'original')

    def test_child_process_group_stopped(self):
        # No waiting for the sleep; stop immediately to validate real process cleanup.
        kwargs={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform=='win32' else {'start_new_session':True}
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
        try:
            stop_worker(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
