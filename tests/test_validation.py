"""Contract and failure tests use explicit fake evidence, never fake inference claims."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.application.validate_photos import validate_photos
from memento3d_head.bootstrap import main
from memento3d_head.domain.contracts import FaceObservation, Inspection, Photo, Settings, ValidationError
from memento3d_head.pipelines.photo_set_validation_v1 import PhotoSetValidationV1
from memento3d_head.settings import load_settings


class FakeInspector:
    def __init__(self, yaws=(0, -40, 40), hashes=('a', 'b', 'c'), issues=()):
        self.yaws, self.hashes, self.issues = yaws, hashes, issues

    def inspect(self, photo, destination, settings):
        index = ('front', 'left', 'right').index(photo.slot)
        face = FaceObservation((.2,.2,.8,.8), self.yaws[index], 0, 0, ())
        return Inspection(photo.slot, self.hashes[index], 900, 1200, 100, 120, face, self.issues)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.photos = tuple(Photo(s, self.root / f'{s}.jpg') for s in ('front','left','right'))

    def assess(self, inspector):
        return PhotoSetValidationV1(inspector).execute(self.photos, self.root, Settings())

    def test_missing_slot_rejected(self):
        with self.assertRaises(ValidationError):
            PhotoSetValidationV1(FakeInspector()).execute(self.photos[:2],self.root,Settings())

    def test_good_geometry_still_requires_review(self):
        result = self.assess(FakeInspector())
        self.assertEqual(result.status, 'NEEDS_HUMAN_REVIEW')
        self.assertFalse(result.identity_verified)
        self.assertFalse(result.reconstruction_ready)

    def test_same_direction_rejected(self):
        self.assertIn('SIDE_VIEWS_NOT_OPPOSITE', self.assess(FakeInspector((0,40,40))).issues)

    def test_duplicate_content_rejected(self):
        self.assertIn('DUPLICATE_IMAGE_BYTES', self.assess(FakeInspector(hashes=('a','a','c'))).issues)

    def test_front_angle_rejected(self):
        self.assertIn('front:NOT_FRONTAL', self.assess(FakeInspector((35,-40,40))).issues)

    def test_adapter_quality_rejections_preserved(self):
        self.assertIn('left:LOW_SHARPNESS', self.assess(FakeInspector(issues=('LOW_SHARPNESS',))).issues)

    def test_missing_face_rejected(self):
        class NoFace(FakeInspector):
            def inspect(self, photo, destination, settings):
                item = super().inspect(photo, destination, settings)
                return replace(item, face=None, issues=('FACE_COUNT_NOT_ONE',))
        self.assertEqual(self.assess(NoFace()).status, 'NEEDS_RETAKE')

    def test_pipeline_failure_reported(self):
        class Broken:
            def execute(self, *args):
                raise ValidationError('INVALID_IMAGE','cannot decode')
        report = validate_photos(self.photos,self.root,Settings(),Broken(),LocalStore(),{})
        self.assertEqual(report['error']['code'], 'INVALID_IMAGE')
        self.assertEqual(json.loads((self.root/'report.json').read_text())['status'], 'FAILED')

    def test_interruption_reported(self):
        class Interrupted:
            def execute(self, *args):
                raise KeyboardInterrupt()
        report = validate_photos(self.photos,self.root,Settings(),Interrupted(),LocalStore(),{})
        self.assertEqual(report['status'], 'CANCELLED')

    def test_config_unknown_and_nonfinite_rejected(self):
        for payload in ({'typo':1},{'min_sharpness':float('nan')},{'min_short_side':True},
                        {'side_min_yaw':60,'side_max_yaw':50},{'pipeline_version':2}):
            file = self.root/'config.json'
            file.write_text(json.dumps(payload))
            with self.assertRaises(ValidationError):
                load_settings(file)

    def test_existing_output_never_overwritten(self):
        for photo in self.photos:
            photo.path.touch()
        report = self.root/'report.json'
        report.write_text('original')
        code = main(['validate-photos','--front',str(self.photos[0].path),
                     '--left',str(self.photos[1].path),'--right',str(self.photos[2].path),
                     '--output',str(self.root),'--source-kind','unknown',
                     '--model',str(self.root/'missing.task'),'--model-sha256','a'*64])
        self.assertEqual(code,1)
        self.assertEqual(report.read_text(),'original')

    def test_model_failure_has_report(self):
        for photo in self.photos:
            photo.path.touch()
        output=self.root/'new'
        code=main(['validate-photos','--front',str(self.photos[0].path),
                   '--left',str(self.photos[1].path),'--right',str(self.photos[2].path),
                   '--output',str(output),'--source-kind','unknown',
                   '--model',str(self.root/'missing.task'),'--model-sha256','a'*64])
        self.assertEqual(code,1)
        self.assertEqual(json.loads((output/'report.json').read_text())['status'],'FAILED')


if __name__ == '__main__':
    unittest.main()
