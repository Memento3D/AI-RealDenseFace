"""Adapter tests use a fake detector solely for image-ingest behavior."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
from memento3d_head.adapters.mediapipe_inspector import MediaPipeInspector, rotation_angles
from memento3d_head.domain.contracts import Photo, Settings, ValidationError


class ImageAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.adapter=MediaPipeInspector.__new__(MediaPipeInspector)
        self.adapter.mp=SimpleNamespace(Image=Mock(),ImageFormat=SimpleNamespace(SRGB=1))
        self.adapter.landmarker=Mock()
        self.adapter.landmarker.detect.return_value=SimpleNamespace(face_landmarks=[])

    def test_exif_rotation_and_no_face(self):
        path=self.root/'portrait.jpg'
        image=Image.new('RGB',(800,1000),'gray')
        exif=Image.Exif()
        exif[274]=6
        image.save(path,exif=exif)
        result=self.adapter.inspect(Photo('front',path),self.root,Settings())
        self.assertEqual((result.oriented_width,result.oriented_height),(1000,800))
        self.assertIn('FACE_COUNT_NOT_ONE',result.issues)
        self.assertTrue((self.root/'front.normalized.png').is_file())

    def test_invalid_format_rejected(self):
        path=self.root/'fake.jpg'
        Image.new('RGB',(800,1000)).save(path,format='BMP')
        with self.assertRaises(ValidationError):
            self.adapter.inspect(Photo('front',path),self.root,Settings())

    def test_pixel_limit_before_inference(self):
        path=self.root/'large.png'
        Image.new('RGB',(800,1000)).save(path)
        with self.assertRaises(ValidationError):
            self.adapter.inspect(Photo('front',path),self.root,Settings(max_pixels=100))
        self.adapter.landmarker.detect.assert_not_called()

    def test_rotation_signs(self):
        for angle in (-40,0,40):
            theta=np.deg2rad(angle)
            matrix=np.eye(4)
            matrix[:3,:3]=[[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]
            yaw,pitch,roll=rotation_angles(matrix)
            self.assertAlmostEqual(yaw,angle)
            self.assertAlmostEqual(pitch,0)
            self.assertAlmostEqual(roll,0)
