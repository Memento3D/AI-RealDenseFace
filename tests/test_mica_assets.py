"""Asset integrity contracts; dummy bytes are never passed to model inference."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from memento3d_head.adapters.mica_assets import REQUIRED, pin_assets, verify_config
from memento3d_head.domain.contracts import ValidationError


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in REQUIRED:
            target=self.root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b'contract fixture only')
        self.detector=self.root/'detector.onnx'
        self.detector.write_bytes(b'not a real model')
        self.config=self.root/'config.json'
        self.mock=patch('memento3d_head.adapters.mica_assets.inspect_repo',return_value={'missing':[]})
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def pin(self):
        return pin_assets(self.root,Path(sys.executable),self.detector,self.config)

    def test_changed_asset_rejected(self):
        self.pin()
        self.detector.write_bytes(b'changed')
        with self.assertRaisesRegex(ValidationError,'Asset changed'):
            verify_config(self.config)

    def test_existing_pin_never_overwritten(self):
        self.pin()
        original=self.config.read_bytes()
        with self.assertRaises(FileExistsError):
            self.pin()
        self.assertEqual(self.config.read_bytes(),original)

    def test_asset_layout_cannot_be_redirected(self):
        self.pin()
        config=json.loads(self.config.read_text())
        config['assets'][REQUIRED[0]]['path']=str(self.detector)
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValidationError,'fixed upstream layout'):
            verify_config(self.config)
