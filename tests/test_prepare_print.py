import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from PIL import Image
from tools.headapi.prepare_print import prepare


class PreparePrintTests(unittest.TestCase):
    def images(self,root):
        for slot in ('front','left','right'):
            Image.new('RGB',(32,32),'gray').save(root/(slot+'.png'))

    def test_missing_views_rejected_before_creating_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'out'
            with self.assertRaises(ValueError):prepare(root,output,30)
            self.assertFalse(output.exists())

    def test_existing_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.images(root);output=root/'out';output.mkdir()
            (output/'sentinel').write_text('retained')
            with self.assertRaises(FileExistsError):prepare(root,output,30)
            self.assertEqual((output/'sentinel').read_text(),'retained')

    def test_stage_failure_retains_failed_manifest_and_never_claims_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.images(root);output=root/'out'
            with patch('tools.headapi.prepare_print.subprocess.run',return_value=Mock(returncode=1)):
                with self.assertRaises(RuntimeError):prepare(root,output,30)
            report=json.loads((output/'manifest.json').read_text())
            self.assertEqual(report['status'],'FAILED')
            self.assertFalse(report['ready_for_machine'])
            self.assertFalse((output/'head_hair_pedestal_mm.stl').exists())
