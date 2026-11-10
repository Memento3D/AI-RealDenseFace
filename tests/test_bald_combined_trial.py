import unittest
import tempfile
from pathlib import Path
from tools.headapi.bald_combined_trial import validate_stage,run


class BaldCombinedTests(unittest.TestCase):
    def test_stage_rejects_wrong_policy_geometry_or_failure(self):
        for row in [dict(status='FAILED',policy='p',geometry_changed=False),
                    dict(status='DONE',policy='other',geometry_changed=False),
                    dict(status='DONE',policy='p',geometry_changed=True)]:
            with self.subTest(row=row),self.assertRaises(ValueError):validate_stage(row,'DONE','p')

    def test_stage_rejects_changed_bound_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_text('new')
            with self.assertRaises(ValueError):
                validate_stage(dict(status='DONE',policy='p',geometry_changed=False,input_sha256={str(p):'old'}),'DONE','p')

    def test_existing_attempt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):run(d,d,Path(__import__('sys').executable))
