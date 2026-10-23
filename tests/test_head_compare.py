import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from memento3d_head.adapters.head_compare import write_comparison
from memento3d_head.adapters.head_store import sha256
from memento3d_head.domain.contracts import ValidationError

class HeadCompareTests(unittest.TestCase):
    def fixture(self):
        return np.array([[-10.,-10.,0.],[10.,-10.,0.],[0.,10.,0.],[0.,0.,20.]]),np.array([[0,1,2],[0,3,1],[1,3,2],[2,3,0]])

    def test_shared_bounds_real_delta_and_embedded_no_network_payload(self):
        base,faces=self.fixture();candidate=base.copy();candidate[0,0]+=1.25
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'viewer.html';info=write_comparison(path,base,candidate,faces)
            html=path.read_text();payload=json.loads(html.split('<script id="model-data" type="application/json">')[1].split('</script>')[0])
            self.assertAlmostEqual(info['max_displacement_mm'],1.25)
            np.testing.assert_array_equal(payload['baseline']['vertices'],base)
            np.testing.assert_array_equal(payload['candidate']['vertices'],candidate)
            np.testing.assert_allclose(payload['center_mm'],(base.min(0)+base.max(0))*.5)
            self.assertTrue(info['shared_scale']);self.assertFalse(info['geometry_exaggerated'])
            self.assertIn("connect-src 'none'",html);self.assertNotIn('__HEAD_DATA__',html)
            self.assertNotIn('https://',html);self.assertNotIn('fetch(',html)

    def test_invalid_geometry_fails(self):
        base,faces=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):write_comparison(Path(directory)/'x.html',base,base[:-1],faces)
            bad=base.copy();bad[0,0]=float('nan')
            with self.assertRaises(ValidationError):write_comparison(Path(directory)/'x.html',base,bad,faces)

    def test_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.head_compare_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['compare-heads','--head-run','h','--candidate-run','c','--output','o']),0)
        execute.assert_called_once()

    def test_source_bound_export_rejects_changed_mesh_and_preserves_attempt(self):
        from memento3d_head.head_compare_commands import execute
        base,faces=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);head=root/'head';candidate=root/'candidate';head.mkdir();candidate.mkdir()
            obj=''.join('v %s %s %s\n'%tuple(point) for point in base)+''.join('f %d %d %d\n'%tuple(face+1) for face in faces)
            (head/'head.obj').write_text(obj);(candidate/'head.candidate.obj').write_text(obj)
            (head/'report.json').write_text(json.dumps({'status':'GEOMETRY_EXPORTED','artifacts':{'head.obj':{'sha256':sha256(head/'head.obj')}}}))
            (candidate/'report.json').write_text(json.dumps({'status':'JAW_EXPERIMENT_EXPORTED','pipeline_type':'bounded_frontal_jaw_proxy_fit',
                'input_sha256':{'head':sha256(head/'report.json'),'mesh':sha256(head/'head.obj')},
                'artifacts':{'head.candidate.obj':{'sha256':sha256(candidate/'head.candidate.obj')}}}))
            args=SimpleNamespace(head_run=head,candidate_run=candidate,output=root/'out')
            self.assertEqual(execute(args),0);before=sha256(args.output/'report.json')
            self.assertEqual(execute(args),1);self.assertEqual(before,sha256(args.output/'report.json'))
            (candidate/'head.candidate.obj').write_text(obj+'\n');args.output=root/'bad'
            self.assertEqual(execute(args),1);self.assertFalse(args.output.exists())
