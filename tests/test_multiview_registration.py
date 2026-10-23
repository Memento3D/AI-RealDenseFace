import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from memento3d_head.adapters.head_store import sha256
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.multiview_commands import bind_sources

class MultiViewRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        head=root/'head';valid=root/'valid';head.mkdir();valid.mkdir()
        paths={slot:root/(slot+'.jpg') for slot in ('front','left','right')}
        for slot,path in paths.items():path.write_bytes(slot.encode())
        (head/'head.obj').write_text('original mesh',encoding='utf-8')
        validation={'status':'NEEDS_HUMAN_REVIEW','assessment':{'issues':[],
            'photos':[{'slot':slot,'source_sha256':sha256(path)} for slot,path in paths.items()]}}
        (valid/'report.json').write_text(json.dumps(validation),encoding='utf-8')
        report={'status':'GEOMETRY_EXPORTED','input':{
            'source_sha256':sha256(paths['front']),
            'validation_report_sha256':sha256(valid/'report.json')},
            'artifacts':{'head.obj':{'sha256':sha256(head/'head.obj')}}}
        (head/'report.json').write_text(json.dumps(report),encoding='utf-8')
        return head,valid,paths

    def test_bind_exact_three_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            head,valid,paths=self.fixture(Path(directory))
            mesh,evidence=bind_sources(head,valid,paths)
            self.assertEqual(mesh,head/'head.obj')
            self.assertEqual(set(evidence['photos']),{'front','left','right'})

    def test_swapped_sides_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            head,valid,paths=self.fixture(Path(directory))
            paths['left'],paths['right']=paths['right'],paths['left']
            with self.assertRaises(ValidationError) as error:bind_sources(head,valid,paths)
            self.assertEqual(error.exception.code,'PHOTO_SET_MISMATCH')

    def test_changed_mesh_or_validation_rejected(self):
        for target in ('mesh','validation'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as directory:
                head,valid,paths=self.fixture(Path(directory))
                path=head/'head.obj' if target=='mesh' else valid/'report.json'
                path.write_text('{}',encoding='utf-8')
                with self.assertRaises(ValidationError):bind_sources(head,valid,paths)

    def test_cli_dispatch_all_three_paths(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.multiview_commands.execute',return_value=0) as execute:
            result=main(['register-views','--head-run','head','--validation-run','validation',
                         '--front','front.jpg','--left','left.jpg','--right','right.jpg',
                         '--model','model.task','--model-sha256','a'*64,'--mica-config','mica.json','--output','out'])
        self.assertEqual(result,0)
        self.assertEqual(str(execute.call_args.args[0].left),'left.jpg')
        self.assertEqual(str(execute.call_args.args[0].right),'right.jpg')
