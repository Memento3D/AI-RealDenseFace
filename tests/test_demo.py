import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from memento3d_head.adapters.head_store import sha256
from memento3d_head.demo_commands import execute,find_photos


class DemoTests(unittest.TestCase):
    def setup_case(self,root):
        photos=root/'input'/'01';photos.mkdir(parents=True)
        for slot in ('front','left','right'):Image.new('RGB',(100,100),'red').save(photos/(slot+'.png'))
        model=root/'model.task';model.write_bytes(b'test model')
        return SimpleNamespace(input_dir=photos,output=root/'out',model=model,model_sha256=sha256(model),mica_config=root/'mica.json',config=root/'settings.json',no_open=True)

    def test_names_are_case_insensitive_but_ambiguous_files_fail(self):
        from memento3d_head.domain.contracts import ValidationError
        with tempfile.TemporaryDirectory() as tmp:
            args=self.setup_case(Path(tmp));(args.input_dir/'front.png').rename(args.input_dir/'FRONT.PNG')
            self.assertEqual(set(find_photos(args.input_dir)),{'front','left','right'})
            Image.new('RGB',(10,10)).save(args.input_dir/'front.jpg')
            with self.assertRaises(ValidationError):find_photos(args.input_dir)

    def fake_stage(self,name,arguments,output):
        # Test doubles for heavy inference only. Color generation below is real.
        output.mkdir();args=list(map(str,arguments))
        def option(flag):return Path(args[args.index(flag)+1])
        if name=='01_validation':report={'status':'NEEDS_HUMAN_REVIEW'}
        elif name=='02_head':
            (output/'head.obj').write_text('v -10 -10 100\nv 10 -10 100\nv 0 10 100\nv 0 0 120\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
            (output/'head.ply').write_bytes(b'fixture');(output/'identity.npy').write_bytes(b'fixture')
            report={'status':'GEOMETRY_EXPORTED','artifacts':{'head.obj':{'sha256':sha256(output/'head.obj')}}}
        else:
            head=option('--head-run');view={'camera_matrix':[[100.,0,50],[0,100.,50],[0,0,1]],'rotation_vector':[0.,0.,0.],'translation_mm':[0.,0.,0.]}
            landmarks=[[.1,.1,0],[.5,.1,0],[.9,.1,0],[.9,.5,0],[.9,.9,0],[.5,.9,0],[.1,.9,0],[.1,.5,0],[.4,.4,0],[.6,.6,0]]
            for slot in ('front','left','right'):
                with Image.open(option('--'+slot)) as image:image.save(output/(slot+'.normalized.png'))
            report={'status':'REGISTRATIONS_EXPORTED','input':{'mesh':sha256(head/'head.obj'),'head_report':sha256(head/'report.json')},'views':{s:view for s in ('front','left','right')},'assessment':{'photos':[{'slot':s,'oriented_width':100,'oriented_height':100,'face':{'landmarks':landmarks}} for s in ('front','left','right')]},'artifacts':{s+'.normalized.png':{'sha256':sha256(output/(s+'.normalized.png'))} for s in ('front','left','right')}}
        (output/'report.json').write_text(json.dumps(report));return report

    def test_fresh_folder_to_actual_color_viewer_no_stale_jaw_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.setup_case(Path(tmp));original={s:sha256(p) for s,p in find_photos(args.input_dir).items()}
            with patch('memento3d_head.adapters.mica_assets.verify_config'),patch('memento3d_head.settings.load_settings'),patch('memento3d_head.demo_commands.run_stage',side_effect=self.fake_stage) as stages,patch('memento3d_head.demo_commands.review_photos',return_value='Test reviewer'):
                self.assertEqual(execute(args),0)
                self.assertEqual(stages.call_count,3)
                report=json.loads((args.output/'report.json').read_text())
                self.assertEqual(report['status'],'DEMO_PREVIEW_EXPORTED')
                self.assertFalse(report['manual_jaw_fit_performed'])
                self.assertTrue((args.output/'final'/'multiview_preview.html').is_file())
                self.assertEqual(sha256(args.output/'02_head'/'head.obj'),sha256(args.output/'final'/'head.obj'))
                before=sha256(args.output/'report.json')
                self.assertEqual(execute(args),1)
                self.assertEqual(before,sha256(args.output/'report.json'))
            self.assertEqual(original,{s:sha256(p) for s,p in find_photos(args.input_dir).items()})

    def test_failed_validation_never_reviews_or_starts_mica(self):
        from memento3d_head.domain.contracts import ValidationError
        with tempfile.TemporaryDirectory() as tmp:
            args=self.setup_case(Path(tmp))
            with patch('memento3d_head.adapters.mica_assets.verify_config'),patch('memento3d_head.settings.load_settings'),patch('memento3d_head.demo_commands.run_stage',side_effect=ValidationError('STAGE_FAILED','retake')) as stage,patch('memento3d_head.demo_commands.review_photos') as review:
                self.assertEqual(execute(args),1);self.assertEqual(stage.call_count,1);review.assert_not_called()
                self.assertFalse((args.output/'02_head').exists())

    def test_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.demo_commands.execute',return_value=0) as demo:
            self.assertEqual(main(['demo','--input-dir','input/02']),0)
            self.assertEqual(demo.call_args.args[0].input_dir,Path('input/02'))
