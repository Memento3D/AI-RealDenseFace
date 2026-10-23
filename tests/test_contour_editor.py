import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from memento3d_head.domain.contour_annotations import validate_annotations
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.adapters.visible_contour_editor import projected_outer_boundary,write_editor

class ContourEditorTests(unittest.TestCase):
    def fixture(self):
        manifest={'workspace_id':'test','source_binding':{'mesh':'abc'},'views':{
            slot:{'image_size':[100,80],'image_sha256':slot+'sha','outlines':{'baseline':[],'candidate':[]}} for slot in ('front','left','right')}}
        data={'schema_version':1,'pipeline_type':'manual_visible_contours','workspace_id':'test',
              'source_binding':{'mesh':'abc'},'visible_only_reviewed':True,'views':{
                  slot:{'image_size':[100,80],'image_sha256':slot+'sha','curves':[
                      {'id':slot+'-1','kind':'visible_profile','points_px':[[10,10],[20,25],[30,40]]}]} for slot in ('front','left','right')}}
        return manifest,data

    def test_valid_curves_are_not_claimed_semantically_verified(self):
        manifest,data=self.fixture();r=validate_annotations(data,manifest)
        self.assertEqual(r['status'],'ANNOTATIONS_STRUCTURALLY_VALID')
        self.assertFalse(r['semantic_accuracy_verified']);self.assertFalse(r['ready_for_automatic_fitting'])

    def test_stale_binding_and_wrong_dimensions_rejected(self):
        for target in ('source','size'):
            manifest,data=self.fixture()
            if target=='source':data['source_binding']['mesh']='wrong'
            else:data['views']['front']['image_size']=[200,160]
            with self.assertRaises(ValidationError):validate_annotations(data,manifest)

    def test_invalid_and_incomplete_curves_rejected(self):
        for change in ('nan','outside','duplicate','empty_view','unchecked'):
            manifest,data=self.fixture()
            points=data['views']['front']['curves'][0]['points_px']
            if change=='nan':points[0][0]=float('nan')
            elif change=='outside':points[0][0]=100
            elif change=='duplicate':points[1]=points[0].copy()
            elif change=='empty_view':data['views']['left']['curves']=[]
            else:data['visible_only_reviewed']=False
            with self.subTest(change=change),self.assertRaises(ValidationError):validate_annotations(data,manifest)

    def test_foreground_boundary_does_not_draw_hidden_internal_triangle(self):
        vertices=np.array([[-10.,-10.,100.],[10.,-10.,100.],[10.,10.,100.],[-10.,10.,100.],
                           [-2.,-2.,110.],[2.,-2.,110.],[0.,2.,110.]])
        faces=np.array([[0,1,2],[0,2,3]])
        view={'camera_matrix':[[100.,0,50],[0,100.,40],[0,0,1]],'rotation_vector':[0.,0,0],
              'translation_mm':[0.,0,0]}
        before=vertices.copy()
        outer=projected_outer_boundary(vertices,faces,view,(100,80))
        more=projected_outer_boundary(vertices,np.vstack((faces,[4,5,6])),view,(100,80))
        self.assertEqual(outer,more);self.assertEqual(len(outer),1)
        np.testing.assert_array_equal(vertices,before)

    def test_html_embeds_images_and_escapes_metadata(self):
        manifest,_=self.fixture();manifest['source_binding']['note']='</script><script>bad()</script>'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'photo.png';Image.new('RGB',(100,80),'white').save(image)
            write_editor(root/'editor.html',manifest,{slot:image for slot in ('front','left','right')})
            html=(root/'editor.html').read_text()
            self.assertIn('data:image/png;base64,',html);self.assertNotIn('__CONTOUR_DATA__',html)
            self.assertNotIn('</script><script>bad()',html);self.assertIn("connect-src 'none'",html)

    def test_top_level_dispatch_both_commands(self):
        from memento3d_head.bootstrap import main
        commands=[['prepare-contours','--registration-run','run','--head-run','head','--shape-run','shape','--output','out'],
                  ['validate-contours','--workspace','workspace','--annotations','data.json','--output','out']]
        for argv in commands:
            with patch('memento3d_head.contour_editor_commands.execute',return_value=0) as execute:
                self.assertEqual(main(argv),0)
            execute.assert_called_once()

    def test_prepare_and_validate_real_workspace_without_overwriting(self):
        from types import SimpleNamespace
        from memento3d_head.contour_editor_commands import execute
        from memento3d_head.adapters.head_store import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);head=root/'head';shape=root/'shape';run=root/'run'
            for path in (head,shape,run):path.mkdir()
            obj='v -10 -10 100\nv 10 -10 100\nv 0 10 100\nv 0 0 120\nf 1 2 3\nf 1 4 2\nf 2 4 3\nf 3 4 1\n'
            mesh=head/'head.obj';mesh.write_text(obj);candidate=shape/'head.candidate.obj';candidate.write_text(obj)
            h={'artifacts':{'head.obj':{'sha256':sha256(mesh)}}};(head/'report.json').write_text(json.dumps(h))
            view={'camera_matrix':[[100.,0,50],[0,100.,40],[0,0,1]],'rotation_vector':[0.,0,0],'translation_mm':[0.,0,0]}
            r={'status':'REGISTRATIONS_EXPORTED','pipeline_type':'three_view_camera_registration',
               'input':{'head_report':sha256(head/'report.json'),'mesh':sha256(mesh)},'views':{},'artifacts':{}}
            for slot in ('front','left','right'):
                photo=run/f'{slot}.normalized.png';Image.new('RGB',(100,80),'white').save(photo)
                r['artifacts'][photo.name]={'sha256':sha256(photo)};r['views'][slot]=view
            (run/'report.json').write_text(json.dumps(r))
            s={'status':'SHAPE_CANDIDATE_EXPORTED','pipeline_type':'bounded_multiview_identity_fit',
               'input':{'registration_report_sha256':sha256(run/'report.json'),'head_obj_sha256':sha256(mesh)},
               'artifacts':{'head.candidate.obj':{'sha256':sha256(candidate)}}}
            (shape/'report.json').write_text(json.dumps(s))
            out=root/'editor';args=SimpleNamespace(command='prepare-contours',registration_run=run,head_run=head,shape_run=shape,output=out)
            original_hash=sha256(mesh)
            self.assertEqual(execute(args),0)
            saved_hash=sha256(out/'manifest.json');self.assertTrue((out/'contour_editor.html').is_file())
            self.assertEqual(execute(args),1);self.assertEqual(saved_hash,sha256(out/'manifest.json'))
            manifest=json.loads((out/'manifest.json').read_text());_,data=self.fixture()
            data['workspace_id']=manifest['workspace_id'];data['source_binding']=manifest['source_binding']
            for slot in ('front','left','right'):
                data['views'][slot]['image_sha256']=manifest['views'][slot]['image_sha256']
            annotations=root/'curves.json';annotations.write_text(json.dumps(data))
            valid=root/'validated';args=SimpleNamespace(command='validate-contours',workspace=out,annotations=annotations,output=valid)
            self.assertEqual(execute(args),0)
            result=json.loads((valid/'report.json').read_text())
            self.assertEqual(result['status'],'ANNOTATIONS_STRUCTURALLY_VALID')
            self.assertEqual(sha256(mesh),original_hash)

            # Exercise comparison against a real prepared/validated synthetic workspace.
            data['views']['front']['curves'][0]['kind']='jaw_left_image'
            second=copy.deepcopy(data['views']['front']['curves'][0])
            second['id']='front-2';second['kind']='jaw_right_image'
            second['points_px']=[[60,30],[60,35],[60,40]]
            data['views']['front']['curves'].append(second)
            annotations.write_text(json.dumps(data));args.output=root/'validated2'
            self.assertEqual(execute(args),0)
            from memento3d_head.manual_comparison_commands import execute as compare
            comparison=SimpleNamespace(workspace=out,validation_run=args.output,registration_run=run,output=root/'comparison')
            self.assertEqual(compare(comparison),0)
            comparison_report=json.loads((comparison.output/'report.json').read_text())
            self.assertFalse(comparison_report['geometry_modified'])
            self.assertFalse(comparison_report['ready_for_automatic_fitting'])
            self.assertEqual(sha256(mesh),original_hash)
            self.assertTrue((comparison.output/'left.manual_comparison.png').is_file())
            self.assertEqual(compare(comparison),1)
            # New jaw comparison must retain missing support, without distances.
            comparison.command='compare-jaw-contours';comparison.output=root/'jaw-comparison'
            comparison.mica_config=root/'mica.json';comparison.mica_config.write_text('{}')
            def prepare_fake(args,manifest,registration,output,sources):
                for name in ('face_regions.npz','face_mask_worker.log','surface_chains.json'):
                    (output/name).write_text('{}')
                sources[args.mica_config]=sha256(args.mica_config)
                return {slot:{'baseline':[],'candidate':[]} for slot in ('front','left','right')},{}
            with patch('memento3d_head.adapters.jaw_sources.prepare_jaw_outlines',side_effect=prepare_fake):
                self.assertEqual(compare(comparison),0)
            checked=json.loads((comparison.output/'report.json').read_text())
            self.assertFalse(checked['all_curves_supported'])
            self.assertEqual(checked['supported_curve_counts']['front']['baseline'],0)
            self.assertIsNone(checked['views']['front']['curves'][0]['candidate_minus_baseline_px'])
            self.assertIn('surface_chains.json',checked['artifacts'])
            # Tampering must fail even when the annotation JSON itself is well formed.
            snapshot=args.output/'visible_contours.snapshot.json'
            snapshot.write_text(snapshot.read_text()+' ')
            comparison.output=root/'tampered'
            self.assertEqual(compare(comparison),1)
            self.assertFalse(comparison.output.exists())
