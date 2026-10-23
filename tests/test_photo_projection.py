import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from memento3d_head.adapters.photo_projection import depth_buffer,face_hull_mask,project_photo,write_photo_viewer,export_photo_obj
from memento3d_head.domain.contracts import ValidationError

class PhotoProjectionTests(unittest.TestCase):
    def fixture(self):
        vertices=np.array([[-10.,-10.,100.],[10.,-10.,100.],[0.,10.,100.],[-10.,-10.,110.],[10.,-10.,110.],[0.,10.,110.]])
        faces=np.array([[0,2,1],[3,5,4]])
        view={'camera_matrix':[[100.,0,50],[0,100.,50],[0,0,1]],'rotation_vector':[0.,0,0],'translation_mm':[0.,0,0]}
        landmarks=[[.1,.1,0],[.5,.1,0],[.9,.1,0],[.9,.5,0],[.9,.9,0],[.5,.9,0],[.1,.9,0],[.1,.5,0],[.4,.4,0],[.6,.6,0]]
        return vertices,faces,view,landmarks

    def test_full_mesh_depth_rejects_back_triangle(self):
        v,f,view,landmarks=self.fixture();before=v.copy()
        coordinates,covered,info=project_photo(v,f,view,(100,100),landmarks)
        np.testing.assert_array_equal(covered,[True,False]);np.testing.assert_array_equal(before,v)
        self.assertEqual(info['covered_triangles'],1);self.assertFalse(info['lighting_removed'])
        np.testing.assert_allclose(coordinates[0],[40.,60.,100.])

    def test_perspective_depth_matches_ray_plane_intersection(self):
        v=np.array([[-20.,-20.,100.],[20.,-20.,100.],[0.,20.,200.]])
        k=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
        _,depth=depth_buffer(v,np.array([[0,1,2]]),k,(100,100))
        expected=150/(1-2.5*.5/100)
        self.assertAlmostEqual(depth[50,50],expected,places=8)

    def test_mask_backfacing_and_behind_camera_rejected(self):
        v,f,view,landmarks=self.fixture()
        _,covered,_=project_photo(v,f[:,::-1],view,(100,100),landmarks)
        self.assertFalse(covered.any())
        small=[[x*.01,y*.01,z] for x,y,z in landmarks]
        _,covered,_=project_photo(v,f,view,(100,100),small);self.assertFalse(covered.any())
        v[:,2]=-1
        with self.assertRaises(ValidationError):project_photo(v,f,view,(100,100),landmarks)

    def test_mask_erosion_and_invalid_landmarks(self):
        _,_,_,landmarks=self.fixture();mask=face_hull_mask(landmarks,(100,100),4)
        self.assertTrue(mask[50,50]);self.assertFalse(mask[11,50])
        with self.assertRaises(ValidationError):face_hull_mask([[.5,.5,0]]*10,(100,100))

    def test_html_and_obj_preserve_geometry_and_separate_unknown_material(self):
        v,f,view,landmarks=self.fixture();coords,covered,_=project_photo(v,f,view,(100,100),landmarks)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);photo=root/'front.photo.png';Image.new('RGB',(100,100),'red').save(photo)
            write_photo_viewer(root/'preview.html',v,f,coords,covered,photo,{'source':'test'})
            html=(root/'preview.html').read_text();self.assertNotIn('__HEAD_DATA__',html);self.assertIn('data:image/png;base64,',html)
            payload=json.loads(html.split('<script id="model-data" type="application/json">')[1].split('</script>')[0])
            np.testing.assert_array_equal(payload['baseline']['vertices'],v)
            np.testing.assert_array_equal(payload['candidate']['vertices'],v)
            self.assertEqual(len(payload['photo_faces']),1)
            export_photo_obj(root,v,f,coords,covered)
            obj=(root/'head.photo.obj').read_text();self.assertIn('usemtl unknown',obj);self.assertIn('usemtl photo',obj)
            self.assertEqual(sum(line.startswith('v ') for line in obj.splitlines()),len(v))

    def test_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.photo_projection_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['preview-photo-color','--candidate-run','c','--registration-run','r','--output','o']),0)
        execute.assert_called_once()

    def test_source_bound_cli_export_and_tamper_rejection(self):
        from types import SimpleNamespace
        from memento3d_head.photo_projection_commands import execute
        from memento3d_head.adapters.head_store import sha256
        from memento3d_head.adapters.smooth_preview import read_geometry
        _,_,view,landmarks=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);candidate=root/'candidate';registration=root/'registration';candidate.mkdir();registration.mkdir()
            mesh=candidate/'head.candidate.obj'
            mesh.write_text('v -10 -10 100\nv 10 -10 100\nv 0 10 100\nv 0 0 120\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
            photo=registration/'front.normalized.png';Image.new('RGB',(100,100),'red').save(photo)
            r={'pipeline_type':'three_view_camera_registration','status':'REGISTRATIONS_EXPORTED',
               'artifacts':{'front.normalized.png':{'sha256':sha256(photo)}},'views':{'front':view},
               'assessment':{'photos':[{'slot':'front','oriented_width':100,'oriented_height':100,'face':{'landmarks':landmarks}}]}}
            (registration/'report.json').write_text(json.dumps(r))
            c={'pipeline_type':'bounded_frontal_jaw_proxy_fit','status':'JAW_EXPERIMENT_EXPORTED',
               'input_sha256':{'registration':sha256(registration/'report.json'),'photo_front':sha256(photo)},
               'artifacts':{'head.candidate.obj':{'sha256':sha256(mesh)}}}
            (candidate/'report.json').write_text(json.dumps(c))
            args=SimpleNamespace(candidate_run=candidate,registration_run=registration,output=root/'out')
            original=sha256(mesh);self.assertEqual(execute(args),0)
            report=json.loads((args.output/'report.json').read_text());self.assertFalse(report['geometry_modified'])
            self.assertFalse(report['albedo_recovered']);self.assertEqual(report['projection']['covered_triangles'],1)
            np.testing.assert_array_equal(read_geometry(mesh)[0],read_geometry(args.output/'head.photo.obj')[0])
            saved=sha256(args.output/'report.json');self.assertEqual(execute(args),1);self.assertEqual(saved,sha256(args.output/'report.json'))
            self.assertEqual(original,sha256(mesh))
            Image.new('RGB',(100,100),'blue').save(photo);args.output=root/'tampered'
            self.assertEqual(execute(args),1);self.assertFalse(args.output.exists())

    def test_sloped_triangle_compares_depth_on_same_pixel_ray(self):
        v=np.array([[-20.,-20.,100.],[20.,-20.,100.],[0.,20.,145.]])
        f=np.array([[0,2,1]])
        _,_,view,landmarks=self.fixture()
        _,covered,info=project_photo(v,f,view,(100,100),landmarks)
        self.assertTrue(covered[0])
        self.assertEqual(info['policy']['version'],2)
