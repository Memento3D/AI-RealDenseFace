import copy
import unittest
from unittest.mock import patch
import numpy as np
from memento3d_head.adapters.jaw_fitting import fit_jaw,project
from memento3d_head.domain.contracts import ValidationError

class JawFitTests(unittest.TestCase):
    def fixture(self):
        base=np.array([[-10.,-10.,100.],[10.,-10.,100.],[0.,10.,100.],[0.,0.,120.]])
        faces=np.array([[0,1,2],[0,3,1],[1,3,2],[2,3,0]])
        basis=np.zeros((4,3,8));basis[:,0,0]=1.
        indices=np.zeros((1,68),int);weights=np.zeros((1,68,3));weights[:,:,0]=1.
        view={'camera_matrix':[[100.,0,50],[0,100.,50],[0,0,1]],'rotation_vector':[0.,0,0],
              'translation_mm':[0.,0,0],'landmarks':[{'observed_px':[40.,40.]} for _ in range(7)]}
        views={slot:copy.deepcopy(view) for slot in ('front','left','right')}
        ids=[0,2,1];target=project(base[ids],view)+[1.,0.]
        curves=[{'kind':kind,'points_px':target.tolist()} for kind in ('jaw_left_image','jaw_right_image')]
        return base,basis,faces,indices,weights,views,ids,curves

    def test_fit_reduces_jaw_while_enforcing_landmark_and_geometry_guards(self):
        args=self.fixture();original=args[0].copy();views=copy.deepcopy(args[5])
        mesh,delta,info=fit_jaw(*args)
        self.assertGreater(delta[0],0)
        for curve in info['frontal_training_curves']:self.assertLess(curve['candidate_mean_px'],curve['baseline_mean_px'])
        for view in info['landmarks'].values():
            self.assertLessEqual(view['candidate_mean_px'],view['baseline_mean_px']+.5+1e-10)
            self.assertTrue(np.all(np.array(view['candidate_per_point_px'])<=np.array(view['baseline_per_point_px'])+1.+1e-10))
        self.assertLessEqual(np.max(np.abs(delta)),.5+1e-12)
        self.assertLessEqual(info['max_vertex_displacement_mm'],3.)
        self.assertTrue(all(a>b for a,b in zip(info['objective_history'],info['objective_history'][1:])))
        np.testing.assert_array_equal(args[0],original);self.assertEqual(args[5],views)

    def test_zero_basis_returns_unchanged_geometry(self):
        args=list(self.fixture());args[1][:]=0
        mesh,delta,info=fit_jaw(*args)
        np.testing.assert_array_equal(mesh,args[0]);np.testing.assert_array_equal(delta,np.zeros(8))

    def test_unsupported_jaw_and_nonfinite_basis_fail(self):
        for mode in ('support','nan'):
            args=list(self.fixture())
            if mode=='support':args[-1][0]['points_px']=[[0,0],[0,50],[0,99]]
            else:args[1][0,0,0]=float('nan')
            with self.assertRaises(ValidationError):fit_jaw(*args)

    def test_cli_dispatch(self):
        from memento3d_head.bootstrap import main
        argv=['fit-jaw']
        for name in ('head-run','shape-run','registration-run','workspace','validation-run','jaw-comparison-run','output'):
            argv+=['--'+name,name]
        with patch('memento3d_head.jaw_fit_commands.execute',return_value=0) as execute:
            self.assertEqual(main(argv),0)
        execute.assert_called_once()

    def test_source_bound_cli_exports_and_rejects_overwrite_and_changed_basis(self):
        import tempfile,json
        from pathlib import Path
        from types import SimpleNamespace
        from PIL import Image
        from memento3d_head.adapters.head_store import sha256
        from memento3d_head.jaw_fit_commands import execute
        base,basis,faces,indices,weights,views,ids,curves=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(**{name:root/name for name in ('head_run','shape_run','registration_run','workspace','validation_run','jaw_comparison_run','output')})
            for name,path in vars(args).items():
                if name!='output':path.mkdir()
            write=lambda p,x:p.write_text(json.dumps(x))
            mesh=args.head_run/'head.obj'
            with mesh.open('w') as stream:
                for point in base:stream.write('v %s %s %s\n'%tuple(point))
                for face in faces:stream.write('f %d %d %d\n'%tuple(face+1))
            identity=args.head_run/'identity.npy';np.save(identity,np.zeros(300,np.float32))
            h={'artifacts':{'head.obj':{'sha256':sha256(mesh)},'identity.npy':{'sha256':sha256(identity)}}}
            write(args.head_run/'report.json',h)
            b=args.shape_run/'shape_basis.npz'
            np.savez(b,base=base,basis=basis,faces=faces,landmark_faces=indices,barycentric=weights)
            write(args.shape_run/'report.json',{'status':'SHAPE_CANDIDATE_EXPORTED','artifacts':{'shape_basis.npz':{'sha256':sha256(b)}}})
            photos={}
            for slot in views:
                p=args.registration_run/f'{slot}.normalized.png';Image.new('RGB',(100,100),'white').save(p);photos[slot]=sha256(p)
            r={'status':'REGISTRATIONS_EXPORTED','views':views,'input':{'mesh':sha256(mesh),'head_report':sha256(args.head_run/'report.json')}}
            write(args.registration_run/'report.json',r)
            binding={'head_obj_sha256':sha256(mesh),'head_report_sha256':sha256(args.head_run/'report.json'),
                     'shape_report_sha256':sha256(args.shape_run/'report.json'),'registration_report_sha256':sha256(args.registration_run/'report.json'),'photos':photos}
            manifest={'workspace_id':'synthetic','source_binding':binding,'views':{slot:{'image_size':[100,100],'image_sha256':photos[slot]} for slot in views}}
            write(args.workspace/'manifest.json',manifest)
            write(args.workspace/'report.json',{'status':'CONTOUR_EDITOR_EXPORTED','artifacts':{'manifest.json':{'sha256':sha256(args.workspace/'manifest.json')}}})
            data={'schema_version':1,'pipeline_type':'manual_visible_contours','workspace_id':'synthetic','source_binding':binding,'visible_only_reviewed':True,'views':{}}
            for slot in views:
                values=copy.deepcopy(curves if slot=='front' else [curves[0]])
                for i,curve in enumerate(values):
                    curve['id']=slot+str(i)
                    if slot!='front':curve['kind']='visible_profile'
                data['views'][slot]={'image_size':[100,100],'image_sha256':photos[slot],'curves':values}
            write(args.validation_run/'visible_contours.snapshot.json',data)
            write(args.validation_run/'report.json',{'status':'ANNOTATIONS_STRUCTURALLY_VALID','source_manifest_sha256':sha256(args.workspace/'manifest.json'),'artifacts':{'visible_contours.snapshot.json':{'sha256':sha256(args.validation_run/'visible_contours.snapshot.json')}}})
            c={'status':'MANUAL_CONTOUR_COMPARISON_EXPORTED','pipeline_version':4,'boundary_source':'lower_semantic_face_patch_boundary_proxy',
               'input':{'manifest_sha256':sha256(args.workspace/'manifest.json'),'validation_report_sha256':sha256(args.validation_run/'report.json'),
                        'annotations_snapshot_sha256':sha256(args.validation_run/'visible_contours.snapshot.json'),'registration_report_sha256':sha256(args.registration_run/'report.json')},
               'views':{'front':{'curves':[{'baseline':{'status':'ANATOMICAL_PROXY_SUPPORTED'}}]*2}},
               'surface_extraction':{'front':{'baseline':{'boundary_vertices':ids}}}}
            write(args.jaw_comparison_run/'report.json',c)
            self.assertEqual(execute(args),0)
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['status'],'JAW_EXPERIMENT_EXPORTED')
            self.assertFalse(report['automatic_acceptance']);self.assertFalse(report['cameras_modified'])
            saved=sha256(args.output/'report.json');self.assertEqual(execute(args),1)
            self.assertEqual(saved,sha256(args.output/'report.json'))
            with b.open('ab') as stream:stream.write(b'changed')
            args.output=root/'tampered-output'
            self.assertEqual(execute(args),1);self.assertFalse(args.output.exists())
