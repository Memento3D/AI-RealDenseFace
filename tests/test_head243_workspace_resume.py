import sys,unittest,tempfile,json,hashlib,re
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from run_head24 import resume_artifacts

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class WorkspaceTests(unittest.TestCase):
    def test_lists_produce_front_and_side_weights(self):
        from build_head24_workspace import texture_weights
        covered=[[True,False,False],[True,True,False],[True,True,False]]
        facing=[[.8,.8,.8],[.9,.6,.9],[.9,.9,.9]]
        w,h=texture_weights(covered,facing)
        self.assertEqual(w.shape,(3,3,3))
        self.assertTrue((w[0,:,1:]==0).all());self.assertTrue((h[1,:,2]==1).all())
        self.assertTrue((h[2]==0).all())
    def test_different_triangle_counts_rejected(self):
        from build_head24_workspace import texture_weights
        with self.assertRaises(ValueError):texture_weights([[True]*3]*3,[[.8]*4]*3)
    def test_nonfinite_facing_rejected(self):
        from build_head24_workspace import texture_weights
        with self.assertRaises(ValueError):texture_weights([[True]]*3,[[.8],[np.nan],[.8]])
    def test_workspace_writer_with_real_npz_images_and_visibility(self):
        from build_head24_workspace import run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fit=root/'fit';fit.mkdir();crops=root/'crops';crops.mkdir();out=root/'workspace'
            # Fixture geometry is a visible grid, not a reconstruction or FLAME substitute.
            n=12;v=np.array([[x,y,10.] for y in np.linspace(-2,2,n) for x in np.linspace(-2,2,n)])
            faces=[]
            for y in range(n-1):
                for x in range(n-1):
                    a=y*n+x;b=a+1;c=a+n;d=c+1
                    faces.extend([[a,c,b],[b,c,d]])
            faces=np.asarray(faces)
            K=np.array([[400.,0,512],[0,400,512],[0,0,1]])
            P=np.column_stack((K,np.zeros(3)))
            np.savez(fit/'projection_bundle.npz',vertices=v,view_vertices=np.array([v]*3),faces=faces,cameras=np.array([P]*3),camera_points=np.array([v]*3),intrinsics=np.array([K]*3))
            crop_report={'views':{}}
            for slot in ('front','left','right'):
                Image.new('RGB',(1024,1024),(180,140,120)).save(crops/(slot+'.png'))
                crop_report['views'][slot]={'crop_sha256':sha(crops/(slot+'.png'))}
            (crops/'report.json').write_text(json.dumps(crop_report))
            (fit/'report.json').write_text(json.dumps({'status':'FIT_GATES_PASSED','projection_bundle_sha256':sha(fit/'projection_bundle.npz'),'crop_report_sha256':sha(crops/'report.json')}))
            run(fit,crops,out)
            text=(out/'head.workspace.html').read_text();payload=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',text,re.S)[1])
            self.assertEqual(np.asarray(payload['weights']).shape,(len(faces)*3,3))
            self.assertEqual(np.asarray(payload['camera_contract']['cameras']).shape,(3,3,4))
            self.assertTrue(np.array_equal(payload['vertices'],v[faces].reshape(-1,3)))
            self.assertEqual(json.loads((out/'report.json').read_text())['status'],'TEXTURE_REVIEW_READY')

class ResumeTests(unittest.TestCase):
    def make_run(self,root):
        crops=root/'01_crops';fit=root/'02_fit';crops.mkdir();fit.mkdir();views={}
        for slot in ('front','left','right'):
            p=crops/(slot+'.png');p.write_bytes(slot.encode());views[slot]={'crop_sha256':sha(p)}
        (crops/'report.json').write_text(json.dumps({'status':'CROPS_PREPARED','views':views}))
        (fit/'projection_bundle.npz').write_bytes(b'fixture: hash gate only; real NPZ tested separately')
        (fit/'report.json').write_text(json.dumps({'status':'FIT_GATES_PASSED','crop_report_sha256':sha(crops/'report.json'),'projection_bundle_sha256':sha(fit/'projection_bundle.npz')}))
        (root/'pipeline.report.json').write_text(json.dumps({'input':'input/05','status':'FAILED','stages':[{'stage':'01_prepare','exit_code':0},{'stage':'02_fit','exit_code':0},{'stage':'03_workspace','exit_code':1}]}))
    def test_reuse_completed_fit_without_changing_source(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_run(root);before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            crops,fit,info=resume_artifacts(root)
            self.assertEqual(fit,root/'02_fit');self.assertEqual(info['reused_verified_artifacts'],['01_crops','02_fit'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
    def test_changed_crop_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_run(root);(root/'01_crops/right.png').write_bytes(b'other person')
            with self.assertRaisesRegex(ValueError,'crop image checksum'):resume_artifacts(root)
    def test_incomplete_fit_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_run(root);(root/'02_fit/report.json').write_text(json.dumps({'status':'FAILED'}))
            with self.assertRaisesRegex(ValueError,'gates'):resume_artifacts(root)

class ResumeStageTests(unittest.TestCase):
    make_run=ResumeTests.make_run
    def test_mocked_resume_invokes_only_workspace_and_figurine(self):
        # Explicit subprocess doubles verify scheduling only. Workspace I/O is tested above.
        from unittest.mock import patch
        import run_head24
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'previous';source.mkdir();self.make_run(source);calls=[]
            class Process:
                stdout=[]
                def wait(self):return 0
            def worker(command,**kwargs):
                command=list(map(str,command));calls.append(Path(command[1]).name)
                if calls[-1]=='build_head24_workspace.py':
                    dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True)
                    (dest/'head.workspace.html').write_text('explicit mocked workspace fixture')
                elif calls[-1]=='run_pipeline_v24.py':
                    dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True)
                    (dest/'report.json').write_text(json.dumps({'status':'DEMO_EXPORTED'}))
                    (dest/'review.html').write_text('explicit mocked review fixture')
                else:raise AssertionError('Resume scheduled detector or GPU fitting: '+str(command))
                return Process()
            with patch.object(run_head24,'ROOT',root),patch.object(run_head24,'check_manifest'),patch.object(run_head24.subprocess,'Popen',side_effect=worker),patch.object(sys,'argv',['run_head24.py','--resume',str(source)]):
                self.assertEqual(run_head24.main(),0)
            self.assertEqual(calls,['build_head24_workspace.py','run_pipeline_v24.py'])
            self.assertTrue((source/'02_fit/projection_bundle.npz').is_file())

if __name__=='__main__':unittest.main()
