import base64,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import control
from prepare_subject import hair_canvas,prepare,sha

class BenchmarkTests(unittest.TestCase):
    def test_hardware_plan_is_honest(self):
        hw={'gpus':[]}
        self.assertEqual(control.plan(hw,['triposr'])[0]['status'],'BLOCKED_GPU_NOT_VISIBLE')
        hw['gpus']=[{'total_mib':6144}]
        self.assertEqual(control.plan(hw,['triposr'])[0]['status'],'READY_TO_ATTEMPT')
        self.assertIn('DEFERRED',control.plan(hw,['difflocks'])[0]['status'])
        hw['gpus']=[{'total_mib':4096}]
        self.assertEqual(control.plan(hw,['hunyuan-mini'])[0]['status'],'BLOCKED_LOW_VRAM')

    def test_cross_os_paths_are_separate(self):
        m={'face_reference':r'D:\project with spaces\face.glb','face_sha256':'abc','hair_rgb':'/tmp/input.png'}
        with patch.object(control.subprocess,'check_output',return_value='/mnt/d/project with spaces/face.glb\n') as cmd:
            n=control.normalize_subject(m)
        self.assertEqual(n['face_reference'],'/mnt/d/project with spaces/face.glb')
        self.assertEqual(m['face_reference'],r'D:\project with spaces\face.glb')
        self.assertEqual(n['face_sha256'],'abc');self.assertEqual(n['hair_rgb'],'/tmp/input.png')
        self.assertEqual(cmd.call_args.args[0],['wslpath','-a','-u',m['face_reference']])

    def test_mask_crop_and_rejection(self):
        labels=np.zeros((100,160),dtype='uint8');labels[20:60,40:120]=17
        rgba,rgb,box=hair_canvas(Image.new('RGB',(160,100),(20,30,40)),labels)
        self.assertEqual(box,[40,20,120,60]);self.assertEqual(rgba.size,(512,512))
        self.assertEqual(rgb.getpixel((0,0)),(128,128,128));self.assertEqual(rgba.getpixel((0,0))[3],0)
        self.assertEqual(rgb.getpixel((256,256)),(20,30,40))
        with self.assertRaises(ValueError):hair_canvas(Image.new('RGB',(160,100)),np.zeros_like(labels))

    def test_original_face_bytes_and_checksum_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);run=d/'old';shared=run/'shared';shared.mkdir(parents=True)
            b=io.BytesIO();Image.new('RGB',(50,50),(35,25,20)).save(b,format='PNG')
            workspace=d/'head.workspace.html';workspace.write_text('<script id="data" type="application/json">'+json.dumps({'images':['data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()]})+'</script>')
            labels=np.zeros((50,50),dtype='uint8');labels[5:25,10:40]=17;Image.fromarray(labels).save(shared/'view_0.labels.png')
            source=shared/'source_reference.glb';source.write_bytes(b'original face sentinel - unchanged')
            (run/'report.json').write_text(json.dumps({'face':{'workspace_sha256':sha(workspace)}}));(run/'profile.used.json').write_text('{}')
            checks={str(p.relative_to(run)):sha(p) for p in shared.iterdir()};(run/'checksums.json').write_text(json.dumps(checks))
            m=prepare(d,workspace,run,d/'output')
            self.assertEqual(sha(Path(m['face_reference'])),sha(source));self.assertEqual(sha(Path(m['viewer_run'])/'shared/canonical_reference.glb'),sha(source))
            self.assertFalse(m['geometry_or_skin_modified']);self.assertFalse(m['ready_for_machine'])
            source.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):prepare(d,workspace,run,d/'bad-output')

    def test_report_shows_plan_on_inspect_and_failure_on_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);data={'plan':[{'model':'triposr','status':'BLOCKED_GPU_NOT_VISIBLE'}],'runs':[]}
            control.render_report(out,data);self.assertIn('BLOCKED_GPU_NOT_VISIBLE',(out/'report.html').read_text())
            data['runs']=[{'model':'triposr','status':'FAILED','error':'out of memory'}];control.render_report(out,data)
            self.assertIn('out of memory',(out/'report.html').read_text())

    def test_failed_subprocess_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'test.log'
            with self.assertRaisesRegex(RuntimeError,'failed'):control.command([sys.executable,'-c','print("diagnostic");raise SystemExit(7)'],log)
            self.assertIn('diagnostic',log.read_text())

    def test_cpu_surface_axis_adapter(self):
        # Lightweight tensor facade checks adapter ordering without installing CUDA/PyTorch.
        import types
        from run_model import install_cpu_mc
        class Tensor:
            def __init__(self,a):self.a=np.asarray(a)
            def detach(self):return self
            def float(self):return self
            def cpu(self):return self
            def numpy(self):return self.a
        fake=types.SimpleNamespace(tensor=lambda a,dtype=None:Tensor(a),float32=np.float32,int64=np.int64)
        grid=np.indices((24,28,32));center=np.array([8,13,21]);field=((grid-center[:,None,None,None])**2).sum(0)-16
        with patch.dict(sys.modules,{'torch':fake}):
            install_cpu_mc();v,f=sys.modules['torchmcubes'].marching_cubes(Tensor(field),0)
        # Upstream TripoSR then reverses the axes itself.
        restored=v.a[:,[2,1,0]]
        self.assertTrue(np.allclose((restored.min(0)+restored.max(0))/2,center,atol=.1))
        self.assertGreater(len(f.a),0)

if __name__=='__main__':unittest.main(verbosity=2)
