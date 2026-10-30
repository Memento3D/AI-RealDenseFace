import unittest,tempfile,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tools.triposr_probe.surface import cpu_surface
from tools.triposr_probe.common import check_repo,write,sha,CODE_REVISION

class SurfaceTests(unittest.TestCase):
 def test_ellipsoid_axes_and_positive_volume_after_upstream_axis_swap(self):
  import trimesh
  x,y,z=np.mgrid[:40,:32,:24];density=1-((x-23)/10)**2-((y-12)/7)**2-((z-9)/4)**2
  returned,faces=cpu_surface(density,0);vertices=returned[:,[2,1,0]];mesh=trimesh.Trimesh(vertices,faces,process=False)
  np.testing.assert_allclose(mesh.bounds.mean(0),[23,12,9],atol=.2);np.testing.assert_allclose(mesh.extents,[20,14,8],atol=.3);self.assertTrue(mesh.is_watertight);self.assertGreater(mesh.volume,0)
 def test_missing_isosurface_never_returns_dummy_mesh(self):
  with self.assertRaises(ValueError):cpu_surface(np.ones((8,8,8)),25)
 def test_nonfinite_density_rejected(self):
  a=np.zeros((8,8,8));a[0]=np.nan
  with self.assertRaises(ValueError):cpu_surface(a,0)

class CropTests(unittest.TestCase):
 def test_observed_face_and_margin_preserved(self):
  from tools.triposr_probe.prepare import crop_box
  box=crop_box([100,100,200,240]);json.dumps(box);x0,y0,x1,y1=box;self.assertLess(x0,100);self.assertLess(y0,40);self.assertGreater(x1,200);self.assertGreater(y1,240);self.assertEqual(x1-x0,y1-y0)
 def test_invalid_face_box_rejected(self):
  from tools.triposr_probe.prepare import crop_box
  with self.assertRaises(ValueError):crop_box([100,100,90,200])

class ProvenanceTests(unittest.TestCase):
 def test_changed_upstream_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'system.py').write_text('original');write(p/'probe.source.json',{'revision':CODE_REVISION,'files':{'system.py':sha(p/'system.py')}});check_repo(p);(p/'system.py').write_text('changed')
   with self.assertRaisesRegex(ValueError,'changed'):check_repo(p)
 def test_preview_does_not_modify_export_geometry(self):
  import trimesh
  from tools.triposr_probe.worker import preview
  mesh=trimesh.creation.icosphere(subdivisions=1);original=mesh.vertices.copy()
  with tempfile.TemporaryDirectory() as d:preview(mesh,Path(d)/'preview.png',64);self.assertTrue((Path(d)/'preview.png').is_file())
  np.testing.assert_array_equal(mesh.vertices,original)

class BootstrapRecoveryTests(unittest.TestCase):
 def fixture(self,root,name='TripoSR-probe.download'):
  p=Path(root)/'vendor'/name;p.mkdir(parents=True);(p/'system.py').write_text('source fixture');write(p/'probe.source.json',{'revision':CODE_REVISION,'files':{'system.py':sha(p/'system.py')}});return p
 def test_valid_legacy_staging_reused_without_download_or_rename(self):
  from unittest.mock import patch
  from tools.triposr_probe import bootstrap
  with tempfile.TemporaryDirectory() as d:
   stage=self.fixture(d)
   with patch.object(bootstrap,'ROOT',Path(d)),patch.object(bootstrap.urllib.request,'urlopen',side_effect=AssertionError('Must reuse bytes')),patch.object(Path,'rename',side_effect=PermissionError('WinError5')):
    self.assertEqual(bootstrap.run(),stage)
   self.assertTrue((stage/'system.py').is_file())
 def test_empty_final_directory_does_not_block_verified_staging(self):
  from tools.triposr_probe.common import source_repo
  with tempfile.TemporaryDirectory() as d:
   stage=self.fixture(d);(Path(d)/'vendor/TripoSR-probe').mkdir();self.assertEqual(source_repo(d),stage)
 def test_changed_staging_not_promoted_or_redownloaded(self):
  from unittest.mock import patch
  from tools.triposr_probe import bootstrap
  with tempfile.TemporaryDirectory() as d:
   stage=self.fixture(d);(stage/'system.py').write_text('changed')
   with patch.object(bootstrap,'ROOT',Path(d)),patch.object(bootstrap.urllib.request,'urlopen',side_effect=AssertionError('Do not replace changed source')):
    with self.assertRaisesRegex(ValueError,'changed'):bootstrap.run()
   self.assertEqual((stage/'system.py').read_text(),'changed')
 def test_unverified_tree_preserved_and_rejected(self):
  from tools.triposr_probe.common import source_repo
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'vendor/TripoSR-probe';p.mkdir(parents=True);(p/'notes.txt').write_text('keep')
   with self.assertRaisesRegex(ValueError,'unverified'):source_repo(d)
   self.assertEqual((p/'notes.txt').read_text(),'keep')
 def test_empty_manifest_is_not_an_installation(self):
  with tempfile.TemporaryDirectory() as d:
   write(Path(d)/'probe.source.json',{'revision':CODE_REVISION,'files':{}})
   with self.assertRaisesRegex(ValueError,'Empty'):check_repo(d)
 def test_fresh_install_never_renames_directory(self):
  import io,zipfile,types
  from unittest.mock import patch
  from tools.triposr_probe import bootstrap
  data=io.BytesIO()
  with zipfile.ZipFile(data,'w') as z:z.writestr('TripoSR-pin/tsr/system.py','fresh fixture')
  with tempfile.TemporaryDirectory() as d:
   with patch.object(bootstrap,'ROOT',Path(d)),patch.object(bootstrap.shutil,'disk_usage',return_value=types.SimpleNamespace(free=20*1024**3)),patch.object(bootstrap.urllib.request,'urlopen',return_value=io.BytesIO(data.getvalue())),patch.object(Path,'rename',side_effect=PermissionError('WinError5')):
    repo=bootstrap.run()
   self.assertEqual(repo.name,'TripoSR-probe-source');check_repo(repo)
 def test_valid_final_tree_is_preferred(self):
  from tools.triposr_probe.common import source_repo
  with tempfile.TemporaryDirectory() as d:
   self.fixture(d);final=self.fixture(d,'TripoSR-probe');self.assertEqual(source_repo(d),final)


class AuditTests(unittest.TestCase):
 def test_audit_preserves_source_and_verifies_report(self):
  import trimesh
  from tools.triposr_probe.audit import audit
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'raw_native.glb';trimesh.creation.icosphere(subdivisions=1).export(path)
   digest=sha(path);report=root/'report.json';write(report,{'output_sha256':{path.name:digest}})
   result=audit(path,root/'audit',report)
   self.assertTrue(result['source_bytes_unchanged']);self.assertTrue(result['provenance']['mesh_hash_verified'])
   self.assertEqual(sha(path),digest);self.assertFalse(result['ready_for_machine']);self.assertFalse(result['hair_isolated'])
   self.assertTrue((root/'audit/geometry_gray.png').is_file())
 def test_wrong_report_rejects_before_output(self):
  import trimesh
  from tools.triposr_probe.audit import audit
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'raw_native.glb';trimesh.creation.box().export(path)
   write(root/'report.json',{'output_sha256':{path.name:'wrong'}})
   with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):audit(path,root/'audit',root/'report.json')
   self.assertFalse((root/'audit').exists())
 def test_scene_node_transforms_are_applied(self):
  import trimesh
  from tools.triposr_probe.audit import load_mesh
  with tempfile.TemporaryDirectory() as d:
   scene=trimesh.Scene();t=np.eye(4);t[0,3]=8
   scene.add_geometry(trimesh.creation.box(),transform=t)
   path=Path(d)/'scene.glb';scene.export(path)
   np.testing.assert_allclose(load_mesh(path).bounds.mean(0),[8,0,0])
 def test_renderer_does_not_mutate_geometry_or_colors(self):
  import trimesh
  from tools.triposr_probe.audit import render
  mesh=trimesh.creation.icosphere(subdivisions=1);mesh.visual.vertex_colors=[220,30,20,255]
  vertices=mesh.vertices.copy();faces=mesh.faces.copy();colors=mesh.visual.vertex_colors.copy()
  with tempfile.TemporaryDirectory() as d:
   render(mesh,Path(d)/'gray.png',True,64);render(mesh,Path(d)/'color.png',False,64)
  np.testing.assert_array_equal(mesh.vertices,vertices);np.testing.assert_array_equal(mesh.faces,faces);np.testing.assert_array_equal(mesh.visual.vertex_colors,colors)
 def test_metric_reports_open_boundary_without_repair(self):
  import trimesh
  from tools.triposr_probe.audit import metrics
  mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],process=False)
  result=metrics(mesh);self.assertEqual(result['boundary_edges'],3);self.assertFalse(result['watertight']);self.assertEqual(len(mesh.faces),1)
 def test_existing_audit_is_never_overwritten(self):
  import trimesh
  from tools.triposr_probe.audit import audit
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'raw_native.glb';trimesh.creation.box().export(path);(root/'audit').mkdir()
   with self.assertRaises(FileExistsError):audit(path,root/'audit')
 def test_verified_run_rejects_changed_prepared_pixels(self):
  from tools.triposr_probe.review import verified_raw
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'raw_native.glb').write_bytes(b'mesh');(root/'model_input.png').write_bytes(b'pixels')
   write(root/'report.json',{'status':'RAW_PROPOSAL_EXPORTED','output_sha256':{n:sha(root/n) for n in ('raw_native.glb','model_input.png')}})
   self.assertEqual(verified_raw(root),root)
   (root/'model_input.png').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):verified_raw(root)
 def test_failed_review_records_failed_status(self):
  from tools.triposr_probe.review import run
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'raw_native.glb').write_bytes(b'not-a-real-glb');(root/'model_input.png').write_bytes(b'pixels')
   write(root/'report.json',{'status':'RAW_PROPOSAL_EXPORTED','output_sha256':{n:sha(root/n) for n in ('raw_native.glb','model_input.png')}})
   with self.assertRaises(Exception):run(root,output=root/'new')
   result=json.loads((root/'new/review.report.json').read_text());self.assertEqual(result['status'],'FAILED')



class WorkerComparisonTests(unittest.TestCase):
 def exercise(self, oom_resolution=None):
  import contextlib, types, trimesh
  from PIL import Image
  from unittest.mock import patch
  from tools.triposr_probe import worker, surface
  calls=[];codes=object()
  class OOM(RuntimeError):pass
  cuda=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'synthetic test device',
   get_device_properties=lambda n:types.SimpleNamespace(total_memory=1),synchronize=lambda:None,
   reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:1,empty_cache=lambda:None,OutOfMemoryError=OOM)
  torch=types.SimpleNamespace(cuda=cuda,inference_mode=contextlib.nullcontext,__version__='test',version=types.SimpleNamespace(cuda='test'))
  class Model:
   def __init__(self):self.renderer=types.SimpleNamespace(set_chunk_size=lambda n:None)
   def eval(self):return self
   def to(self,d):return self
   def __call__(self,images,device):calls.append(('forward',len(images)));return codes
   def extract_mesh(self,received,has_color,resolution,threshold):
    selftest.assertIs(received,codes);selftest.assertEqual(threshold,25.0);calls.append(('extract',resolution))
    if resolution==oom_resolution:raise OOM('simulated VRAM exhaustion')
    m=trimesh.creation.icosphere(subdivisions=0);m.visual.vertex_colors=[120,100,80,255];return [m]
  selftest=self
  def forbidden(*a,**kw):raise AssertionError('Prepared input must bypass foreground removal')
  modules={n:types.ModuleType(n) for n in ['torch','rembg','huggingface_hub','tsr','tsr.system','tsr.utils','tsr.models','tsr.models.tokenizers','tsr.models.tokenizers.image']}
  modules['torch']=torch;modules['rembg'].new_session=forbidden
  modules['tsr.system'].TSR=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Model())
  modules['tsr.utils'].remove_background=forbidden;modules['tsr.utils'].resize_foreground=forbidden
  modules['tsr.models.tokenizers.image'].hf_hub_download=forbidden
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);image=root/'pixels.png';Image.new('RGB',(32,32),(100,120,140)).save(image)
   for name in ['config.yaml','model.ckpt']:(root/name).write_bytes(b'fixture')
   modules['huggingface_hub'].hf_hub_download=lambda repo,name,**kw:str(root/name)
   args=types.SimpleNamespace(output=root/'output',image=image,prepared_image=True,resolutions=[128,192,256],resolution=128,chunk_size=2048)
   with patch.dict(sys.modules,modules),patch.object(surface,'register_backend',return_value='test CPU mesh backend'),patch.object(worker,'source_repo',return_value=root),patch.object(worker,'check_repo'):
    worker.run(args)
   result=json.loads((root/'output/report.json').read_text())
   self.assertEqual(calls.count(('forward',1)),1)
   self.assertEqual([r for kind,r in calls if kind=='extract'],[128,192,256])
   self.assertEqual(sha(image),result['source_image_sha256']);self.assertTrue(result['source_image_unchanged'])
   self.assertFalse(result['source_face_preserved']);self.assertFalse(result['ready_for_machine'])
   self.assertTrue((root/'output/resolution_128/raw_native.glb').is_file())
   self.assertTrue((root/'output/resolution_256/raw_native.glb').is_file())
   return result
 def test_shared_field_and_prepared_pixels_with_one_forward(self):
  result=self.exercise();self.assertEqual(result['status'],'RAW_PROPOSAL_EXPORTED')
  self.assertEqual([v['status'] for v in result['variants']],['EXPORTED']*3)
 def test_oom_retains_other_exports_without_success_claim(self):
  result=self.exercise(192);self.assertEqual(result['status'],'RAW_PROPOSALS_PARTIAL')
  self.assertEqual([v['status'] for v in result['variants']],['EXPORTED','CUDA_OUT_OF_MEMORY','EXPORTED'])

if __name__=='__main__':unittest.main()
