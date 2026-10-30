"""Folder discovery, asset lifecycle, immutable head buffers and preview HTTP API."""
import json,sys,tempfile,time,threading,unittest,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hair_studio import Studio,handler,confined,transform_mesh,validate_transform,latest_run,sha
from http.server import ThreadingHTTPServer
from run_pipeline_v2 import glb
from run_figurine_v3 import export_print_stl
from add_graduation_cap import load
from figurine_preview import accessor

class HairStudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.project=self.root/'project';self.assets=self.project/'assets/hair';self.assets.mkdir(parents=True)
        self.run=self.project/'runs/pipeline_v22_fixture';(self.run/'shared').mkdir(parents=True);(self.run/'selected').mkdir()
        head=trimesh.creation.icosphere(subdivisions=2,radius=10);eyes=[]
        for x in [-3,3]:
            e=trimesh.creation.icosphere(subdivisions=1,radius=1);e.apply_translation([x,3,7]);eyes.append(e)
        glb(self.run/'shared/canonical_reference.glb',[('Head',head,[.7,.6,.5,1]),('EyeL',eyes[0],[1,1,1,1]),('EyeR',eyes[1],[1,1,1,1])])
        (self.run/'report.json').write_text(json.dumps({'version':'Memento3D V2.2','selected':{'candidate':'fixture'}}))
        (self.run/'profile.used.json').write_text(json.dumps({'fit':{'anchor_overlap_mm':.22,'roof_overlap_mm':.2,'cage_bandwidth_mm':1.3}}))
        self.studio=Studio(self.project,self.run,stable_seconds=0)
    def tearDown(self):self.studio.close();self.temp.cleanup()
    def add(self,name='new style.stl'):
        p=self.assets/name;trimesh.creation.box([2,1,2]).export(p);return p
    def await_job(self,job):
        end=time.monotonic()+8
        while time.monotonic()<end:
            current=self.studio.status(job['id'])
            if current['state'] in ['ready','error']:return current
            time.sleep(.01)
        self.fail('Preview job timed out')
    def test_added_asset_appears_without_registry_and_deleted_asset_disappears(self):
        self.assertEqual(self.studio.inventory(),[]);p=self.add();first=self.studio.inventory();self.assertEqual(len(first),1)
        self.assertEqual(first[0]['mode'],'approximate');p.unlink();self.assertEqual(self.studio.inventory(),[])
        with self.assertRaises(ValueError):self.studio.request(first[0]['id'],first[0]['revision'])
    def test_file_content_change_invalidates_cached_revision(self):
        p=self.add();a=self.studio.inventory()[0];trimesh.creation.box([3,1,2]).export(p);b=self.studio.inventory()[0]
        self.assertEqual(a['id'],b['id']);self.assertNotEqual(a['revision'],b['revision'])
        with self.assertRaisesRegex(ValueError,'changed'):self.studio.request(a['id'],a['revision'])
    def test_invalid_sidecar_isolated_and_old_entry_removed(self):
        p=self.add();good=self.studio.inventory()[0];p.with_suffix('.hair.json').write_text('{invalid')
        self.add('other.STL');rows=self.studio.inventory();self.assertEqual(sorted(x['state'] for x in rows),['available','error'])
        with self.assertRaises(ValueError):self.studio.request(good['id'],good['revision'])
    def test_partial_copy_not_selectable_until_stable(self):
        self.studio.stable=.025;self.add();row=self.studio.inventory()[0];self.assertEqual(row['state'],'copying')
        with self.assertRaises(ValueError):self.studio.request(row['id'],row['revision'])
        time.sleep(.03);self.assertEqual(self.studio.inventory()[0]['state'],'available')
    def test_asset_export_keeps_original_head_positions_exact(self):
        self.add();a=self.studio.inventory()[0];before=sha(self.studio.face_path);job=self.await_job(self.studio.request(a['id'],a['revision']))
        self.assertEqual(job['state'],'ready',job);self.assertFalse(job['ready_for_machine'])
        raw=self.studio.export({'job':job['id'],'transform':{'scale':1.1,'height_mm':.5,'yaw_deg':20}})
        p=self.root/'export.glb';p.write_bytes(raw);d,b=load(p);old,bb=load(self.studio.face_path)
        np.testing.assert_array_equal(accessor(d,b,0),accessor(old,bb,0));self.assertEqual(before,sha(self.studio.face_path))
        self.assertTrue(d['asset']['extras']['preview_only']);self.assertFalse(d['asset']['extras']['ready_for_machine'])
    def test_current_processed_hair_stl_axes_restore_exactly(self):
        path=self.add();mesh=trimesh.creation.box([6,2,5]);mesh.apply_translation([0,7,-1])
        export_print_stl(mesh,self.run/'selected/hair_mm.stl',self.studio.base_y)
        self.studio.current={'fit':{'asset_sha256':sha(path)}}
        a=self.studio.inventory()[0];self.assertEqual(a['mode'],'current');job=self.await_job(self.studio.request(a['id'],a['revision']))
        self.assertEqual(job['state'],'ready',job);m=trimesh.load(self.studio.cache/(job['id']+'.ply'),force='mesh')
        np.testing.assert_allclose(m.bounds,mesh.bounds,atol=2e-6)
    def test_symlink_and_traversal_cannot_leave_asset_folder(self):
        with self.assertRaises(ValueError):confined(self.assets,'../secret.stl')
        outside=self.root/'outside.stl';trimesh.creation.box().export(outside)
        try:(self.assets/'linked.stl').symlink_to(outside)
        except OSError:self.skipTest('Symlink creation unavailable')
        self.assertEqual(self.studio.inventory()[0]['state'],'error')
    def test_preview_transform_bounds_and_nonfinite_rejected(self):
        for value in [{'scale':0},{'yaw_deg':181},{'height_mm':6},{'scale':float('nan')}]:
            with self.assertRaises(ValueError):validate_transform(value)
    def test_transform_pivot_matches_client_scale_rotate_translate(self):
        mesh=trimesh.creation.box();pivot=np.array([1.,2.,3.]);old=mesh.vertices.copy()
        moved=transform_mesh(mesh,{'scale':1.2,'height_mm':.4,'yaw_deg':90},pivot)
        expected=(old-pivot)*1.2;expected=expected[:,[2,1,0]]*np.array([1,1,-1])+pivot+[0,.4,0]
        np.testing.assert_allclose(moved.vertices,expected,atol=1e-12);np.testing.assert_array_equal(mesh.vertices,old)
    def test_corrupt_asset_fails_job_without_changing_head(self):
        p=self.assets/'broken.stl';p.write_bytes(b'not a mesh');a=self.studio.inventory()[0];before=sha(self.studio.face_path)
        job=self.await_job(self.studio.request(a['id'],a['revision']));self.assertEqual(job['state'],'error');self.assertEqual(sha(self.studio.face_path),before)
    def test_http_inventory_detects_new_file_and_rejects_cross_origin_export(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.studio));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            def fetch(path):return json.load(urllib.request.urlopen(base+path))
            self.assertEqual(fetch('/api/assets')['assets'],[]);self.add();self.assertEqual(len(fetch('/api/assets')['assets']),1)
            req=urllib.request.Request(base+'/api/export',data=b'{}',headers={'Origin':'https://example.com','Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,403)
            self.assertEqual(fetch('/api/info')['head_sha256'],self.studio.face_hash)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
