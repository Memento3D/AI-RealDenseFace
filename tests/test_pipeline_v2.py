"""Critical V2 invariants; no simulated human/physical-print success claims."""
import copy,io,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import trimesh
from figurine2_boolean import solid_union
from figurine2_render import srgb_to_linear,linear_to_srgb
from figurine2_face import visible_teeth,pose_head,sha
from figurine2_hair import read_library,validate_custom,normalize_asset,fit_hair
from figurine2_evaluation import mesh_audit,chord_screen,candidate_ledger,evaluation_summary,vertex_fans
from run_pipeline_v2 import validate,glb
from add_graduation_cap import load
from figurine_preview import accessor

ROOT=Path(__file__).resolve().parents[1]

class PipelineV2Tests(unittest.TestCase):
    def profile(self):return json.loads((ROOT/'configs/pipeline-v2-04.json').read_text())
    def test_overlapping_solids_survive_stl_roundtrip(self):
        a=trimesh.creation.icosphere(subdivisions=2);b=a.copy();b.apply_translation([.7,0,0])
        result=solid_union([a,b]);r=trimesh.load(io.BytesIO(result.export(file_type='stl')),file_type='stl')
        self.assertTrue(r.is_volume);self.assertEqual(len(r.split()),1)
    def test_detached_material_is_rejected(self):
        a=trimesh.creation.box();b=a.copy();b.apply_translation([3,0,0])
        with self.assertRaisesRegex(ValueError,'Detached'):solid_union([a,b])
    def test_open_input_rejected_before_union(self):
        a=trimesh.creation.box();a.update_faces(np.arange(len(a.faces)-1))
        with self.assertRaisesRegex(ValueError,'closed'):solid_union([a])
    def test_color_transfer_roundtrip_preserves_palette(self):
        colors=np.asarray([p['rgb'] for p in self.profile()['skin_palette']])/255
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(colors)),colors,atol=1e-12)
    def test_glb_colors_are_linear_not_srgb_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'test.glb';glb(file,[('skin',trimesh.creation.box(),[.5,.4,.3,1.])])
            doc,blob=load(file);p=doc['meshes'][0]['primitives'][0]
            colors=accessor(doc,blob,p['attributes']['COLOR_0'])
            np.testing.assert_allclose(colors[0,:3],srgb_to_linear([.5,.4,.3]),rtol=1e-6)
    def test_white_background_is_not_teeth(self):
        self.assertFalse(visible_teeth(np.full((20,20,3),255),np.zeros((20,20),int)))
    def test_visible_mouth_white_region_can_propose_teeth(self):
        image=np.full((20,20,3),80);label=np.zeros((20,20),int)
        label[8:13,5:15]=11;image[8:13,5:15]=220
        self.assertTrue(visible_teeth(image,label))
    def test_small_specular_patch_does_not_force_teeth(self):
        image=np.full((20,20,3),50);label=np.zeros((20,20),int);label[5:15,5:15]=11
        image[8,8:12]=250;self.assertFalse(visible_teeth(image,label))
    def test_source_expression_preserves_coordinates(self):
        source=trimesh.creation.icosphere();before=source.vertices.copy()
        head,r=pose_head({'head':source,'anchor':np.zeros(3),'width':2},'source',self.profile())
        np.testing.assert_array_equal(head.vertices,before);self.assertEqual(r['max_displacement_mm'],0)
    def test_expression_is_bounded_and_does_not_mutate_source(self):
        source=trimesh.creation.icosphere();before=source.vertices.copy();profile=self.profile()
        head,r=pose_head({'head':source,'anchor':np.zeros(3),'width':2},'soft_smile',profile)
        self.assertLessEqual(r['max_displacement_mm'],profile['expression']['max_displacement_mm'])
        np.testing.assert_array_equal(source.vertices,before)
    def test_custom_nan_and_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):validate_custom({'volume':float('nan')})
        with self.assertRaises(ValueError):validate_custom({'yaw_deg':16})
    def test_profile_rejects_excessive_eye_and_expression_changes(self):
        a=self.profile();a['eyes']['size_factor']=1.5
        with self.assertRaises(ValueError):validate(a)
        a=self.profile();a['expression']['corner_lift_mm']=2
        with self.assertRaises(ValueError):validate(a)
    def test_supplied_assets_validate_without_modifying_source(self):
        entries=read_library(ROOT/'assets/hair/library.json',ROOT)
        for a in entries:
            path=ROOT/a['path'];before=sha(path);m=normalize_asset(a,ROOT)
            self.assertTrue(m.is_volume);self.assertEqual(before,sha(path))
    def test_asset_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=trimesh.creation.box();m.export(root/'source.stl')
            a=copy.deepcopy(read_library(ROOT/'assets/hair/library.json',ROOT)[0]);a['path']='source.stl';a['sha256']='0'*64
            p=root/'registry.json';p.write_text(json.dumps({'schema_version':1,'assets':[a]}))
            with self.assertRaisesRegex(ValueError,'altered'):read_library(p,root)
    def test_asset_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=copy.deepcopy(read_library(ROOT/'assets/hair/library.json',ROOT)[0]);a['path']='../escape.stl'
            p=root/'registry.json';p.write_text(json.dumps({'schema_version':1,'assets':[a]}))
            with self.assertRaisesRegex(ValueError,'inside project'):read_library(p,root)
    def test_out_of_range_head_rejected_instead_of_stretching_asset(self):
        asset=read_library(ROOT/'assets/hair/library.json',ROOT)[0];huge=trimesh.creation.icosphere(radius=100)
        with self.assertRaisesRegex(ValueError,'scale range'):fit_hair(asset,ROOT,huge,0,self.profile()['fit'])
    def test_closed_mesh_is_not_a_thickness_certificate(self):
        mesh=trimesh.creation.box(extents=[2,2,.1]);audit=mesh_audit(mesh,'test mm')
        screen=chord_screen(mesh,32,.4)
        self.assertTrue(audit['watertight']);self.assertFalse(audit['minimum_wall_thickness_verified'])
        self.assertFalse(screen['minimum_wall_thickness_verified']);self.assertGreater(screen['flagged_samples'],0)
    def test_chord_screen_budget_is_bounded(self):
        with self.assertRaises(ValueError):chord_screen(trimesh.creation.box(),10000)
    def test_vertex_link_finds_pinched_mesh_missed_by_edge_watertightness(self):
        a=trimesh.creation.box();b=trimesh.creation.box();b.apply_translation([1,1,1])
        joined=trimesh.util.concatenate([a,b]);joined.merge_vertices()
        self.assertTrue(joined.is_watertight)
        self.assertEqual(vertex_fans(joined)['nonmanifold_vertex_count'],1)
        self.assertFalse(mesh_audit(joined,'test mm')['closed_surface_manifold'])
    def test_sphere_has_one_vertex_fan_at_every_vertex(self):
        audit=mesh_audit(trimesh.creation.icosphere(subdivisions=1),'test mm')
        self.assertTrue(audit['closed_surface_manifold']);self.assertEqual(audit['euler_characteristic'],2)
    def test_failure_ledger_never_passes_machine_readiness(self):
        ledger=candidate_ledger({'status':'FAILED','error':'disconnected hair'})
        self.assertEqual(ledger['status'],'REJECTED');self.assertFalse(ledger['ready_for_machine'])
    def test_expert_and_physical_evidence_not_invented(self):
        report={'candidates':[],'selected':None,'input_assessment':{}}
        summary=evaluation_summary(report)
        self.assertEqual(summary['human_expert_reviews_completed'],0)
        self.assertEqual(summary['independent_validation_subjects'],0)
        self.assertFalse(summary['ready_for_machine']);self.assertFalse(summary['physical_print_verified'])

if __name__=='__main__':unittest.main()
