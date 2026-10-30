"""Numeric behavior, automatic anchors and honest manufacturing gates."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
    import trimesh
    from headlab_algorithms import laplacian_edit,arap_edit,catmull_rom,normal_ray_screen,fit_offsets
    from headlab_printcheck import external_driver
    from run_headlab_v4 import validate,manufacture_gate,geometry
    AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional figurine dependencies unavailable')
class HeadLabTests(unittest.TestCase):
    def test_laplacian_keeps_automatic_anchors_exact(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        target=mesh.vertices.copy();target[:,1]+=.1
        fixed=np.array([0,1,2]);result,_=laplacian_edit(mesh,target,fixed)
        np.testing.assert_allclose(result[fixed],target[fixed],atol=1e-10)
        np.testing.assert_allclose(result,target,atol=1e-8)

    def test_arap_preserves_a_rigid_rotation(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        rot=trimesh.transformations.rotation_matrix(.3,[0,1,0])[:3,:3]
        target=mesh.vertices@rot.T+np.array([.2,.1,0])
        result,report=arap_edit(mesh,target,np.array([0,1,2]),iterations=5)
        np.testing.assert_allclose(result,target,atol=1e-8)
        self.assertLess(report['objective_history'][-1],1e-12)

    def test_arap_nonrigid_objective_does_not_increase(self):
        mesh=trimesh.creation.icosphere(subdivisions=1);target=mesh.vertices.copy()
        target[target[:,1]>.3,0]+=.2
        result,report=arap_edit(mesh,target,np.array([0,1,2]),iterations=8)
        energies=np.array(report['objective_history'])
        self.assertTrue(np.all(np.diff(energies)<1e-8))
        np.testing.assert_allclose(result[[0,1,2]],target[[0,1,2]],atol=1e-10)

    def test_centripetal_curve_hits_controls(self):
        controls=np.array([[0,0,0],[1,2,0],[2,1,0],[3,1,0.]])
        curve=catmull_rom(controls,8)
        np.testing.assert_allclose(curve[::8],controls,atol=1e-10)

    def test_duplicate_curve_controls_rejected(self):
        with self.assertRaisesRegex(ValueError,'Repeated'):catmull_rom([[0,0,0],[0,0,0]])

    def test_chord_screen_detects_thin_box_without_certifying_it(self):
        box=trimesh.creation.box(extents=[2,2,.2])
        report=normal_ray_screen(box,12,.4)
        self.assertAlmostEqual(report['min_sampled_chord_mm'],.2,places=5)
        self.assertGreater(report['sampled_chords_below_trial_threshold'],0)
        self.assertFalse(report['minimum_wall_thickness_verified'])

    def test_open_mesh_does_not_get_a_thickness_pass(self):
        box=trimesh.creation.box();box.update_faces(np.arange(len(box.faces)-1))
        report=normal_ray_screen(box)
        self.assertEqual(report['status'],'SKIPPED_OPEN_MESH')
        self.assertFalse(report['minimum_wall_thickness_verified'])

    def test_no_driver_never_claims_machine_ready(self):
        report=external_driver(Path('missing.stl'),Path('unused'),Path('missing.json'),None)
        self.assertFalse(report['ready_for_machine']);self.assertEqual(report['status'],'NOT_CONFIGURED')

    def test_manufacturing_gate_cannot_be_satisfied_by_topology_alone(self):
        checks={'topology_pass':True,'self_intersections_verified':False,'minimum_wall_thickness_verified':False}
        self.assertFalse(manufacture_gate(checks,{'ready_for_machine':True}))

    def test_driver_rejects_missing_checks_even_with_matching_hashes(self):
        from headlab_printcheck import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stl=root/'a.stl';stl.write_text('test');profile=root/'p.json';profile.write_text('{}')
            body={'input_sha256':file_hash(stl),'profile_sha256':file_hash(profile)}
            code='import pathlib,json;pathlib.Path(__import__("sys").argv[1],"report.json").write_text('+repr(json.dumps(body))+')'
            with self.assertRaisesRegex(ValueError,'verify all'):
                external_driver(stl,root/'output',profile,{'command':[sys.executable,'-c',code,'{output}']})

    def test_profile_rejects_missing_repeat_budget_and_bad_offset(self):
        root=Path(__file__).resolve().parents[1]
        profile=json.loads((root/'configs/headlab04.v4.json').read_text())
        with self.assertRaises(ValueError):validate(profile,0)
        profile['fit']['min_offset_mm']=2
        with self.assertRaises(ValueError):validate(profile,1)

if __name__=='__main__':unittest.main()
