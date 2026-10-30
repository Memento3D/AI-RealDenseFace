"""CPU geometry contracts for the optional print-first prototype."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
    import trimesh
    import manifold3d
    from figurine_geometry import close_boundaries,recover_projector,mesh_from_volume,smile_deform,teeth_band,solid_report
    from run_figurine_v3 import validate_config
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional figurine CPU dependencies are not installed')
class FigurineTests(unittest.TestCase):
    def test_camera_recovery_preserves_y_flip_and_pixels(self):
        rng=np.random.default_rng(4);v=rng.normal(size=(100,3))
        m=np.array([[.08,0,.03,.5],[0,.12,.02,.4],[0,0,.03,2.]])
        q=np.column_stack((v,np.ones(100)))@m.T
        p,report=recover_projector(v,q,(800,600),np.ones(100,bool))
        pred=np.column_stack((v,np.ones(100)))@p.T
        expected=np.column_stack((q[:,0]/q[:,2]*800,(1-q[:,1]/q[:,2])*600))
        np.testing.assert_allclose(pred[:,:2]/pred[:,2,None],expected,atol=1e-10)
        self.assertLess(report['p95_px'],1e-8)

    def test_bad_projector_rejected(self):
        rng=np.random.default_rng(5);v=rng.normal(size=(100,3));q=rng.uniform(.1,1,(100,3))
        with self.assertRaises(ValueError):recover_projector(v,q,(800,600),np.ones(100,bool))

    def test_boundary_cap_is_closed_and_reported(self):
        box=trimesh.creation.box()
        opened=trimesh.Trimesh(box.vertices,box.faces[2:],process=False)
        closed,caps=close_boundaries(opened)
        self.assertTrue(closed.is_watertight);self.assertTrue(caps)

    def test_voxel_output_is_single_closed_solid(self):
        volume=np.zeros((20,20,20),bool);volume[4:16,4:16,4:16]=True
        result=solid_report(mesh_from_volume(volume,np.zeros(3),.2))
        self.assertTrue(result['watertight']);self.assertEqual(result['connected_components'],1)
        self.assertFalse(result['physical_print_verified'])

    def test_smile_localized_and_explicitly_synthetic(self):
        points=np.array([[0,.15,0],[0,-.15,0],[0,10,0],[0,0,-10]],float)
        out,report=smile_deform(points,np.zeros(3),4,dict(gap_mm=.8,corner_lift_mm=.2))
        self.assertGreater(out[0,1],points[0,1]);self.assertLess(out[1,1],points[1,1])
        np.testing.assert_allclose(out[2:],points[2:],atol=1e-8)
        self.assertIn('synthetic',report['expression_origin'])

    def test_teeth_are_one_continuous_solid(self):
        mesh,color=teeth_band(np.zeros(3),5,dict(gap_mm=.8))
        self.assertTrue(mesh.is_watertight);self.assertEqual(len(mesh.split()),1)
        self.assertEqual(color.shape,(len(mesh.vertices),4))

@unittest.skipUnless(AVAILABLE,'optional figurine CPU dependencies are not installed')
class PrintCoordinateTests(unittest.TestCase):
    def test_surface_union_fills_enclosed_voids(self):
        from figurine_geometry import union_print_parts
        outer=trimesh.creation.box(extents=[10,10,10])
        inner=trimesh.creation.box(extents=[2,2,2])
        shell=trimesh.boolean.difference([outer,inner],engine='manifold')
        result=union_print_parts([shell])
        self.assertTrue(result.is_watertight)
        self.assertEqual(len(result.split()),1)
        self.assertAlmostEqual(result.volume,1000,places=4)
        self.assertTrue(result.metadata['surface_processing']['filled_enclosed_cavities'])

    def test_surface_union_rejects_detached_parts(self):
        from figurine_geometry import union_print_parts
        a=trimesh.creation.box();b=a.copy();b.apply_translation([4,0,0])
        with self.assertRaisesRegex(ValueError,'Detached'):
            union_print_parts([a,b])

    def test_stl_mm_z_up_preserves_dimensions(self):
        import tempfile
        from run_figurine_v3 import export_print_stl
        source=trimesh.creation.box(extents=[12,35,20])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'head.stl';export_print_stl(source,path,-17.5)
            result=trimesh.load(path)
        np.testing.assert_allclose(result.extents,[12,20,35],atol=1e-5)
        self.assertAlmostEqual(result.bounds[0,2],0,places=5)

    def test_profile_rejects_invalid_feature_parameters(self):
        import json
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/figurine04.json').read_text())
        validate_config(c)
        c['hair']['base_mm']=-1
        with self.assertRaises(ValueError):validate_config(c)

if __name__=='__main__':unittest.main()
