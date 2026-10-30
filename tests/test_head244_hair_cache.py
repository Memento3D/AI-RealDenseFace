import sys,unittest,tempfile,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from head24_cache import cache_key,save_hair_cache,load_hair_cache

class HairCacheTests(unittest.TestCase):
    def fixture(self):
        mesh=trimesh.creation.icosphere(subdivisions=1,radius=2)
        face={'head':mesh,'eye_centers':np.array([[-.5,0,1],[.5,0,1]])}
        asset={'id':'21778','sha256':'a'*64};profile={'fit':{'use_cage':True},'compact_fit':{'body_mm':.5}}
        return mesh,cache_key(face,asset,profile,ROOT)
    def test_roundtrip_exact_mesh_and_policy(self):
        mesh,key=self.fixture()
        with tempfile.TemporaryDirectory() as t:
            save_hair_cache(t,'21778',key,mesh,{'a':1},mesh,{'b':2})
            a,_,b,_=load_hair_cache(t,'21778',key)
            self.assertTrue(np.array_equal(a.vertices,mesh.vertices));self.assertTrue(np.array_equal(b.faces,mesh.faces))
    def test_other_head_cache_rejected(self):
        mesh,key=self.fixture()
        with tempfile.TemporaryDirectory() as t:
            save_hair_cache(t,'21778',key,mesh,{},mesh,{})
            other=copy.deepcopy(key);other['head_geometry_sha256']='other'
            with self.assertRaises(ValueError):load_hair_cache(t,'21778',other)
    def test_tampered_cache_rejected(self):
        mesh,key=self.fixture()
        with tempfile.TemporaryDirectory() as t:
            save_hair_cache(t,'21778',key,mesh,{},mesh,{})
            with (Path(t)/'hair_checkpoint_21778.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):load_hair_cache(t,'21778',key)
    def test_sideburn_handle_selection_respects_protected_region(self):
        import figurine24_hair as module
        class Hair:
            def __init__(self,p):self.vertices=p.copy()
            def copy(self):return Hair(self.vertices)
            @property
            def is_volume(self):return True  # Explicit solver-selection double, not a mesh audit.
        p=np.array([[1.,-.8+i*.005,-.3] for i in range(20)]+[[1.+i*.002,i*.002,.4+i*.006] for i in range(40)])
        anchor={'side':1,'center':[1,0,0],'low':[1,-2,0],'high':[1,1,0]}
        settings={'max_extra_mm':1,'front_of_ear_mm':.45,'tip_band_mm':1.1,'region_radius_mm':3.8,'ear_top_clearance_mm':.5,'max_total_extension_mm':2.2,'tip_taper_factor':.35,'tip_rounding_sigma_mm':.35,'max_fore_aft_adjust_mm':1}
        face={'eye_centers':np.array([[0,0,0],[1,0,0]])};head=trimesh.creation.icosphere(radius=2)
        def solver(mesh,region,handles,target):
            self.assertTrue(region[handles].all());self.assertTrue((mesh.vertices[handles,2]>.1).all())
            return np.zeros_like(mesh.vertices)
        with patch.object(module,'ear_anchors',return_value=[anchor]),patch.object(module,'laplacian_displacement',side_effect=solver):
            _,report=module.extend_sideburns(Hair(p),head,face,.4,settings)
        self.assertGreater(report['sides'][0]['handle_count'],3)

if __name__=='__main__':unittest.main()
