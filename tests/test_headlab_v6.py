"""Checks for evidence semantics, occlusion, repair safeguards and cache tampering."""
import sys,tempfile,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import trimesh
from scipy import sparse
from headlab_repair6 import classify_edges,sample_evidence,solve_field,accepts,region_mask,validate_settings
from headlab_cache6 import load_prepared
from headlab_raster import surface_depth_fast


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.limits={'max_view_iou_loss':.025,'max_region_excess_increase':.015}
        self.settings={'min_facing':.3,'mask_uncertainty_px':2.5,'rear_weight':2.,'silhouette_negative_weight':.45}
    def metrics(self,score=.3,iou=.8,excess=.1):
        return {'score':score,'views':[{'view':'front','iou':iou,'lower_excess_fraction':excess}]}
    def test_background_edge_is_not_contact(self):
        mask=np.zeros((20,20),bool);mask[3:10,5:15]=True
        labels=np.where(mask,17,0).astype(np.uint8)
        edges=classify_edges(mask,labels)
        self.assertFalse(edges['contact'].any());self.assertTrue(edges['silhouette'].any())
    def test_skin_contact_and_outer_silhouette_are_distinct(self):
        mask=np.zeros((20,20),bool);mask[3:10,5:15]=True
        labels=np.where(mask,17,0).astype(np.uint8);labels[10:15,5:15]=1
        edges=classify_edges(mask,labels)
        self.assertTrue(edges['contact'][9,10]);self.assertTrue(edges['silhouette'][3,10])
        self.assertFalse((edges['contact']&edges['silhouette']).any())
    def test_synthetic_rear_label_never_becomes_skin(self):
        mask=np.zeros((20,20),bool);mask[3:10,5:15]=True
        edges=classify_edges(mask,np.where(mask,17,1).astype(np.uint8),rear=True)
        self.assertFalse(edges['skin'].any());self.assertFalse(edges['contact'].any())
    def test_occluded_vertices_have_zero_weight(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        camera=np.array([[30,0,32,160],[0,-30,32,160],[0,0,1,5.]])
        mask=np.ones((64,64),bool);depth=surface_depth_fast(mesh.vertices,mesh.faces,camera,(64,64))
        shared={'cameras':[camera],'masks':[mask],'labels':[np.full((64,64),17,dtype=np.uint8)],'depths':[depth]}
        original=mesh.vertices.copy();p,w,n,d=sample_evidence(mesh,shared,self.settings)
        self.assertTrue((w[0,mesh.vertices[:,2]>.8]==0).all())
        self.assertGreater(w[0,mesh.vertices[:,2]<-.8].sum(),0)
        np.testing.assert_array_equal(original,mesh.vertices)
    def test_veto_reduces_conflicting_positive_without_forcing_symmetry(self):
        prior=np.ones(3)*.8;p=np.array([[1.,1.,1.],[0.,1.,1.]])
        w=np.ones((2,3));negative=np.array([[0.,0.,0.],[1.,0.,0.]])
        result=solve_field(prior,p,w,negative,sparse.csr_matrix((3,3)),.2,2.,np.zeros(3,bool))
        self.assertLess(result[0],.5);self.assertGreater(result[1],.8)
    def test_synthetic_closure_cannot_sprout_hair(self):
        a=solve_field(np.ones(3),np.ones((1,3)),np.ones((1,3)),np.zeros((1,3)),sparse.csr_matrix((3,3)),.2,1.,np.array([True,False,False]))
        self.assertEqual(a[0],0);self.assertGreater(a[1],.5)
    def test_better_average_cannot_hide_view_regression(self):
        ok,reason=accepts(self.metrics(.1,.7),self.metrics(),self.limits)
        self.assertFalse(ok);self.assertIn('view_regression',reason)
    def test_better_average_cannot_hide_lower_region_regression(self):
        ok,reason=accepts(self.metrics(.1,.8,.2),self.metrics(),self.limits)
        self.assertFalse(ok);self.assertIn('region_regression',reason)
    def test_equal_objective_falls_back(self):
        self.assertFalse(accepts(self.metrics(),self.metrics(),self.limits)[0])
    def test_nan_objective_is_rejected(self):
        self.assertFalse(accepts(self.metrics(float('nan')),self.metrics(),self.limits)[0])
    def test_zone_is_relative_to_photo_not_fixed_hair_length(self):
        a=np.zeros((20,20),bool);a[2:10,4:12]=True
        b=np.zeros((20,20),bool);b[6:14,4:12]=True
        self.assertEqual(np.flatnonzero(region_mask(a,'lower')[:,0])[0]+4,np.flatnonzero(region_mask(b,'lower')[:,0])[0])
    def test_tampered_cache_is_rejected_before_npz_load(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'bad').write_text('bad')
            (folder/'cache.json').write_text(json.dumps({'schema_version':1,'reference_head_neck_mm':20,'mask_dilation_px':1,'files':{'bad':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'checksum'):load_prepared(folder,{'reference_head_neck_mm':20,'mask_dilation_px':1})
    def test_hair_at_projected_scalp_rim_is_weaker_root_evidence(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        camera=np.array([[30,0,32,160],[0,-30,32,160],[0,0,1,5.]])
        mask=np.ones((64,64),bool);depth=surface_depth_fast(mesh.vertices,mesh.faces,camera,(64,64))
        shared={'cameras':[camera],'masks':[mask],'labels':[np.full((64,64),17,dtype=np.uint8)],'depths':[depth]}
        a=sample_evidence(mesh,shared,dict(self.settings,scalp_interior_px=0))[1]
        b=sample_evidence(mesh,shared,dict(self.settings,scalp_interior_px=100,silhouette_root_weight=.08))[1]
        np.testing.assert_allclose(b,a*.08,atol=1e-12)
    def test_repair_budget_cannot_grow_without_bound(self):
        config=dict(self.settings,**self.limits,regularization_trials=[.1]*5,veto_trials=[1]*3)
        with self.assertRaisesRegex(ValueError,'budget'):validate_settings(config)

if __name__=='__main__':unittest.main()
