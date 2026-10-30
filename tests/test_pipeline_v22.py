"""Regression for canonical skin, masked source features and local geometry editing."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from figurine22_appearance import canonical_photo
from figurine22_hair import laplacian_displacement,shell_fit,extend_sideburns,reference_correspondence,sideburn_targets
ROOT=Path(__file__).resolve().parents[1]

class PipelineV22Tests(unittest.TestCase):
    def test_shadowed_and_lit_skin_share_exact_same_base(self):
        source=np.full((64,64,3),[90,60,50],np.uint8);source[:,32:]=[220,185,165]
        labels=np.ones((64,64),np.uint8);reference=np.array([195,162,154])/255
        out,alpha=canonical_photo(Image.fromarray(source),labels,reference)
        np.testing.assert_array_equal(out,np.tile([195,162,154],(64,64,1)))
        self.assertFalse(alpha.any())
    def test_ears_neck_nose_and_hidden_fill_share_base(self):
        labels=np.tile(np.array([0,1,7,8,10,14,17,18]),(30,4));source=np.random.default_rng(3).integers(0,255,(*labels.shape,3),dtype=np.uint8)
        out,a=canonical_photo(Image.fromarray(source),labels,np.array([.7,.6,.5]))
        np.testing.assert_array_equal(out,np.tile(np.rint(np.array([.7,.6,.5])*255).astype(np.uint8),(*labels.shape,1)))
    def test_feature_core_preserves_source_and_skin_next_to_it_does_not_inherit_shadow(self):
        source=np.full((40,40,3),[60,40,30],np.uint8);labels=np.ones((40,40),np.uint8)
        labels[10:20,10:30]=2;source[10:20,10:30]=[35,25,20]
        reference=np.array([195,162,154])/255;out,a=canonical_photo(Image.fromarray(source),labels,reference)
        np.testing.assert_array_equal(out[13:17,13:27],source[13:17,13:27])
        np.testing.assert_array_equal(out[9,15],[195,162,154])
        self.assertTrue((a[labels==1]==0).all())
    def test_laplacian_moves_handles_and_leaves_outside_exactly_fixed(self):
        mesh=trimesh.creation.icosphere(subdivisions=3);p=mesh.vertices.copy()
        region=p[:,1]>.2;handles=np.flatnonzero(p[:,1]>.95)
        target=np.tile([0.,.15,0.],(len(handles),1));d=laplacian_displacement(mesh,region,handles,target)
        np.testing.assert_array_equal(d[~region],0)
        np.testing.assert_allclose(d[handles],target,atol=.001)
        np.testing.assert_array_equal(mesh.vertices,p)
    def test_laplacian_zero_edit_is_identity(self):
        m=trimesh.creation.icosphere(subdivisions=2);region=m.vertices[:,1]>-.3;handles=np.flatnonzero(m.vertices[:,1]>.8)
        d=laplacian_displacement(m,region,handles,np.zeros((len(handles),3)))
        np.testing.assert_allclose(d,0,atol=1e-12)
    def test_sparse_or_invalid_anchor_region_rejected(self):
        m=trimesh.creation.icosphere(subdivisions=2);region=np.zeros(len(m.vertices),bool)
        with self.assertRaises(ValueError):laplacian_displacement(m,region,[0,1,2],np.zeros((3,3)))
    def test_sideburn_tip_is_rounded_and_narrowed_instead_of_a_flat_block(self):
        settings=json.loads((ROOT/'configs/pipeline-v22-04.json').read_text())['sideburns']
        points=np.array([[5.,4.,.6],[5.,4.,1.],[5.,4.,1.4]])
        delta=sideburn_targets(points,np.array([5.,4.,1.]),.9,1.,settings)
        moved=points+delta
        self.assertLess(moved[1,1],moved[0,1]);self.assertLess(moved[1,1],moved[2,1])
        self.assertAlmostEqual(np.ptp(moved[:,2]),np.ptp(points[:,2])*settings['tip_taper_factor'])
        np.testing.assert_array_equal(delta[:,0],0)
    def test_sideburn_nan_and_excessive_extension_rejected_before_geometry(self):
        for x in [float('nan'),-1,5]:
            with self.assertRaises(ValueError):extend_sideburns(None,None,None,x,{'max_extra_mm':1})
    def test_reference_cache_cannot_cross_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'report.json').write_text(json.dumps({'face':{'workspace_sha256':'other'}}))
            with self.assertRaisesRegex(ValueError,'workspace mismatch'):
                reference_correspondence(path,{'report':{'workspace_sha256':'current'}},None,None,None)
    def test_shell_rejects_invalid_design_controls(self):
        settings=json.loads((ROOT/'configs/pipeline-v22-04.json').read_text())['compact_fit']
        settings['body_mm']=float('nan')
        with self.assertRaises(ValueError):shell_fit(None,None,0,settings)

if __name__=='__main__':unittest.main()
