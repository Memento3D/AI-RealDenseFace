import unittest
import numpy as np
from tools.customization.neck_natural_math import soften_neck_base,fit_skin_field,skin_field,crown_similarity,neck_base_mask
from tools.headapi.assembly_math import transform_points


class NaturalNeckMathTests(unittest.TestCase):
    def test_lower_boundary_is_included_without_selecting_face_boundary(self):
        vertices=np.array([[0,0,.76],[0,0,.78],[0,0,.88],[0,0,.91]])
        mask=neck_base_mask(vertices,[0],[1,2],.795)
        np.testing.assert_array_equal(mask,[True,True,False,False])
        with self.assertRaises(ValueError):neck_base_mask(vertices,[9],[1],.795)
    def test_only_lower_semantic_neck_can_move(self):
        theta=np.linspace(0,2*np.pi,24,endpoint=False)
        rows=[]
        for z,r in [(.755,.08),(.77,.07),(.79,.04),(.81,.04)]:
            rows.extend(np.c_[np.cos(theta)*r,np.sin(theta)*r,np.full(24,z)])
        v=np.array(rows);mask=np.ones(len(v),bool);mask[-24:]=False
        result,report=soften_neck_base(v,mask,.765,transition=.022)
        np.testing.assert_array_equal(result[-24:],v[-24:])
        np.testing.assert_array_equal(result[v[:,2]>=.787],v[v[:,2]>=.787])
        np.testing.assert_array_equal(result[:,2],v[:,2])
        self.assertGreater(report['max_neck_displacement_m'],0)

    def test_bad_geometry_fails_before_export(self):
        for points,mask in [(np.zeros((3,3)),np.ones(4,bool)),(np.full((40,3),np.nan),np.ones(40,bool))]:
            with self.assertRaises(ValueError):soften_neck_base(points,mask,.765)
        with self.assertRaises(ValueError):soften_neck_base(np.zeros((40,3)),np.ones(40,bool),.765)

    def test_skin_field_is_smooth_bounded_and_robust_to_outliers(self):
        rng=np.random.default_rng(11);points=rng.uniform(-.04,.04,(500,3))
        true=np.array([160,125,110])+points[:,2,None]*150
        observed=true+rng.normal(0,1,(500,3));observed[:20]=[8,8,8]
        model=fit_skin_field(points,observed)
        prediction=skin_field(points,model)
        self.assertLess(np.median(np.linalg.norm(prediction-true,axis=1)),3)
        self.assertTrue(np.isfinite(prediction).all());self.assertTrue(((prediction>=0)&(prediction<=255)).all())
        p=np.array([[0,0,0],[0,0,.00001]])
        self.assertLess(np.linalg.norm(np.diff(skin_field(p,model),axis=0)),.1)

    def test_similarity_keeps_attachment_and_facial_proportions(self):
        pivot=np.array([-.02,.06,.765]);v=np.array([[-.08,.0,.85],[.04,0,.85],[-.02,0,.95]])
        matrix=crown_similarity(v,.97,pivot);result=transform_points(v,matrix)
        np.testing.assert_allclose(transform_points(pivot[None],matrix)[0],pivot,atol=1e-12)
        self.assertAlmostEqual(result[:,2].max(),.97)
        before=np.linalg.norm(v[1]-v[0])/np.linalg.norm(v[2]-v[0])
        after=np.linalg.norm(result[1]-result[0])/np.linalg.norm(result[2]-result[0])
        self.assertAlmostEqual(before,after)
        with self.assertRaises(ValueError):crown_similarity(v,1.4,pivot)

    def test_no_color_fallback_without_skin_evidence(self):
        with self.assertRaises(ValueError):fit_skin_field(np.zeros((4,3)),np.zeros((4,3)))

if __name__=='__main__':unittest.main()
