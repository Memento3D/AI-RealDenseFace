import unittest
import numpy as np
from tools.customization.lip_natural_trial import bounded_lip_candidate


def fixture():
    xs=np.linspace(-.05,.05,11);vertices=[];faces=[];lookup={}
    for ys in [np.array([.0005,.01,.02,.03,.04,.05]),np.array([-.05,-.04,-.03,-.02,-.01,-.0005])]:
        offset=len(vertices)
        for y in ys:
            for x in xs:
                lookup[(round(float(x),4),round(float(y),4))]=len(vertices);vertices.append([x,y,.1])
        for row in range(len(ys)-1):
            for col in range(len(xs)-1):
                a=offset+row*len(xs)+col;b=a+1;c=a+len(xs);d=c+1
                faces.extend([[a,b,c],[b,d,c]])
    vertices=np.array(vertices);faces=np.array(faces)
    ids=[lookup[(x,y)] for x,y in [(-.01,.0005),(0.,.0005),(.01,.0005),(-.01,-.0005),(0.,-.0005),(.01,-.0005),(-.02,.0005),(.02,.0005)]]
    fi=np.zeros(68,int);weights=np.zeros((68,3));weights[:,0]=1
    for lm,vertex in zip([61,62,63,67,66,65,60,64],ids):
        face,corner=np.argwhere(faces==vertex)[0];fi[lm]=face;weights[lm]=0;weights[lm,corner]=1
    masks={name:np.array([],int) for name in ['eye_region','nose','neck','left_ear','right_ear']};masks['lips']=np.array(ids[:6])
    return vertices,faces,{'full_faces':fi,'full_weights':weights},masks,ids


class LipNaturalTrialTests(unittest.TestCase):
    def test_bounded_closure_keeps_other_face_regions_fixed(self):
        v,f,e,m,ids=fixture();result,alpha,report=bounded_lip_candidate(v,f,e,m)
        self.assertTrue(report['guard']['passed'])
        self.assertLess(result[ids[1],1]-result[ids[4],1],v[ids[1],1]-v[ids[4],1])
        np.testing.assert_array_equal(result[alpha==0],v[alpha==0])
        np.testing.assert_array_equal(result[:,[0,2]],v[:,[0,2]])
        self.assertEqual(report['guard']['normal_reversals'],0)

    def test_invalid_binding_is_rejected(self):
        v,f,e,m,_=fixture();e['full_weights'][62]=np.nan
        with self.assertRaises(ValueError):bounded_lip_candidate(v,f,e,m)

    def test_already_closed_binding_is_rejected(self):
        v,f,e,m,_=fixture()
        for a,b in [(61,67),(62,66),(63,65)]:e['full_faces'][b]=e['full_faces'][a];e['full_weights'][b]=e['full_weights'][a]
        with self.assertRaises(ValueError):bounded_lip_candidate(v,f,e,m)

if __name__=='__main__':unittest.main()
