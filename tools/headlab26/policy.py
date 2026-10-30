"""Array-only observation policies: no hidden-eye reconstruction or blanket symmetry."""
import numpy as np

def observation_confidence(coords,labels,settings):
 p=np.asarray(coords,float);label=np.asarray(labels)
 if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Invalid predicted vertex coordinates')
 x,y=np.floor(p).astype(int).T;h,w=label.shape
 valid=(x>=0)&(y>=0)&(x<w)&(y<h);classes=np.full(len(p),-1);classes[valid]=label[y[valid],x[valid]]
 confidence=np.ones(len(p));confidence[classes==-1]=settings['outside_weight'];confidence[np.isin(classes,[0,17])]=settings['nonface_weight'];confidence[classes==6]=settings['glasses_weight']
 if not np.isfinite(confidence).all() or confidence.min()<=0 or confidence.max()>1:raise ValueError('Invalid confidence policy')
 return confidence,classes

def adjusted_uncertainty(cv,dv,confidence,eye_ids,eye_boost_delta=0):
 cv=np.asarray(cv,float).copy();dv=np.asarray(dv,float).copy();c=np.asarray(confidence,float)
 if cv.shape!=dv.shape or cv.shape!=(len(c),1) or not np.isfinite(cv).all() or not np.isfinite(dv).all() or not np.isfinite(c).all() or (c<=0).any() or (c>1).any():raise ValueError('Invalid uncertainty arrays')
 # c scales the whitened residual amplitude. IRLS objective weights must
 # be square-rooted before being combined with this amplitude policy.
 penalty=-2*np.log(c)
 cv[:,0]+=penalty;dv[:,0]+=penalty
 ids=np.asarray(eye_ids,int)
 # Never re-boost a point known to be covered by glasses/hair/background.
 ids=ids[c[ids]>=.9];cv[ids,0]+=eye_boost_delta
 return cv.astype('float32'),dv.astype('float32')

def robust_confidence(projected,target,delta_px):
 p=np.asarray(projected,float);t=np.asarray(target,float)
 if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or not np.isfinite(t).all() or delta_px<=0:raise ValueError('Invalid robust residual inputs')
 residual=np.linalg.norm(p-t,axis=1);return np.minimum(1.,delta_px/np.maximum(residual,1e-8)),residual

def eye_aperture(points):
 p=np.asarray(points,float)
 if p.ndim!=2 or p.shape[1]!=2 or len(p)<4 or not np.isfinite(p).all():raise ValueError('Invalid eyelid sample')
 lo,hi=np.quantile(p,[.05,.95],axis=0);return float((hi[1]-lo[1])/max(hi[0]-lo[0],1e-8))
