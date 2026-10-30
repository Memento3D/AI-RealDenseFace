"""Robust low-frequency source matching; not intrinsic albedo recovery."""
import numpy as np


def rgb(linear):
    return np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)


def solve(features,target):
    beta=np.zeros((features.shape[1],3))
    regularizer=np.diag([.002,.04,.04,.04])
    for _ in range(8):
        error=features@beta-target
        weight=np.minimum(1,.12/np.maximum(np.linalg.norm(error,axis=1),1e-8))
        beta=np.linalg.solve(features.T@(features*weight[:,None])+regularizer,
            features.T@(target*weight[:,None]))
    return beta


def estimate(vertices,coordinates,weights,images,gains,skin,sample,linear,components):
    count=len(vertices);lo=vertices.min(0);extent=np.maximum(np.ptp(vertices,axis=0),1e-6)
    xyz=(vertices-(lo+extent/2))/extent
    features=np.column_stack((np.ones(count),xyz))
    colors=np.array([sample(images[j],coordinates[j].reshape(-1,3)[:,:2]/coordinates[j].reshape(-1,3)[:,2,None]) for j in range(3)])
    values=linear(colors)
    valid=weights.reshape(-1,3)>1e-8
    component=np.repeat(components,3)
    # Geometry/color heuristic deliberately conservative: exclude eyes, central
    # nose/mouth column and eyebrow-height area from fitting. Not a skin parser.
    eye_y=np.mean(vertices[component>0,1])
    skin_region=(component==0)&(vertices[:,1]<eye_y-.015*extent[1])&(vertices[:,1]>eye_y-.27*extent[1])&(np.abs(xyz[:,0])>.08)&(np.abs(xyz[:,0])<.4)
    plausible=np.array([(c[:,0]>.22)&(c[:,0]<.91)&(c[:,1]>.12)&(c[:,2]>.1)&(c[:,0]>c[:,2]*1.04)&(np.max(c,axis=1)-np.min(c,axis=1)<.5) for c in colors])
    # Deduplicate expanded triangle corners before fitting/evaluation.
    _,unique=np.unique(vertices,axis=0,return_index=True)
    correction=np.tile(np.asarray(gains)[:,None,:],(1,count,1));correction[0]=1
    reports=[];biases=np.zeros((3,3))
    for j in (1,2):
        ids=unique[skin_region[unique]&valid[unique,0]&valid[unique,j]&plausible[0,unique]&plausible[j,unique]]
        item=dict(source=j,overlap_skin_vertices=int(len(ids)),accepted=False,mask_method='geometric and color heuristic, not semantic verification')
        if len(ids)>=30:
            train=ids[np.arange(len(ids))%4!=0];test=ids[np.arange(len(ids))%4==0]
            target=np.log(np.maximum(values[0,train],.008))-np.log(np.maximum(values[j,train],.008))
            if len(ids)<80:
                beta=np.zeros((4,3));beta[0]=np.median(target,axis=0)
                item['fit_type']='constant robust gain, limited overlap'
            else:
                beta=solve(features[train],target)
                item['fit_type']='linear spatial log gain'
            factors=np.clip(np.exp(features@beta),.65,1.55)
            before=np.linalg.norm(colors[0,test]-rgb(values[j,test]*gains[j]),axis=1)
            after=np.linalg.norm(colors[0,test]-rgb(values[j,test]*factors[test]),axis=1)
            # A bounded additive linear-light correction is a fallback for
            # low-overlap exposure/color casts that gains alone cannot match.
            bias=np.clip(np.median(values[0,train]-values[j,train]*gains[j],axis=0),-.045,.045)
            bias_error=np.linalg.norm(colors[0,test]-rgb(np.maximum(values[j,test]*gains[j]+bias,0)),axis=1)
            if np.median(bias_error)<np.median(after):
                after=bias_error;factors=np.tile(gains[j],(count,1));item['fit_type']='bounded additive linear-light correction';biases[j]=bias
            accepted=float(np.median(after))<float(np.median(before))
            if accepted:correction[j]=factors
            else:biases[j]=0
            item.update(accepted=accepted,held_out_vertices=len(test),median_rgb_distance_before=float(np.median(before)),median_rgb_distance_after=float(np.median(after)),gain_range=[float(factors.min()),float(factors.max())])
        reports.append(item)
    return correction,biases,dict(method='robust_source_color_candidates',reference='front',sources=reports,
        linear_bias=biases.tolist(),
        gain_limits=[.65,1.55],albedo_recovered=False,skin_color_truth_verified=False,
        evaluation='held-out overlap used for candidate selection, not independent color ground truth')
