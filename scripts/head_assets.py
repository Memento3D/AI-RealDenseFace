"""Original short-hair and rectangular-frame templates; not image reconstruction."""
import numpy as np
from add_graduation_cap import bridge,tube


def short_hair(head,front_hairline=None):
    lo,hi=head.min(0),head.max(0);height=hi[1]-lo[1]
    scalp=head[head[:,1]>hi[1]-.24*height]
    cx=(scalp[:,0].min()+scalp[:,0].max())/2
    cz=(scalp[:,2].min()+scalp[:,2].max())/2
    origin=np.array([cx,hi[1]-.36*height,cz])
    scale=np.array([np.ptp(scalp[:,0])*.52,hi[1]-origin[1],np.ptp(scalp[:,2])*.53])
    angles=np.arange(96)*2*np.pi/96
    front=np.maximum(np.cos(angles),0);rear=np.maximum(-np.cos(angles),0)
    line=hi[1]-.31*height+.12*height*front-.22*height*rear
    if front_hairline is not None:
        line+=(front_hairline-(hi[1]-.19*height))*front**3
    line-=.008*height*(np.sin(angles*9+.6)**2)*front**5
    end=np.arccos(np.clip((line-origin[1])/scale[1],-.95,.95))
    rings=[]
    relative=(head-origin)/scale
    rad=np.linalg.norm(relative,axis=1)
    direction=relative/np.maximum(rad[:,None],1e-8)
    candidate=(head[:,1]>hi[1]-.60*height)&(rad>.5)
    direction=direction[candidate];rad=rad[candidate]
    for k in range(32):
        theta=.025+(end-.025)*k/31
        unit=np.column_stack((np.sin(theta)*np.sin(angles),np.cos(theta),np.sin(theta)*np.cos(angles)))
        dot=unit@direction.T
        nearest=np.argpartition(dot,-5,axis=1)[:,-5:]
        score=np.take_along_axis(dot,nearest,axis=1)
        w=np.exp((score-score.max(1)[:,None])*120)
        radius=(rad[nearest]*w).sum(1)/w.sum(1)
        offset=.018+.005*np.sin(angles*19+theta*3)**2
        rings.append(origin+(unit*(radius+offset)[:,None])*scale)
    outer=np.array(rings);inner=origin+(outer-origin)*.985
    triangles=np.concatenate((bridge(outer),bridge(inner)[:,::-1],bridge([outer[0],inner[0]]),bridge([outer[-1],inner[-1]])))
    # Close the tiny crown opening with a small fan.
    for ring,reverse in ((outer[0],True),(inner[0],False)):
        center=ring.mean(0);center[1]=hi[1]+(.02 if reverse else .005)*height
        fan=np.array([[center,ring[i],ring[(i+1)%len(ring)]] for i in range(len(ring))])
        triangles=np.concatenate((triangles,fan[:,::-1] if reverse else fan))
    return [('Hair_ShortTemplate',triangles,[.012,.010,.009])]


def glasses(eyes,head):
    centers=sorted([np.array(c) for c in eyes],key=lambda c:c[0])
    width=centers[1][0]-centers[0][0]
    rx,ry=width*.40,width*.29
    front=max(c[2] for c in centers)+width*.45
    parts=[]
    for i,c in enumerate(centers):
        angle=np.linspace(0,2*np.pi,97)
        # Rounded rectangular superellipse matching the reference's broad style.
        x=np.sign(np.cos(angle))*np.abs(np.cos(angle))**.55
        y=np.sign(np.sin(angle))*np.abs(np.sin(angle))**.55
        path=np.column_stack((c[0]+rx*x,c[1]+ry*y,np.full(len(angle),front)))
        # Closed frame tube: duplicate endpoint is permitted as distinct tube rings.
        parts.append((f'Glasses_Frame_{i}',tube(path,width*.023,12),[.012,.014,.018]))
    left,right=centers
    parts.append(('Glasses_Bridge',tube([[left[0]+rx,left[1]+ry*.2,front],[np.mean([left[0],right[0]]),np.mean([left[1],right[1]])+ry*.45,front+.002],[right[0]-rx,right[1]+ry*.2,front]],width*.019),[.012,.014,.018]))
    hmin,hmax=head.min(0),head.max(0)
    for i,c in enumerate(centers):
        sign=-1 if i==0 else 1
        outer=c[0]+sign*rx
        ear_x=hmin[0]*.87 if sign<0 else hmax[0]*.87
        path=[[outer,c[1]+ry*.45,front],[ear_x,c[1]+ry*.4,front-.04],[ear_x,c[1]+ry*.25,-.045],[ear_x-sign*.003,c[1]-ry*.4,-.06]]
        parts.append((f'Glasses_Temple_{i}',tube(path,width*.014),[.12,.13,.15]))
    return parts
