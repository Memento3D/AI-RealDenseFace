"""Versioned asset registry and cavity-aware bounded radial-cage fitting.

The exterior is a supplied style, not recovered hair. No image mask is lifted
onto the neck. New assets require a one-time orientation/cavity metadata setup.
"""
import json,re
from pathlib import Path
import numpy as np
import trimesh
from figurine2_face import sha

def validate_custom(custom):
    for key,bound in [('volume',.20),('fringe_lift_mm',1.),('nape_lift_mm',1.),('yaw_deg',15.)]:
        value=custom.get(key,0)
        if not np.isfinite(value) or abs(value)>bound:raise ValueError('Custom control outside limit: '+key)

def read_library(path,root):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('schema_version')!=1:raise ValueError('Unsupported hair library schema')
    ids=[a['id'] for a in data['assets']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate hair asset IDs')
    for item in data['assets']:
        if not isinstance(item['id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',item['id']):raise ValueError('Unsafe hair asset ID')
        p=(Path(root)/item['path']).resolve()
        if not p.is_relative_to(Path(root).resolve()):raise ValueError('Asset path must stay inside project')
        if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Missing or altered asset '+item['id'])
        rotation=np.asarray(item['source_to_y_up'],float)
        if rotation.shape!=(3,3) or not np.allclose(rotation@rotation.T,np.eye(3)) or np.linalg.det(rotation)<.99:raise ValueError('Asset orientation must be a proper rotation')
        if not 1<item['inner_radius']<100 or item['inner_roof_y']<=item['source_base_y']:raise ValueError('Invalid cavity metadata')
        limits=item['limits']
        if not 0<limits['scale_min']<limits['scale_max']<=4 or not 0<limits['cage_displacement_mm']<=3:raise ValueError('Invalid asset deformation limits')
        validate_custom(item.get('custom',{}))
    return data['assets']

def nearest_ray(mesh,origins,directions):
    hit=np.full(len(origins),np.nan)
    points,rayids,faceids=mesh.ray.intersects_location(origins,directions,multiple_hits=False)
    distance=np.linalg.norm(points-origins[rayids],axis=1);hit[rayids]=distance
    return hit

def normalize_asset(asset,root):
    mesh=trimesh.load(Path(root)/asset['path'],force='mesh',process=True)
    if not mesh.is_volume or not mesh.is_winding_consistent or len(mesh.split())!=1:raise ValueError('Hair asset must be one closed oriented volume')
    mesh.vertices=mesh.vertices@np.asarray(asset['source_to_y_up']).T
    # Fix only numerical center offset of the LEGO-compatible source.
    mesh.vertices[:,0]-=(mesh.bounds[0,0]+mesh.bounds[1,0])/2
    return mesh

def fit_hair(asset,root,head,eye_y,settings,method='cage'):
    source=normalize_asset(asset,root);v=np.asarray(source.vertices).copy()
    top=float(head.bounds[1,1]);height=float(head.extents[1])
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*height]
    lo,hi=np.quantile(upper,[.005,.995],axis=0)
    center=np.array([(lo[0]+hi[0])/2,0.,(lo[2]+hi[2])/2])
    desired_radius=(hi[[0,2]]-lo[[0,2]])/2-settings['anchor_overlap_mm']
    bottom=eye_y-.1*height
    scale=np.array([desired_radius[0]/asset['inner_radius'],
                    (top-settings['roof_overlap_mm']-bottom)/(asset['inner_roof_y']-asset['source_base_y']),
                    desired_radius[1]/asset['inner_radius']])
    limits=asset['limits']
    if (scale<limits['scale_min']).any() or (scale>limits['scale_max']).any():raise ValueError('Head outside declared asset scale range')
    center[1]=bottom-scale[1]*asset['source_base_y']
    affine=v*scale+center;output=affine.copy();controls=[];deltas=[]
    if method not in ['affine','cage']:raise ValueError('Unknown fitting method')
    if method=='cage':
        na=36;levels=np.linspace(asset['inner_roof_y']*.20,asset['inner_roof_y']*.82,4)
        angles=np.tile(np.linspace(-np.pi,np.pi,na,endpoint=False),len(levels));ys=np.repeat(levels,na)
        directions=np.column_stack((np.sin(angles),np.zeros(len(angles)),np.cos(angles)))
        origins=np.column_stack((np.zeros(len(ys)),ys,np.zeros(len(ys))))
        radius=nearest_ray(source,origins,directions)
        valid=np.isfinite(radius)&(radius>2)&(radius<asset['inner_radius']*1.5)
        origins=origins[valid];directions=directions[valid];radius=radius[valid];angles=angles[valid]
        source_controls=origins+radius[:,None]*directions
        fitted_controls=source_controls*scale+center
        # Avoid fitting cavity to ear protrusions or the nose below the scalp.
        target_y=np.maximum(fitted_controls[:,1],eye_y+.07*height)
        ray_origins=np.column_stack((np.full(len(target_y),center[0]),target_y,np.full(len(target_y),center[2])))
        target_radius=nearest_ray(head,ray_origins,directions)
        current_radius=np.linalg.norm((fitted_controls-ray_origins)[:,[0,2]],axis=1)
        valid=np.isfinite(target_radius)&(target_radius>1)
        source_controls=source_controls[valid];angles=angles[valid]
        delta=target_radius[valid]-settings['anchor_overlap_mm']-current_radius[valid]
        delta=np.clip(delta,-limits['cage_displacement_mm'],limits['cage_displacement_mm'])
        if len(delta)<30:raise ValueError('Insufficient cavity/scalp correspondence')
        theta=np.arctan2(v[:,0],v[:,2]);displacement=np.zeros(len(v))
        # Smooth periodic angular/height interpolation. Shared radial offset moves
        # adjacent inside/outside surfaces together; still recheck after boolean.
        for first in range(0,len(v),4000):
            sl=slice(first,first+4000)
            angular=np.angle(np.exp(1j*(theta[sl,None]-angles[None,:])))
            dy=(v[sl,1,None]-source_controls[None,:,1])*scale[1]
            weights=np.exp(-.5*(angular/.40)**2-.5*(dy/settings['cage_bandwidth_mm'])**2)
            displacement[sl]=(weights@delta)/(weights.sum(1)+1e-12)
        radial=affine-center;radial[:,1]=0;radial/=np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-8)
        # Fade radial offsets at the centerline to avoid a singular axis.
        fade=np.clip(np.linalg.norm(v[:,[0,2]],axis=1)/2,0,1)
        output+=radial*(displacement*fade)[:,None]
        controls=source_controls.tolist();deltas=delta.tolist()
    custom=asset.get('custom',{})
    radial=output-center;radial[:,1]=0
    output+=radial*custom.get('volume',0)
    theta=np.arctan2(v[:,0],v[:,2]);low=np.clip((asset['inner_roof_y']*.65-v[:,1])/(asset['inner_roof_y']*.65),0,1)
    front=np.maximum(np.cos(theta),0)**3;back=np.maximum(-np.cos(theta),0)**3
    output[:,1]+=low*(front*custom.get('fringe_lift_mm',0)+back*custom.get('nape_lift_mm',0))
    yaw=np.deg2rad(custom.get('yaw_deg',0));rotation=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    output=(output-center)@rotation.T+center
    result=source.copy();result.vertices=output
    if not np.isfinite(output).all() or not result.is_volume or not result.is_winding_consistent:raise ValueError('Fitted hair lost closed positive-volume topology')
    report={'asset_id':asset['id'],'asset_sha256':asset['sha256'],'method':method,
        'scale_xyz':scale.tolist(),'translation_mm':center.tolist(),'control_count':len(controls),
        'control_offsets_mm':deltas,'max_added_cage_displacement_mm':float(np.linalg.norm(output-affine,axis=1).max()),
        'custom':custom,'source_faces':len(source.faces),'fitted_faces':len(result.faces),
        'style':'supplied reusable asset; not a reconstructed personal hairstyle',
        'interface':'solid union after bounded cavity fit; not a detachable mechanical connector',
        'minimum_thickness_preservation_verified':False,'self_intersection_verified':False}
    return result,report
