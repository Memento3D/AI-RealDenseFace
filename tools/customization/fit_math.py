"""Offline accessory fitting in an anatomical Z-up frame (metres).

Input landmarks must be the 68 FLAME barycentric surface bindings in the same
world frame as the head. They are not image pixel coordinates. Source assets
need a reviewed forward/up convention; a bounding box cannot infer semantics.
"""
import numpy as np


def head_frame(landmarks):
    p=np.asarray(landmarks,dtype=float)
    if p.shape!=(68,3) or not np.isfinite(p).all():
        raise ValueError('Expected 68 finite world-space landmarks')
    left=p[[36,39]].mean(0);right=p[[42,45]].mean(0)
    origin=(left+right)/2;span=np.linalg.norm(right-left)
    if not .015<span<.10:raise ValueError('Unsupported eye spacing in metres')
    x=(right-left)/span
    z=origin-p[8];z-=x*np.dot(z,x)
    if np.linalg.norm(z)<span*.3:raise ValueError('Degenerate chin/eye frame')
    z/=np.linalg.norm(z);y=np.cross(z,x)
    basis=np.column_stack([x,y,z])
    # The nose must face away from the back-facing axis.
    if np.dot(p[30]-origin,y)>=0:raise ValueError('Landmark orientation is inconsistent')
    return origin,basis,span


def glasses_transform(landmarks,profile):
    origin,basis,span=head_frame(landmarks)
    left=np.asarray(profile['left_lens_center_m'],float)
    right=np.asarray(profile['right_lens_center_m'],float)
    if left.shape!=(3,) or right.shape!=(3,) or not np.isfinite([left,right]).all():
        raise ValueError('Invalid source lens anchors')
    source_span=np.linalg.norm(right-left)
    if source_span<.01:raise ValueError('Degenerate source lens anchors')
    # A small deliberate lens decentration keeps the hinges outside the temples.
    scale=span/source_span*float(profile.get('lens_spacing_factor',1.12))
    if not .3<=scale<=1.8:raise ValueError('Accessory outside calibrated scale range')
    clearance=float(profile.get('bridge_clearance_m',.002))
    if not 0<=clearance<=.008:raise ValueError('Unsupported bridge clearance')
    target=origin+basis[:,2]*.001
    # Bridge height belongs to the source model; evaluate the nose at that level.
    bridge_z=float(profile['bridge_height_above_lens_m'])*scale+.001
    p=np.asarray(landmarks,float);nose=p[[27,28,29,30]]
    heights=(nose-origin)@basis[:,2];depths=(nose-origin)@basis[:,1]
    order=np.argsort(heights)
    depth=float(np.interp(bridge_z,heights[order],depths[order]))
    target+=basis[:,1]*(depth-clearance)
    matrix=np.eye(4);matrix[:3,:3]=basis*scale
    matrix[:3,3]=target-matrix[:3,:3]@((left+right)/2)
    return matrix,{'scale':scale,'eye_spacing_m':span,'bridge_clearance_m':clearance,
                   'world_eye_midpoint':origin.tolist(),'world_basis':basis.tolist(),
                   'lens_spacing_factor':float(profile.get('lens_spacing_factor',1.12))}


def validate_options(headwear):
    if headwear not in ('cap','hair'):raise ValueError('Choose exactly one: cap OR hair')
    return headwear


def profile_for_head(profile,ear_anchors,landmarks,source_vertices):
    """Balance pupil spacing with temple width, within a reviewed size limit."""
    _,basis,span=head_frame(landmarks);ears=np.asarray(ear_anchors,float)
    if ears.shape!=(2,3) or not np.isfinite(ears).all():raise ValueError('Invalid semantic ear anchors')
    width=np.ptp(np.asarray(source_vertices,float)[:,0])
    if width<.02:raise ValueError('Degenerate frame width')
    ear_span=abs(float((ears[1]-ears[0])@basis[:,0]))
    old_factor=float(profile.get('lens_spacing_factor',1.12))
    lens_span=np.linalg.norm(np.asarray(profile['right_lens_center_m'])-profile['left_lens_center_m'])
    initial=span/lens_span*old_factor;requested=max(initial,.96*ear_span/width)
    ratio=float(requested/initial);result=dict(profile)
    if ratio<=1.2:result['lens_spacing_factor']=old_factor*ratio
    return result,{'ear_span_m':ear_span,'target_frame_fraction_of_ear_span':.96,
                   'requested_uniform_upscale':ratio,'maximum_uniform_upscale':1.2,
                   'requires_wider_asset':ratio>1.2,'applied_uniform_upscale':ratio if ratio<=1.2 else 1.}


def transform_points(points,matrix):
    p=np.asarray(points,float);m=np.asarray(matrix,float)
    return p@m[:3,:3].T+m[:3,3]
