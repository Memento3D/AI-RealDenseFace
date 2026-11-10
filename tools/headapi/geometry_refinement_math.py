"""Bounded camera updates and mesh guards for experimental local refinement."""
import numpy as np
import cv2


def project(points,camera):
    h=np.c_[points,np.ones(len(points))]@camera.T
    return h[:,:2]/h[:,2,None]


def camera_update(camera,vertices,parameters,height):
    k,r,c,*_=cv2.decomposeProjectionMatrix(camera);k/=k[2,2];center=(c[:3]/c[3]).reshape(3)
    t=-r@center;mean=(vertices@r.T+t).mean(0)
    rotation=cv2.Rodrigues(np.asarray(parameters[:3])*np.radians(1.5))[0]
    translation=np.asarray(parameters[3:])*height*np.array([.005,.005,.01])
    return k@np.c_[rotation@r,rotation@(t-mean)+mean+translation]


def mesh_guard(base,candidate,faces,height):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    old_cross=np.cross(base[:,faces[:,1]]-base[:,faces[:,0]],base[:,faces[:,2]]-base[:,faces[:,0]])
    new_cross=np.cross(candidate[:,faces[:,1]]-candidate[:,faces[:,0]],candidate[:,faces[:,2]]-candidate[:,faces[:,0]])
    old_area=np.linalg.norm(old_cross,axis=-1);new_area=np.linalg.norm(new_cross,axis=-1)
    usable=old_area>1e-14
    cosine=np.sum(old_cross*new_cross,axis=-1)/np.maximum(old_area*new_area,1e-30)
    old_edge=np.linalg.norm(base[:,edges[:,0]]-base[:,edges[:,1]],axis=-1)
    new_edge=np.linalg.norm(candidate[:,edges[:,0]]-candidate[:,edges[:,1]],axis=-1)
    valid_edge=old_edge>1e-10
    displacement=float(np.linalg.norm(candidate-base,axis=-1).max())
    normal_change=float(np.degrees(np.arccos(np.clip(cosine[usable],-1,1))).max())
    edge_change=float(np.max(np.abs(new_edge[valid_edge]/old_edge[valid_edge]-1)))
    area_ratio=float((new_area[usable]/old_area[usable]).min())
    reversals=int((cosine[usable]<=0).sum())
    passed=bool(displacement<=.003*height and normal_change<=30 and edge_change<=.15 and area_ratio>=.2 and reversals==0)
    return {'passed':passed,'maximum_displacement_native':displacement,'maximum_displacement_head_height_fraction':displacement/height,'maximum_normal_change_degrees':normal_change,'maximum_edge_fraction_change':edge_change,'minimum_area_ratio':area_ratio,'normal_reversals':reversals,'baseline_degenerate_faces_excluded':int((~usable).sum()),'self_intersections_verified':False}
