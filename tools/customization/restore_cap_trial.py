"""Offline template-specific recovery of original cap/material/pose.

Source triangles/UVs are selected, never remodeled or recolored. The mask is
an annotated trial for men.glb, not a general semantic segmentation model.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra,connected_components

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.headapi.service import sha,write
from tools.headapi.assembly_math import AXES

def select_source(mesh,mask):
    ids,inverse=np.unique(mesh.faces[mask].ravel(),return_inverse=True)
    return trimesh.Trimesh(mesh.vertices[ids],inverse.reshape(-1,3),
        vertex_normals=mesh.vertex_normals[ids],process=False,
        visual=trimesh.visual.TextureVisuals(uv=mesh.visual.uv[ids],material=mesh.visual.material))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/prod-win-01/src/GLB/men.glb'
    if sha(source)!='96496c2d10e2896a780fb060c2171773000df22a8649e15d66096d26080db84d':raise ValueError('Original body pin changed')
    m=trimesh.load(source,force='mesh',process=False);m.apply_transform(AXES)
    v,first,inverse=np.unique(np.round(m.vertices,7),axis=0,return_index=True,return_inverse=True)
    faces=inverse[m.faces]
    image=np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
    uv=m.visual.uv
    x=np.clip((uv[:,0]*(image.shape[1]-1)).round().astype(int),0,image.shape[1]-1)
    y=np.clip(((1-uv[:,1])*(image.shape[0]-1)).round().astype(int),0,image.shape[0]-1)
    original_rgb=image[y,x].astype(float)/255
    rgb=np.zeros((len(v),3));np.add.at(rgb,inverse,original_rgb);rgb/=np.bincount(inverse)[:,None]
    active=(v[:,2]>.85)&(v[:,0]>-.23)&(v[:,0]<.15)&(v[:,1]>-.15)&(v[:,1]<.20)
    eligible=active[faces].all(1);ff=faces[eligible]
    edges=np.unique(np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1),axis=0)
    cost=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)*(1+25*np.linalg.norm(rgb[edges[:,0]]-rgb[edges[:,1]],axis=1))
    graph=coo_matrix((np.r_[cost,cost],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    hi=rgb.max(1);lo=rgb.min(1)
    # Dark cloth can be slightly warm in the scanned atlas. Only strong,
    # bright skin evidence is a protected hand/face seed.
    skin=(rgb[:,0]>rgb[:,1]*1.07)&(rgb[:,0]>rgb[:,2]*1.18)&(hi>.45)
    cap_seed=active&~skin&(hi<.65)&(hi>.10)&(v[:,2]>.96)&(v[:,0]>-.13)&(v[:,0]<.10)
    hair_seed=active&(v[:,2]<.93)&(hi<.09)&(v[:,0]>-.10)&(v[:,0]<.075)
    protected=active&(skin|hair_seed|(v[:,2]<.875))
    cap_seed&=~protected
    if cap_seed.sum()<50 or protected.sum()<50:raise ValueError('Insufficient annotated seed evidence')
    dc=dijkstra(graph,directed=False,indices=np.flatnonzero(cap_seed),min_only=True)
    dp=dijkstra(graph,directed=False,indices=np.flatnonzero(protected),min_only=True)
    cap_v=(dc<dp)&np.isfinite(dc)&active
    cap=np.zeros(len(m.faces),bool);cap[eligible]=cap_v[ff].sum(1)>=2
    # Recover the original cream tassel using its annotated hanging region and
    # source colors; keep it separate from the dark cap.
    centers=m.vertices[m.faces].mean(1)
    face_rgb=original_rgb[m.faces].mean(1)
    # Template-specific rim annotation in the source's world frame. The cap
    # and hand are connected in this scanned template; graph labels alone
    # leave a ragged cap and holes around the pinching fingers.
    rim=.923-.20*(centers[:,1]+.02)
    region=(centers[:,0]>-.18)&(centers[:,0]<.12)&(centers[:,1]>-.065)&(centers[:,1]<.24)&(centers[:,2]>rim)
    strong_skin=(face_rgb[:,0]>face_rgb[:,1]*1.07)&(face_rgb[:,0]>face_rgb[:,2]*1.18)&(face_rgb.max(1)>.45)
    strong_skin&=(centers[:,1]<.04)
    cap=region&~strong_skin
    tassel=(centers[:,0]>-.18)&(centers[:,0]<-.085)&(centers[:,1]>-.10)&(centers[:,1]<.075)&(centers[:,2]>.73)&(centers[:,2]<.99)
    tassel&=(face_rgb.min(1)>.38)&(face_rgb[:,1]>face_rgb[:,0]*.78)&(face_rgb[:,2]>face_rgb[:,0]*.58)
    tassel&=~cap
    # Recover original hand triangles that earlier head/cap removal discarded.
    hand=(centers[:,0]<-.045)&(centers[:,0]>-.26)&(centers[:,2]>.88)&(centers[:,1]>-.13)&(centers[:,1]<.11)
    hand&=(face_rgb[:,0]>face_rgb[:,1]*1.07)&(face_rgb[:,0]>face_rgb[:,2]*1.18)&(face_rgb.max(1)>.45)
    with np.load(ROOT/'runs/assembly-cohesion-01/restored-selection-v5.npz',allow_pickle=False) as prior:
        removed=prior['removed_original_faces']
    if removed.shape!=cap.shape:raise ValueError('Original body selection topology differs')
    keep=(~removed|hand)&~cap&~tassel
    body=select_source(m,keep)
    # The historical selector discards detached original-head remnants. Retain
    # the grounded body component; cap/tassel are restored as explicit nodes.
    vv,iv=np.unique(np.round(body.vertices,7),axis=0,return_inverse=True);bf=iv[body.faces]
    e=np.concatenate([bf[:,[0,1]],bf[:,[1,2]],bf[:,[2,0]]])
    g=coo_matrix((np.ones(len(e),np.uint8),(e[:,0],e[:,1])),shape=(len(vv),len(vv))).tocsr()
    _,cc=connected_components(g,directed=False);label=cc[bf[:,0]]
    ids,counts=np.unique(label,return_counts=True);body=select_source(body,label==ids[counts.argmax()])
    current=ROOT/'runs/headapi-assembly/55b83418-5c1a-4608-9dda-429d786d5e51/render/scene.glb'
    scene=trimesh.load(current,force='scene',process=False)
    result=trimesh.Scene()
    for node in ('Personalized_Complete_Head','Neck_Transition'):
        if node=='Neck_Transition':continue
        matrix,name=scene.graph[node];mesh=scene.geometry[name].copy(include_cache=True);mesh.apply_transform(matrix);mesh.apply_transform(AXES)
        if node=='Personalized_Complete_Head':
            mesh.apply_translation([0,0,.018])
        else:
            # Keep the body-side ring fixed and move the upper ring with head.
            z=mesh.vertices[:,2];weight=(z.max()-z)/(z.max()-z.min())
            mesh.vertices[:,2]+=weight*.018
        result.add_geometry(mesh,node_name=node,geom_name=node)
    for node,mesh in [('Body_Original_With_Hand',body),('Cap_Original',select_source(m,cap)),('Tassel_Original',select_source(m,tassel))]:
        result.add_geometry(mesh,node_name=node,geom_name=node)
    # The restored original body has a different neck boundary from the old
    # procedural-body preset. Bind the connector to the actual two rings.
    from tools.headapi.assembly_cohesion_trial import boundary_loops,bridge_rings,smooth_seam_normals
    head=result.geometry['Personalized_Complete_Head']
    body_rings=[r for r in boundary_loops(body.vertices,body.faces) if .75<r[:,2].mean()<.84 and abs(r[:,0].mean())<.06 and len(r)>30]
    head_rings=[r for r in boundary_loops(head.vertices,head.faces) if np.ptp(r[:,2])<1e-5 and abs(r[:,2].mean()-.783)<1e-5]
    if len(body_rings)!=1 or len(head_rings)!=1:raise ValueError('Ambiguous actual neck boundary rings')
    nv,nf=bridge_rings(body_rings[0],head_rings[0]);neck=trimesh.Trimesh(nv,nf,process=False)
    atlas=np.asarray(head.visual.material.baseColorTexture.convert('RGB'))
    sample=head.visual.uv[(head.vertices[:,2]<.799)&(head.vertices[:,1]<.06)]
    ix=np.clip((sample[:,0]*(atlas.shape[1]-1)).astype(int),0,atlas.shape[1]-1)
    iy=np.clip(((1-sample[:,1])*(atlas.shape[0]-1)).astype(int),0,atlas.shape[0]-1)
    color=np.r_[np.median(atlas[iy,ix],axis=0),255].astype(np.uint8)
    neck.visual.vertex_colors=np.tile(color,(len(nv),1));neck.vertex_normals=smooth_seam_normals(nv,nf)
    result.add_geometry(neck,node_name='Neck_Transition',geom_name='Neck_Transition')
    result.apply_transform(np.linalg.inv(AXES));result.export(out/'cap-only.glb')
    np.savez_compressed(out/'original-face-selection.npz',cap=cap,tassel=tassel,hand=hand,body=keep)
    write(out/'report.json',{'policy':'original_cap_source_triangle_recovery_trial_v1','status':'EXPORTED_NEEDS_VISUAL_REVIEW',
        'source_sha256':sha(source),'current_job_glb_sha256':sha(current),'cap_faces':int(cap.sum()),'tassel_faces':int(tassel.sum()),'head_translation_working_m':[0,0,.018],
        'hand_restored_faces':int((removed&hand).sum()),'cap_seed_vertices':int(cap_seed.sum()),
        'original_cap_geometry_recolored':False,'source_uv_and_material_reused':True,'cap_scaled_or_reposed':False,
        'hair_mesh_included':False,'old_template_head_included':False,'ready_for_machine':False,
        'limitations':['Template-specific segmentation candidate requires visual audit; no print or universal fit certification']})
    print('cap',int(cap.sum()),'tassel',int(tassel.sum()),'restored hand',int((removed&hand).sum()),flush=True)
if __name__=='__main__':main()
