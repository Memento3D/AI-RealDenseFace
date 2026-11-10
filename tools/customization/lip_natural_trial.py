"""Bounded local lip experiment, with fixed identity/cameras and explicit ROI."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
from scipy.sparse import coo_matrix,diags,eye
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.customization.fit_trial import world_parts
from tools.customization.neck_natural_math import soften_neck_base,smoothstep,neck_base_mask
from tools.headapi.assembly_math import AXES,transform_points
from tools.headapi.assembly_cohesion_trial import smooth_seam_normals
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.geometry_refinement_math import mesh_guard,project
from tools.headapi.service import sha,write


def bounded_lip_candidate(vertices,faces,embedding,masks):
    v=np.asarray(vertices,float);f=np.asarray(faces,int);height=float(np.ptp(v[:,1]));n=len(v)
    e=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)
    rows=np.r_[e[:,0],e[:,1]];cols=np.r_[e[:,1],e[:,0]]
    graph=coo_matrix((np.r_[lengths,lengths],(rows,cols)),shape=(n,n)).tocsr()
    distance=dijkstra(graph,directed=False,indices=np.asarray(masks['lips']),min_only=True)
    radius=.045*height;active=distance<radius
    for label in ['eye_region','nose','neck','left_ear','right_ear']:active[np.asarray(masks[label],int)]=False
    ids=np.flatnonzero(active);adj=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    L=eye(n)-diags(1/np.maximum(np.asarray(adj.sum(1)).ravel(),1))@adj;A=L[:,ids]
    fi=embedding['full_faces'];weights=embedding['full_weights']
    if np.asarray(fi).shape!=(68,) or np.asarray(weights).shape!=(68,3) or not np.isfinite(weights).all() or not np.allclose(weights.sum(1),1):raise ValueError('Invalid landmark binding')
    C=np.zeros((8,n));landmark_ids=[61,62,63,67,66,65,60,64]
    for row,index in enumerate(landmark_ids):np.add.at(C[row],f[fi[index]],weights[index])
    landmarks=C@v;targets=np.zeros(8)
    if np.mean(abs(landmarks[:3,1]-landmarks[3:6,1]))<height*.0001:raise ValueError('No supported open gap to correct')
    for a,b in [(0,3),(1,4),(2,5)]:
        mid=(landmarks[a,1]+landmarks[b,1])/2
        targets[a]=mid-landmarks[a,1];targets[b]=mid-landmarks[b,1]
    B=C[:,ids];system=A.T@A+eye(len(ids))*.01+coo_matrix(4000*(B.T@B)).tocsr()
    attempts=[]
    for fraction in [.65,.50,.35,.20]:
        displacement=spsolve(system,4000*B.T@(targets*fraction))
        candidate=v.copy();candidate[ids,1]+=displacement
        guard=mesh_guard(v[None],candidate[None],f,height)
        attempts.append({'fraction':fraction,'guard':guard})
        if guard['passed']:
            if not np.array_equal(v[~active],candidate[~active]):raise ValueError('Changed geometry outside mouth ROI')
            blend=1-smoothstep(distance/radius);blend[~active]=0
            return candidate,blend,{'active_vertices':len(ids),'fraction_of_requested_closure':fraction,'guard':guard,'attempts':attempts}
    raise ValueError('All local lip candidates failed deformation guards: '+json.dumps(attempts))


def reproject_mouth(atlas,candidate,faces,uv,quality_path,job,vertex_alpha):
    with np.load(quality_path,allow_pickle=False) as d:index=d['face_index']
    alpha=atlas_positions(index,np.repeat(vertex_alpha[:,None],3,axis=1),faces,uv)[...,0]
    valid=(index>=0)&np.isfinite(alpha)&(alpha>.001)
    positions=atlas_positions(index,candidate,faces,uv)
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as d:camera=d['cameras'][0]
    import cv2
    center=cv2.decomposeProjectionMatrix(camera)[2];center=(center[:3]/center[3]).reshape(3)
    mesh=trimesh.Trimesh(candidate,faces,process=False)
    front=np.einsum('ij,ij->i',mesh.face_normals,center-mesh.triangles_center)>0
    valid&=front[np.maximum(index,0)]
    y,x=np.where(valid);pixels=project(positions[y,x],camera)
    image=np.asarray(Image.open(job/'rdf_crops/front.png').convert('RGB'))
    good=(pixels[:,0]>=0)&(pixels[:,0]<image.shape[1]-1)&(pixels[:,1]>=0)&(pixels[:,1]<image.shape[0]-1)
    y,x,pixels=y[good],x[good],pixels[good];lo=np.floor(pixels).astype(int);w=pixels-lo
    a=image[lo[:,1],lo[:,0]].astype(float);b=image[lo[:,1],lo[:,0]+1];c=image[lo[:,1]+1,lo[:,0]];d=image[lo[:,1]+1,lo[:,0]+1]
    sampled=a*(1-w[:,0,None])*(1-w[:,1,None])+b*w[:,0,None]*(1-w[:,1,None])+c*(1-w[:,0,None])*w[:,1,None]+d*w[:,0,None]*w[:,1,None]
    result=atlas.copy();weight=alpha[y,x,None]
    result[y,x]=np.rint(atlas[y,x]*(1-weight)+sampled*weight).clip(0,255).astype(np.uint8)
    changed=np.any(result!=atlas,axis=-1)
    if not np.array_equal(result[~valid],atlas[~valid]):raise ValueError('Changed texture outside mouth ROI')
    return result,changed


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    job=ROOT/'runs/headapi-assembly/55b83418-5c1a-4608-9dda-429d786d5e51';source=ROOT/'runs/neck-natural-review-20261105/attempt-03/restore-neck-tone'
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as d:v=d['vertices'];f=d['faces'];camera=d['cameras'][0]
    with np.load(ROOT/'runs/prod-win-02/input04-production-02/color/embedding.npz',allow_pickle=False) as d:embedding={k:d[k] for k in d.files}
    with (ROOT/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl').open('rb') as stream:masks=pickle.load(stream,encoding='latin1')
    observed=np.asarray(json.loads((job/'rdf_crops/report.json').read_text())['views']['front']['landmarks_crop'])
    source_lm=np.einsum('ij,ijk->ik',embedding['full_weights'],v[f[embedding['full_faces']]])
    source_xy=project(source_lm,camera);source_gap=np.linalg.norm(source_xy[62]-source_xy[66]);observed_gap=np.linalg.norm(observed[13]-observed[14])
    if source_gap<1 or observed_gap>source_gap*.6:raise ValueError('Source photo does not support a bounded closure experiment')
    candidate,alpha,report=bounded_lip_candidate(v,f,embedding,masks)
    with np.load(ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz',allow_pickle=False) as d:
        if not np.array_equal(f,d['faces']):raise ValueError('UV topology mismatch')
        uv=d['uvs'][d['uv_faces']]
    atlas=np.asarray(Image.open(source.parent/'head-neck-atlas.png').convert('RGB'))
    result,changed=reproject_mouth(atlas,candidate,f,uv,job/'rdf_uv/raw/quality_maps.npz',job,alpha)
    Image.fromarray(result).save(a.output/'head-neck-mouth-atlas.png');Image.fromarray(changed.astype(np.uint8)*255).save(a.output/'mouth-change-mask.png')
    M=np.array(json.loads((job/'render/assembly-report.json').read_text())['matrix_native_to_working'])
    world=transform_points(candidate,M)+[0,0,.018];mask=neck_base_mask(transform_points(v,M),masks['neck'],masks['boundary'],.795)
    prior=json.loads((source/'report.json').read_text())['neck_conditioning']
    world,_=soften_neck_base(world,mask,.765,target_center=prior['center_xy'],target_radii=prior['radii_xy'])
    head=trimesh.Trimesh(world[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv.reshape(-1,2),material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(result),metallicFactor=0,roughnessFactor=1,doubleSided=True)))
    hv,hf,hu=trimesh.intersections.slice_faces_plane(head.vertices,head.faces,[0,0,1],[0,0,.765],uv=head.visual.uv)
    head=trimesh.Trimesh(hv,hf,vertex_normals=smooth_seam_normals(hv,hf),process=False,visual=trimesh.visual.TextureVisuals(uv=hu,material=head.visual.material))
    scene=trimesh.Scene();scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
    for name,mesh in world_parts(source/'scene.glb').items():
        if name!='Personalized_Complete_Head':scene.add_geometry(mesh,node_name=name,geom_name=name)
    scene.apply_transform(np.linalg.inv(AXES));scene.export(a.output/'scene.glb')
    def landmarks(vertices):return np.einsum('ij,ijk->ik',embedding['full_weights'],vertices[f[embedding['full_faces']]])
    before=project(landmarks(v),camera);after=project(landmarks(candidate),camera)
    observed=np.asarray(json.loads((job/'rdf_crops/report.json').read_text())['views']['front']['landmarks_crop'])
    report.update({'status':'LOCAL_LIP_REVIEW_REQUIRED','source_sha256':sha(source/'scene.glb'),'latent_identity_parameters_unchanged':True,
        'geometry_outside_mouth_roi_unchanged':True,'texture_outside_mouth_roi_unchanged':True,'mouth_texture_reprojected_from_front_photo':True,
        'changed_mouth_texels':int(changed.sum()),'front_projected_inner_lip_gap_before_px':float(np.linalg.norm(before[62]-before[66])),
        'front_projected_inner_lip_gap_after_px':float(np.linalg.norm(after[62]-after[66])),
        'front_observed_mediapipe_inner_lip_gap_px':float(np.linalg.norm(observed[13]-observed[14])),
        'api_integrated':False,'print_ready':False,'limitations':['Local post-fit deformation, not a new identity or expression reconstruction',
        'FLAME68 and MediaPipe landmarks are approximate anatomical correspondences','No independent likeness or lip self-intersection validation']})
    write(a.output/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
