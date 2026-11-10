"""Offline natural-neck candidates against the existing customization review."""
import argparse,json,pickle,sys,copy
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.customization.fit_trial import world_parts,material_digest
from tools.customization.neck_natural_math import soften_neck_base,fit_skin_field,skin_field,smoothstep,crown_similarity,neck_base_mask
from tools.headapi.assembly_math import AXES,transform_points
from tools.headapi.assembly_cohesion_trial import smooth_seam_normals,boundary_loops
from tools.headapi.demo_appearance import atlas_positions
from tools.headapi.neck_semantic_trial import barycentric_pixels
from tools.headapi.hair_parser import HairParser,PARSER_SHA
from tools.headapi.service import sha,write


def texture_candidate(job,out,vertices,faces,neck_ids,uv):
    original=np.array(Image.open(job/'render/appearance/head-atlas.png').convert('RGB'))
    with np.load(job/'rdf_uv/raw/quality_maps.npz',allow_pickle=False) as data:
        index=data['face_index'];confidence=data['confidence']
    owner=np.asarray(Image.open(job/'rdf_uv/raw/coverage.png'))//80
    positions=atlas_positions(index,vertices,faces,uv)
    membership=np.isin(np.arange(len(vertices)),neck_ids).astype(float)
    fraction=atlas_positions(index,np.repeat(membership[:,None],3,axis=1),faces,uv)[...,0]
    valid=index>=0;neck=valid&(fraction>.001)
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as data:views=data['view_vertices'];cameras=data['cameras']
    dominant=np.zeros(index.shape,np.uint8);parser_path=Path('D:/AI-cache/face-parsing/resnet18.onnx');parser=HairParser(parser_path)
    for i,slot in enumerate(['front','left','right']):
        image=Image.open(job/'rdf_crops'/(slot+'.png')).convert('RGB');semantic=parser(image)
        Image.fromarray(semantic).save(out/(slot+'-semantic.png'))
        yy,xx=np.where(neck&(owner==i+1))
        for start in range(0,len(yy),32768):
            y,x=yy[start:start+32768],xx[start:start+32768];fi=index[y,x]
            weights=barycentric_pixels(y,x,fi,uv,index.shape[0]);points=np.einsum('ni,nij->nj',weights,views[i][faces[fi]])
            projected=np.c_[points,np.ones(len(points))]@cameras[i].T
            xy=np.floor(projected[:,:2]/projected[:,2,None]).astype(int)
            good=(projected[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<image.width)&(xy[:,1]>=0)&(xy[:,1]<image.height)
            dominant[y[good],x[good]]=semantic[xy[good,1],xy[good,0]]
    reference=neck&(fraction>.6)&np.isin(dominant,[1,14])&(owner>0)&(confidence>.25)
    reference&=(original.mean(-1)>35)&(original.mean(-1)<225)
    yy,xx=np.where(reference)
    if len(yy)<64:raise ValueError('Insufficient semantically observed neck skin; no invented success fallback')
    step=max(1,len(yy)//20000);yy=yy[::step];xx=xx[::step]
    model=fit_skin_field(positions[yy,xx],original[yy,xx])
    result=original.copy();y,x=np.where(neck)
    blend=smoothstep((fraction[y,x]-.05)/.8)
    colors=skin_field(positions[y,x],model)
    result[y,x]=np.rint(original[y,x]*(1-blend[:,None])+colors*blend[:,None]).clip(0,255).astype(np.uint8)
    changed=np.any(result!=original,axis=-1)
    if not np.array_equal(result[~neck],original[~neck]):raise ValueError('Facial/scalp texels changed')
    Image.fromarray(result).save(out/'head-neck-atlas.png')
    Image.fromarray(changed.astype(np.uint8)*255).save(out/'neck-texture-change-mask.png')
    write(out/'texture-report.json',{'policy':'semantic_neck_low_frequency_skin_estimate_v1','reference_texels':int(reference.sum()),
        'changed_neck_texels':int(changed.sum()),'changed_observed_neck_texels':int((changed&(owner>0)).sum()),
        'non_neck_texture_pixels_byte_equal':True,'neck_is_estimated_not_recovered_albedo':True,
        'parser_sha256':sha(parser_path),'dominant_neck_labels':{str(k):int((dominant[neck]==k).sum()) for k in np.unique(dominant[neck])}})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    job=ROOT/'runs/headapi-assembly/55b83418-5c1a-4608-9dda-429d786d5e51'
    baseline=ROOT/'runs/customization-prep-20261105/fit-review-06/cap-glasses-03/scene.glb'
    parts=world_parts(baseline);old=parts['Personalized_Complete_Head']
    assembly=json.loads((job/'render/assembly-report.json').read_text());M=np.asarray(assembly['matrix_native_to_working'])
    bundle=job/'rdf/projection_bundle.npz';uv_path=ROOT/'vendor/RealDenseFace/weights/flame/flame_assets.npz';mask_path=ROOT/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl'
    for path in [bundle,uv_path,mask_path]:
        if sha(path)!=assembly['input_sha256'][str(path)]:raise ValueError('Pinned job dependency changed')
    with np.load(bundle,allow_pickle=False) as data:v=data['vertices'];f=data['faces']
    with np.load(uv_path,allow_pickle=False) as data:
        if not np.array_equal(f,data['faces']):raise ValueError('UV topology mismatch')
        uv=data['uvs'][data['uv_faces']]
    with mask_path.open('rb') as stream:semantic=pickle.load(stream,encoding='latin1')
    native_work=transform_points(v,M)
    mask=neck_base_mask(native_work,semantic['neck'],semantic['boundary'],.795);neck_ids=np.flatnonzero(mask)
    atlas=texture_candidate(job,out,v,f,neck_ids,uv)
    source_atlas=Image.open(job/'render/appearance/head-atlas.png').convert('RGB')
    pivot=np.array([-.022,.060,.765]);S=crown_similarity(native_work,old.vertices[:,2].max(),pivot)
    body=parts['Body_Original_With_Hand']
    rings=[ring for ring in boundary_loops(body.vertices,body.faces) if .75<ring[:,2].mean()<.84 and abs(ring[:,0].mean())<.06 and len(ring)>30]
    if len(rings)!=1:raise ValueError('Ambiguous collar aperture')
    collar_center=[float(np.median(native_work[mask,0])),float(rings[0][:,1].mean())]
    collar_radii=np.ptp(rings[0][:,:2],axis=0)*.4
    cases=[]
    for name,transform,retexture in [('restore-neck-only',np.eye(4),False),('restore-neck-tone',np.eye(4),True),('balanced-neck-tone',S,True)]:
        folder=out/name;folder.mkdir();world=transform_points(native_work,transform)
        if name!='balanced-neck-tone':world+=[0,0,.018]
        original_world=world.copy();world,conditioning=soften_neck_base(world,mask,.765,target_center=collar_center,target_radii=collar_radii)
        original_face=original_world[~mask].copy()
        # Keep the source RDF neck, trimming only the hidden shoulder base.
        corners=world[f].reshape(-1,3);normals=smooth_seam_normals(world,f)[f].reshape(-1,3)
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas) if retexture else source_atlas.copy(),metallicFactor=0,roughnessFactor=1,doubleSided=True)
        head=trimesh.Trimesh(corners,np.arange(f.size).reshape(-1,3),vertex_normals=normals,process=False,
                visual=trimesh.visual.TextureVisuals(uv=uv.reshape(-1,2),material=material))
        hv,hf,hu=trimesh.intersections.slice_faces_plane(head.vertices,head.faces,[0,0,1],[0,0,.765],uv=head.visual.uv)
        head=trimesh.Trimesh(hv,hf,process=False,visual=trimesh.visual.TextureVisuals(uv=hu,material=material))
        head.vertex_normals=smooth_seam_normals(hv,hf)
        if not np.array_equal(world[~mask],original_face):raise ValueError('Facial shape changed')
        scene=trimesh.Scene();scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
        for node,mesh in parts.items():
            if node in ('Personalized_Complete_Head','Neck_Transition'):continue
            mesh=mesh.copy(include_cache=True)
            if node.startswith('Glasses') and name=='balanced-neck-tone':
                mesh.apply_translation([0,0,-.018]);mesh.apply_transform(S)
            scene.add_geometry(mesh,node_name=node,geom_name=node)
        scene.apply_transform(np.linalg.inv(AXES));scene.export(folder/'scene.glb')
        report={'case':name,'status':'LOCAL_REVIEW_REQUIRED','source_baseline_sha256':sha(baseline),'source_bundle_sha256':sha(bundle),
                'face_geometry_preserved_up_to_reported_similarity':True,'source_face_expression_changed':False,
                'head_uniform_scale':float(transform[0,0]),'matrix_from_native_working':transform.tolist(),
                'extra_head_translation_working_m':[0,0,.018] if name!='balanced-neck-tone' else [0,0,0],
                'neck_geometry_restored_from_current_job':True,'neck_conditioning':conditioning,
                'old_flat_connector_removed':True,'neck_tone_estimated':retexture,'non_neck_texture_pixels_preserved':True,
                'non_head_parts_changed':False,'glasses_follow_head_similarity':name=='balanced-neck-tone','api_integrated':False,
                'print_ready':False,'limitations':['Neck hidden color is estimated; original photo lighting remains in facial texture',
                'Uniform size/pose candidate, not recovery of expression or face identity','Open surfaces overlap under collar; no print/collision certification',
                'Original cap-hand segmentation issues remain outside this neck-only change']}
        write(folder/'report.json',report);cases.append(report);print(name,'scale',transform[0,0],flush=True)
    write(out/'report.json',{'policy':'raw_neck_preserved_local_comparison_v1','source_baseline':str(baseline),'source_sha256':sha(baseline),'cases':cases})

if __name__=='__main__':main()
