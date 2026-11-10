"""Offline multi-option fitting/review batch; never updates running API pins."""
import argparse,copy,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.customization.fit_math import head_frame,glasses_transform,transform_points,validate_options,profile_for_head
from tools.headapi.assembly_math import AXES,eye_alignment
from tools.headapi.service import sha,write


def world_parts(path):
    scene=trimesh.load(path,force='scene',process=False);parts={}
    for node in scene.graph.nodes_geometry:
        matrix,name=scene.graph[node];mesh=scene.geometry[name].copy(include_cache=True)
        mesh.apply_transform(matrix);mesh.apply_transform(AXES);parts[node]=mesh
    return parts


def surface_details(mesh,points):
    """Approximate local signed distances, not a closed-solid collision test."""
    points=np.asarray(points);tree=cKDTree(mesh.triangles_center)
    _,indices=tree.query(points,k=min(32,len(mesh.faces)))
    candidates=mesh.triangles[indices]
    query=np.broadcast_to(points[:,None,:],candidates.shape[:2]+(3,))
    closest=trimesh.triangles.closest_point(candidates.reshape(-1,3,3),query.reshape(-1,3)).reshape(query.shape)
    lengths=np.linalg.norm(query-closest,axis=2);best=lengths.argmin(1)
    face_ids=indices[np.arange(len(points)),best]
    delta=points-closest[np.arange(len(points)),best]
    signed=np.einsum('ij,ij->i',delta,mesh.face_normals[face_ids])
    return closest[np.arange(len(points)),best],mesh.face_normals[face_ids],signed


def surface_samples(mesh,points):
    _,_,signed=surface_details(mesh,points)
    return {'samples':len(points),'nearest_triangle_candidates':32,
            'signed_clearance_quantiles_mm':(np.quantile(signed,[0,.1,.5,.9,1])*1000).tolist(),
            'samples_below_minus_1mm':int((signed<-.001).sum()),
            'scope':'Nearest 32 triangle centers; local normal sign only, open/nonmanifold surfaces allowed; not full mesh collision certification'}


def ear_support_anchors(native_vertices,matrix,offset,landmarks):
    origin,basis,_=head_frame(landmarks)
    mask_file=ROOT/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl'
    with mask_file.open('rb') as stream:masks=pickle.load(stream,encoding='latin1')
    anchors=[]
    for name in ['left_ear','right_ear']:
        world=transform_points(np.asarray(native_vertices)[masks[name]],matrix)+offset
        local=(world-origin)@basis
        crest=world[local[:,2]>=np.quantile(local[:,2],.88)]
        anchors.append(np.median(crest,axis=0)+basis[:,2]*.0015)
    return np.asarray(sorted(anchors,key=lambda point:np.dot(point-origin,basis[:,0])))


def refine_temples(mesh,source_vertices,head,landmarks,profile,ear_anchors=None):
    """Bounded outward arm adjustment; the lens/frame anchors stay fixed."""
    origin,basis,span=head_frame(landmarks)
    source_vertices=np.asarray(source_vertices)
    front=float(profile['left_lens_center_m'][1])
    arms=(source_vertices[:,1]>front+.006)&(abs(source_vertices[:,0])>.025)
    original=mesh.vertices.copy();moved=original.copy();ear_adjustments=[]
    if ear_anchors is not None:
        ear_anchors=np.asarray(ear_anchors,float)
        if ear_anchors.shape!=(2,3) or not np.isfinite(ear_anchors).all():raise ValueError('Invalid semantic ear anchors')
        rear=float(np.max(source_vertices[:,1]));length=max(rear-front,1e-5)
        t=np.clip((source_vertices[:,1]-front)/length,0,1)
        weight=np.clip(t/.65,0,1);weight=weight*weight*(3-2*weight);weight[~arms]=0
        for side,anchor in zip([-1,1],ear_anchors):
            local=(original-origin)@basis;target=(anchor-origin)@basis
            support=arms&(source_vertices[:,0]*side>0)&(abs(local[:,1]-target[1])<.015)
            if not support.any():continue
            rise=float(np.clip(target[2]-np.median(local[support,2]),-.012,.012))
            selected=source_vertices[:,0]*side>0
            moved[selected]+=basis[:,2]*(weight[selected]*rise)[:,None]
            ear_adjustments.append({'side':side,'support_anchor_m':np.asarray(anchor).tolist(),'support_rise_mm':rise*1000})
    raised=moved.copy()
    # Search actual surface clearance within the existing 8 mm arm budget.
    # The former four local-normal steps could stop before convergence or skip
    # vertices at curved ear surfaces. Keep front anchors and vertical fit fixed.
    indices=np.flatnonzero(arms)
    _,_,initial_signed=surface_details(head,raised)
    best_signed=initial_signed.copy()
    target_clearance=.0015
    side=np.where((raised-origin)@basis[:,0]>=0,1.,-1.)
    direction=side[:,None]*basis[:,0]
    for distance in np.linspace(.0005,.008,16):
        pending=indices[best_signed[indices]<target_clearance]
        if not len(pending):break
        candidate=raised[pending]+direction[pending]*distance
        _,_,signed=surface_details(head,candidate)
        better=signed>best_signed[pending]
        selected=pending[better]
        moved[selected]=candidate[better]
        best_signed[selected]=signed[better]
    mesh.vertices=moved
    return {'arm_vertices':int(arms.sum()),'moved_vertices':int((np.linalg.norm(moved-original,axis=1)>1e-7).sum()),
            'max_displacement_mm':float(np.linalg.norm(moved-original,axis=1).max()*1000),
            'max_outward_spread_mm':float(np.linalg.norm(moved-raised,axis=1).max()*1000),
            'ear_support_adjustment':ear_adjustments,'max_outward_allowed_mm':8,'max_vertical_allowed_mm':12,
            'scope':'Semantic ear-crest initialization and bounded clearance search, arms only; no global collision guarantee',
            'clearance_policy':'bounded_lateral_search_v1'}


def fit_front_clearance(glasses,head,landmarks,profile,matrix):
    """Move the complete frame forward by at most 6 mm before arm fitting."""
    _,basis,_=head_frame(landmarks);front=float(profile['left_lens_center_m'][1])
    points=np.vstack([m.vertices[m.vertices[:,1]<front+.006] for m in glasses.values()])
    world=transform_points(points,matrix);chosen=0.;before=None;after=None
    for offset in np.linspace(0,.006,13):
        _,_,signed=surface_details(head,world-basis[:,1]*offset)
        if before is None:before=float(signed.min())
        chosen=float(offset);after=float(signed.min())
        if after>=.0003:break
    adjusted=matrix.copy();adjusted[:3,3]-=basis[:,1]*chosen
    return adjusted,{'forward_offset_mm':chosen*1000,'front_min_local_clearance_before_mm':before*1000,
                    'front_min_local_clearance_after_mm':after*1000,'maximum_forward_offset_mm':6,
                    'requires_manual_review':after<.0003}


def make_hair(path,head,landmarks):
    source=world_parts(path);all_v=np.vstack([m.vertices for m in source.values()])
    origin,basis,span=head_frame(landmarks);local=(head.vertices-origin)@basis
    scalp=local[local[:,2]>.012];lower=scalp.min(0);upper=scalp.max(0)
    target_min=np.array([lower[0]-.003,lower[1]-.004,-.018])
    target_max=np.array([upper[0]+.003,upper[1]+.004,local[:,2].max()+.008])
    lo=all_v.min(0);hi=all_v.max(0);scale=(target_max-target_min)/(hi-lo)
    for mesh in source.values():
        v=(mesh.vertices-lo)*scale+target_min
        mesh.vertices=v@basis.T+origin
        # Smooth shading of the low-poly candidate; this does not add detail.
        mesh.vertex_normals=mesh.vertex_normals
    return source,{'axis_scale':scale.tolist(),'source_bounds':[lo.tolist(),hi.tolist()],
                   'target_bounds_in_head_frame':[target_min.tolist(),target_max.tolist()],
                   'policy':'Reviewed scalp-envelope bounding-box initialization; collision review required'}


def material_digest(mesh):
    mat=getattr(mesh.visual,'material',None);image=getattr(mat,'baseColorTexture',None)
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest() if image else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--cap-attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    profiles=json.loads(a.profiles.read_text(encoding='utf-8'));catalog=ROOT/'assets/custom-library/prepared-20261105'
    cap_report=json.loads((a.cap_attempt/'report.json').read_text())
    if cap_report['current_job_glb_sha256']!=sha(a.job/'render/scene.glb'):
        raise ValueError('Cap preparation belongs to a different job output')
    parts=world_parts(a.cap_attempt/'cap-only.glb');head=parts['Personalized_Complete_Head']
    bundle=a.job/'rdf/projection_bundle.npz';embedding=ROOT/'runs/prod-win-02/input04-production-02/color/embedding.npz'
    matrix=np.array(json.loads((a.job/'render/assembly-report.json').read_text())['matrix_native_to_working'])
    offset=np.asarray(cap_report['head_translation_working_m'])
    with np.load(bundle,allow_pickle=False) as b,np.load(embedding,allow_pickle=False) as e:
        lm=np.einsum('ij,ijk->ik',e['full_weights'],b['vertices'][b['faces'][e['full_faces']]])
    lm=transform_points(lm,matrix)+offset
    with np.load(bundle,allow_pickle=False) as b:ear_anchors=ear_support_anchors(b['vertices'],matrix,offset,lm)
    write(a.output/'landmarks.json',{'world_working_z_up_m':lm.tolist(),'source_bundle_sha256':sha(bundle),
        'embedding_sha256':sha(embedding),'native_to_working':matrix.tolist(),'head_translation_working_m':offset.tolist(),
        'ear_support_anchors':ear_anchors.tolist(),'ear_masks_sha256':sha(ROOT/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl')})
    cases=[];clean_body=world_parts(a.job/'render/scene.glb')
    catalog_index={item['id']:item for item in json.loads((catalog/'catalog.json').read_text())['assets']}
    for headwear,hair_id,glasses_id in [('cap',None,'glasses-03'),('hair','micket-side-part','glasses-03'),
                                      ('hair','micket-upcomb','glasses-03'),('cap',None,'glasses-01')]:
        validate_options(headwear);case_id=headwear+'-'+(hair_id+'-' if hair_id else '')+glasses_id
        for asset_id in [glasses_id]+([hair_id] if hair_id else []):
            if sha(catalog/asset_id/'asset.glb')!=catalog_index[asset_id]['sha256']:raise ValueError('Source accessory pin changed')
        folder=a.output/case_id;folder.mkdir();result=trimesh.Scene();hair={};hair_audit=None
        for node,mesh in parts.items():
            if headwear=='hair' and node in ('Cap_Original','Tassel_Original'):continue
            if headwear=='hair' and node=='Body_Original_With_Hand':
                for clean_node in ('Body_Cuff_Prepared','Complete_Hand_Posed_Static','Wrist_Lining_Join'):
                    result.add_geometry(clean_body[clean_node].copy(include_cache=True),node_name=clean_node,geom_name=clean_node)
                continue
            result.add_geometry(mesh.copy(include_cache=True),node_name=node,geom_name=node)
        if headwear=='hair':
            hair,hair_audit=make_hair(catalog/hair_id/'asset.glb',head,lm)
            for i,mesh in enumerate(hair.values()):result.add_geometry(mesh,node_name=f'Hair_Option_{i}',geom_name=f'Hair_Option_{i}')
        glasses=world_parts(catalog/glasses_id/'asset.glb')
        profile,width_audit=profile_for_head(profiles['glasses'][glasses_id],ear_anchors,lm,np.vstack([m.vertices for m in glasses.values()]))
        M,fit=glasses_transform(lm,profile)
        M,front_audit=fit_front_clearance(glasses,head,lm,profile,M)
        arm_audits=[];kept_glasses=[]
        for i,mesh in enumerate(glasses.values()):
            source_vertices=mesh.vertices.copy()
            if glasses_id=='glasses-03':
                old=mesh.visual.material
                lens=getattr(old,'alphaMode',None)=='BLEND'
                # Open-lens fitting preview avoids foggy alpha blending in
                # simple GLB viewers. Original clear-lens geometry is retained
                # in the source library; this variant previews the frame.
                if lens:continue
                mesh.visual.material=trimesh.visual.material.PBRMaterial(
                    name='Neutral clear preview lens' if lens else 'Charcoal preview frame',
                    baseColorFactor=[215,225,235,35] if lens else [24,25,28,255],
                    metallicFactor=0 if lens else .15,roughnessFactor=.12 if lens else .4,
                    alphaMode='BLEND' if lens else 'OPAQUE',doubleSided=True)
            mesh.apply_transform(M)
            arm_audits.append(refine_temples(mesh,source_vertices,head,lm,profile,ear_anchors))
            kept_glasses.append(mesh)
            result.add_geometry(mesh,node_name=f'Glasses_{i}',geom_name=f'Glasses_{i}')
        glass_vertices=np.vstack([m.vertices for m in kept_glasses]);surface=surface_samples(head,glass_vertices)
        hair_distance=None
        if hair:
            hv=np.vstack([m.vertices for m in hair.values()]);dist,_=cKDTree(hv).query(glass_vertices)
            hair_distance={'nearest_vertex_min_mm':float(dist.min()*1000),
                           'glasses_vertices_within_2mm':int((dist<.002).sum()),
                           'scope':'Vertex proximity warning only; triangle intersections may still occur'}
        result.apply_transform(np.linalg.inv(AXES));target=folder/'scene.glb';result.export(target)
        accessories=trimesh.Scene()
        for i,m in enumerate(kept_glasses):accessories.add_geometry(m.copy(include_cache=True),node_name=f'Glasses_{i}',geom_name=f'Glasses_{i}')
        for i,m in enumerate(hair.values()):accessories.add_geometry(m.copy(include_cache=True),node_name=f'Hair_{i}',geom_name=f'Hair_{i}')
        accessories.apply_transform(np.linalg.inv(AXES));accessories.export(folder/'accessories.glb')
        check=world_parts(target)['Personalized_Complete_Head']
        unchanged=np.allclose(check.vertices,head.vertices,atol=1e-7,rtol=0) and np.array_equal(check.faces,head.faces)
        uv_same=np.array_equal(check.visual.uv.astype(np.float32),head.visual.uv.astype(np.float32))
        texture_same=material_digest(check)==material_digest(head)
        if not (unchanged and uv_same and texture_same):raise ValueError('Accessory fitting changed the personalized head')
        report={'case':case_id,'headwear':headwear,'hair_asset':hair_id,'glasses_asset':glasses_id,
                'status':'LOCAL_REVIEW_REQUIRED','fit':fit,'matrix_source_working_to_world':M.tolist(),
                'head_geometry_unchanged_after_export':unchanged,'head_uv_unchanged':uv_same,'head_texture_unchanged':texture_same,
                'glasses_head_sampled_clearance':surface,'glasses_hair_proximity':hair_distance,'hair_fit':hair_audit,
                'frame_width_adjustment':width_audit,'front_clearance_adjustment':front_audit,'temple_refinement':arm_audits,'glasses_triangle_count':sum(len(m.faces) for m in kept_glasses),
                'hair_triangle_count':sum(len(m.faces) for m in hair.values()),'glb_sha256':sha(target),'glb_bytes':target.stat().st_size,
                'accessories_glb_bytes':(folder/'accessories.glb').stat().st_size,
                'source_asset_sha256':sha(catalog/glasses_id/'asset.glb'),
                'derived_appearance':'glasses-03 charcoal frame with open lenses for fitting preview; originals retained',
                'body_hand_policy':'Original pinching hand with cap; existing complete posed hand/body preset for hair option',
                'ready_for_machine':False,'api_integrated':False,
                'limitations':['Single captured face; generalization requires more real subjects',
                               'Headwear template segmentation is manually annotated; not a general segmenter',
                               'Hair-only option retains graduation hand pose; hand holds empty space',
                               'Hair initialization and approximate clearance are diagnostic, not a collision certificate']}
        write(folder/'report.json',report);cases.append(report);print(case_id,surface,flush=True)
    write(a.output/'report.json',{'policy':'local_accessory_landmark_fit_v1','cases':cases,'elapsed_seconds':time.monotonic()-started,
        'inputs':{'bundle_sha256':sha(bundle),'profiles_sha256':sha(a.profiles),'cap_sha256':sha(a.cap_attempt/'cap-only.glb')},
        'api_configuration_changed':False})

if __name__=='__main__':main()
