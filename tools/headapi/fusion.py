"""MVP assembly: immutable RDF geometry + semantically selected TripoSR hair surface.
No boolean union, print-ready claim or identity guarantee is made.
"""
import argparse,json,hashlib,html
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from .service import sha,write
from .fusion_geometry import raster,sample_surface,similarity,transform_points,preview_assembly
from tools.triposr_probe.audit import load_mesh,render
PARSER_SHA='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
# Stable features spanning upper face, nose and mouth; nonrigid mismatches are robustly downweighted.
CONTROL_IDS=[4,6,33,133,263,362,168,197,195,5,98,327,61,291,78,308,152,234,454,127,356,50,280]


def verify_job(job):
    job=Path(job);r=json.loads((job/'rdf/report.json').read_text());raw=json.loads((job/'raw/report.json').read_text());crop=json.loads((job/'rdf_crops/report.json').read_text())
    if r.get('status')!='FIT_GATES_PASSED' or raw.get('status')!='RAW_PROPOSAL_EXPORTED':raise ValueError('MVP requires completed RDF and TripoSR components')
    if sha(job/'rdf/projection_bundle.npz')!=r['projection_bundle_sha256']:raise ValueError('RDF bundle hash mismatch')
    if sha(job/'rdf_crops/report.json')!=r['crop_report_sha256']:raise ValueError('RDF crop report hash mismatch')
    for slot in r['views_used']:
        if sha(job/'rdf_crops'/(slot+'.png'))!=crop['views'][slot]['crop_sha256']:raise ValueError('Crop changed: '+slot)
    for name in ['raw_native.glb','model_input.png']:
        if sha(job/'raw'/name)!=raw['output_sha256'][name]:raise ValueError('Raw artifact changed: '+name)
    # Bind both branches to the same normalized front upload, not merely similar geometry.
    observation=json.loads((job/'observation/report.json').read_text())
    if observation['source_sha256']!=crop['views']['front']['original_sha256']:raise ValueError('RDF and TripoSR front source mismatch')
    if sha(job/'observation/head_crop.png')!=raw['source_image_sha256']:raise ValueError('TripoSR observation changed')
    return r,crop


def prepare(job,out):
    job=Path(job);out=Path(out);verify_job(job);out.mkdir(parents=True,exist_ok=False)
    raw=load_mesh(job/'raw/raw_native.glb')
    # This axis prior only chooses a frontal raster. The solved similarity is separate.
    native=raw.vertices.copy();oriented=native[:,[1,2,0]]
    rgb,points,indices=raster(oriented,raw.faces,raw.visual.vertex_colors,512)
    Image.fromarray(rgb).save(out/'registration_front.png')
    np.savez_compressed(out/'registration_surface.npz',points=points)
    write(out/'prepare.json',{'source_mesh_sha256':sha(job/'raw/raw_native.glb'),'render_sha256':sha(out/'registration_front.png'),'rdf_bundle_sha256':sha(job/'rdf/projection_bundle.npz')})


class Parser:
    def __init__(self,path):
        if sha(path)!=PARSER_SHA:raise ValueError('Face parser pin mismatch')
        import onnxruntime as ort
        opt=ort.SessionOptions();opt.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
    def __call__(self,image):
        image=Image.fromarray(np.asarray(image,dtype=np.uint8)) if not isinstance(image,Image.Image) else image
        a=np.asarray(image.convert('RGB').resize((512,512),Image.Resampling.BILINEAR),np.float32)/255
        a=((a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        logits=self.session.run(None,{self.session.get_inputs()[0].name:np.ascontiguousarray(a)})[0]
        if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():raise ValueError('Unexpected parser output')
        return np.asarray(Image.fromarray(logits[0].argmax(0).astype('uint8')).resize(image.size,Image.Resampling.NEAREST))


def hair_surface(raw,parser,out,growth_floor,crown_floor):
    oriented=raw.vertices[:,[1,2,0]];n=len(raw.faces);hair=np.zeros(n);total=np.zeros(n);protected=np.zeros(n,bool)
    for angle in [0,90,180,270]:
        rgb,_,ids=raster(oriented,raw.faces,raw.visual.vertex_colors,384,angle=angle)
        label=parser(rgb);valid=ids>=0
        # A parser which sees no hair on a profile is not a reliable negative vote.
        if (label==17).sum()>200:
            total+=np.bincount(ids[valid],minlength=n)
            hair+=np.bincount(ids[valid&(label==17)],minlength=n)
        if angle==0:
            face_votes=np.bincount(ids[valid&np.isin(label,list(range(1,14)))],minlength=n)
            visible=np.bincount(ids[valid],minlength=n)
            protected=(face_votes/np.maximum(visible,1))>.45
        Image.fromarray(label).save(out/('hair_labels_'+str(angle)+'.png'))
    mask=(hair>=2)&(hair/np.maximum(total,1)>.18)
    # Grow from semantic seeds only through dark, adjacent upper-head faces.
    face_rgb=np.asarray(raw.visual.vertex_colors)[:,:3][raw.faces].mean(1)
    seed_y=oriented[raw.faces][mask].mean(1)[:,1]
    if len(seed_y):
        limit=np.quantile(face_rgb[mask].mean(1),.90)+35
        eligible=(face_rgb.mean(1)<min(limit,135))&(oriented[raw.faces].mean(1)[:,1]>growth_floor)&~protected
        for _ in range(32):
            adj=raw.face_adjacency;neighbors=np.zeros(n,bool);neighbors[adj[mask[adj[:,0]],1]]=True;neighbors[adj[mask[adj[:,1]],0]]=True
            expanded=mask|(neighbors&eligible)
            if np.array_equal(expanded,mask):break
            mask=expanded
    if mask.sum()<50:raise ValueError('MVP_HAIR_UNRESOLVED: insufficient semantic hair surface')
    # MVP short-hair crown completion: the generated roof belongs to an inferred
    # cap only after confident hair seeds exist. Preserve frontal observed skin.
    mask |= oriented[raw.faces].mean(1)[:,1]>crown_floor
    mask &= ~protected
    selected=raw.submesh([np.flatnonzero(mask)],append=True,repair=False)
    # Discard isolated parsing specks; keep large components without assuming one hairstyle topology.
    pieces=selected.split(only_watertight=False);largest=max(p.area for p in pieces)
    selected=trimesh.util.concatenate([p for p in pieces if p.area>=largest*.03])
    return selected,{'method':'four orthographic synthetic-view parser votes; largest-relative components retained',
                     'parser_sha256':PARSER_SHA,'selected_triangles':len(selected.faces),
                     'crown_completion_inferred':True,'crown_scope':'generated upper-head geometry above eye-relative threshold, conditioned on hair seeds; not observed hair ground truth','frontal_semantic_face_protected':True,'growth_scope':'upper head only, below-eye floor from landmarks; lower existing semantic seeds retained','ground_truth_verified':False,'hidden_hair_shape_inferred_by_triposr':True,'surface_only':True}


def head_colors(vertices,faces,views,cameras,slots,job,parser):
    labels=[];images=[];candidates=[]
    for slot in slots:
        rgb=np.asarray(Image.open(job/'rdf_crops'/(slot+'.png')).convert('RGB'));label=parser(rgb);labels.append(label);images.append(rgb)
        skin=ndimage.binary_erosion(label==1,iterations=4);pixels=rgb[skin]
        if len(pixels)>100:
            lum=pixels@np.array([.2126,.7152,.0722]);lo,hi=np.quantile(lum,[.65,.90]);eligible=pixels[(lum>=lo)&(lum<=hi)]
            if len(eligible):candidates.append((float(np.median(eligible@np.array([.2126,.7152,.0722]))),slot,np.median(eligible,axis=0)))
    if not candidates:raise ValueError('MVP_SKIN_REFERENCE_UNRESOLVED')
    _,slot,base=max(candidates,key=lambda x:x[0]);colors=np.tile(np.r_[base,255],(len(vertices),1)).astype('uint8');best=np.full(len(vertices),-np.inf)
    for i,rgb in enumerate(images):
        q=np.column_stack((views[i],np.ones(len(vertices))))@cameras[i].T;positive=q[:,2]>1e-8;xy=np.zeros((len(vertices),2),int);xy[positive]=np.rint(q[positive,:2]/q[positive,2,None]).astype(int)
        valid=positive&(xy[:,0]>=0)&(xy[:,0]<rgb.shape[1])&(xy[:,1]>=0)&(xy[:,1]<rgb.shape[0]);ids=np.flatnonzero(valid)
        pixel_labels=np.zeros(len(vertices),int);pixel_labels[ids]=labels[i][xy[ids,1],xy[ids,0]]
        camera_center=-np.linalg.solve(cameras[i,:,:3],cameras[i,:,3]);direction=camera_center-views[i];direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
        normals=trimesh.Trimesh(views[i],faces,process=False).vertex_normals;score=(normals*direction).sum(1)
        # Keep brows, eyes and lips; do not paint hair or clothing onto the scalp.
        valid&=np.isin(pixel_labels,[2,3,4,5,11,12,13])&(score>.2)&(score>best)
        colors[valid,:3]=rgb[xy[valid,1],xy[valid,0]];best[valid]=score[valid]
    return colors,{'base_rgb':base.tolist(),'source_view':slot,'method':'bright non-extreme eroded semantic skin band, best-lit view; approximate appearance, not recovered albedo','feature_regions':'brows, eyes, mouth/lips projected; no reconstructed gaze','color_truth_verified':False}


def assemble(job,out,parser_model):
    job=Path(job);out=Path(out);rdf,crops=verify_job(job)
    prepared=json.loads((out/'prepare.json').read_text());landmarks=json.loads((out/'raw_landmarks.json').read_text())
    if prepared['source_mesh_sha256']!=sha(job/'raw/raw_native.glb') or prepared['rdf_bundle_sha256']!=sha(job/'rdf/projection_bundle.npz'):raise ValueError('Assembly input changed')
    if landmarks['image_sha256']!=sha(out/'registration_front.png'):raise ValueError('Landmarks do not belong to render')
    with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as b:v=b['vertices'].copy();faces=b['faces'].copy();views=b['view_vertices'];cameras=b['cameras']
    before=hashlib.sha256(v.tobytes()+faces.tobytes()).hexdigest()
    with np.load(out/'registration_surface.npz',allow_pickle=False) as b:raw_points=b['points']
    lm=np.asarray(landmarks['landmarks'])[CONTROL_IDS];source,valid1=sample_surface(raw_points,lm)
    _,target_points,_=raster(v,faces,np.tile([180,180,180],(len(v),1)),1024,camera=cameras[0])
    target,valid2=sample_surface(target_points,np.asarray(crops['views']['front']['landmarks_crop'])[CONTROL_IDS])
    valid=valid1&valid2;matrix,errors=similarity(source[valid],target[valid]);height=float(np.ptp(v[:,1]));p95=float(np.quantile(errors,.95)/height)
    if p95>.15:raise ValueError('MVP_ALIGNMENT_REJECTED: normalized control residual exceeds provisional .15 head-height gate')
    raw=load_mesh(job/'raw/raw_native.glb');parser=Parser(parser_model);eye_points,eye_valid=sample_surface(raw_points,np.asarray(landmarks['landmarks'])[[33,133,263,362]])
    if not eye_valid.all():raise ValueError('MVP eye controls unresolved')
    eye_distance=np.linalg.norm(eye_points[:2].mean(0)-eye_points[2:].mean(0))
    growth_floor=float(eye_points[:,1].mean()-eye_distance*1.2)
    hair,hair_info=hair_surface(raw,parser,out,growth_floor,float(eye_points[:,1].mean()+eye_distance*.60))
    hair.vertices=transform_points(hair.vertices[:,[1,2,0]],matrix)
    # Match the scalp on the HAIR side only. Local closest-surface normals are
    # an MVP clearance approximation, not a certified signed distance field.
    scalp=max(trimesh.Trimesh(v,faces,process=False).split(only_watertight=False),key=lambda m:len(m.faces)).convex_hull
    original_hair=hair.vertices.copy()
    for _ in range(2):
        closest,distance,triangle=trimesh.proximity.closest_point(scalp,hair.vertices)
        normal=scalp.face_normals[triangle];signed=((hair.vertices-closest)*normal).sum(1)
        move=signed<height*.008
        adjusted=hair.vertices.copy();adjusted[move]=closest[move]+normal[move]*height*.008;hair.vertices=adjusted
    displacement=np.linalg.norm(hair.vertices-original_hair,axis=1)
    hair_info['scalp_clearance']={'method':'two convex-envelope closest-surface normal projections on hair only','requested_offset_fraction_head_height':.008,'moved_vertices':int((displacement>1e-8).sum()),'p95_displacement_fraction_head_height':float(np.quantile(displacement,.95)/height),'collision_free_verified':False}
    hair.visual.vertex_colors=np.tile(np.r_[np.median(np.asarray(hair.visual.vertex_colors)[:,:3],axis=0),255],(len(hair.vertices),1)).astype('uint8')
    # Matrix is fitted only on facial controls; RDF vertices are never moved.
    colors,skin_info=head_colors(v,faces,views,cameras,rdf['views_used'],job,parser)
    head=trimesh.Trimesh(v,faces,process=False,vertex_colors=colors)
    scene=trimesh.Scene();scene.add_geometry(head,geom_name='RDF_Head_Preserved',node_name='RDF_Head_Preserved');scene.add_geometry(hair,geom_name='TripoSR_Hair_Surface',node_name='TripoSR_Hair_Surface');scene.export(out/'mvp_head.glb')
    hair.export(out/'hair_aligned.glb')
    display=trimesh.util.concatenate([head,hair])
    preview_assembly(display,out/'mvp_color.png');preview_assembly(display,out/'mvp_gray.png',gray=True)
    after=hashlib.sha256(v.tobytes()+faces.tobytes()).hexdigest()
    report={'status':'MVP_ASSEMBLED','hair_isolated':True,'hair_segmentation_verified':False,'hybrid_assembly_performed':True,'geometry_union_performed':False,'ready_for_machine':False,'face_identity_verified':False,
            'rdf_geometry_unchanged':before==after,'rdf_geometry_sha256':before,'units':'RDF native units; physical scale unverified',
            'alignment':{'raw_axis_prior':[1,2,0],'oriented_raw_to_rdf':matrix.tolist(),'control_pairs':int(valid.sum()),'residual_median_native':float(np.median(errors)),'residual_p95_fraction_head_height':p95,'independent_registration_validation':False},
            'hair':hair_info,'skin':skin_info,'source':{'raw_mesh_sha256':sha(job/'raw/raw_native.glb'),'rdf_bundle_sha256':sha(job/'rdf/projection_bundle.npz'),'parser_sha256':sha(parser_model)},
            'limitations':['hair is an open surface component; not a manufacturing solid','overlap/gap at scalp not independently verified','no Boolean union or print calibration','hair back shape inferred from one photo; side photos used by RDF only']}
    report['output_sha256']={name:sha(out/name) for name in ['mvp_head.glb','hair_aligned.glb','mvp_color.png','mvp_gray.png']};write(out/'assembly.report.json',report)
    (out/'review.html').write_text('<meta charset="utf-8"><title>MVP head</title><style>body{background:#171c24;color:white;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#acf}</style><h1>MVP assembled: RDF head + TripoSR hair</h1><p>Two aligned components in one GLB. Surface assembly, not a watertight print mesh.</p><img src="mvp_color.png"><img src="mvp_gray.png"><p><a href="mvp_head.glb">mvp_head.glb</a> · <a href="assembly.report.json">Report</a></p><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','assemble'],required=True);p.add_argument('--parser',type=Path);a=p.parse_args()
    if a.phase=='prepare':prepare(a.job,a.output)
    else:assemble(a.job,a.output,a.parser)
