"""Source-identity head, semantic feature paint and restrained expression presets.

No skin albedo or anatomical expression accuracy is claimed from uncalibrated
photos. All styling parameters and source hashes are exported for review.
"""
import base64,io,json,re,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab,deltaE_ciede2000
import trimesh
from eye_components import identify
from figurine_geometry import weld,close_boundaries,recover_projector,mouth_anchor
from figurine_surface31 import painted_eye
from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256,crop_box,preprocess,restore_crop

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def visible_teeth(rgb,labels):
    mouth=(labels==11)
    pixels=np.asarray(rgb,float)[mouth]
    if len(pixels)<20:return False
    if pixels.max()>1:pixels=pixels/255
    white=(pixels.mean(1)>.68)&(np.ptp(pixels,axis=1)<.18)
    return bool(white.sum()>=20 and white.mean()>.18)

def load_face(workspace,model,profile,output):
    raw=Path(workspace).read_text(encoding='utf-8-sig')
    match=re.search(r'<script id="data" type="application/json">(.*?)</script>',raw,re.S)
    if match is None:raise ValueError('Expected head.workspace.html with embedded source data')
    data=json.loads(match.group(1));source=np.asarray(data['vertices'],float)
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all():raise ValueError('Invalid source geometry')
    components,eye_info=identify(source);head_mask=np.repeat(components==0,3)
    lo,hi=source[head_mask].min(0),source[head_mask].max(0);center=(lo+hi)/2
    scale=profile['head_neck_mm']/(hi[1]-lo[1]);points=(source-center)*scale
    head=weld(points[head_mask]);closed,caps=close_boundaries(head)
    eye_centers=(np.asarray(eye_info['eye_centers_native'])-center)*scale
    images=[Image.open(io.BytesIO(base64.b64decode(s.split(',',1)[1]))).convert('RGB') for s in data['images']]
    q=np.asarray(data['coordinates'],float).reshape(3,-1,3)
    from head24_policy import validate_camera_contract,project,InputRejected
    contract=data.get('camera_contract',{})
    if contract.get('version')!=1:raise ValueError('Head24 requires explicit per-view camera contract')
    explicit=np.asarray(contract['cameras'],float)
    view_vertices=np.asarray(contract['view_vertices'],float)
    if explicit.shape!=(3,3,4) or view_vertices.shape!=(3,len(source),3):raise ValueError('Invalid camera contract shape')
    if sha(model)!=MODEL_SHA256:raise ValueError('Parser checksum mismatch')
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    cameras=[];labels=[];masks=[];camera_reports=[]
    transform=np.eye(4);transform[:3,:3]/=scale;transform[:3,3]=center
    upper=source[:,1]>np.asarray(eye_info['eye_centers_native'])[:,1].mean()+.02*(hi[1]-lo[1])
    for i,image in enumerate(images):
        box=crop_box(q[i],image.size);print('V2 semantic preparation:',i,flush=True)
        logits=session.run(None,{session.get_inputs()[0].name:preprocess(image.crop(box))})[0]
        if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():raise ValueError('Invalid parser output')
        label=restore_crop(logits[0].argmax(0).astype('uint8'),box,image.size)
        report=validate_camera_contract(view_vertices[i],q[i],explicit[i],image.size)
        from head24_pose import cranial_camera
        camera,pose_report=cranial_camera(source,view_vertices[i],explicit[i],head_mask,np.asarray(eye_info['eye_centers_native']))
        report['cranial_pose']=pose_report
        report['cranial_pose']['cranial_holdout_p95_mm']=pose_report['cranial_holdout_p95_native']*scale
        camera=camera@transform
        cameras.append(camera);labels.append(label);masks.append(label==17);camera_reports.append(report)
        Image.fromarray(label).save(output/('view_'+str(i)+'.labels.png'))
    # Front visible skin only, robust luminance trimming; not intrinsic albedo.
    rgb=np.asarray(images[0],float)/255;skin_pixels=rgb[ndi.binary_erosion(labels[0]==1,iterations=3)]
    if len(skin_pixels)<100:raise ValueError('Insufficient front skin pixels')
    luminance=skin_pixels@np.array([.2126,.7152,.0722]);a,b=np.quantile(luminance,[.25,.80])
    pixels=skin_pixels[(luminance>=a)&(luminance<=b)];estimate=np.median(pixels,axis=0)
    palette=np.asarray([p['rgb'] for p in profile['skin_palette']],float)/255
    labs=rgb2lab(palette.reshape(1,-1,3)).reshape(-1,3);target=rgb2lab(estimate.reshape(1,1,3))[0,0]
    distance=deltaE_ciede2000(labs,np.tile(target,(len(labs),1)));choice=int(np.argmin(distance))
    if profile.get('skin_id') is not None:
        ids=[p['id'] for p in profile['skin_palette']]
        if profile['skin_id'] not in ids:raise ValueError('Unknown skin palette ID')
        choice=ids.index(profile['skin_id'])
    skin=palette[choice]
    front_weights=np.asarray(data['weights']).reshape(-1,3)[:,0]
    from head24_features import front_observation,mouth_region,observed_mouth_anchor
    observation=front_observation(data,images[0])
    try:
        anchor,width,anchor_report=mouth_anchor(source,q[0],labels[0],head_mask,front_weights,scale,center)
    except ValueError as error:
        if observation is None:raise
        anchor,width,anchor_report=observed_mouth_anchor(source,q[0],mouth_region(observation,images[0].size),head_mask,front_weights,scale,center)
        anchor_report['semantic_anchor_rejection']=str(error)
    # Recover exact original mouth cap loop, distinct from source neck cap.
    mouth_cap=min(caps,key=lambda c:np.linalg.norm(np.asarray(c['center_mm'])-anchor))
    teeth_visible=visible_teeth(np.asarray(images[0]),labels[0])
    report={'workspace_sha256':sha(workspace),'parser_sha256':sha(model),'head_neck_mm':profile['head_neck_mm'],
        'native_to_mm':scale,'source_center_native':center.tolist(),'source_topology':eye_info,
        'camera_fit':camera_reports,'mouth_anchor':anchor_report,'mouth_width_mm':float(width),
        'skin':{'estimated_rgb':np.round(estimate*255).astype(int).tolist(),'palette_id':profile['skin_palette'][choice]['id'],
                'selected_rgb':np.round(skin*255).astype(int).tolist(),'deltaE00_to_photo_estimate':float(distance[choice]),
                'method':'eroded front semantic skin, middle-luminance median, CIELAB/CIEDE2000 palette proposal',
                'intrinsic_albedo_verified':False,'illumination_calibrated':False,'physical_color_match_verified':False},
        'teeth':{'source_visible_proposal':teeth_visible,'mode':profile.get('teeth','auto'),'design':'continuous painted oral-cap region with optional color dividers; not dental geometry'},
        'boundary_caps':caps,'source_face_accuracy_verified':False}
    return dict(data=data,points=points,components=components,head_corners=head_mask,head=closed,source_head=head,
                caps=caps,mouth_cap=mouth_cap,anchor=anchor,width=width,eye_centers=eye_centers,
                cameras=cameras,images=images,labels=labels,masks=masks,skin=skin,teeth_visible=teeth_visible,report=report)

def pose_head(face,expression,profile):
    head=face['head'].copy();p=head.vertices.copy();anchor=face['anchor'];width=face['width']
    if expression=='source':return head,{'origin':'source geometry preserved','max_displacement_mm':0.}
    if expression!='soft_smile':raise ValueError('Unsupported expression')
    x,y,z=(p-anchor).T
    front=np.exp(-np.power(z/max(width*.6,.1),4));side=np.clip(abs(x)/(width*.5),0,1)
    lip=np.exp(-np.power(x/(width*.65),6)-np.power(y/(width*.28),4))*front
    # Small coordinated corner and cheek lift, never separate pasted mouth geometry.
    lift=profile['expression']['corner_lift_mm']*side**2*lip
    cheeks=np.exp(-((abs(x)-width*.65)/(width*.35))**2-((y-width*.24)/(width*.4))**2)*front
    head.vertices[:,1]+=lift+profile['expression']['cheek_lift_mm']*cheeks
    shift=np.linalg.norm(head.vertices-p,axis=1)
    if shift.max()>profile['expression']['max_displacement_mm']:raise ValueError('Expression displacement budget exceeded')
    if not head.is_volume:raise ValueError('Expression broke source topology')
    return head,{'origin':'bounded synthetic corner/cheek displacement preset; not reconstructed smile',
                 'max_displacement_mm':float(shift.max()),'modified_vertices':int((shift>1e-5).sum())}

def painted_head(face,head,profile):
    # Refine exactly for smooth color boundaries, without moving the surface.
    p,f=trimesh.remesh.subdivide(head.vertices,head.faces)
    p,f=trimesh.remesh.subdivide(p,f)
    mesh=trimesh.Trimesh(p,f,process=False)
    colors=np.tile(np.r_[face['skin'],1.],(len(p),1));camera=face['cameras'][0]
    q=np.column_stack((p,np.ones(len(p))))@camera.T;xy=q[:,:2]/q[:,2,None]
    h,w=face['labels'][0].shape;x=np.floor(xy[:,0]).astype(int);y=np.floor(xy[:,1]).astype(int)
    valid=(q[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h)&(mesh.vertex_normals[:,2]>.25)
    lab=np.zeros(len(p),int);ids=np.flatnonzero(valid);lab[ids]=face['labels'][0][y[ids],x[ids]]
    lip=np.asarray(profile['face_colors']['lips'])/255;brow=np.asarray(profile['face_colors']['brows'])/255
    # Anti-aliased feature paint; no blocky per-triangle photo texture/shadows.
    for label_ids,color,strength in [([12,13],lip,.85),([2,3],brow,.95)]:
        field=ndi.gaussian_filter(np.isin(face['labels'][0],label_ids).astype(float),1.1)
        weight=np.zeros(len(p));weight[ids]=ndi.map_coordinates(field,[xy[ids,1],xy[ids,0]],order=1,mode='nearest')
        weight*=strength;colors[:,:3]=colors[:,:3]*(1-weight[:,None])+color*weight[:,None]
    # Inner oral closure is painted dark, not a free-floating tooth strip.
    original=len(face['source_head'].faces);new_caps=np.arange(len(head.faces))>=original
    capids=np.unique(np.asarray(f)[np.repeat(new_caps,16)])
    mouth=np.linalg.norm(p-face['mouth_cap']['center_mm'],axis=1)<max(face['width']*.5,.4)
    oral=np.zeros(len(p),bool);oral[capids]=True;oral&=mouth
    colors[oral,:3]=np.asarray(profile['face_colors']['mouth'])/255
    show=profile.get('teeth','auto')=='show' or (profile.get('teeth','auto')=='auto' and face.get('teeth_visible',False))
    if show:
        center=np.asarray(face['mouth_cap']['center_mm']);width=face['width']
        band=oral&(p[:,1]>=center[1])&(abs(p[:,0]-center[0])<width*.38)
        colors[band,:3]=[.88,.85,.77]
        phase=np.mod(p[:,0]-center[0]+width*.5,width*.20)
        colors[band&(phase<.035),:3]=[.52,.46,.40]
    return mesh,colors

def eyes(face,profile):
    result=[]
    for index,center in enumerate(face['eye_centers']):
        pts=face['points'][np.repeat(face['components']==index+1,3)];radii=np.ptp(pts,axis=0)/2
        radii*=profile['eyes']['size_factor']
        # Shared forward gaze, not two separately guessed pupil directions.
        mesh,color=painted_eye(center,radii,profile['eyes'],np.array([0.,0.,1.]),index)
        result.append(('Eye_'+str(index),mesh,color))
    return result
