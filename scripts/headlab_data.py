"""Shared photo preparation for a head-only controlled algorithm experiment."""
import base64,hashlib,io,json,re,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from eye_components import identify
from run_figurine_v3 import digest
from figurine_geometry import weld,close_boundaries,recover_projector
from figurine_hairline32 import (refine_source_surface,collect_observations,
    propagate_unknown,stabilize_labels,clip_surface)
from im2_diagnostics import surface_depth
from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256,crop_box,preprocess,restore_crop


def prepare(workspace,model,output,profile):
    started=time.perf_counter();output.mkdir()
    if digest(model)!=MODEL_SHA256:raise ValueError('Pinned parser checksum differs')
    match=re.search(r'<script id="data" type="application/json">(.*?)</script>',workspace.read_text(encoding='utf-8-sig'),re.S)
    if match is None:raise ValueError('Workspace has no embedded head data')
    data=json.loads(match.group(1));vertices=np.asarray(data['vertices'],float)
    coordinates=np.asarray(data['coordinates'],float).reshape(3,-1,3)
    if not np.isfinite(vertices).all() or not np.isfinite(coordinates).all():raise ValueError('Nonfinite workspace')
    components,eyes=identify(vertices);corners=np.repeat(components==0,3)
    low,high=vertices[corners].min(0),vertices[corners].max(0);center=(low+high)/2
    scale=profile['reference_head_neck_mm']/(high[1]-low[1]);mm=(vertices-center)*scale
    head=weld(mm[corners]);closed,caps=close_boundaries(head)
    eye_centers=(np.asarray(eyes['eye_centers_native'])-center)*scale
    upper=vertices[:,1]>np.asarray(eyes['eye_centers_native'])[:,1].mean()+.02*(high[1]-low[1])
    images=[Image.open(io.BytesIO(base64.b64decode(x.split(',',1)[1]))).convert('RGB') for x in data['images']]
    front_hash=hashlib.sha256(np.asarray(images[0]).tobytes()).hexdigest()
    if profile.get('front_rgb_sha256') and profile['front_rgb_sha256']!=front_hash:raise ValueError('Front photo differs from study profile')
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    transform=np.eye(4);transform[:3,:3]/=scale;transform[:3,3]=center
    cameras=[];masks=[];depths=[];labels=[];camera_reports=[]
    for index,(name,image) in enumerate(zip(['front','left','right'],images)):
        print('Shared parser:',name,flush=True)
        box=crop_box(coordinates[index],image.size)
        result=session.run(None,{session.get_inputs()[0].name:preprocess(image.crop(box))})[0]
        if result.shape!=(1,19,512,512) or not np.isfinite(result).all():raise ValueError('Invalid parser tensor')
        label=restore_crop(result[0].argmax(0).astype(np.uint8),box,image.size)
        small=(512,round(512*image.height/image.width))
        mask=np.asarray(Image.fromarray(np.uint8(label==17)*255).resize(small,Image.Resampling.NEAREST))>0
        mask=ndi.binary_fill_holes(mask)
        dilation=profile['mask_dilation_px']
        if dilation:mask=ndi.binary_dilation(mask,iterations=dilation)
        if mask.sum()<100:raise ValueError('Not enough hair pixels in '+name)
        camera,report=recover_projector(vertices,coordinates[index],image.size,upper)
        camera[0]*=small[0]/image.width;camera[1]*=small[1]/image.height;camera=camera@transform
        depth=surface_depth(closed.vertices,closed.faces,camera,small)
        cameras.append(camera);masks.append(mask);depths.append(depth);labels.append(label);camera_reports.append(report)
        Image.fromarray(np.uint8(mask)*255).save(output/(name+'.mask.png'))
        rgb=np.asarray(image.resize(small)).copy();rgb[mask]=np.uint8(rgb[mask]*.45+np.array([30,210,70])*.55)
        Image.fromarray(rgb).save(output/(name+'.mask_overlay.png'))
    work,closure_vertices=refine_source_surface(closed,len(head.faces))
    observed,known,weights,records=collect_observations(work,cameras,masks,depths,labels)
    fixed=known.copy();values=observed.copy();fixed[closure_vertices]=True;values[closure_vertices]=0
    field=propagate_unknown(work,values,fixed);field[closure_vertices]=0
    field,removed=stabilize_labels(work,field,observed,known)
    patch,normals,patch_known,boundary=clip_surface(work,field,known)
    skin,skin_normals,_,skin_boundary=clip_surface(work,1-field,known)
    np.savez_compressed(output/'shared_arrays.npz',closed_vertices=closed.vertices,closed_faces=closed.faces,
        patch_vertices=patch.vertices,patch_faces=patch.faces,normals=normals,known=patch_known,boundary=boundary,
        skin_vertices=skin.vertices,skin_faces=skin.faces,skin_normals=skin_normals,skin_boundary=skin_boundary,
        cameras=np.asarray(cameras),field=field,source_mm=mm,head_corners=corners)
    provenance={'workspace_sha256':digest(workspace),'parser_sha256':digest(model),'front_rgb_sha256':front_hash,
        'reference_head_neck_mm':profile['reference_head_neck_mm'],'native_to_reference_mm':scale,'source_center_native':center.tolist(),
        'camera_fit':camera_reports,'observed_surface_vertices':int(known.sum()),'unknown_surface_vertices':int((~known).sum()),
        'removed_unsupported_vertices':removed,'synthetic_closure_vertices_excluded':len(closure_vertices),'boundary_caps':caps,
        'scope':'same input photos and recovered cameras for all candidates; not independent calibration; no new head reconstruction'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return dict(data=data,source_mm=mm,components=components,closed=closed,patch=patch,normals=normals,known=patch_known,
        boundary=boundary,skin=skin,skin_normals=skin_normals,skin_boundary=skin_boundary,cameras=cameras,masks=masks,
        depths=depths,labels=labels,records=records,images=images,eye_centers=eye_centers,provenance=provenance,prepare_seconds=time.perf_counter()-started)
