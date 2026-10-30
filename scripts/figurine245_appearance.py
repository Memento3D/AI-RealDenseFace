"""Continuous front-photo eye UVs and bounded eyelid detail on canonical skin.

No geometry change. Eye photo stays unlit by default; specular is an explicit
preview variant, not calibrated albedo recovery or a printable color feature.
"""
import io,math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab,lab2rgb
from add_graduation_cap import load,save
from export_workspace_glb import export,sample
from figurine_preview import assemble,accessor
from head245_skin import skin_reference
from figurine22_appearance import embed_atlas,assemble_appearance as assemble_previous

FEATURES=(2,3,11,12,13)

def prepared_front(image,labels,reference,settings,observation=None,mouth_fallback=False):
    rgb=np.asarray(image,float)/255;labels=np.asarray(labels).copy()
    brow_fallback=False
    if observation is not None and (np.count_nonzero(labels==2)<30 or np.count_nonzero(labels==3)<30):
        from PIL import ImageDraw
        for label_id,indices in [(2,[70,63,105,66,107,55,65,52,53,46]),(3,[336,296,334,293,300,276,283,282,295,285])]:
            if np.count_nonzero(labels==label_id)>=30:continue
            mask=Image.new('L',image.size);ImageDraw.Draw(mask).polygon([tuple(observation[j]) for j in indices],fill=255)
            labels[np.asarray(mask)>0]=label_id;brow_fallback=True
    feature=np.isin(labels,FEATURES);eyes=np.isin(labels,[4,5])
    eye_semantics_available=bool(eyes.sum()>=20)
    glasses=(labels==6)
    if not eye_semantics_available and not (glasses.sum()>=20 and observation is not None):
        raise ValueError('Insufficient visible eye semantics; no image-bound glasses observation')
    source_color=rgb.copy();source_color[np.isin(labels,[2,3])]=np.array([49,39,32])/255.;mouth_support=None;mouth_color_note=None
    if observation is not None and mouth_fallback:
        from head24_features import mouth_region
        mouth_support=mouth_region(observation,image.size)
        ring=ndi.binary_dilation(mouth_support,iterations=12)&~mouth_support&(labels==1)
        if ring.sum()<30:raise ValueError('Insufficient local observed skin for mouth-region normalization')
        local=np.median(rgb[ring],axis=0)
        source_color[mouth_support]=np.clip(rgb[mouth_support]+reference-local,0,1)
        feature=feature|mouth_support
        mouth_color_note={'method':'local observed-skin additive color offset and soft ROI feather; not albedo recovery','source_ring_pixels':int(ring.sum()),'source_local_skin_rgb8':np.rint(local*255).astype(int).tolist()}
    distance=ndi.distance_transform_edt(~eyes) if eye_semantics_available else np.full(labels.shape,np.inf)
    support=(labels==1)&(distance<settings['radius_px'])
    fade=np.clip(1-distance/settings['radius_px'],0,1);fade=fade*fade*(3-2*fade)*support
    source_L=rgb2lab(rgb)[:,:,0]
    residual=source_L-(.65*ndi.gaussian_filter(source_L,settings['sigma_small_px'])+.35*ndi.gaussian_filter(source_L,settings['sigma_large_px']))
    delta=np.clip(residual,-settings['dark_L_limit'],settings['bright_L_limit'])*fade*settings['strength']
    lab=np.tile(rgb2lab(reference.reshape(1,1,3)),(image.height,image.width,1));lab[:,:,0]+=delta
    base=lab2rgb(lab);alpha=np.clip(ndi.distance_transform_edt(feature)/1.5,0,1)
    if mouth_support is not None:
        # Region support is a placement aid, not a hard painted quadrilateral.
        soft=np.clip(ndi.distance_transform_edt(mouth_support)/12.,0,1)
        soft=soft*soft*(3-2*soft)
        alpha[mouth_support]=soft[mouth_support]
    result=base*(1-alpha[:,:,None])+source_color*alpha[:,:,None]
    image8=np.rint(np.clip(result,0,1)*255).astype(np.uint8)
    active=feature|support;image8[~active]=np.rint(reference*255).astype(np.uint8)
    return image8,active.astype(float),{'detail_pixels':int(support.sum()),'min_added_L':float(delta.min()),'max_added_L':float(delta.max()),
        'scope':'bounded grayscale source residual in semantic-skin band around eyes only; not intrinsic albedo or shadow-free detail',
        'eye_semantics_available':eye_semantics_available,'glasses_pixels':int(glasses.sum()),'hidden_eye_details_recovered':False,'landmark_brow_fallback_used':brow_fallback,'landmark_mouth_region_used':mouth_support is not None,'mouth_color_normalization':mouth_color_note,'outside_feature_band_base_max_error_rgb8':int(abs(image8[~active].astype(int)-np.rint(reference*255).astype(int)).max())}

def append_array(doc,blob,array,kind):
    a=np.asarray(array,dtype='<f4')
    while len(blob)%4:blob.append(0)
    doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':a.nbytes,'target':34962});blob.extend(a.tobytes())
    doc['accessors'].append({'bufferView':len(doc['bufferViews'])-1,'componentType':5126,'count':len(a),'type':kind})
    return len(doc['accessors'])-1

def append_png(doc,blob,image):
    b=io.BytesIO();image.save(b,format='PNG');raw=b.getvalue()
    while len(blob)%4:blob.append(0)
    doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(raw)});blob.extend(raw)
    doc.setdefault('images',[]).append({'bufferView':len(doc['bufferViews'])-1,'mimeType':'image/png'})
    doc.setdefault('textures',[]).append({'source':len(doc['images'])-1,'sampler':0})
    return len(doc['textures'])-1

def continuous_eye_uv(doc,blob,face,output):
    q=np.asarray(face['data']['coordinates'])[0];w,h=face['images'][0].size
    xy=np.c_[q[:,0]/q[:,2]*w,(1-q[:,1]/q[:,2])*h]
    reports=[]
    for component in [1,2]:
        corners=np.repeat(face['components']==component,3);points=xy[corners]
        lo=np.floor(points.min(0)-[6,6]).astype(int);hi=np.ceil(points.max(0)+[6,6]).astype(int)
        if (lo<0).any() or (hi>np.array([w,h])).any():raise ValueError('Eye projection leaves front image')
        crop=face['images'][0].crop((*lo,*hi));crop.save(output/('eye_'+str(component)+'.source.png'))
        uv=np.zeros((len(q),2));uv[corners]=(xy[corners]-lo)/(hi-lo)
        # Duplicated triangle corners sharing a source vertex must share UV exactly.
        _,groups=np.unique(np.round(face['points'][corners],9),axis=0,return_inverse=True)
        lower=np.full((groups.max()+1,2),np.inf);upper=np.full_like(lower,-np.inf)
        np.minimum.at(lower,groups,uv[corners]);np.maximum.at(upper,groups,uv[corners])
        seam=float(np.max(upper-lower))
        if seam>2e-5:raise ValueError('Front projection is discontinuous at welded eye vertices')
        idx=append_array(doc,blob,uv,'VEC2');texture=append_png(doc,blob,crop)
        count=0
        for node in doc['nodes']:
            if 'Eye_ModelX'+('Negative' if component==1 else 'Positive') in node.get('name',''):
                for p in doc['meshes'][node['mesh']]['primitives']:
                    p['attributes']['TEXCOORD_0']=idx;material=doc['materials'][p['material']]
                    material['pbrMetallicRoughness']['baseColorTexture']={'index':texture}
                    material.setdefault('extensions',{})['KHR_materials_unlit']={};count+=1
        if count!=1:raise ValueError('Expected exactly one known source eye primitive')
        reports.append({'component':component,'crop_xyxy':[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],
            'source_crop_pixels':list(crop.size),'welded_vertex_uv_max_discontinuity':seam,
            'source_policy':'one continuous front-photo projection; no per-triangle visibility switch or cross-view blend',
            'geometry_changed':False,'gaze_estimated':False})
    return reports

def canonical_appearance(workspace,face,profile,output):
    cell=profile['appearance23']['atlas_cell'];original=output/'source_native.glb';export(workspace,original,cell,True,True,True)
    reference,patches,report=skin_reference(face['images'],face['labels'],profile['skin245']);patch=patches[report['selected_index']]
    from head24_features import front_observation
    observation=front_observation(face['data'],face['images'][0])
    base=np.rint(reference*255).astype(np.uint8);photo,active,detail=prepared_front(face['images'][0],face['labels'][0],reference,profile['appearance23']['eyelid'],observation,bool(face['report']['mouth_anchor'].get('semantic_anchor_rejection')))
    Image.fromarray(photo).save(output/'front.canonical_detail.png')
    overlay=np.asarray(face['images'][report['selected_index']]).copy();overlay[patch]=np.uint8(overlay[patch]*.5+np.array([30,220,150])*.5);Image.fromarray(overlay).save(output/'skin_reference_patch.png')
    doc,blob=load(original);count=len(face['components']);grid=math.ceil(math.sqrt(count));side=grid*cell;atlas=np.tile(base,(side,side,1))
    yy,xx=np.mgrid[:cell,:cell];b=np.clip((xx.ravel()+.5-2)/(cell-4),0,1);c=np.clip((yy.ravel()+.5-2)/(cell-4),0,1);total=np.maximum(b+c,1);b,c=b/total,c/total;bary=np.c_[1-b-c,b,c]
    coords=np.asarray(face['data']['coordinates'])[0].reshape(-1,3,3);normals=np.asarray(face['data']['normals']).reshape(-1,3,3)
    prepared=photo.astype(float)/255;active_rgb=np.repeat(active[:,:,None],3,axis=2);pure_count=0
    visible_front=np.asarray(face['data']['weights']).reshape(-1,3,3)[:,:,0].max(1)>1e-8
    for i,component in enumerate(face['components']):
        if component>0:continue
        q=bary@coords[i];uv=q[:,:2]/q[:,2,None];n=bary@normals[i];n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-10)
        front=np.clip((n[:,2]-.2)/.25,0,1)*visible_front[i]
        color=sample(prepared,uv);coverage=sample(active_rgb,uv)[:,0]*front
        color=reference[None,:]+front[:,None]*(color-reference[None,:]);exact=coverage<1e-7;color[exact]=reference
        row,col=divmod(i,grid);atlas[row*cell:(row+1)*cell,col*cell:(col+1)*cell]=np.rint(np.clip(color,0,1)*255).astype(np.uint8).reshape(cell,cell,3)
        pure_count+=int(exact.sum())
    embed_atlas(doc,blob,atlas)
    for i,m in enumerate(doc['materials']):
        if i==0:m['name']='CanonicalSkin_BoundedEyelidDetail';m.get('extensions',{}).pop('KHR_materials_unlit',None)
        m['pbrMetallicRoughness'].update(metallicFactor=0,roughnessFactor=.85)
    from head245_glasses import design_eye_material
    eyes=design_eye_material(doc,blob,face,profile['glasses245']) if profile['glasses245']['enabled'] and (face['labels'][0]==6).sum()>=20 else continuous_eye_uv(doc,blob,face,output)
    canonical=output/'canonical_native.glb';save(canonical,doc,blob)
    for m in doc['materials']:m.setdefault('extensions',{})['KHR_materials_unlit']={}
    unlit=output/'canonical_unlit_native.glb';save(unlit,doc,blob)
    for src,name in [(original,'source_reference'),(canonical,'canonical_reference'),(unlit,'canonical_unlit')]:
        assemble(src,output/(name+'.glb'),face['points'],np.asarray(face['data']['normals']),[],False)
    report.update(method='multiview observed skin; normalized feature masks; explicit eye design where glasses occlude',
        skin_base_rgb8=base.tolist(),atlas_cell=cell,atlas_pure_base_pixels=pure_count,eyelid_detail=detail,eye_projection=eyes,
        source_geometry_modified=False,front_visibility_enforced=True,whole_triangle_feature_restore=False,source_eye_material_unlit=True,eye_under_glasses_is_design_estimate=bool(profile['glasses245']['enabled'] and (face['labels'][0]==6).sum()>=20),
        intrinsic_albedo_recovered=False,skin_color_truth_verified=False)
    face['skin']=reference;face['report']['skin']=report
    return canonical,unlit,report


def assemble_appearance(source,destination,face,hair,hair_rgb):
    error=assemble_previous(source,destination,face,hair,hair_rgb)
    doc,blob=load(destination);doc['asset']['extras']['pipeline']='Memento3D Head24.5'
    doc['asset']['extras']['skin']='canonical base with bounded eyelid detail; continuous source-photo eye UV'
    tmp=destination.with_suffix('.v23.glb');save(tmp,doc,blob);tmp.replace(destination)
    return error


def export_eye_pbr_probe(source,destination,roughness=.16):
    if not .05<=roughness<=.4:raise ValueError('Eye probe roughness outside range')
    doc,blob=load(source)
    for material in doc['materials']:
        if 'Eye' in material.get('name',''):
            material.get('extensions',{}).pop('KHR_materials_unlit',None)
            material['pbrMetallicRoughness']['roughnessFactor']=roughness
    doc['asset'].setdefault('extras',{})['eye_material_scope']='PBR comparison only; source photo still contains lighting'
    save(destination,doc,blob)
