"""One canonical skin base, pixel-level source features, unchanged source geometry.

A stylized design albedo, not recovered calibrated reflectance. No original
whole-triangle patches are restored, so photo shadows cannot leak via skin tiles.
"""
import io,math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from add_graduation_cap import load,save
from export_workspace_glb import export,sample
from figurine_preview import assemble,accessor
from figurine21_appearance import bright_reference
from figurine2_render import srgb_to_linear

FEATURES=(2,3,4,5,6,11,12,13)

def canonical_photo(image,labels,reference):
    source=np.asarray(image,float)/255
    feature=np.isin(labels,FEATURES)
    # Transition stays INSIDE feature regions: adjacent semantic skin is exact base.
    alpha=np.clip(ndi.distance_transform_edt(feature)/1.5,0,1)
    result=reference[None,None,:]*(1-alpha[:,:,None])+source*alpha[:,:,None]
    return np.rint(result*255).astype(np.uint8),alpha

def embed_atlas(doc,blob,atlas):
    buffer=io.BytesIO();Image.fromarray(atlas).save(buffer,format='PNG');raw=buffer.getvalue()
    while len(blob)%4:blob.append(0)
    doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(raw)});blob.extend(raw)
    doc['images'][0]['bufferView']=len(doc['bufferViews'])-1


def canonical_appearance(workspace,face,profile,output):
    original=output/'source_native.glb';export(workspace,original,16,True,True,True)
    reference,patch,report=bright_reference(face['images'][0],face['labels'][0],profile['bright_skin'])
    base=np.rint(reference*255).astype(np.uint8)
    rgb=np.asarray(face['images'][0]).copy();rgb[patch]=np.uint8(rgb[patch]*.5+np.array([30,220,150])*.5)
    Image.fromarray(rgb).save(output/'skin_reference_patch.png')
    prepared=[];alphas=[];pixel_reports=[]
    for i,(im,labels) in enumerate(zip(face['images'],face['labels'])):
        photo,a=canonical_photo(im,labels,reference);prepared.append(photo.astype(float)/255);alphas.append(a)
        pure=~np.isin(labels,FEATURES)
        error=int(np.abs(photo[pure].astype(int)-base.astype(int)).max())
        if error:raise ValueError('Semantic skin base is not uniform')
        pixel_reports.append({'view':i,'pure_base_pixels':int(pure.sum()),'max_base_error_rgb8':error})
        Image.fromarray(photo).save(output/('view_'+str(i)+'.canonical.png'))
    alpha_images=[np.repeat(a[:,:,None],3,axis=2) for a in alphas]
    doc,blob=load(original);view=doc['bufferViews'][doc['images'][0]['bufferView']];off=view.get('byteOffset',0)
    before=np.asarray(Image.open(io.BytesIO(blob[off:off+view['byteLength']])).convert('RGB'))
    atlas=np.tile(base,(before.shape[0],before.shape[1],1));grid=math.ceil(math.sqrt(len(face['components'])));cell=atlas.shape[0]//grid
    yy,xx=np.mgrid[:cell,:cell];b=np.clip((xx.ravel()+.5-2)/(cell-4),0,1);c=np.clip((yy.ravel()+.5-2)/(cell-4),0,1)
    total=np.maximum(b+c,1);b,c=b/total,c/total;bary=np.column_stack((1-b-c,b,c))
    coords=np.asarray(face['data']['coordinates']).reshape(3,-1,3,3)
    weights=np.asarray(face['data']['weights']).reshape(-1,3,3)
    touched=0;pure_pixels=0;eye_pixels=0
    for i,component in enumerate(face['components']):
        row,col=divmod(i,grid);sl=(slice(row*cell,(row+1)*cell),slice(col*cell,(col+1)*cell))
        if component>0:
            atlas[sl]=before[sl];eye_pixels+=cell*cell;continue
        w=bary@weights[i];support=np.maximum(w,0)
        # Prefer the front where it already has valid support, preserving personal feature contours.
        front=support[:,0]>1e-6;support[front,1:]*=.05
        sums=support.sum(1);color=np.zeros((len(bary),3));coverage=np.zeros(len(bary))
        for j in range(3):
            q=bary@coords[j,i];uv=q[:,:2]/q[:,2,None]
            photo=sample(prepared[j],uv)
            a=sample(alpha_images[j],uv)[:,0]
            color+=photo*support[:,j,None];coverage+=a*support[:,j]
        color/=np.maximum(sums[:,None],1e-12);coverage/=np.maximum(sums,1e-12)
        color[sums<1e-6]=reference
        exact=(coverage<1e-7)|(sums<1e-6);color[exact]=reference
        tile=np.rint(np.clip(color,0,1)*255).astype(np.uint8)
        pure_pixels+=int(exact.sum());touched+=int((~exact).sum())
        if exact.any() and np.max(np.abs(tile[exact].astype(int)-base.astype(int)))!=0:raise ValueError('Atlas skin base changed')
        atlas[sl]=tile.reshape(cell,cell,3)
    embed_atlas(doc,blob,atlas)
    for material_index,m in enumerate(doc['materials']):
        m['name']='CanonicalSkin_SourceFeatures' if m is doc['materials'][0] else m.get('name','SourceEye')
        if material_index==0:m.get('extensions',{}).pop('KHR_materials_unlit',None)
        else:m.setdefault('extensions',{})['KHR_materials_unlit']={}
        m['pbrMetallicRoughness'].update(metallicFactor=0,roughnessFactor=.85)
    canonical=output/'canonical_native.glb';save(canonical,doc,blob)
    # Explicit diagnostic material; same atlas, only the lighting model changes.
    for m in doc['materials']:m.setdefault('extensions',{})['KHR_materials_unlit']={}
    unlit=output/'canonical_unlit_native.glb';save(unlit,doc,blob)
    normals=np.asarray(face['data']['normals'])
    for src,name in [(original,'source_reference'),(canonical,'canonical_reference'),(unlit,'canonical_unlit')]:
        assemble(src,output/(name+'.glb'),face['points'],normals,[],False)
    report.update(method='single design skin base across face/ears/neck; per-pixel source feature mask',
        skin_L_lift=0,source_features_retained=True,whole_triangle_feature_restore=False,
        source_geometry_modified=False,skin_base_rgb8=base.tolist(),photos=pixel_reports,
        atlas_pure_base_pixels=pure_pixels,atlas_feature_pixels=touched,source_eye_atlas_pixels=eye_pixels,source_eye_material_unlit=True,
        atlas_pure_base_max_error_rgb8=0,skin_uniform_in_material=True,feature_boundary_policy='feather inside semantic feature only',
        shading='PBR lighting is separate; canonical_unlit.glb is the base-color diagnostic',
        intrinsic_albedo_recovered=False,skin_color_truth_verified=False)
    face['skin']=reference;face['report']['skin']=report
    return canonical,unlit,report


def assemble_appearance(source,destination,face,hair,hair_rgb):
    color=np.r_[srgb_to_linear(np.asarray(hair_rgb)/255),1.]
    assemble(source,destination,face['points'],np.asarray(face['data']['normals']),
             [('Fitted_Hair',hair,np.tile(color,(len(hair.vertices),1)))],False)
    doc,blob=load(destination);error=float(np.max(abs(accessor(doc,blob,0)*1000-face['points'])))
    if error>2e-5:raise ValueError('Source face or eye positions changed')
    doc['asset']['extras']={'pipeline':'Memento3D V2.2','source_face_export_error_mm':error,'skin':'canonical design base; source feature masks'}
    temporary=destination.with_suffix('.final.glb');save(temporary,doc,blob);temporary.replace(destination)
    return error
