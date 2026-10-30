"""Restore source facial appearance; bright skin is a design reference, not albedo.

Keep source eyes, brows and lips rather than repainting generic replacements.
Only bounded skin lightness adjustment and hidden scalp fill are permitted.
"""
import io,math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab,lab2rgb
from add_graduation_cap import load,save
from export_workspace_glb import export
from figurine_preview import assemble,accessor


def bright_reference(image,labels,settings):
    rgb=np.asarray(image,float)/255
    valid=ndi.binary_erosion(labels==1,iterations=3)
    valid&=(rgb.max(2)<settings['max_channel'])&(rgb.min(2)>.04)
    if valid.sum()<100:raise ValueError('Insufficient reliable face-skin pixels')
    L=rgb2lab(rgb)[:,:,0];lo,hi=np.quantile(L[valid],[settings['low_quantile'],settings['high_quantile']])
    bright=valid&(L>=lo)&(L<=hi)
    groups,n=ndi.label(bright);counts=np.bincount(groups.ravel());counts[0]=0
    if n==0 or counts.max()<settings['minimum_patch_pixels']:
        raise ValueError('No sufficiently large bright skin patch; do not guess a skin color')
    patch=groups==counts.argmax();color=np.median(rgb[patch],axis=0)
    median=np.median(rgb[valid],axis=0)
    Lref=float(rgb2lab(color.reshape(1,1,3))[0,0,0]);Lmedian=float(np.median(L[valid]))
    delta=float(np.clip((Lref-Lmedian)*settings['lift_fraction'],0,settings['maximum_L_lift']))
    yy,xx=np.nonzero(patch)
    report={'method':'largest connected bright semantic-skin patch, clipped highlights excluded',
            'reference_rgb':np.rint(color*255).astype(int).tolist(),'reference_rgb_float':color.tolist(),
            'median_skin_rgb_before':np.rint(median*255).astype(int).tolist(),
            'reference_L':Lref,'median_skin_L':Lmedian,'skin_L_lift':delta,
            'patch_pixels':int(patch.sum()),'patch_bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'quantile_range':[settings['low_quantile'],settings['high_quantile']],
            'palette_quantized':False,'illumination_calibrated':False,'true_skin_albedo_verified':False}
    return color,patch,report


def normalize_photo(image,label,reference,settings):
    rgb=np.asarray(image,float)/255;lab=rgb2lab(rgb)
    skin=np.isin(label,[1,7,8,10,14]);features=np.isin(label,[2,3,4,5,6,11,12,13])
    sigma=max(4.,image.width/1280*18.)
    mass=ndi.gaussian_filter(skin.astype(float),sigma)
    illumination=ndi.gaussian_filter(lab[:,:,0]*skin,sigma)/np.maximum(mass,1e-8)
    target=float(rgb2lab(reference.reshape(1,1,3))[0,0,0])
    lift=np.clip(target-illumination,0,settings['maximum_L_lift'])*settings['lift_fraction']
    feather=np.clip(ndi.distance_transform_edt(skin)/max(3.,image.width/1280*8.),0,1)
    lab[:,:,0]=np.clip(lab[:,:,0]+lift*feather,0,100)
    adjusted=np.rint(np.clip(lab2rgb(lab),0,1)*255).astype(np.uint8)
    adjusted[~skin]=np.asarray(image)[~skin]
    # Source hair pigments must not masquerade as geometry on the bald head.
    adjusted[label==17]=np.rint(reference*255).astype(np.uint8)
    adjusted[features]=np.asarray(image)[features]
    if not np.array_equal(adjusted[features],np.asarray(image)[features]):raise ValueError('Source feature pixels changed')
    return Image.fromarray(adjusted),{'method':'smooth masked low-frequency skin-lightness lift',
        'protected_feature_pixels':int(features.sum()),'protected_pixels_identical':True,
        'max_L_lift_applied':float((lift*feather).max()),'no_skin_palette_quantization':True}


def source_appearance(workspace,face,profile,output):
    import base64,copy,json
    original=output/'source_native.glb'
    export(workspace,original,16,True,True,True)
    reference,patch,report=bright_reference(face['images'][0],face['labels'][0],profile['bright_skin'])
    rgb=np.asarray(face['images'][0]).copy();rgb[patch]=np.uint8(rgb[patch]*.5+np.array([30,220,150])*.5)
    Image.fromarray(rgb).save(output/'bright_skin_patch.png')
    data=copy.deepcopy(face['data']);processed=[];normalization=[]
    for image,label in zip(face['images'],face['labels']):
        adjusted,measurement=normalize_photo(image,label,reference,profile['bright_skin'])
        buffer=io.BytesIO();adjusted.save(buffer,format='PNG')
        processed.append('data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))
        normalization.append(measurement)
    data['images']=processed;data['skin']=reference.tolist()
    temporary=output/'_appearance_workspace.html'
    temporary.write_text('<script id="data" type="application/json">'+json.dumps(data,separators=(',',':'))+'</script>',encoding='utf-8')
    adjusted=output/'bright_source_native.glb';export(temporary,adjusted,16,True,True,True);temporary.unlink()
    original_doc,original_blob=load(original);doc,blob=load(adjusted)
    def atlas_from(d,b):
        image=d['images'][0];view=d['bufferViews'][image['bufferView']];off=view.get('byteOffset',0)
        return np.asarray(Image.open(io.BytesIO(b[off:off+view['byteLength']])).convert('RGB')).copy()
    before=atlas_from(original_doc,original_blob);atlas=atlas_from(doc,blob)
    count=len(face['components']);grid=math.ceil(math.sqrt(count));cell=atlas.shape[0]//grid
    q=np.asarray(face['data']['coordinates'])[0].reshape(-1,3,3)
    h,w=face['labels'][0].shape;x=np.floor(q[:,:,0]/q[:,:,2]*w).astype(int);y=np.floor((1-q[:,:,1]/q[:,:,2])*h).astype(int)
    valid=(x>=0)&(x<w)&(y>=0)&(y<h);lab=np.zeros(x.shape,np.uint8);lab[valid]=face['labels'][0][y[valid],x[valid]]
    weights=np.asarray(face['data']['weights']).reshape(-1,3,3).mean(1)
    protected=(face['components']>0)|((weights[:,0]>1e-8)&np.isin(lab,[2,3,4,5,11,12,13]).any(1))
    for i in np.flatnonzero(protected):
        row,column=divmod(int(i),grid)
        atlas[row*cell:(row+1)*cell,column*cell:(column+1)*cell]=before[row*cell:(row+1)*cell,column*cell:(column+1)*cell]
    buffer=io.BytesIO();Image.fromarray(atlas).save(buffer,format='PNG');raw=buffer.getvalue()
    while len(blob)%4:blob.append(0)
    doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(raw)});blob.extend(raw)
    doc['images'][0]['bufferView']=len(doc['bufferViews'])-1
    meta=output/'bright_source_native.final.glb';save(meta,doc,blob);meta.replace(adjusted)
    normals=np.asarray(face['data']['normals'],float)
    assemble(original,output/'source_reference.glb',face['points'],normals,[],False)
    assemble(adjusted,output/'bright_reference.glb',face['points'],normals,[],False)
    report.update(source_features_retained=True,protected_feature_triangles=int(protected.sum()),
                  protected_feature_tiles_byte_identical=True,photo_normalization=normalization,
                  synthetic_brow_or_lip_repaint=False,replacement_iris_or_eyeball=False,
                  expression='source geometry unchanged')
    face['skin']=reference;face['report']['skin']=report
    return adjusted,report


def assemble_appearance(source,destination,face,hair,hair_rgb):
    from figurine2_render import srgb_to_linear
    color=np.r_[srgb_to_linear(np.asarray(hair_rgb)/255),1.]
    assemble(source,destination,face['points'],np.asarray(face['data']['normals']),
             [('Compact_Hair',hair,np.tile(color,(len(hair.vertices),1)))],False)
    doc,blob=load(destination)
    error=float(np.max(abs(accessor(doc,blob,0)*1000-face['points'])))
    doc['asset']['extras']={'length_unit':'meter','pipeline':'Memento3D V2.1','source_face_position_export_error_mm':error,
                            'appearance':'source facial features retained; bounded skin lightness lift'}
    temporary=destination.with_suffix('.metadata.glb')
    save(temporary,doc,blob);temporary.replace(destination)
    if error>2e-5:raise ValueError('Source face/eye geometry changed in appearance export')
    return error
