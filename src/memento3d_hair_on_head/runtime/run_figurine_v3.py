"""One CPU run from an existing workspace: source face + printable stylized details.
Experimental geometry, not a trained model or a verified physical print.
"""
import argparse
import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import trimesh

SUPPORTED_FIGURINE_DESIGN_VERSIONS=('3.0','3.1','3.2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256, crop_box, preprocess, restore_crop
from eye_components import identify
from export_workspace_glb import export
from figurine_geometry import (weld,close_boundaries,recover_projector,grid_for,voxel_solid,
    mesh_from_volume,make_hair,eye_mesh,mouth_anchor,smile_deform,teeth_band,solid_report,union_print_parts)
from figurine_preview import assemble,review
from im2_diagnostics import surface_depth


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def largest(volume):
    labels,n=ndi.label(volume)
    counts=np.bincount(labels.ravel());counts[0]=0
    if n==0:raise ValueError('Empty volume')
    keep=counts.argmax();removed=int(counts.sum()-counts[keep])
    return labels==keep,{'removed_components':n-1,'removed_voxels':removed,'retained_voxels':int(counts[keep])}


def export_print_stl(mesh,path,base_y):
    """STL uses mm with +Z up; GLB stays +Y up in meters."""
    out=mesh.copy();p=np.asarray(out.vertices).copy()
    out.vertices=np.column_stack((p[:,0],-p[:,2],p[:,1]-base_y))
    out.export(path)


def validate_config(c):
    if c.get('schema_version')!=1:raise ValueError('Unsupported config version')
    if not 15<=c['head_neck_height_mm']<=80:raise ValueError('Trial height must be 15..80 mm')
    if not .12<=c['voxel_mm']<=.5:raise ValueError('voxel_mm must be .12.. .5')
    h=c['hair'];e=c['eyes'];s=c['smile']
    if not (0<h['base_mm']<=3 and 0<=h['ridge_mm']<=2 and .2<=h['anchor_mm']<=1.5 and 3<=h['clumps']<=20 and abs(h['part'])<=1 and abs(h['sweep'])<=4):raise ValueError('Hair parameter out of range')
    if not (.25<=e['iris_ratio']<=.9 and .2<=e['pupil_ratio']<=.8 and 0<=e['highlight_ratio']<=.15 and abs(e['gaze_yaw_deg'])<=25 and abs(e['gaze_pitch_deg'])<=20):raise ValueError('Eye parameter out of range')
    if not (.3<=s['gap_mm']<=1.5 and 0<=s['corner_lift_mm']<=.6):raise ValueError('Smile parameter out of range')
    if c.get('design_version') not in (None,'3.0','3.1','3.2'):raise ValueError('Unsupported design version')
    if c.get('design_version') in ('3.1','3.2'):
        if not 0<=h.get('groove_mm',.15)<=h['base_mm']*.5:raise ValueError('Hair groove depth must be <= half of base_mm')
        for field in ['per_eye_yaw_deg','per_eye_pitch_deg']:
            values=e.get(field,[0,0])
            if len(values)!=2 or not all(np.isfinite(x) and abs(x)<=15 for x in values):raise ValueError('Per-eye offsets must have two values within +/-15 degrees')
    for rgb in [h['rgb'],e['sclera_rgb'],e['iris_rgb']]:
        if len(rgb)!=3 or not all(0<=x<=255 for x in rgb):raise ValueError('RGB must be 0..255')


def run(workspace,model,output,config):
    started=time.perf_counter()
    validate_config(config)
    if output.exists():raise ValueError('Output already exists; use a new directory')
    if digest(model)!=MODEL_SHA256:raise ValueError('Parser checksum mismatch; do not use an unknown model')
    data=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',workspace.read_text(encoding='utf-8-sig'),re.S).group(1))
    v=np.asarray(data['vertices'],float);q=np.asarray(data['coordinates'],float).reshape(3,-1,3)
    weights=np.asarray(data['weights'],float).reshape(-1,3)
    if not all(np.isfinite(x).all() for x in [v,q,weights]):raise ValueError('Nonfinite workspace arrays')
    components,eye_info=identify(v);head_corners=np.repeat(components==0,3)
    low,high=v[head_corners].min(0),v[head_corners].max(0);center=(low+high)/2
    scale=config['head_neck_height_mm']/(high[1]-low[1]);mm=(v-center)*scale
    eye_centers=(np.asarray(eye_info['eye_centers_native'])-center)*scale
    eye_y=eye_centers[:,1].mean();upper=v[:,1]>np.asarray(eye_info['eye_centers_native'])[:,1].mean()+.02*(high[1]-low[1])
    images=[Image.open(io.BytesIO(base64.b64decode(x.split(',',1)[1]))).convert('RGB') for x in data['images']]
    rgbhash=hashlib.sha256(np.asarray(images[0]).tobytes()).hexdigest()
    expected=config.get('front_rgb_sha256','')
    if expected and expected!=rgbhash:raise ValueError('Front image pixel checksum differs from the profile')
    output.mkdir(parents=True)
    (output/'config.used.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    cams=[];masks=[];camera_reports=[];labels=[];mask_reports=[]
    transform=np.eye(4);transform[:3,:3]/=scale;transform[:3,3]=center
    for i,(name,image) in enumerate(zip(['front','left','right'],images)):
        print('Parsing photo:',name,flush=True)
        box=crop_box(q[i],image.size)
        pred=session.run(None,{session.get_inputs()[0].name:preprocess(image.crop(box))})[0]
        if pred.shape!=(1,19,512,512) or not np.isfinite(pred).all():raise ValueError('Unexpected parser output')
        label=restore_crop(pred[0].argmax(0).astype(np.uint8),box,image.size);labels.append(label)
        small=(512,round(512*image.height/image.width))
        mask=np.asarray(Image.fromarray((label==17).astype(np.uint8)*255).resize(small,Image.Resampling.NEAREST))>0
        mask=ndi.binary_fill_holes(mask)
        if config['hair']['mask_dilation_px']>0:mask=ndi.binary_dilation(mask,iterations=int(config['hair']['mask_dilation_px']))
        if mask.sum()<100:raise ValueError('Hair mask too small: '+name)
        cam,rep=recover_projector(v,q[i],image.size,upper)
        cam[0]*=small[0]/image.width;cam[1]*=small[1]/image.height
        cams.append(cam@transform);camera_reports.append(rep);masks.append(mask)
        Image.fromarray((mask*255).astype(np.uint8)).save(output/(name+'.hair_mask.png'))
        overlay=np.asarray(image.resize(small)).copy();overlay[mask]=(overlay[mask]*.45+np.array([30,210,70])*.55).astype(np.uint8)
        Image.fromarray(overlay).save(output/(name+'.hair_overlay.png'))
        mask_reports.append({'view':name,'rgb_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),'hair_pixels':int(mask.sum()),'crop':box,'reviewed':False})
    print('Closing source head and building a solid hair design...',flush=True)
    head=weld(mm[head_corners]);closed,caps=close_boundaries(head)
    is31=config.get('design_version') in ('3.1','3.2')
    is32=config.get('design_version')=='3.2'
    eye_reports=[]
    if is31:
        from figurine_surface31 import continuous_hair,photo_eye_direction,painted_eye
        depths=[surface_depth(closed.vertices,closed.faces,cam,(mask.shape[1],mask.shape[0])) for cam,mask in zip(cams,masks)]
        if is32:
            from figurine_hairline32 import detected_hair
            hair,hair_report=detected_hair(closed,cams,masks,depths,labels,images,output,config['hair'],source_face_count=len(head.faces))
        else:
            hair,hair_report=continuous_hair(closed,eye_y,cams,masks,depths,config['hair'])
    else:
        origin,shape=grid_for(closed,config['voxel_mm'],config['hair']['base_mm']+config['hair']['ridge_mm']+1.5)
        headvol=voxel_solid(closed,origin,shape,config['voxel_mm'])
        depths=[surface_depth(closed.vertices,closed.faces,cam,(mask.shape[1],mask.shape[0])) for cam,mask in zip(cams,masks)]
        hairvol,hair_report=make_hair(headvol,origin,config['voxel_mm'],eye_y,closed.bounds,cams,masks,depths,config['hair'])
        before=int(hairvol.sum())
        # Close narrow segmentation notches as a declared design regularizer.
        hairvol |= ndi.binary_closing(hairvol,iterations=4)
        hair_report['notch_closing_added_voxels']=int(hairvol.sum())-before
        hair_report['notch_closing_radius_mm']=config['voxel_mm']*4
        hairvol,clean=largest(hairvol);hair_report['cleanup']=clean
        hair=mesh_from_volume(hairvol,origin,config['voxel_mm'])
    export_print_stl(hair,output/'hair_design_mm.stl',closed.bounds[0,1]-config['voxel_mm'])
    additions=[('Hair_PhotoConstrained',hair,np.tile([*np.asarray(config['hair']['rgb'])/255,1],(len(hair.vertices),1)))]
    eyes=[]
    for i,ec in enumerate(eye_centers,1):
        source_eye=mm[np.repeat(components==i,3)]
        radii=np.ptp(source_eye,axis=0)/2
        if is31:
            direction,er=photo_eye_direction(images[0],labels[0],cams[0],ec,radii)
            mesh,color=painted_eye(ec,radii,config['eyes'],direction,i-1)
            eye_reports.append(dict(eye_index=i,direction_unit=direction.tolist(),**er))
        else:
            mesh,color=eye_mesh(ec,radii,config['eyes'])
        eyes.append(mesh);additions.append(('Eye_Custom_'+str(i),mesh,color))
    anchor,width,mouth=mouth_anchor(v,q[0],labels[0],head_corners,weights[:,0],scale,center)
    native_glb=output/'_native.glb'
    print('Baking existing face colors; no face model retraining...',flush=True)
    export(workspace,native_glb,16,True,True,True)
    source_normals=np.asarray(data['normals'],float)
    assemble(native_glb,output/'source_baseline.glb',mm,source_normals,[])
    variants=[('source_baseline',output/'source_baseline.glb')]
    reports={};union_results={}
    cases=[('source_expression',mm,additions,{'expression_origin':'source workspace; face coordinates unchanged','max_displacement_mm':0})]
    if config['smile']['enabled']:
        posed,expression=smile_deform(mm,anchor,width,config['smile'])
        # Project a shallow continuous colored relief onto the posed lip surface.
        # This avoids loose teeth/fangs and treats teeth as a stylized paint feature.
        from figurine_preview import front_surface_sampler
        surface_z=front_surface_sampler(posed[head_corners])
        teeth,teethcolor=teeth_band(anchor,width,config['smile'],surface_z)
        expression['teeth_design']='continuous shallow painted relief on posed lip surface; not anatomical dental reconstruction'
        cases.append(('smile_custom',posed,additions+[('Teeth_ContinuousBand',teeth,teethcolor)],expression))
    for name,posed,parts,expression in cases:
        print('Exporting',name,flush=True)
        normals=source_normals if name=='source_expression' else weld(posed).vertex_normals[np.unique(posed,axis=0,return_inverse=True)[1]]
        assemble(native_glb,output/(name+'.glb'),posed,normals,parts,True)
        variants.append((name,output/(name+'.glb')))
        h,cap=close_boundaries(weld(posed[head_corners]))
        # Preserve the source head surface; voxelization is restricted to hair design.
        if is32:
            exterior=hair.metadata['print_head_mesh'].copy()
            if name=='smile_custom':exterior.vertices=smile_deform(exterior.vertices,anchor,width,config['smile'])[0]
            mesh=union_print_parts([exterior]+[weld(part.vertices[part.faces].reshape(-1,3)) if part_name.startswith('Eye_') else part for part_name,part,_ in parts if not part_name.startswith('Hair_')],allow_tiny_surface_slivers=True)
        else:
            mesh=union_print_parts([h]+[weld(part.vertices[part.faces].reshape(-1,3)) if part_name.startswith('Eye_') else part for part_name,part,_ in parts])
        report=solid_report(mesh);report.update(expression=expression,boundary_caps=cap)
        if is31:report['surface_processing']['hair_surface']=hair_report['method']
        # Never emit a success label for disconnected or open final geometry.
        if not report['watertight'] or report['connected_components']!=1:
            (output/(name+'.failed_geometry.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
            raise ValueError('Print candidate failed mesh topology checks: '+name)
        export_print_stl(mesh,output/(name+'.print_mm.stl'),closed.bounds[0,1]-config['voxel_mm']);reports[name]=report
        gray=np.tile([.7,.7,.7,1.],(len(mesh.vertices),1))
        # Standalone monochrome GLB exposes actual print geometry, including merged details.
        from add_graduation_cap import save
        from figurine_preview import add_mesh
        doc={'asset':{'version':'2.0'},'scenes':[{'nodes':[]}],'scene':0,'nodes':[],'meshes':[],'materials':[],'accessors':[],'bufferViews':[]};blob=bytearray()
        add_mesh(doc,blob,'PrintCandidate_mm_to_m',mesh,gray);save(output/(name+'.print_shape.glb'),doc,blob)
        union_results[name]=output/(name+'.print_shape.glb')
    if is32:
        support_colors=np.tile([*np.asarray(config['hair']['rgb'])/255,1],(len(hair.vertices),1))
        support_colors[~hair.metadata['observed_support']]=[1.,.48,.08,1.]
        support_parts=[('Hair_ObservedVsInterpolated',hair,support_colors)]+[part for part in additions if not part[0].startswith('Hair_')]
        assemble(native_glb,output/'hair_observation_support.glb',mm,source_normals,support_parts,True)
        variants.append(('hair_observation_support',output/'hair_observation_support.glb'))
    native_glb.unlink();native_glb.with_suffix('.report.json').unlink(missing_ok=True)
    report={'status':'FIGURINE_V32_DETECTED_BOUNDARY_EXPORTED' if is32 else 'FIGURINE_V31_FEASIBILITY_EXPORTED' if is31 else 'FIGURINE_V3_FEASIBILITY_EXPORTED','design_version':config.get('design_version','3.0'),'workspace_sha256':digest(workspace),'parser_sha256':digest(model),'front_rgb_sha256':rgbhash,
        'units':{'glb':'meter, +Y up','stl':'millimeter, +Z up; common base below source neck'},'geometry_export_seconds':round(time.perf_counter()-started,3),'scale':{'source_head_neck_height_mm':config['head_neck_height_mm'],'native_to_mm':float(scale),'source_center_native':center.tolist()},
        'source_face_preserved_in':'source_expression.glb','masks':mask_reports,'camera_fit':camera_reports,'mouth_anchor':mouth,
        'hair':hair_report,'geometry':reports,'eyes':{'photo_candidate_attempted':is31,'gaze_estimated':any(x.get('candidate_pixel') is not None for x in eye_reports),'gaze':'photo candidate or camera-facing fallback plus manual offsets' if is31 else 'config preset','per_eye':eye_reports,'eyelid_geometry':'source preserved','colors':'painting guide'},
        'limitations':['No physical print or slicer validation','Source head reconstruction accuracy unverified','Unobserved scalp labels are interpolated, not detected; no back-image camera fit' if is32 else 'Hair hidden regions use a continuous design prior; no back-image camera fit','Smile is a design preset, not a reconstructed expression','Custom teeth are a shallow painted relief; smile anatomy and source lip texture are not independently verified','Resin prints do not reproduce preview colors','Hairline and eye gaze candidates require visual review; source face uses a surface boolean'],
        'ready_for_production':False}
    report['execution_scope']='Measured CPU process through geometry export; excludes static review rendering; not a Windows/GPU benchmark'
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'paint_guide.json').write_text(json.dumps({'skin_rgb_source':(np.asarray(data['skin'])*255).round().astype(int).tolist(),'hair_rgb':config['hair']['rgb'],'eyes':config['eyes'],'teeth':'continuous ivory band, optional subtle painted dividers','physical_color_match_verified':False},indent=2),encoding='utf-8')
    review(output,variants+[(name+'.print_shape',p) for name,p in union_results.items()],report)
    manifest={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    (output/'checksums.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(output),'production_verified':False},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,default=ROOT/'configs/figurine04.json');p.add_argument('--workspace',type=Path);p.add_argument('--parser-model',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();config=json.loads(a.profile.read_text(encoding='utf-8-sig'))
    workspace=a.workspace or ROOT/config['workspace'];model=a.parser_model or Path(config['parser_model'])
    output=a.output or ROOT/'runs'/(('figurine_v32_' if config.get('design_version')=='3.2' else 'figurine_v31_' if config.get('design_version')=='3.1' else 'figurine_v3_')+time.strftime('%Y%m%d_%H%M%S'))
    try:run(workspace,model,output,config)
    except Exception as exc:
        if output.is_dir():(output/'failure.json').write_text(json.dumps({'status':'FAILED','error':str(exc)},indent=2),encoding='utf-8')
        raise

if __name__=='__main__':main()
