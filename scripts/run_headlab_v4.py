"""One-command head algorithm matrix. Experimental geometry, not a print guarantee."""
import argparse,csv,hashlib,html,json,platform,sys,time,traceback,threading,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from headlab_algorithms import SOURCES,taper,fit_offsets,laplacian_edit,arap_edit
from headlab_data import prepare
from headlab_printcheck import screen,external_driver
from figurine_geometry import weld,union_print_parts
from figurine_surface31 import painted_eye,photo_eye_direction
from figurine_preview import add_mesh,assemble,render
from run_figurine_v3 import export_print_stl,digest
from export_workspace_glb import export
from add_graduation_cap import save
from im2_diagnostics import surface_depth

VARIANTS=[('baseline',None,None),('linear_laplacian','linear','laplacian'),('robust_laplacian','soft_l1','laplacian'),('linear_arap','linear','arap'),('robust_arap','soft_l1','arap')]


class Memory:
    def __init__(self):self.peak=None;self.stop=threading.Event()
    def __enter__(self):
        try:
            import psutil
            process=psutil.Process(os.getpid())
            self.peak=process.memory_info().rss
            def poll():
                while not self.stop.wait(.05):
                    try:self.peak=max(self.peak,process.memory_info().rss)
                    except Exception:self.stop.set()
            self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        except Exception:self.thread=None;self.peak=None
        return self
    def __exit__(self,*args):
        self.stop.set()
        if self.thread:self.thread.join()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def validate(profile,repeats):
    if profile.get('schema_version')!=1:raise ValueError('Unsupported HeadLab profile')
    if not 1<=repeats<=10:raise ValueError('Repeats must be 1..10')
    heights=profile['head_neck_heights_mm']
    if not heights or not all(10<=x<=40 for x in heights):raise ValueError('Study head+neck heights must be 10..40 mm')
    if len(set(heights))!=len(heights):raise ValueError('Duplicate study heights')
    fit=profile['fit']
    if not 4<=fit['controls']<=32 or not 5<=fit['max_nfev']<=200:raise ValueError('Fit budget outside bounds')
    if not 0<fit['min_offset_mm']<profile['hair']['base_mm']<fit['max_offset_mm']<=3:raise ValueError('Invalid hair offset bounds')
    if fit['pixel_scale']<=0 or fit['prior_mm']<=0:raise ValueError('Invalid residual scale')
    if not .02<=profile['hair']['seam_mm']<profile['hair']['base_mm']:raise ValueError('Invalid seam')
    if not 1<=profile['arap_iterations']<=30 or not .001<=profile['deformation_weight']<=10:raise ValueError('Invalid deformation policy')


def geometry(patch,skin,normals,skin_normals,boundary,skin_boundary,outer,seam):
    vertices=np.asarray(patch.vertices);n=len(vertices)
    # Separate hair is diagnostic; full exterior never relies on a fragile inward ribbon.
    inset=np.full(n,.02);inner=vertices-normals*inset[:,None]
    sides=np.asarray([[b,a,a+n] for a,b in boundary]+[[b,a+n,b+n] for a,b in boundary])
    hair=trimesh.Trimesh(np.vstack((outer,inner)),np.vstack((patch.faces,patch.faces[:,::-1]+n,sides)),process=False)
    hair.fix_normals()
    skin_points=np.asarray(skin.vertices).copy();ids=np.unique(skin_boundary)
    skin_points[ids]+=skin_normals[ids]*seam
    triangles=np.vstack((outer[patch.faces],skin_points[skin.faces]))
    p,inv=np.unique(np.round(triangles.reshape(-1,3),8),axis=0,return_inverse=True)
    exterior=trimesh.Trimesh(p,inv.reshape(-1,3),process=False);exterior.fix_normals()
    if not exterior.is_volume:raise ValueError('Hair/skin seam failed exterior topology checks')
    return hair,exterior


def photo_metrics(folder,outer,patch,cameras,masks,depths,images):
    rows=[]
    for name,camera,mask,depth,image in zip(['front','left','right'],cameras,masks,depths,images):
        size=(mask.shape[1],mask.shape[0]);hair_depth=surface_depth(outer,patch.faces,camera,size)
        predicted=np.isfinite(hair_depth)&((hair_depth<=depth+.02*np.linalg.norm(camera[2,:3]))|~np.isfinite(depth))
        intersection=int((predicted&mask).sum());union=int((predicted|mask).sum())
        false_positive=int((predicted&~mask).sum());false_negative=int((mask&~predicted).sum())
        rgb=np.asarray(image.resize(size)).copy();rgb[mask]=np.uint8(rgb[mask]*.5+np.array([35,210,75])*.5)
        edge=predicted&~__import__('scipy').ndimage.binary_erosion(predicted)
        rgb[edge]=[235,65,210];Image.fromarray(rgb).save(folder/(name+'.fit.png'))
        rows.append({'view':name,'visible_hair_mask_iou':intersection/max(union,1),'false_positive_pixels':false_positive,'false_negative_pixels':false_negative,'scope':'training-photo consistency using recovered cameras and detected masks; NOT independent likeness or printability'})
    return rows


def save_gray(path,mesh):
    doc={'asset':{'version':'2.0'},'scenes':[{'nodes':[]}],'scene':0,'nodes':[],'meshes':[],'materials':[],'accessors':[],'bufferViews':[]};blob=bytearray()
    add_mesh(doc,blob,'HeadAlgorithmCandidate',mesh,np.tile([.7,.7,.7,1],(len(mesh.vertices),1)));save(path,doc,blob)


def preview(folder,frame,size):
    for name in ['appearance','geometry']:
        sheet=Image.new('RGB',(size*3,size+22),(24,28,35));draw=ImageDraw.Draw(sheet)
        for j,yaw in enumerate([0,65,180]):
            draw.text((size*j+8,3),str(yaw)+' deg',fill='white')
            sheet.paste(render(folder/(name+'.glb'),yaw,size,frame=frame),(size*j,22))
        sheet.save(folder/(name+'.png'))


def manufacture_gate(checks,driver):
    # Even a successful external driver cannot replace unresolved geometry checks.
    return bool(checks['topology_pass'] and checks['self_intersections_verified'] and checks['minimum_wall_thickness_verified'] and driver.get('ready_for_machine'))


def report_html(output,report):
    esc=html.escape;rows=[];cards=[]
    for item in report['candidates']:
        key=item['id'];folder=key+'/'
        if item['status']=='FAILED':
            rows.append('<tr><td>'+esc(key)+'</td><td colspan="7">FAILED: '+esc(item['error'])+'</td></tr>');continue
        t=item['timing']['algorithm_median_seconds'];iou=item['metrics']['mean_visible_hair_mask_iou']
        rows.append(f'<tr><td><a href="#{esc(key)}">{esc(key)}</a></td><td>{t:.3f}</td><td>{iou:.3f}</td><td>{item["metrics"]["displacement_p95_mm"]:.3f}</td><td>{item["geometry"]["topology_pass"]}</td><td>{item["geometry"]["normal_ray_screen"]["min_sampled_chord_mm"]}</td><td>{item["fit"].get("converged","N/A")}</td><td>NOT VERIFIED</td></tr>')
        cards.append('<section id="'+esc(key)+'"><h2>'+esc(key)+'</h2><p>Same camera framing within each height. Colors require painting. Magenta = projected candidate hair edge, green = detected mask.</p><img src="'+folder+'geometry.png"><img src="'+folder+'appearance.png"><div>'+''.join('<img width="320" src="'+folder+view+'.fit.png">' for view in ['front','left','right'])+'</div><p><a href="'+folder+'head_mm.stl">STL mm</a> · <a href="'+folder+'appearance.glb">GLB meters</a> · <a href="'+folder+'candidate.json">Full metrics</a></p><details><summary>Measurements and limitations</summary><pre>'+esc(json.dumps(item,indent=2))+'</pre></details></section>')
    sources=''.join('<li><a href="'+esc(x['url'],quote=True)+'">'+esc(x['title'])+'</a>: '+esc(x['use'])+'</li>' for x in SOURCES)
    text='''<!doctype html><meta charset="utf-8"><title>Memento HeadLab V4</title><style>body{background:#161b22;color:#e3e9ef;font:16px system-ui;margin:24px}a{color:#79baff}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #48515d}img{max-width:100%;height:auto}section{background:#222b35;padding:18px;margin:22px 0}pre{white-space:pre-wrap} .warning{background:#513b17;padding:16px}</style><h1>HeadLab V4 — controlled algorithm comparison</h1><p class="warning">HEAD ONLY, not a complete 80–100 mm figurine. Study head+neck sizes are hypotheses, not an approved body proportion. No automatic winner. All manufacturing readiness remains unverified until geometry, supports, sliced layers and a calibrated profile are verified. No physical print evidence is included.</p><p>Shared input and cameras; repeated deterministic solves measure timing, not additional subjects. Mask IoU is consistency with fitting photos, not independent likeness. Uniform Laplacian is an adaptation, not full original LSE. ARAP is a local/global implementation. Hair topology remains short attached scalp hair; long/tied styles are not implemented here.</p>'''
    text+='<table><tr><th>Candidate</th><th>Algorithm median s</th><th>Mask IoU</th><th>Move p95 mm</th><th>Topology</th><th>Min sampled chord mm</th><th>Fit converged</th><th>Print ready</th></tr>'+''.join(rows)+'</table><p>Timing excludes shared parsing, appearance baking, export, mesh screening and static previews. See batch.json for end-to-end wall time and per-stage times. Chord samples do not measure true minimum thickness.</p>'+''.join(cards)+'<h2>Sources and implementation scope</h2><ul>'+sources+'</ul><h2>Batch provenance</h2><pre>'+esc(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))+'</pre>'
    (output/'compare.html').write_text(text,encoding='utf-8')


def run(workspace,model,output,profile,repeats,profile_path):
    validate(profile,repeats)
    if output.exists():raise ValueError('Output already exists; choose a new directory')
    output.mkdir(parents=True);start=time.perf_counter()
    write_json(output/'study.used.json',profile);write_json(output/'sources.json',SOURCES)
    machine_path=output/'printer_profile.json';write_json(machine_path,profile['printer'])
    shared=prepare(workspace,model,output/'shared',profile)
    bake_start=time.perf_counter();native=output/'shared/source_native.glb'
    print('Shared appearance bake...',flush=True);export(workspace,native,16,True,True,True)
    bake_seconds=time.perf_counter()-bake_start
    report={'schema_version':1,'status':'RUNNING','scope':'head-only algorithm experiment from existing reconstructed workspace; not a new image-to-head model',
        'provenance':shared['provenance'],'python':platform.python_version(),'platform':platform.platform(),'shared_prepare_seconds':shared['prepare_seconds'],'shared_bake_seconds':bake_seconds,
        'repeats':repeats,'repeat_scope':'deterministic same-case solves, not independent people; sequential fixed order; no claim of cross-platform performance',
        'study_head_neck_heights_mm':profile['head_neck_heights_mm'],'full_figurine_height_goal_mm':[80,100],'body_and_base_generated':False,'printer_profile':profile['printer'],'candidates':[],
        'sources':'sources.json','ready_for_machine':False,'physical_print_verified':False}
    # Bake scale data once, generate the same eyes for every method at each height.
    reference=profile['reference_head_neck_mm'];source_normals=np.asarray(shared['data']['normals'],float)
    for height in profile['head_neck_heights_mm']:
        factor=height/reference;patch=shared['patch'].copy();patch.vertices*=factor;skin=shared['skin'].copy();skin.vertices*=factor
        closed=shared['closed'].copy();closed.vertices*=factor
        cameras=[c.copy() for c in shared['cameras']]
        for c in cameras:c[:,:3]/=factor
        settings=profile['fit'].copy()
        # All variants share identical absolute manufacturing-study parameters.
        seam=profile['hair']['seam_mm'];base=profile['hair']['base_mm']
        blend=taper(patch,shared['boundary'],profile['hair']['edge_taper_mm'])
        fixed=np.unique(shared['boundary']);rest=patch.copy()
        rest.vertices=patch.vertices+shared['normals']*(seam+(base-seam)*blend)[:,None]
        eyes=[]
        for index,center in enumerate(shared['eye_centers']*factor):
            points=shared['source_mm'][np.repeat(shared['components']==index+1,3)]*factor
            radii=np.ptp(points,axis=0)/2
            direction,_=photo_eye_direction(shared['images'][0],shared['labels'][0],cameras[0],center,radii)
            eye,color=painted_eye(center,radii,profile['eyes'],direction,index)
            eyes.append(('Eye_Custom_'+str(index+1),eye,color))
        frame=([0,0,0],float(max(closed.extents)+2*settings['max_offset_mm'])*1.2)
        for name,loss,edit in VARIANTS:
            key=('h'+str(height).replace('.','p')+'_'+name);folder=output/key;folder.mkdir();print('Candidate:',key,flush=True)
            try:
                times=[];max_repeat_difference=0;previous=None;fit_report={'method':'V3.2-style constant offset baseline','converged':None};edit_report={'method':'none'}
                with Memory() as memory:
                    for repeat in range(repeats):
                        tick=time.perf_counter()
                        if loss is None:outer=rest.vertices.copy()
                        else:
                            target,fit_report=fit_offsets(patch,shared['normals'],blend,shared['known'],cameras,shared['masks'],shared['depths'],base,seam,settings,loss)
                            if edit=='laplacian':outer,edit_report=laplacian_edit(rest,target,fixed,profile['deformation_weight'])
                            else:outer,edit_report=arap_edit(rest,target,fixed,profile['deformation_weight'],profile['arap_iterations'])
                        times.append(time.perf_counter()-tick)
                        if previous is not None:max_repeat_difference=max(max_repeat_difference,float(np.max(np.linalg.norm(outer-previous,axis=1))))
                        previous=outer.copy()
                    displacement=np.linalg.norm(outer-rest.vertices,axis=1)
                    if displacement.max()>profile['max_hair_change_mm']:raise ValueError('Hair change exceeds study displacement limit')
                    if not np.allclose(outer[fixed],rest.vertices[fixed],atol=1e-8):raise ValueError('Hairline seam anchors moved')
                    export_start=time.perf_counter()
                    hair,exterior=geometry(patch,skin,shared['normals'],shared['skin_normals'],shared['boundary'],shared['skin_boundary'],outer,seam)
                    solid=union_print_parts([exterior]+[weld(e.vertices[e.faces].reshape(-1,3)) for _,e,_ in eyes],allow_tiny_surface_slivers=True)
                    stl=folder/'head_mm.stl';export_print_stl(solid,stl,closed.bounds[0,1]-.1)
                    save_gray(folder/'geometry.glb',solid)
                    parts=[('Hair_AlgorithmCandidate',hair,np.tile([*np.asarray(profile['hair']['rgb'])/255,1],(len(hair.vertices),1)))]+eyes
                    mm=shared['source_mm']*factor
                    assemble(native,folder/'appearance.glb',mm,source_normals,parts,True)
                    geometry_seconds=time.perf_counter()-export_start
                    photo_start=time.perf_counter();views=photo_metrics(folder,outer,patch,cameras,shared['masks'],shared['depths'],shared['images']);photo_seconds=time.perf_counter()-photo_start
                    checks=screen(stl,profile['printer']);driver=external_driver(stl,folder/'manufacturing',machine_path,profile.get('manufacturing_driver'))
                    ready=manufacture_gate(checks,driver)
                row={'id':key,'status':'EXPERIMENT_EXPORTED','variant':name,'head_neck_height_mm':height,
                    'fit':fit_report,'edit':edit_report,'timing':{'algorithm_seconds':times,'algorithm_median_seconds':float(np.median(times)),'algorithm_min_seconds':float(min(times)),'export_union_seconds':geometry_seconds,'photo_metrics_seconds':photo_seconds,'peak_process_rss_mb':memory.peak/1024**2 if memory.peak is not None else None,'rss_scope':'whole current process during candidate; not exclusive allocation or GPU memory'},
                    'metrics':{'photo_views':views,'mean_visible_hair_mask_iou':float(np.mean([v['visible_hair_mask_iou'] for v in views])),'displacement_max_mm':float(displacement.max()),'displacement_p95_mm':float(np.quantile(displacement,.95)),'fixed_boundary_max_error_mm':float(np.max(abs(outer[fixed]-rest.vertices[fixed]))),'repeat_max_vertex_difference_mm':max_repeat_difference,'source_face_coordinates_changed':False,'scope':'original appearance head unchanged; union eye/hair intersections alter printable exterior'},
                    'geometry':checks,'manufacturing_driver':driver,'ready_for_machine':ready,'physical_print_verified':False,
                    'limitations':['short attached scalp hair only; no automatic long/tied topology','same fitting photos used for consistency metrics','unobserved hairline inherited harmonic interpolation, not detected','normal ray chords can miss thin features','no exact self-intersection or minimum-wall verification','no FEA or physical print evidence']}
                write_json(folder/'candidate.json',row)
                preview_start=time.perf_counter();preview(folder,frame,profile['preview_size']);row['timing']['preview_seconds']=time.perf_counter()-preview_start
                write_json(folder/'candidate.json',row);report['candidates'].append(row)
            except Exception as exc:
                failure={'id':key,'status':'FAILED','error':str(exc),'ready_for_machine':False};write_json(folder/'failure.json',failure)
                (folder/'traceback.txt').write_text(traceback.format_exc());report['candidates'].append(failure);print('Candidate FAILED:',key,str(exc),flush=True)
            report['elapsed_seconds']=time.perf_counter()-start;write_json(output/'batch.json',report);report_html(output,report)
    report['status']='BENCHMARK_COMPLETE' if all(x['status']!='FAILED' for x in report['candidates']) else 'BENCHMARK_PARTIAL_FAILURE'
    report['end_to_end_seconds']=time.perf_counter()-start
    write_json(output/'batch.json',report);report_html(output,report)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['id','status','head_neck_mm','algorithm_median_s','photo_mask_iou','move_p95_mm','topology_pass','min_sampled_chord_mm','ready_for_machine'])
        for row in report['candidates']:
            if row['status']=='FAILED':writer.writerow([row['id'],row['status'],'','','','','','',False]);continue
            writer.writerow([row['id'],row['status'],row['head_neck_height_mm'],row['timing']['algorithm_median_seconds'],row['metrics']['mean_visible_hair_mask_iou'],row['metrics']['displacement_p95_mm'],row['geometry']['topology_pass'],row['geometry']['normal_ray_screen']['min_sampled_chord_mm'],row['ready_for_machine']])
    write_json(output/'checksums.json',{str(p.relative_to(output)).replace('\\','/'):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':report['status'],'candidates':len(report['candidates']),'output':str(output),'ready_for_machine':False},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,default=ROOT/'configs/headlab04.v4.json');parser.add_argument('--workspace',type=Path);parser.add_argument('--parser-model',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--repeats',type=int);parser.add_argument('--heights-mm',nargs='+',type=float)
    args=parser.parse_args();profile=json.loads(args.profile.read_text(encoding='utf-8-sig'))
    if args.heights_mm is not None:profile['head_neck_heights_mm']=args.heights_mm
    workspace=args.workspace or ROOT/profile['workspace'];model=args.parser_model or Path(profile['parser_model'])
    output=args.output or ROOT/'runs'/('headlab_v4_'+time.strftime('%Y%m%d_%H%M%S'))
    result=run(workspace,model,output,profile,args.repeats if args.repeats is not None else profile['repeats'],args.profile)
    return 0 if result['status']=='BENCHMARK_COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
