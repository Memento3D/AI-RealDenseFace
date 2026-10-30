"""HeadLab V6: classified boundary evidence, automatic local footprint repair,
fast CPU previews, spatial geometry diagnostics and experimental auto supports.
Not a calibrated production system or a raw-photo head reconstruction model.
"""
import argparse,csv,html,json,platform,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from headlab_data import prepare
from headlab_fusion5 import add_back,fused_surface
from headlab_repair6 import repair_surface,evaluate,accepts
from headlab_cache6 import save_prepared,load_prepared
from headlab_raster import surface_depth_fast,render_fast
from headlab_algorithms import SOURCES,taper,fit_offsets,laplacian_edit,arap_edit
from headlab_diagnostics5 import diagnose
from headlab_skin5 import clean_skin_atlas
from headlab_support5 import prepare_supports
from run_headlab_v4 import VARIANTS,Memory,validate,geometry,save_gray,write_json,manufacture_gate
from headlab_printcheck import external_driver
from figurine_geometry import weld,union_print_parts
from figurine_surface31 import painted_eye,photo_eye_direction
from figurine_preview import assemble,accessor
from export_workspace_glb import export
from run_figurine_v3 import digest,export_print_stl
from add_graduation_cap import load


def photo_metrics5(folder,outer,patch,shared,cameras,view_count):
    from scipy import ndimage as ndi
    rows=[]
    for name,camera,mask,depth,image in zip(['front','left','right','back'],cameras,shared['masks'],shared['depths'],shared['images']):
        size=(mask.shape[1],mask.shape[0]);z=surface_depth_fast(outer,patch.faces,camera,size)
        prediction=np.isfinite(z)&((z<=depth+.02*np.linalg.norm(camera[2,:3]))|~np.isfinite(depth))
        union=int((prediction|mask).sum());intersection=int((prediction&mask).sum())
        rgb=np.asarray(image.resize(size)).copy();rgb[mask]=np.uint8(rgb[mask]*.5+np.array([30,210,70])*.5)
        rgb[prediction&~ndi.binary_erosion(prediction)]=[235,65,210];Image.fromarray(rgb).save(folder/(name+'.fit.png'))
        rows.append({'view':name,'iou':intersection/max(union,1),'false_positive_pixels':int((prediction&~mask).sum()),'false_negative_pixels':int((mask&~prediction).sum()),'used_for_fitting':name!='back' or view_count==4})
    return rows


def previews(folder,frame,size):
    for name in ['appearance','geometry']:
        sheet=Image.new('RGB',(size*3,size+22),(23,28,35));draw=ImageDraw.Draw(sheet)
        for index,yaw in enumerate([0,65,180]):
            draw.text((size*index+8,3),str(yaw)+' deg',fill='white')
            sheet.paste(render_fast(folder/(name+'.glb'),yaw,size,frame),(index*size,22))
        sheet.save(folder/(name+'.png'))


def nominated_candidate(rows,front_loss_limit):
    # Screening nomination for downstream experiments, NOT a best/printable claim.
    valid=[r for r in rows if r['status']=='EXPERIMENT_EXPORTED' and r['geometry']['topology_pass']]
    if not valid:return None
    baseline=next(r for r in valid if r['variant']=='baseline')
    base_front=baseline['metrics']['view_ious']['front'];eligible=[]
    for row in valid:
        if row['fit'].get('converged') is False:continue
        if row['metrics']['view_ious']['front']<base_front-front_loss_limit:continue
        eligible.append(row)
    return max(eligible,key=lambda r:r['metrics']['common_three_view_iou']+.5*r['metrics']['view_ious']['back']) if eligible else baseline


def html_report(output,report):
    esc=html.escape;rows=[];cards=[]
    for c in report['candidates']:
        if c['status']=='FAILED':rows.append('<tr><td>'+esc(c['id'])+'</td><td colspan="8">'+esc(c['error'])+'</td></tr>');continue
        metrics=c['metrics'];iou=metrics['view_ious']
        rows.append(f'<tr><td><a href="#{esc(c["id"])}">{esc(c["id"])}</a></td><td>{iou["front"]:.3f}</td><td>{iou["left"]:.3f}</td><td>{iou["right"]:.3f}</td><td>{iou["back"]:.3f}</td><td>{c["timing"]["algorithm_median_seconds"]:.3f}</td><td>{c["fit"].get("controls_at_bounds",0)}</td><td>{c["geometry"]["topology_pass"]}</td><td>UNVERIFIED</td></tr>')
        folder=c['id']+'/'
        cards.append('<section id="'+esc(c['id'])+'"><h2>'+esc(c['id'])+'</h2><img src="'+folder+'geometry.png"><img src="'+folder+'appearance.png"><div>'+''.join('<img width="320" src="'+folder+name+'.fit.png">' for name in ['front','left','right','back'])+'</div><p><a href="'+folder+'head_mm.stl">STL mm</a> · <a href="'+folder+'appearance.glb">GLB meters</a> · <a href="'+folder+'screening_map.glb">Located screening markers</a> · <a href="'+folder+'candidate.json">Metrics</a></p><details><summary>Full report</summary><pre>'+esc(json.dumps(c,indent=2))+'</pre></details></section>')
    supports=''.join('<li>'+esc(x['candidate'])+': <a href="'+x['folder']+'/support_report.json">report</a> · <a href="'+x['folder']+'/assembly_unioned_not_verified_mm.stl">experimental assembly</a></li>' for x in report.get('support_preparation',[]) if x.get('status')=='EXPORTED')
    text='''<!doctype html><meta charset="utf-8"><title>HeadLab V6 integrated comparison</title><style>body{background:#161b22;color:#e3e9ef;font:16px system-ui;margin:24px}a{color:#79baff}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #4b5562}img{max-width:100%;height:auto}section{background:#222c36;padding:16px;margin:20px 0}pre{white-space:pre-wrap}.warning{background:#513b17;padding:16px}</style><h1>HeadLab V6 — V5 baseline versus automatic repair</h1><p class="warning">HEAD ONLY. Rear camera is an approximate fit with pose priors, not independent calibration. Dark-hair rear segmentation has a restricted domain. Supports are experimental, not verified printer layers or a machine file. No physical print certification.</p><p>Selected research exports: <a href="selected/">selected/</a>. This is not a machine-ready print file.</p><h2>Boundary evidence</h2><p>Cyan = detected hair/skin contact. Amber = silhouette. Magenta = ambiguous. Rear contour is not a measured root.</p><img width="512" src="repair/left.edge_types.png"><img width="512" src="repair/back.edge_types.png"><h2>Back observation and registration</h2><img width="640" src="shared/back.registration.png"><p>Cyan: approximate image anchors. Red: fitted projection. Ear-region anchors are not verified anatomical landmarks. Hair-top offset is a nuisance variable, not measured hair thickness. Low reprojection error alone does not establish camera accuracy.</p>'''
    for selection in report.get('selected_outputs',[]):
        prefix='head_'+str(selection['head_neck_mm']).replace('.','p')+'mm'
        text+='<h2>Automatic selection: '+esc(str(selection['head_neck_mm']))+' mm</h2><p>'+esc(selection['candidate'])+' | '+esc(selection['status'])+'</p><img src="'+prefix+'_before_after.png"><p><a href="selected/'+prefix+'/selection.json">Selection and unresolved checks</a> | <a href="selected/'+prefix+'/head_mm.stl">Research STL</a> | <a href="selected/'+prefix+'/appearance.glb">Appearance GLB</a></p>'
    text+='<p>Both groups use all four views. v5 preserves the previous footprint; v6 classifies boundary evidence and tries automatic footprint repairs. All photo scores are training consistency. Appearance cleanup is reported separately and cannot count as a geometry gain.</p><table><tr><th>Candidate</th><th>Front</th><th>Left</th><th>Right</th><th>Back</th><th>Algorithm s</th><th>Bound controls</th><th>Topology</th><th>Machine-ready</th></tr>'+''.join(rows)+'</table><p>Geometry/render/export are timed separately. Images use common framing per height and the vectorized z-buffer. Colored diagnostic points never enter print STL.</p>'+''.join(cards)+'<h2>Automatic support-preparation experiments</h2><ul>'+supports+'</ul><h2>Repair evidence</h2><p><a href="repair/repair.json">Repair trials and acceptance</a> | <a href="repair/repair_map.glb">Removed/added region map</a></p><h2>Batch report</h2><pre>'+esc(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))+'</pre>'
    (output/'compare.html').write_text(text,encoding='utf-8')


def run(workspace,model,back,output,profile,repeats,prepared_run=None):
    validate(profile,repeats)
    fusion=profile['fusion'];support=profile['support_preparation']
    if not (.1<=fusion['back_weight']<=10 and 0<=fusion['boundary_regularization']<=3):raise ValueError('Invalid fusion settings')
    if len(profile['view_weights'])!=4 or not all(np.isfinite(x) and x>0 for x in profile['view_weights']):raise ValueError('Four positive view weights required')
    if not (.08<=support['voxel_pitch_mm']<=.4 and 1<=support['max_supports']<=128 and 0<support['tip_radius_mm']<support['shaft_radius_mm']<1 and support['lift_mm']>1):raise ValueError('Invalid experimental support profile')
    if output.exists():raise ValueError('Output exists; never overwrite previous results')
    if prepared_run is None:
        if not back.is_file():raise ValueError('Back image missing: '+str(back))
        if profile.get('back_sha256') and digest(back)!=profile['back_sha256']:raise ValueError('Back image differs from pinned case profile')
    output.mkdir(parents=True);start=time.perf_counter();write_json(output/'study.used.json',profile);write_json(output/'sources.json',SOURCES)
    native=output/'shared/source_native.glb'
    if prepared_run is None:
        shared=prepare(workspace,model,output/'shared',profile)
        add_back(shared,back,output/'shared')
        tick=time.perf_counter();export(workspace,native,16,True,True,True);bake=time.perf_counter()-tick
        save_prepared(shared,output/'prepared',native,profile,workspace,model,back)
        provenance={'workspace_sha256':digest(workspace),'parser_sha256':digest(model),'back_sha256':digest(back)}
    else:
        shared,meta=load_prepared(prepared_run,profile)
        (output/'shared').mkdir();shutil.copy2(prepared_run/'source_native.glb',native);bake=0
        for name in ['back.registration.png','back.mask.png','back.observation.json']:
            original=prepared_run.parent/'shared'/name
            if original.is_file():shutil.copy2(original,output/'shared'/name)
        provenance={k:meta[k] for k in ['workspace_sha256','parser_sha256','back_sha256']}
    print('V6: measure baseline and run bounded region-repair trials',flush=True)
    original=fused_surface(shared,4,output/'fusion_v5',profile['fusion'])
    repaired=repair_surface(shared,original,output/'repair',profile['repair'])
    groups={'v5':original,'v6':repaired}
    report={'schema_version':1,'version':'HeadLab V6','status':'RUNNING',**provenance,
        'platform':platform.platform(),'python':platform.python_version(),'cache_reused':prepared_run is not None,'back_observation':shared['back_report'],'fusion':{k:v['fusion'] for k,v in groups.items()},
        'shared_prepare_seconds':shared['prepare_seconds'],'shared_bake_seconds':bake,'repeats':repeats,'study_heights_mm':profile['head_neck_heights_mm'],
        'body_and_base_generated':False,'full_figurine_goal_mm':[80,100],'printer_profile':profile['printer'],'candidates':[],'support_preparation':[],
        'ready_for_machine':False,'physical_print_verified':False,'scope':'existing head reconstruction; short attached hair; no general long/tied hairstyle inference; no resin calibration or machine-format slicing'}
    group_natives={}
    for tag,group in groups.items():
        path=output/'shared'/('source_'+tag+'.glb')
        report.setdefault('appearance_cleanup',{})[tag]=clean_skin_atlas(native,path,shared,group)
        group_natives[tag]=path
    machine_profile=output/'printer_profile.json';write_json(machine_profile,profile['printer'])
    source_normals=np.asarray(shared['data']['normals'],float)
    for height in profile['head_neck_heights_mm']:
        factor=height/profile['reference_head_neck_mm'];closed=shared['closed'].copy();closed.vertices*=factor
        all_cameras=[c.copy() for c in shared['cameras']]
        for camera in all_cameras:camera[:,:3]/=factor
        eye_parts=[]
        for index,center in enumerate(shared['eye_centers']*factor):
            eye_points=shared['source_mm'][np.repeat(shared['components']==index+1,3)]*factor;radii=np.ptp(eye_points,axis=0)/2
            direction,_=photo_eye_direction(shared['images'][0],shared['labels'][0],all_cameras[0],center,radii)
            mesh,color=painted_eye(center,radii,profile['eyes'],direction,index);eye_parts.append(('Eye_Custom_'+str(index+1),mesh,color))
        frame=([0,0,0],float(max(closed.extents)+2*profile['fit']['max_offset_mm'])*1.2)
        for tag,group in groups.items():
            patch=group['patch'].copy();patch.vertices*=factor;skin=group['skin'].copy();skin.vertices*=factor
            seam=profile['hair']['seam_mm'];base=profile['hair']['base_mm'];blend=taper(patch,group['boundary'],profile['hair']['edge_taper_mm'])
            fixed=np.unique(group['boundary']);rest=patch.copy();rest.vertices=patch.vertices+group['normals']*(seam+(base-seam)*blend)[:,None]
            settings=profile['fit'].copy();settings['view_weights']=profile['view_weights'][:group['fusion']['view_count']]
            for name,loss,edit in VARIANTS:
                key='h'+str(height).replace('.','p')+'_'+tag+'_'+name;folder=output/key;folder.mkdir();print('V6 candidate:',key,flush=True)
                try:
                    times=[];previous=None;repeat_error=0;fit_report={'converged':None,'controls_at_bounds':0,'method':'constant offset baseline'};edit_report={'method':'none'}
                    with Memory() as memory:
                        for repeat in range(repeats):
                            tick=time.perf_counter()
                            if loss is None:outer=rest.vertices.copy()
                            else:
                                count=group['fusion']['view_count']
                                target,fit_report=fit_offsets(patch,group['normals'],blend,group['known'],all_cameras[:count],shared['masks'][:count],shared['depths'][:count],base,seam,settings,loss)
                                theta=np.asarray(fit_report['theta_mm']);fit_report['controls_at_bounds']=int((np.isclose(theta,settings['min_offset_mm'],atol=1e-4)|np.isclose(theta,settings['max_offset_mm'],atol=1e-4)).sum())
                                if edit=='laplacian':outer,edit_report=laplacian_edit(rest,target,fixed,profile['deformation_weight'])
                                else:outer,edit_report=arap_edit(rest,target,fixed,profile['deformation_weight'],profile['arap_iterations'])
                            times.append(time.perf_counter()-tick)
                            if previous is not None:repeat_error=max(repeat_error,float(np.max(np.linalg.norm(outer-previous,axis=1))))
                            previous=outer.copy()
                        displacement=np.linalg.norm(outer-rest.vertices,axis=1)
                        if displacement.max()>profile['max_hair_change_mm']:raise ValueError('Hair displacement exceeds declared study limit')
                        if not np.allclose(outer[fixed],rest.vertices[fixed],atol=1e-8):raise ValueError('Seam anchors changed')
                        tick=time.perf_counter();hair,exterior=geometry(patch,skin,group['normals'],group['skin_normals'],group['boundary'],group['skin_boundary'],outer,seam)
                        solid=union_print_parts([exterior]+[weld(p.vertices[p.faces].reshape(-1,3)) for _,p,_ in eye_parts],allow_tiny_surface_slivers=True)
                        stl=folder/'head_mm.stl';export_print_stl(solid,stl,closed.bounds[0,1]-.1);save_gray(folder/'geometry.glb',solid)
                        additions=[('Hair_Fused',hair,np.tile([*np.asarray(profile['hair']['rgb'])/255,1],(len(hair.vertices),1)))]+eye_parts
                        positions=shared['source_mm']*factor;assemble(group_natives[tag],folder/'appearance.glb',positions,source_normals,additions,True)
                        doc,blob=load(folder/'appearance.glb');position_error=float(np.max(abs(accessor(doc,blob,0)*1000-positions)))
                        export_seconds=time.perf_counter()-tick
                        tick=time.perf_counter();views=photo_metrics5(folder,outer,patch,shared,all_cameras,group['fusion']['view_count']);metric_seconds=time.perf_counter()-tick
                        checks=diagnose(stl,folder,profile['printer']);driver=external_driver(stl,folder/'manufacturing',machine_profile,profile.get('manufacturing_driver'))
                    ious={v['view']:v['iou'] for v in views}
                    row={'id':key,'status':'EXPERIMENT_EXPORTED','group':tag,'variant':name,'head_neck_height_mm':height,'fit':fit_report,'edit':edit_report,
                        'timing':{'algorithm_seconds':times,'algorithm_median_seconds':float(np.median(times)),'export_seconds':export_seconds,'photo_metric_seconds':metric_seconds,'peak_process_rss_mb':memory.peak/1024**2 if memory.peak else None},
                        'regional_quality':evaluate(dict(group,patch=patch,field=group['field']),dict(shared,cameras=all_cameras),outer=outer),
                        'metrics':{'photo_views':views,'view_ious':ious,'common_three_view_iou':float(np.mean([ious[x] for x in ['front','left','right']])),'displacement_p95_mm':float(np.quantile(displacement,.95)),'displacement_max_mm':float(displacement.max()),'repeat_max_vertex_difference_mm':repeat_error,'source_position_export_max_error_mm':position_error},
                        'geometry':checks,'manufacturing_driver':driver,'ready_for_machine':manufacture_gate(checks,driver),'physical_print_verified':False}
                    tick=time.perf_counter();previews(folder,frame,profile['preview_size']);row['timing']['preview_seconds']=time.perf_counter()-tick
                    write_json(folder/'candidate.json',row);report['candidates'].append(row)
                except Exception as exc:
                    failure={'id':key,'status':'FAILED','error':str(exc),'ready_for_machine':False};write_json(folder/'failure.json',failure);(folder/'traceback.txt').write_text(traceback.format_exc());report['candidates'].append(failure)
                    print('V6 FAILED:',key,str(exc),flush=True)
                report['elapsed_seconds']=time.perf_counter()-start;write_json(output/'batch.json',report);html_report(output,report)
    failures=sum(r['status']=='FAILED' for r in report['candidates'])+sum(r['status']=='FAILED' for r in report['support_preparation']);report['status']='BENCHMARK_COMPLETE' if failures==0 else 'BENCHMARK_PARTIAL_FAILURE'
    report['repair']=repaired['fusion']['repair_v6']
    report['pairwise_review']=[]
    for row in report['candidates']:
        if row.get('group')!='v6' or row['status']!='EXPERIMENT_EXPORTED':continue
        base_row=next((b for b in report['candidates'] if b.get('group')=='v5' and b.get('variant')==row['variant'] and b.get('head_neck_height_mm')==row['head_neck_height_mm'] and b['status']=='EXPERIMENT_EXPORTED'),None)
        if base_row is None:continue
        accepted,reason=accepts(row['regional_quality'],base_row['regional_quality'],profile['repair'])
        if not row['geometry']['topology_pass'] or row['fit'].get('converged') is False:accepted,reason=False,'geometry_or_convergence_gate'
        report['pairwise_review'].append({'candidate':row['id'],'baseline':base_row['id'],'quality_proxy_pass':accepted,'reason':reason,'automatic_output':row['id'] if accepted else base_row['id'],'ready_for_machine':False})
    selections=[]
    for height in profile['head_neck_heights_mm']:
        pool=[]
        for pair in report['pairwise_review']:
            row=next(c for c in report['candidates'] if c['id']==pair['automatic_output'])
            if row['head_neck_height_mm']==height and row['geometry']['topology_pass'] and row['fit'].get('converged') is not False:pool.append(row)
        if not pool:continue
        chosen=min(pool,key=lambda c:c['regional_quality']['score'])
        destination=output/'selected'/('head_'+str(height).replace('.','p')+'mm');destination.mkdir(parents=True)
        for name in ['head_mm.stl','appearance.glb','geometry.glb','appearance.png','geometry.png']:
            shutil.copy2(output/chosen['id']/name,destination/name)
        pose=shared['back_report']['registration']
        unresolved=[]
        if pose.get('hair_top_nuisance_at_bound'):unresolved.append('rear_pose_or_hair_volume_prior_at_bound')
        if min(r['iou'] for r in chosen['regional_quality']['views'])<profile['quality_gate']['min_view_iou']:
            unresolved.append('photo_consistency_below_trial_gate')
        selection={'candidate':chosen['id'],'head_neck_mm':height,'ready_for_machine':False,
            'status':'RESEARCH_EXPORT_INPUT_OR_REPRESENTATION_LIMIT' if unresolved else 'RESEARCH_EXPORT_GEOMETRY_CHECKS_ONLY',
            'unresolved':unresolved,'automatic_mesh_edit_required':False,
            'next_input_action':'recapture a rear view with both ears and nape visible at matching camera height, or use calibrated capture' if unresolved else None}
        if profile['support_preparation']['enabled']:
            try:
                result=prepare_supports(destination/'head_mm.stl',destination/'experimental_supports',profile['support_preparation'])
                report['support_preparation'].append({'candidate':chosen['id'],'folder':str((destination/'experimental_supports').relative_to(output)).replace('\\','/'),'status':'EXPORTED','report':result,'nomination_scope':'selected regional-quality research output, experimental supports only'})
            except Exception as exc:
                report['support_preparation'].append({'candidate':chosen['id'],'status':'FAILED','error':str(exc)})
        write_json(destination/'selection.json',selection);selections.append(selection)
        # Same algorithm/height pair, same framing; upper baseline, lower repair.
        base_id=chosen['id'].replace('_v6_','_v5_')
        new_id=chosen['id'].replace('_v5_','_v6_')
        if (output/base_id/'appearance.png').is_file() and (output/new_id/'appearance.png').is_file():
            images=[Image.open(output/k/'appearance.png').convert('RGB') for k in [base_id,new_id]]
            sheet=Image.new('RGB',(images[0].width,images[0].height*2+44),(23,28,35));draw=ImageDraw.Draw(sheet)
            for j,(image,title) in enumerate(zip(images,['V5 footprint baseline','V6 repair proposal (see acceptance report)'])):
                y=j*(image.height+22);draw.text((8,y+3),title,fill='white');sheet.paste(image,(0,y+22))
            sheet.save(output/('head_'+str(height).replace('.','p')+'mm_before_after.png'))
    report['selected_outputs']=selections
    failures=sum(r['status']=='FAILED' for r in report['candidates'])+sum(r['status']=='FAILED' for r in report['support_preparation'])
    report['status']='BENCHMARK_COMPLETE' if failures==0 else 'BENCHMARK_PARTIAL_FAILURE'
    report['automatic_selection_scope']='per-variant non-regression fallback; proxy quality only, machine readiness remains false'
    report['end_to_end_seconds']=time.perf_counter()-start
    write_json(output/'batch.json',report);html_report(output,report)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['id','status','group','head_neck_mm','algorithm_median_s','front_iou','left_iou','right_iou','back_iou','move_p95_mm','controls_at_bounds','topology_pass','ready_for_machine'])
        for c in report['candidates']:
            if c['status']=='FAILED':writer.writerow([c['id'],'FAILED']+['']*11);continue
            iou=c['metrics']['view_ious'];writer.writerow([c['id'],c['status'],c['group'],c['head_neck_height_mm'],c['timing']['algorithm_median_seconds'],*[iou[x] for x in ['front','left','right','back']],c['metrics']['displacement_p95_mm'],c['fit']['controls_at_bounds'],c['geometry']['topology_pass'],False])
    write_json(output/'checksums.json',{str(p.relative_to(output)).replace('\\','/'):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':report['status'],'candidates':len(report['candidates']),'support_experiments':len(report['support_preparation']),'ready_for_machine':False,'output':str(output)},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,default=ROOT/'configs/headlab04.v6.json');p.add_argument('--workspace',type=Path);p.add_argument('--parser-model',type=Path);p.add_argument('--back-image',type=Path);p.add_argument('--output',type=Path);p.add_argument('--repeats',type=int);p.add_argument('--heights-mm',nargs='+',type=float);p.add_argument('--prepared-run',type=Path)
    a=p.parse_args();profile=json.loads(a.profile.read_text(encoding='utf-8-sig'))
    if a.heights_mm:profile['head_neck_heights_mm']=a.heights_mm
    output=a.output or ROOT/'runs'/('headlab_v6_'+time.strftime('%Y%m%d_%H%M%S'))
    r=run(a.workspace or ROOT/profile['workspace'],a.parser_model or Path(profile['parser_model']),a.back_image or ROOT/profile['back_image'],output,profile,a.repeats if a.repeats is not None else profile['repeats'],a.prepared_run)
    return 0 if r['status']=='BENCHMARK_COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
