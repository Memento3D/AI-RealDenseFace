"""Evidence ledger for input, hair, face and manufacturing assessment.

Measured topology, sampled proxies and unverified claims are intentionally
separate. No single weighted score certifies identity or manufacturing.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from PIL import Image
from figurine2_face import sha


def vertex_fans(mesh):
    """Corner adjacency detects pinched vertices missed by edge-only watertightness."""
    faces=np.asarray(mesh.faces);corners=np.arange(faces.size).reshape(-1,3)
    edges=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    endpoint_corners=np.vstack((corners[:,[0,1]],corners[:,[1,2]],corners[:,[2,0]]))
    sort=np.argsort(edges,axis=1);ordered=np.take_along_axis(edges,sort,axis=1)
    endpoint_corners=np.take_along_axis(endpoint_corners,sort,axis=1)
    _,inverse,count=np.unique(ordered,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');start=np.r_[0,np.cumsum(count)[:-1]]
    paired=start[count==2];a=endpoint_corners[order[paired]].ravel();b=endpoint_corners[order[paired+1]].ravel()
    graph=sparse.coo_matrix((np.ones(len(a)*2,dtype=np.uint8),(np.r_[a,b],np.r_[b,a])),shape=(faces.size,faces.size)).tocsr()
    _,labels=connected_components(graph,directed=False)
    unique=np.unique(np.column_stack((faces.ravel(),labels)),axis=0)
    fans=np.bincount(unique[:,0],minlength=len(mesh.vertices))
    bad=np.flatnonzero(fans>1)
    return {'nonmanifold_vertex_count':len(bad),'vertex_ids':bad.tolist(),
            'max_incident_fan_count':int(fans.max()) if len(fans) else 0}


def mesh_audit(mesh,unit_basis):
    edges,count=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    area=np.asarray(mesh.area_faces)
    fans=vertex_fans(mesh);euler=int(len(mesh.vertices)-len(edges)+len(mesh.faces))
    manifold=bool(mesh.is_watertight and not (count>2).any() and fans['nonmanifold_vertex_count']==0 and not (area<1e-12).any())
    return {'vertices':len(mesh.vertices),'triangles':len(mesh.faces),
        'bounds':mesh.bounds.tolist(),'extents':mesh.extents.tolist(),'unit_basis':unit_basis,
        'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
        'positive_volume':bool(mesh.volume>0),'signed_volume':float(mesh.volume),
        'connected_components':len(mesh.split(only_watertight=False)),
        'boundary_edges':int((count==1).sum()),'nonmanifold_edges':int((count>2).sum()),
        'zero_area_faces':int((area<1e-12).sum()),'finite':bool(np.isfinite(mesh.vertices).all()),
        'vertex_link_audit':fans,'euler_characteristic':euler,'closed_surface_manifold':manifold,
        'self_intersections_verified':False,'minimum_wall_thickness_verified':False}


def chord_screen(mesh,samples=128,threshold=.4):
    if not mesh.is_volume:
        return {'status':'NOT_RUN_INVALID_VOLUME','minimum_wall_thickness_verified':False,'samples':[]}
    if not 16<=samples<=256:raise ValueError('Hair chord screen budget must be 16..256')
    # Deterministic area-stratified sample centers; normal-ray chords only.
    ids=np.unique(np.searchsorted(np.cumsum(mesh.area_faces),(np.arange(samples)+.5)/samples*mesh.area))
    start=mesh.triangles_center[ids];direction=-mesh.face_normals[ids]
    eps=max(float(mesh.extents.max())*1e-6,1e-6);origin=start+direction*eps
    points,rayids,faceids=mesh.ray.intersects_location(origin,direction,multiple_hits=True)
    length=np.full(len(ids),np.inf)
    if len(points):
        distance=np.linalg.norm(points-origin[rayids],axis=1)
        valid=(faceids!=ids[rayids])&(distance>eps)
        np.minimum.at(length,rayids[valid],distance[valid]+eps)
    records=[]
    for i,face_id in enumerate(ids):
        value=float(length[i]) if np.isfinite(length[i]) else None
        records.append({'face_id':int(face_id),'point':start[i].tolist(),'inward_normal':direction[i].tolist(),
                        'chord_in_assumed_mm':value,'below_trial_threshold':value is not None and value<threshold})
    good=length[np.isfinite(length)]
    return {'status':'SAMPLED_PROXY_ONLY','method':'area-stratified inward-normal first-exit ray chords',
        'sample_count':len(ids),'unresolved':int((~np.isfinite(length)).sum()),
        'min_sampled_chord_assumed_mm':float(good.min()) if len(good) else None,
        'p05_sampled_chord_assumed_mm':float(np.quantile(good,.05)) if len(good) else None,
        'trial_threshold_mm':threshold,'flagged_samples':sum(r['below_trial_threshold'] for r in records),
        'minimum_wall_thickness_verified':False,'mechanical_strength_verified':False,
        'scope':'can miss thin regions and does not measure exact minimum thickness; threshold not calibrated for Saturn/resin',
        'samples':records}


def audit_assets(assets,root,output,printer):
    output.mkdir();records=[]
    for asset in assets:
        path=Path(root)/asset['path'];mesh=trimesh.load(path,force='mesh',process=True)
        row={'asset_id':asset['id'],'sha256':sha(path),
             'mesh':mesh_audit(mesh,asset['input_units']),
             'thickness_screen':chord_screen(mesh,min(printer['ray_samples'],128),printer['trial_feature_mm']),
             'physical_print_verified':False,'asset_valid_after_all_deformations':False}
        (output/(asset['id']+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8');records.append(row)
    return records


def audit_input(face,profile):
    views=[];failed=[]
    for i,(image,label,camera) in enumerate(zip(face['images'],face['labels'],face['report']['camera_fit'])):
        counts=np.bincount(label.ravel(),minlength=19);hair=int(counts[17]);skin=int(counts[1]);eyes=int(counts[4]+counts[5])
        p95=float(camera['p95_px']);name=['front','left','right'][i]
        if p95>5:failed.append(name+':camera_internal_reprojection_exceeds_5px')
        if i==0 and (skin<100 or eyes<10):failed.append('front:insufficient_semantic_face_support')
        views.append({'view':name,'image_size':list(image.size),'hair_pixels':hair,'skin_pixels':skin,
                      'eye_pixels':eyes,'camera_p95_internal_reprojection_px':p95,
                      'absolute_pose_calibrated':False,'hair_mask_accuracy_verified':False})
    return {'status':'MEASURED_FAILURE' if failed else 'INTERNAL_CHECKS_PASS',
        'blocking_reasons':failed,'views':views,'source_head_volume_valid':bool(face['head'].is_volume),
        'source_head_accuracy_verified':False,'same_hairstyle_across_views_verified':False,
        'expression_consistency_verified':False,'absolute_camera_calibration_verified':False,
        'posterior_photo_used':False,'posterior_policy':'template design prior, not reconstructed or verified nape',
        'threshold_basis':'provisional engineering guards; not a calibrated population standard'}


def candidate_ledger(row):
    if row['status']=='FAILED':
        return {'status':'REJECTED','blocking_reasons':[row['error']],'ready_for_machine':False}
    topology=row['geometry']['topology_pass'];attachment=row['attachment']['attachment_and_front_proxy_pass']
    frontal=row['face_surface_guard']['max_distance_to_posed_source_mm']<=.02
    return {'status':'GEOMETRY_EXPORT_VALID' if topology and attachment and frontal else 'REJECTED',
        'A_hair':{'fit_and_attachment':'SAMPLED_PROXY_PASS' if attachment else 'PROXY_FAILURE',
                  'template_not_photo_reconstruction':True,'hairstyle_likeness_verified':False,
                  'cage_limits_measured':True,'exact_clearance_verified':False},
        'B_face':{'protected_core_surface':'SAMPLED_PROXY_PASS' if frontal else 'PROXY_FAILURE',
                  'expression_origin':row['expression_report']['origin'],'identity_recognition_verified':False,
                  'emotional_appeal_verified':False,'forced_symmetry':False,'population_specific_beauty_ratio_applied':False,
                  'physical_color_match_verified':False},
        'C_manufacturing':{'stl_roundtrip_topology':'MEASURED_PASS' if topology else 'MEASURED_FAILURE',
                           'minimum_wall_thickness':'UNVERIFIED','mechanical_strength':'UNVERIFIED',
                           'machine_slicing':'NOT_IMPLEMENTED','physical_print':'UNVERIFIED'},
        'ready_for_machine':False}


def evaluation_summary(report):
    rows=[candidate_ledger(c) for c in report['candidates']]
    selected=report.get('selected') or {}
    support=selected.get('experimental_supports')
    return {'version':1,'status':'RESEARCH_DEMO_WITH_UNVERIFIED_MANUFACTURING',
        'input':report['input_assessment'],'candidate_ledgers':{c['id']:r for c,r in zip(report['candidates'],rows)},
        'automation':{'head_reconstruction':'UPSTREAM_WORKSPACE_REQUIRED','template_selection':'AUTOMATIC_PROPOSAL_WITH_USER_STYLE_OVERRIDE',
            'hair_fit':'AUTOMATIC_BOUNDED','mesh_repair':'AUTOMATIC_HAIR_ONLY_WHEN_REQUIRED',
            'geometry_checks':'AUTOMATIC','support_preparation':'EXPERIMENTAL_AUTOMATIC' if support else 'NOT_EXPORTED',
            'printer_file_slicing':'NOT_IMPLEMENTED','color_finishing':'NOT_AUTOMATED'},
        'five_perspectives':{
            'customer':{'likeness_and_purchase_appeal':'NOT_EVALUATED'},
            '3d_design':{'topology':'MEASURED','attachment':'SAMPLED_PROXY','aesthetic_harmony':'NOT_EVALUATED'},
            'computer_vision':{'internal_camera_fit':'MEASURED','absolute_calibration':'UNVERIFIED','mask_ground_truth':'UNVERIFIED'},
            'printing':{'geometry':'MEASURED','supports':'VOXEL_PRECHECK_ONLY' if support else 'NOT_EXPORTED','physical_validation':'NOT_EVALUATED'},
            'anatomy_expression':{'core_surface_preservation':'SAMPLED_PROXY','anatomical_accuracy_and_expression_appeal':'NOT_EVALUATED'}},
        'human_expert_reviews_completed':0,'human_study_form':'human_review.schema.json',
        'independent_validation_subjects':0,'actual_demo_subjects':1,
        'color':{'palette':'project-designed 8 swatches, not the Monk scale','glb_vertex_colors':'linear RGB',
                  'eye_texture':'sRGB','illumination_calibrated':False,'physical_color_reproduction_verified':False},
        'ready_for_machine':False,'physical_print_verified':False,
        'do_not_infer':['topology pass implies printable','silhouette IoU implies identity','five viewpoints means five experts reviewed'],
        'next_validation':['calibrate one machine/resin profile','test printed features and selected head',
                           'held-out views and independent subjects','blind human likeness/expression comparison']}


HUMAN_REVIEW_SCHEMA={'schema_version':1,'study_scope':'research/development validation, not a manual mesh-edit step per order',
    'reviews':[],'role_options':['customer','3d_design','computer_vision','printing','anatomy_expression'],
    'rating_fields':{'likeness':{'range':[1,5],'default':None},'harmony':{'range':[1,5],'default':None},
                     'expression':{'range':[1,5],'default':None},'manufacturing_confidence':{'range':[1,5],'default':None}},
    'review_record_fields':['reviewer_code','role','candidate_id','ratings','defects','evidence','notes'],
    'no_scores_prepopulated':True}
