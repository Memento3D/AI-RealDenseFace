"""Recheck saved chord warnings locally; no automatic repair or print certification."""
import json
from pathlib import Path
import hashlib
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]


def run():
    study=ROOT/'runs/experiments/input04-print-neck-a1a6cf56cf464cc5b0058c7597bec41f'
    source=study/'diagnostics/report.json';previous=json.loads(source.read_text())
    output=ROOT/'runs/experiments/input04-thin-triage-01';output.mkdir(parents=True,exist_ok=False)
    result={'schema_version':1,'method':'five neighboring surface-normal chords per saved flag; radius 0.15mm',
        'threshold_mm':.8,'threshold_calibrated':False,'geometry_modified':False,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'variants':{}}
    for name in ['baseline','refined']:
        path=study/name/'head_hair_pedestal_mm.stl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=previous[name]['stl_sha256']:raise ValueError('STL changed')
        mesh=trimesh.load(path,force='mesh',process=True);records=[]
        scene=trimesh.Scene();display=mesh.copy();display.visual.face_colors=[180,180,190,255];scene.add_geometry(display,node_name='source_unchanged')
        for index,flag in enumerate(previous[name]['thin_feature_proxy']['locations']):
            point=np.asarray(flag['position_mm']);normal=np.asarray(flag['normal']);normal/=np.linalg.norm(normal)
            axis=np.array([1.,0,0]) if abs(normal[0])<.9 else np.array([0.,1,0])
            tangent=np.cross(normal,axis);tangent/=np.linalg.norm(tangent);other=np.cross(normal,tangent)
            candidates=np.array([point,point+.15*tangent,point-.15*tangent,point+.15*other,point-.15*other])
            samples,_,surface_faces=mesh.nearest.on_surface(candidates);normals=mesh.face_normals[surface_faces]
            origins=samples-.001*normals
            hits,rays,hit_faces=mesh.ray.intersects_location(origins,-normals,multiple_hits=True)
            distances=np.linalg.norm(hits-origins[rays],axis=1);nearest=np.full(5,np.inf);exit_faces=np.full(5,-1,dtype=int)
            for ray,distance,face in zip(rays,distances,hit_faces):
                if .002<distance<nearest[ray]:nearest[ray]=distance;exit_faces[ray]=face
            opposing=np.array([float(np.dot(normals[i],mesh.face_normals[f])) if f>=0 else None for i,f in enumerate(exit_faces)],dtype=object)
            short=nearest<.8
            opposing_short=[i for i in range(5) if short[i] and opposing[i] is not None and opposing[i]<-.5]
            status='PERSISTENT_OPPOSING_SHORT_CHORD' if len(opposing_short)>=3 else ('MIXED_OR_EDGE_SENSITIVE' if short.any() else 'NOT_REPRODUCED_LOCALLY')
            values=nearest[np.isfinite(nearest)]
            row={'index':index,'source_position_mm':point.tolist(),'source_component_proxy':flag['nearest_source_component'],
                'classification':status,'short_chords':int(short.sum()),'opposing_short_chords':len(opposing_short),
                'local_chords_mm':[float(v) if np.isfinite(v) else None for v in nearest],
                'exit_normal_dot_start_normal':[float(x) if x is not None else None for x in opposing],
                'median_chord_mm':float(np.median(values)) if len(values) else None,
                'location_proxy':'lower_neck_or_pedestal_zone' if point[2]<5 else ('lateral_zone' if abs(point[0])>.32*np.ptp(mesh.vertices[:,0]) else 'central_head_zone'),
                'anatomical_region_verified':False,'repair_applied':False}
            records.append(row)
            sphere=trimesh.creation.icosphere(subdivisions=1,radius=.18);sphere.apply_translation(point)
            sphere.visual.face_colors=[240,50,50,255] if status=='PERSISTENT_OPPOSING_SHORT_CHORD' else [255,170,30,255]
            scene.add_geometry(sphere,node_name='warning_'+str(index))
        scene.export(output/(name+'_warnings.glb'))
        counts={key:sum(r['classification']==key for r in records) for key in ['PERSISTENT_OPPOSING_SHORT_CHORD','MIXED_OR_EDGE_SENSITIVE','NOT_REPRODUCED_LOCALLY']}
        result['variants'][name]={'stl_sha256':previous[name]['stl_sha256'],'counts':counts,'records':records,
            'decision':'REQUIRES_REVIEW','ready_for_machine':False,'physical_print_verified':False,
            'limitations':['normal chords are not exact wall thickness','neighboring rays are not independent samples',
                          'opposing normals reduce grazing ambiguity but do not prove mechanical fragility',
                          'do not clear or repair warnings solely from these labels']}
        print(name+': '+json.dumps(counts),flush=True)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':run()
