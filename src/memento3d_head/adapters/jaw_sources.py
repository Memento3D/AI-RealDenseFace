"""Load verified source geometry and isolate trusted semantic asset decoding."""
import subprocess
from pathlib import Path
import numpy as np
from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.mica_assets import verify_config
from memento3d_head.adapters.smooth_preview import read_geometry
from memento3d_head.adapters.anatomical_jaw import project_anatomical_jaw,lower_face_boundary
from memento3d_head.domain.contracts import ValidationError


def prepare_jaw_outlines(args,manifest,registration,output,sources):
    mesh=args.head_run/'head.obj';candidate=args.shape_run/'head.candidate.obj'
    for path,key in ((mesh,'head_obj_sha256'),(candidate,'candidate_obj_sha256'),
                     (args.head_run/'report.json','head_report_sha256'),
                     (args.shape_run/'report.json','shape_report_sha256')):
        sources[path]=sha256(path)
        if sources[path]!=manifest['source_binding'][key]:
            raise ValidationError('SOURCE_MISMATCH','Head/candidate differs from annotation workspace')
    config=verify_config(args.mica_config);sources[args.mica_config]=sha256(args.mica_config)
    item=config['assets']['data/FLAME2020/FLAME_masks/FLAME_masks.pkl']
    # Only the documented mask semantics/topology are supported.
    if item['sha256']!='ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492':
        raise ValidationError('UNSUPPORTED_FACE_MASK','Expected pinned MICA FLAME2020 semantic masks')
    mask_path=Path(item['path']);sources[mask_path]=item['sha256']
    worker=Path(__file__).with_name('face_mask_worker.py')
    result=subprocess.run([config['python'],str(worker.resolve()),str(mask_path),item['sha256'],str(output/'face_regions.npz')],
                          capture_output=True,text=True,timeout=120)
    (output/'face_mask_worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise ValidationError('MASK_WORKER_FAILED','See face_mask_worker.log')
    with np.load(output/'face_regions.npz',allow_pickle=False) as data:face=data['face'];exclude=data['exclude']
    base,faces=read_geometry(mesh);after,after_faces=read_geometry(candidate)
    if len(base)!=5023 or len(faces)!=9976 or not np.array_equal(faces,after_faces):
        raise ValidationError('TOPOLOGY_MISMATCH','Expected unchanged MICA FLAME2020 topology')
    boundary=lower_face_boundary(base,faces,face)
    outlines={};diagnostics={}
    for slot,view in registration['views'].items():
        outlines[slot]={};diagnostics[slot]={}
        for name,vertices in (('baseline',base),('candidate',after)):
            outlines[slot][name],diagnostics[slot][name]=project_anatomical_jaw(vertices,faces,face,view,boundary=boundary)
    from memento3d_head.adapters.local_store import LocalStore
    LocalStore().write_json(output/'surface_chains.json',{'schema_version':1,
        'coordinate_space':'original_normalized_photo_pixels','views':outlines,
        'source_binding':manifest['source_binding'],
        'warning':'All extracted chains for diagnostics; no correspondence approval'})
    return outlines,diagnostics
