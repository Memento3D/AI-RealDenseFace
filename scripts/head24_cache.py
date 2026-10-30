"""Versioned pre-sideburn checkpoints, bound to geometry, asset, policy and producer code."""
import hashlib,json
from pathlib import Path
import numpy as np
import trimesh


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cache_key(face,asset,profile,root):
    head=face['head'];h=hashlib.sha256()
    for a in (head.vertices,head.faces,face['eye_centers']):h.update(np.ascontiguousarray(a).tobytes())
    paths=['figurine2_hair.py','figurine21_fit.py','figurine22_hair.py','figurine23_hair.py']
    return {'version':1,'head_geometry_sha256':h.hexdigest(),'asset_sha256':asset['sha256'],'asset_id':asset['id'],'fit':profile['fit'],'compact_fit':profile['compact_fit'],'producer_files':{p:sha(root/'scripts'/p) for p in paths}}


def save_hair_cache(folder,asset_id,key,fitted,fit,shell,shell_report):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);stem='hair_checkpoint_'+asset_id
    path=folder/(stem+'.npz')
    np.savez_compressed(path,fitted_vertices=fitted.vertices,fitted_faces=fitted.faces,shell_vertices=shell.vertices,shell_faces=shell.faces)
    metadata={'key':key,'sha256':sha(path),'fit':fit,'shell_report':shell_report}
    (folder/(stem+'.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')


def load_hair_cache(folder,asset_id,key):
    folder=Path(folder);stem='hair_checkpoint_'+asset_id
    metadata=json.loads((folder/(stem+'.json')).read_text(encoding='utf-8-sig'));path=folder/(stem+'.npz')
    if metadata['key']!=key:raise ValueError('Hair checkpoint belongs to another head/asset/policy/producer')
    if sha(path)!=metadata['sha256']:raise ValueError('Hair checkpoint checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        fitted=trimesh.Trimesh(data['fitted_vertices'],data['fitted_faces'],process=False)
        shell=trimesh.Trimesh(data['shell_vertices'],data['shell_faces'],process=False)
    if not fitted.is_volume or not shell.is_volume:raise ValueError('Cached hair is not a closed positive-volume mesh')
    return fitted,metadata['fit'],shell,metadata['shell_report']
