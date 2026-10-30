"""Checksummed JSON/NPZ replay; no pickle, no model execution on cache replay."""
import base64,io,json,shutil
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from run_figurine_v3 import digest


def save_prepared(shared, folder, native, profile, workspace, model, back):
    folder.mkdir()
    arrays={k:shared[k] for k in ['source_mm','components','eye_centers']}
    arrays.update(closed_vertices=shared['closed'].vertices,closed_faces=shared['closed'].faces)
    for i in range(4):
        for k in ['cameras','masks','depths','labels']:arrays[k+'_'+str(i)]=shared[k][i]
        shared['images'][i].save(folder/('image_'+str(i)+'.png'))
    np.savez_compressed(folder/'prepared.npz',**arrays)
    (folder/'workspace_data.json').write_text(json.dumps(shared['data'],separators=(',',':')),encoding='utf-8')
    shutil.copy2(native,folder/'source_native.glb')
    meta={k:shared[k] for k in ['provenance','prepare_seconds','back_report']}
    meta.update(schema_version=1,reference_head_neck_mm=profile['reference_head_neck_mm'],mask_dilation_px=profile['mask_dilation_px'],
                workspace_sha256=digest(workspace),parser_sha256=digest(model),back_sha256=digest(back))
    meta['files']={p.name:digest(p) for p in folder.iterdir() if p.is_file()}
    (folder/'cache.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


def load_prepared(folder, profile):
    folder=Path(folder);meta=json.loads((folder/'cache.json').read_text(encoding='utf-8'))
    if meta['schema_version']!=1:raise ValueError('Unsupported prepared cache')
    for key in ['reference_head_neck_mm','mask_dilation_px']:
        if meta[key]!=profile[key]:raise ValueError('Cache/profile mismatch: '+key)
    if profile.get('back_sha256') and meta['back_sha256']!=profile['back_sha256']:raise ValueError('Cached back image mismatch')
    if profile.get('front_rgb_sha256') and meta['provenance']['front_rgb_sha256']!=profile['front_rgb_sha256']:raise ValueError('Cached front photo mismatch')
    for name,sha in meta['files'].items():
        if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Invalid prepared-cache checksum: '+name)
    required=['prepared.npz','workspace_data.json','source_native.glb']+['image_'+str(i)+'.png' for i in range(4)]
    if not all(name in meta['files'] for name in required):raise ValueError('Incomplete prepared cache manifest')
    with np.load(folder/'prepared.npz',allow_pickle=False) as a:
        result={k:a[k].copy() for k in ['source_mm','components','eye_centers']}
        result['closed']=trimesh.Trimesh(a['closed_vertices'],a['closed_faces'],process=False)
        for k in ['cameras','masks','depths','labels']:result[k]=[a[k+'_'+str(i)].copy() for i in range(4)]
    result.update(data=json.loads((folder/'workspace_data.json').read_text(encoding='utf-8')),
                  images=[Image.open(folder/('image_'+str(i)+'.png')).convert('RGB') for i in range(4)],
                  **{k:meta[k] for k in ['provenance','prepare_seconds','back_report']})
    return result,meta
