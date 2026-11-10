"""Standard-library catalog pins and authenticated per-job artifact contracts."""
import json
from pathlib import Path
from tools.headapi.service import sha

POLICY='mvp_landmark_fit_v1'
IDS={'graduation-cap','hair-side-part','glasses-round'}


def confined(root,relative):
    root=Path(root).resolve();value=Path(relative)
    if value.is_absolute() or '..' in value.parts:raise ValueError('Invalid customization path')
    path=root/value;cursor=path
    while cursor!=root:
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(),'st_file_attributes',0)&0x400):
            raise ValueError('Linked customization asset')
        cursor=cursor.parent
    path=path.resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing customization asset')
    return path


def library(config):
    settings=config.get('customization') or {}
    if settings.get('policy')!=POLICY:raise ValueError('Unsupported customization policy')
    root=(Path(config['root'])/settings['library']).resolve()
    for filename,key in [('catalog.json','catalog_sha256'),('fitting-profile.json','profile_sha256')]:
        if sha(confined(root,filename))!=settings[key]:raise ValueError('Customization catalog/profile pin changed')
    catalog=json.loads((root/'catalog.json').read_text(encoding='utf8'))
    profile=json.loads((root/'fitting-profile.json').read_text(encoding='utf8'))
    assets={a['id']:a for a in catalog['assets']}
    if set(assets)!=IDS or catalog['status']!='MVP_SELECTED_PREVIEW' or profile.get('policy')!=POLICY:
        raise ValueError('Unsupported MVP customization catalog')
    for asset in assets.values():
        if asset['supported_body_preset']!='graduation-standing-base-v1':raise ValueError('Incompatible body preset')
        for field,digest in [('preview_file','preview_sha256'),('thumbnail_file','thumbnail_sha256')]:
            if sha(confined(root,asset[field]))!=asset[digest]:raise ValueError('Customization asset pin changed')
    return root,catalog,profile


def manifest(service,job_id,attempt_id):
    row=service.get(job_id)
    if row.get('attempt_id')!=attempt_id:raise KeyError('Attempt not found')
    if row.get('status')!='MVP_EXPORTED' or not row.get('customization_ready'):
        raise ValueError('Customization is not ready')
    relative=row.get('customization_manifest_artifact')
    if relative!='customization/manifest.json':raise ValueError('Unsupported customization artifact')
    path=service.artifact(job_id,relative)
    if sha(path)!=row.get('customization_manifest_sha256'):raise ValueError('Customization manifest changed')
    data=json.loads(path.read_text(encoding='utf8'))
    if data['job_id']!=job_id or data['attempt_id']!=attempt_id:raise ValueError('Manifest job/attempt mismatch')
    if data['base_model']['sha256']!=sha(service.artifact(job_id,'render/scene.glb')):
        raise ValueError('Customization base changed')
    return data


def asset_file(service,job_id,attempt_id,asset_id):
    data=manifest(service,job_id,attempt_id)
    entry=next((a for a in data['assets'] if a['asset_id']==asset_id),None)
    if entry is None:raise KeyError('Unknown fitted asset')
    if entry['geometry_mode']=='shared_transform':
        root,catalog,_=library(service.config)
        asset=next(a for a in catalog['assets'] if a['id']==asset_id)
        path=confined(root,asset['preview_file'])
    elif entry['geometry_mode']=='fitted_geometry':
        path=service.artifact(job_id,'customization/'+asset_id+'.glb')
    else:raise ValueError('Unsupported fit geometry')
    if sha(path)!=entry['sha256']:raise ValueError('Fitted asset changed')
    return path
