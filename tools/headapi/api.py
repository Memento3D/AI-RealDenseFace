"""Multipart local demo API. The API process never imports a model/GPU SDK."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hmac
import hashlib
import io
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from .service import ROOT, JobService, RDF_SLOTS, SLOTS, load_config

MAX_BYTES=12*1024*1024
MAX_BACKEND_REQUEST_BYTES=64*1024
VIEWS=SLOTS


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,new_url):
        return None


def _parse_uuid(value,name):
    try:return str(uuid.UUID(str(value)))
    except (ValueError,TypeError,AttributeError) as exc:
        raise HTTPException(422,name+' must be a UUID') from exc


def _fetch_remote_image(url,allowed_hosts):
    try:
        parsed=urllib.parse.urlsplit(url)
        port=parsed.port
    except (TypeError,ValueError) as exc:raise HTTPException(422,'Invalid image URL') from exc
    host=(parsed.hostname or '').lower()
    if (parsed.scheme!='https' or not host or parsed.username or parsed.password
            or port not in (None,443) or parsed.fragment
            or host not in allowed_hosts):
        raise HTTPException(422,'Image URL must be HTTPS on an allowed Cloudinary host')
    request=urllib.request.Request(url,headers={'Accept':'image/jpeg,image/png,image/webp'})
    opener=urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request,timeout=20) as response:
            if response.status!=200:raise HTTPException(422,'Unable to retrieve input image')
            declared=response.headers.get('Content-Length')
            if declared and int(declared)>MAX_BYTES:
                raise HTTPException(413,'Input image exceeds 12 MiB')
            raw=response.read(MAX_BYTES+1)
    except HTTPException:raise
    except (urllib.error.URLError,TimeoutError,OSError,ValueError) as exc:
        raise HTTPException(422,'Unable to retrieve input image') from exc
    if len(raw)>MAX_BYTES:raise HTTPException(413,'Input image exceeds 12 MiB')
    try:return normalize(raw)
    except ValueError as exc:raise HTTPException(422,'Invalid input image') from exc


def _backend_status(row):
    state=row.get('status')
    status='queued' if state=='QUEUED' else 'running' if state=='RUNNING' else 'done' if state=='MVP_EXPORTED' else 'failed'
    stage=row.get('current_stage')
    if stage=='01_rdf_observations':stage='preprocessing'
    elif stage=='03b_uv_texture':stage='texturing'
    elif stage in ('02_rdf_fit','02b_rdf_export','03_hair_cap'):stage='shape_generation'
    elif stage in ('04_scene_export','05_customization_fit'):stage='exporting'
    elif stage=='production_reconstruction':stage='shape_generation'
    else:stage=None
    if status in ('done','failed'):stage=None
    errors=row.get('errors') or []
    last_error=errors[-1] if errors else {}
    return {
        'job_id':row.get('job_id'),
        'attempt_id':row.get('attempt_id'),
        'status':status,
        'stage':stage,
        'progress':100 if status=='done' else None,
        'output_kind':row.get('output_kind','assembled_preview'),
        'customization_ready':status=='done' and bool(row.get('customization_ready')),
        'ready_for_machine':False,
        'error_code':last_error.get('code','PIPELINE_FAILED') if status=='failed' else None,
        'error_message':(last_error.get('error') if last_error.get('code') else 'Reconstruction failed during '+str(last_error.get('stage') or 'processing')) if status=='failed' else None,
        'created_at':row.get('created_at'),
        'started_at':row.get('started_at'),
        'completed_at':row.get('completed_at') or row.get('finished_at'),
    }


def normalize(raw):
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ('JPEG','PNG','WEBP'):raise ValueError('JPEG/PNG/WebP only')
            if im.width*im.height>24_000_000 or min(im.size)<64:raise ValueError('Image outside 64px / 24MP bounds')
            format=im.format;im.load();image=ImageOps.exif_transpose(im).convert('RGB')
            stream=io.BytesIO();image.save(stream,format='PNG')
            return {'original':raw,'png':stream.getvalue(),'size':list(image.size),'format':format}
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:raise ValueError('Invalid or unsafe image') from exc


def create_app(service,api_key=None,allowed_image_hosts=None):
    configured_key=api_key if api_key is not None else os.environ.get('HEADAPI_API_KEY')
    hosts=allowed_image_hosts
    if hosts is None:
        hosts=os.environ.get('HEADAPI_ALLOWED_IMAGE_HOSTS','res.cloudinary.com').split(',')
    allowed_hosts={host.strip().lower() for host in hosts if host.strip()}

    def require_backend_key(value):
        if not configured_key:
            raise HTTPException(503,'Backend API authentication is not configured')
        if not value or not hmac.compare_digest(value,configured_key):
            raise HTTPException(401,'Invalid API key')

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:yield
        finally:service.close()
    swagger_api_key=APIKeyHeader(name='X-API-Key',scheme_name='HeadApiKey',auto_error=False,
        description='Enter HEADAPI_API_KEY from your local configuration. Do not include a Bearer prefix.')
    app=FastAPI(title='Memento3D Head API',version='0.3.0',lifespan=lifespan,
                description='Experimental RDF head and image-conditioned solid hair-cap MVP. Not machine-ready.',
                dependencies=[Security(swagger_api_key)])
    @app.middleware('http')
    async def protect_api(request,call_next):
        if request.url.path not in ('/health','/docs','/openapi.json','/redoc'):
            if not configured_key:
                return JSONResponse({'detail':'Backend API authentication is not configured'},status_code=503)
            supplied=request.headers.get('X-API-Key')
            if not supplied or not hmac.compare_digest(supplied,configured_key):
                return JSONResponse({'detail':'Invalid API key'},status_code=401)
        return await call_next(request)
    @app.get('/',response_class=HTMLResponse,include_in_schema=False)
    def home():return (Path(__file__).parent/'index.html').read_text(encoding='utf-8')
    @app.get('/health',openapi_extra={'security':[]})
    def health():
        try:
            service.preflight('mvp')
            backend_ready=True
        except ValueError:
            backend_ready=False
        return {'status':'ok','service_status':'API_RUNNING','backend_ready':backend_ready,
                'gpu_jobs_parallel':1,'ready_for_machine':False,
                'assembly_output_mode':service.config.get('assembly',{}).get('output_mode','assembled_preview'),
                'customization_fit_enabled':bool(service.config.get('customization')),
                'pipeline_policy':service.config.get('pipeline_policy','rdf_hair_cap_mvp'),
                'face_render_policy':service.config.get('face_render_policy','legacy_vertex_color_v1'),
                'presentation_policy':service.config.get('presentation_policy','none')}

    @app.get('/customization/catalog')
    def customization_catalog():
        from tools.customization.contract import library
        try:_,catalog,_=library(service.config)
        except (ValueError,KeyError,OSError) as exc:raise HTTPException(503,'Customization library unavailable') from exc
        return {'schema_version':1,'catalog_version':catalog['selection'],'groups':catalog['groups'],
            'default_selection':catalog['default_selection'],'ready_for_machine':False,
            'assets':[{key:a[key] for key in ('id','version','name','kind','preview_sha256','preview_bytes','thumbnail_sha256','requires_per_job_fit')} for a in catalog['assets']]}

    @app.get('/customization/assets/{asset_id}/{variant}')
    def customization_library_file(asset_id:str,variant:str):
        from tools.customization.contract import library,confined
        if variant not in ('preview','thumbnail'):raise HTTPException(404,'Unknown asset variant')
        try:
            root,catalog,_=library(service.config)
            asset=next((a for a in catalog['assets'] if a['id']==asset_id),None)
            if asset is None:raise HTTPException(404,'Unknown asset')
            path=confined(root,asset['preview_file' if variant=='preview' else 'thumbnail_file'])
        except (ValueError,KeyError,OSError) as exc:raise HTTPException(503,'Customization library unavailable') from exc
        return FileResponse(path,media_type='model/gltf-binary' if variant=='preview' else 'image/png')

    @app.get('/jobs/{job_id}/customization')
    def customization_scene(job_id:str,attempt_id:str):
        from tools.customization.contract import manifest
        try:return manifest(service,job_id,_parse_uuid(attempt_id,'attempt_id'))
        except KeyError as exc:raise HTTPException(404,'Job/attempt not found') from exc
        except (ValueError,OSError) as exc:raise HTTPException(409,'Customization unavailable or changed') from exc

    @app.get('/jobs/{job_id}/customization/assets/{asset_id}')
    def customization_fitted_file(job_id:str,asset_id:str,attempt_id:str):
        from tools.customization.contract import asset_file
        try:path=asset_file(service,job_id,_parse_uuid(attempt_id,'attempt_id'),asset_id)
        except KeyError as exc:raise HTTPException(404,'Job/attempt/asset not found') from exc
        except (ValueError,OSError) as exc:raise HTTPException(409,'Customization unavailable or changed') from exc
        return FileResponse(path,media_type='model/gltf-binary',filename=asset_id+'.glb')

    @app.post('/jobs',status_code=202)
    async def submit_backend(request:Request,x_api_key:Optional[str]=Security(swagger_api_key)):
        require_backend_key(x_api_key)
        content_length=request.headers.get('content-length')
        if content_length and content_length.isdigit() and int(content_length)>MAX_BACKEND_REQUEST_BYTES:
            raise HTTPException(413,'Request body is too large')
        raw_body=await request.body()
        if len(raw_body)>MAX_BACKEND_REQUEST_BYTES:raise HTTPException(413,'Request body is too large')
        try:payload=json.loads(raw_body)
        except (json.JSONDecodeError,UnicodeDecodeError):raise HTTPException(422,'Invalid JSON request')
        if not isinstance(payload,dict):raise HTTPException(422,'JSON object required')
        job_id=_parse_uuid(payload.get('job_id'),'job_id')
        attempt_id=_parse_uuid(payload.get('attempt_id'),'attempt_id')
        profile=payload.get('profile','default')
        if profile!='default':raise HTTPException(422,"Only profile 'default' is supported")
        requested=payload.get('input_images')
        if requested is None:
            if payload.get('input_image_id') is None or not payload.get('input_url'):
                raise HTTPException(422,'input_images or input_image_id/input_url is required')
            requested=[{
                'image_id':payload['input_image_id'],
                'input_url':payload['input_url'],
                'view_hint':payload.get('view_hint','front'),
                'sha256':payload.get('sha256'),
            }]
        if not isinstance(requested,list) or not len(RDF_SLOTS)<=len(requested)<=len(SLOTS):
            raise HTTPException(422,'Provide front, left and right input_images; back is optional')
        validated=[];seen_ids=set();seen_views=set()
        for item in requested:
            if not isinstance(item,dict):raise HTTPException(422,'Each input_images entry must be an object')
            image_id=_parse_uuid(item.get('image_id'),'image_id')
            view=item.get('view_hint')
            if view not in VIEWS:raise HTTPException(422,'Each image requires view_hint: front, left, right, or back')
            if view in seen_views:raise HTTPException(422,'Duplicate view_hint: '+view)
            if image_id in seen_ids:raise HTTPException(422,'Duplicate input image id')
            seen_ids.add(image_id);seen_views.add(view)
            url=item.get('input_url')
            if not isinstance(url,str) or not url:raise HTTPException(422,'input_url is required for every image')
            expected=item.get('sha256')
            if expected is not None and (not isinstance(expected,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',expected)):
                raise HTTPException(422,'sha256 must be a 64-character hexadecimal digest')
            validated.append({'image_id':image_id,'view_hint':view,'input_url':url,
                              'sha256':expected.lower() if expected else None})
        if not set(RDF_SLOTS)<=seen_views:
            raise HTTPException(422,'Provide front, left and right input_images; back is optional')
        try:existing=service.get(job_id)
        except KeyError:existing=None
        if existing is not None:
            old=existing.get('observations',[])
            same_inputs=(existing.get('attempt_id')==attempt_id and existing.get('profile')==profile
                and len(old)==len(validated)
                and all(any(item.get('image_id')==image['image_id']
                            and item.get('view_hint')==image['view_hint']
                            and (image['sha256'] is None or item.get('original_sha256')==image['sha256'])
                            for item in old) for image in validated))
            if not same_inputs:raise HTTPException(409,'Job id already exists for a different attempt or input')
            return _backend_status(existing)
        with ThreadPoolExecutor(max_workers=len(validated),thread_name_prefix='headapi-input') as pool:
            fetched=list(pool.map(lambda item:_fetch_remote_image(item['input_url'],allowed_hosts),validated))
        images={};metadata=[];seen_hashes=set()
        for item,entry in zip(validated,fetched):
            actual_sha256=hashlib.sha256(entry['original']).hexdigest()
            if item['sha256'] is not None and not hmac.compare_digest(actual_sha256,item['sha256']):
                raise HTTPException(422,'Input image checksum mismatch')
            normalized_sha=hashlib.sha256(entry['png']).hexdigest()
            if normalized_sha in seen_hashes:raise HTTPException(422,'Duplicate photo across view slots')
            seen_hashes.add(normalized_sha);images[item['view_hint']]=entry
            metadata.append({'image_id':item['image_id'],'view_hint':item['view_hint'],'sha256':actual_sha256})
        try:
            row=service.submit(images,mode='mvp',job_id=job_id,attempt_id=attempt_id,
                               image_metadata=metadata,profile=profile)
        except OverflowError as exc:raise HTTPException(429,str(exc)) from exc
        except ValueError as exc:raise HTTPException(503,str(exc)) from exc
        return _backend_status(row)

    @app.get('/jobs/{job_id}')
    def backend_status(job_id:str,x_api_key:Optional[str]=Security(swagger_api_key)):
        require_backend_key(x_api_key)
        try:return _backend_status(service.get(job_id))
        except KeyError:raise HTTPException(404,'Unknown job')

    @app.get('/jobs/{job_id}/outputs/glb')
    def backend_glb(job_id:str,attempt_id:str,x_api_key:Optional[str]=Security(swagger_api_key)):
        require_backend_key(x_api_key)
        try:
            row=service.get(job_id)
            if row.get('status')!='MVP_EXPORTED':raise HTTPException(409,'GLB output is not ready')
            if row.get('attempt_id')!=_parse_uuid(attempt_id,'attempt_id'):
                raise HTTPException(404,'Output attempt not found')
            artifact=row.get('scene_artifact','hair/scene.glb')
            if artifact not in ('hair/scene.glb','render/scene.glb'):raise HTTPException(409,'Unsupported scene artifact')
            path=service.artifact(job_id,artifact)
        except KeyError:raise HTTPException(404,'Output not found')
        return FileResponse(path,media_type='model/gltf-binary',filename='scene.glb')

    @app.post('/v1/head/jobs',status_code=202)
    async def submit(front:UploadFile=File(...),left:UploadFile=File(...),right:UploadFile=File(...),
                     back:Optional[UploadFile]=File(None),mode:str=Form('mvp'),attempt_id:Optional[str]=Form(None)):
        if mode!='mvp':raise HTTPException(422,'Only the RDF + hair-cap MVP mode is supported')
        images={}
        for slot,upload in [('front',front),('left',left),('right',right),('back',back)]:
            if upload is None:continue
            raw=await upload.read(MAX_BYTES+1);await upload.close()
            if len(raw)>MAX_BYTES:raise HTTPException(413,slot+': upload exceeds 12 MiB')
            try:images[slot]=normalize(raw)
            except ValueError as exc:raise HTTPException(422,slot+': '+str(exc)) from exc
        hashes=[hashlib.sha256(entry['png']).hexdigest() for entry in images.values()]
        if len(set(hashes))!=len(hashes):raise HTTPException(422,'Duplicate photo across view slots')
        if not set(RDF_SLOTS)<=set(images):
            raise HTTPException(422,'MVP requires front, left and right; back is optional')
        if attempt_id is not None:attempt_id=_parse_uuid(attempt_id,'attempt_id')
        try:row=service.submit(images,mode,attempt_id=attempt_id)
        except OverflowError as exc:raise HTTPException(429,str(exc)) from exc
        except ValueError as exc:raise HTTPException(503,str(exc)) from exc
        return {'job_id':row['job_id'],'status':row['status'],'status_url':'/v1/head/jobs/'+row['job_id']}
    @app.get('/v1/head/jobs/{job_id}')
    def status(job_id:str):
        try:return service.get(job_id)
        except KeyError:raise HTTPException(404,'Unknown job')
    @app.get('/v1/head/jobs/{job_id}/logs',response_class=PlainTextResponse)
    def logs(job_id:str,stage:Optional[str]=None):
        try:return service.logs(job_id,stage)
        except KeyError:raise HTTPException(404,'Job/stage not found')
    @app.get('/v1/head/jobs/{job_id}/files/{relative:path}')
    def artifact(job_id:str,relative:str):
        try:path=service.artifact(job_id,relative)
        except KeyError:raise HTTPException(404,'Artifact unavailable')
        return FileResponse(path)
    return app


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/headapi.json');p.add_argument('--port',type=int,default=8090)
    a=p.parse_args();import uvicorn
    config=load_config(a.config)
    if config.get('pipeline_policy')=='production_direct_bounded_v1':
        from .production_service import ProductionJobService
        service=ProductionJobService(config)
    else:service=JobService(config)
    uvicorn.run(create_app(service),host='0.0.0.0',port=a.port,workers=1)

if __name__=='__main__':main()
