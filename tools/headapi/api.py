"""Multipart local demo API. The API process never imports a model/GPU SDK."""
import argparse
import hashlib
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from .service import ROOT, JobService, load_config

MAX_BYTES=12*1024*1024


def normalize(raw):
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ('JPEG','PNG','WEBP'):raise ValueError('JPEG/PNG/WebP only')
            if im.width*im.height>24_000_000 or min(im.size)<64:raise ValueError('Image outside 64px / 24MP bounds')
            format=im.format;im.load();image=ImageOps.exif_transpose(im).convert('RGB')
            stream=io.BytesIO();image.save(stream,format='PNG')
            return {'original':raw,'png':stream.getvalue(),'size':list(image.size),'format':format}
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:raise ValueError('Invalid or unsafe image') from exc


def create_app(service):
    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:yield
        finally:service.close()
    app=FastAPI(title='Memento3D Head API',version='0.2.0',lifespan=lifespan,
                description='MVP head: front required, optional RDF side views, experimental semantic hair assembly. Not machine-ready.')
    @app.get('/',response_class=HTMLResponse,include_in_schema=False)
    def home():return (Path(__file__).parent/'index.html').read_text(encoding='utf-8')
    @app.get('/health')
    def health():
        readiness={}
        for mode in ('raw','components','mvp'):
            try:service.preflight(mode);readiness[mode]='FILES_PRESENT_NOT_GPU_VALIDATION'
            except ValueError as exc:readiness[mode]=str(exc)
        return {'status':'API_RUNNING','backends':readiness,'gpu_jobs_parallel':1,'ready_for_machine':False}
    @app.post('/v1/head/jobs',status_code=202)
    async def submit(front:UploadFile=File(...),left:Optional[UploadFile]=File(None),right:Optional[UploadFile]=File(None),
                     mode:str=Form('mvp'),resolution:int=Form(128)):
        if mode not in ('raw','components','mvp') or resolution not in (128,192,256):raise HTTPException(422,'Invalid mode or resolution')
        images={}
        for slot,upload in [('front',front),('left',left),('right',right)]:
            if upload is None:continue
            raw=await upload.read(MAX_BYTES+1);await upload.close()
            if len(raw)>MAX_BYTES:raise HTTPException(413,slot+': upload exceeds 12 MiB')
            try:images[slot]=normalize(raw)
            except ValueError as exc:raise HTTPException(422,slot+': '+str(exc)) from exc
        hashes=[hashlib.sha256(entry['png']).hexdigest() for entry in images.values()]
        if len(set(hashes))!=len(hashes):raise HTTPException(422,'Duplicate photo across view slots')
        if mode=='raw' and len(images)>1:raise HTTPException(422,'raw uses front only; select components to use side views for RDF')
        try:row=service.submit(images,mode,resolution)
        except OverflowError as exc:raise HTTPException(429,str(exc)) from exc
        except ValueError as exc:raise HTTPException(503,str(exc)) from exc
        return {'job_id':row['job_id'],'status':row['status'],'status_url':'/v1/head/jobs/'+row['job_id']}
    @app.post('/v1/head/jobs/{job_id}/assemble',status_code=202)
    def assemble_existing(job_id:str):
        try:row=service.submit_assembly(job_id)
        except KeyError:raise HTTPException(404,'Unknown source job')
        except OverflowError as exc:raise HTTPException(429,str(exc)) from exc
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
        return {'job_id':row['job_id'],'parent_job_id':job_id,'status':row['status'],'status_url':'/v1/head/jobs/'+row['job_id']}
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
    uvicorn.run(create_app(JobService(load_config(a.config))),host='127.0.0.1',port=a.port,workers=1)

if __name__=='__main__':main()
