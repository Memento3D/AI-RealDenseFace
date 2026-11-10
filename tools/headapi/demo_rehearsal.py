"""Replay real existing RDF/cap artifacts through new UV stages; not new inference."""
import argparse,json,shutil
from pathlib import Path
from tools.headapi.service import ROOT,sha
from tools.headapi.texture_completion_trial import supervised


def run(output,sources,parser,assets):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);rows=[];cards=[]
    for index,source in enumerate(sources,1):
        source=Path(source).resolve();job=output/('case'+str(index));job.mkdir()
        required=['rdf/report.json','rdf/projection_bundle.npz','rdf_crops/report.json','hair/report.json','hair/hair_cap_native.glb']+['rdf_crops/'+s+'.png' for s in ['front','left','right']]
        for relative in required:
            destination=job/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,destination)
        before={relative:sha(job/relative) for relative in required}
        entry={'case':job.name,'source':str(source),'scope':'replay existing real RDF + cap, no new inference and no web/BE request','source_sha256':before}
        code='from tools.headapi.demo_face_worker import run; run('+','.join(repr(str(p)) for p in [job,job/'rdf_uv',parser,assets])+')'
        entry['uv']=supervised(code,job/'uv-stage.log')
        if entry['uv']['status']=='COMPLETED':
            code='from tools.headapi.demo_scene_worker import run; run('+repr(str(job))+','+repr(str(job/'render'))+')'
            entry['scene']=supervised(code,job/'scene-stage.log')
        for relative,digest in before.items():
            if sha(job/relative)!=digest:raise ValueError('Rehearsal modified source: '+relative)
        if entry.get('scene',{}).get('status')=='COMPLETED':
            report=json.loads((job/'render/report.json').read_text())
            entry['render_report']=report;entry['below_backend_50MiB_limit']=(job/'render/scene.glb').stat().st_size<=50*1024**2
            cards.append('<section><h2>'+job.name+'</h2><p>Real artifact replay; no fresh reconstruction. Geometry/cap preserved, texture embedded.</p><a href="'+job.name+'/render/scene.glb">UV head + hair GLB</a> | <a href="'+job.name+'/rdf_uv/final/report.json">Face report</a> | <a href="'+job.name+'/render/report.json">Scene report</a><p><img src="'+job.name+'/rdf_uv/final/face_textured.png"></p></section>')
        rows.append(entry)
        print(job.name+' '+entry.get('scene',entry['uv'])['status'],flush=True)
    (output/'report.json').write_text(json.dumps({'scope':'offline artifact replay only','new_inference_executed':False,'web_backend_e2e_verified':False,'cases':rows},indent=2),encoding='utf-8')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>API UV rehearsal</title><style>body{font:16px system-ui;background:#20252c;color:white;margin:24px}img{width:100%;max-width:1440px}a{color:#8cf}</style><h1>API render rehearsal</h1><p>Existing real RDF and hair-cap inputs, new generic UV stages. This is not a web/BE end-to-end test or new reconstruction. No machine-readiness claim.</p><a href="report.json">Report</a>'+''.join(cards),encoding='utf-8')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--parser',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);a=p.parse_args();run(a.output,a.sources,a.parser,a.assets)
