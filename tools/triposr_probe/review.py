"""Audit an existing run; optionally compare extraction grids, without recropping."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from .audit import audit
from .common import ROOT, sha, write


def verified_raw(path):
    path=Path(path).resolve()
    raw=path/'raw' if (path/'raw/report.json').is_file() else path
    report=raw/'report.json'
    data=json.loads(report.read_text(encoding='utf-8-sig'))
    if data.get('status')!='RAW_PROPOSAL_EXPORTED':
        raise ValueError('Expected a completed original single-resolution probe run')
    for name in ('raw_native.glb','model_input.png'):
        expected=data.get('output_sha256',{}).get(name)
        if not expected or sha(raw/name)!=expected:
            raise ValueError('SOURCE_HASH_MISMATCH: '+name)
    return raw


def run(source, compare=None, output=None):
    raw=verified_raw(source)
    if output is None:
        output=ROOT/'runs'/('triposr_review_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result={'status':'AUDITING','original_report_sha256':sha(raw/'report.json'),
            'original_mesh_sha256':sha(raw/'raw_native.glb'),
            'prepared_input_sha256':sha(raw/'model_input.png'),
            'source':str(raw),'source_geometry_modified':False,'ready_for_machine':False,
            'hair_isolated':False,'hybrid_assembly_performed':False,
            'rdf_face_required_for_hybrid':True}
    try:
        result['audit']=audit(raw/'raw_native.glb',output/'audit',raw/'report.json')
        if compare:
            args=[sys.executable,'-m','tools.triposr_probe.worker','--image',str(raw/'model_input.png'),
                  '--prepared-image','--output',str(output/'comparison'),'--resolutions']
            args += [str(r) for r in sorted(set(compare))]
            done=subprocess.run(args,cwd=ROOT,check=False)
            result['comparison_exit_code']=done.returncode
            result['status']='AUDIT_AND_COMPARISON_EXPORTED' if done.returncode==0 else 'COMPARISON_FAILED_OR_PARTIAL'
        else:
            result['status']='AUDIT_EXPORTED'
        return result
    except Exception as exc:
        result.update(status='FAILED',error=str(exc))
        raise
    finally:
        write(output/'review.report.json',result)
        print('Audit:',output/'audit/review.html',flush=True)
        if compare: print('Comparison:',output/'comparison/review.html',flush=True)
        print('Report:',output/'review.report.json',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--compare',type=int,nargs='+',choices=[128,192,256]);a=p.parse_args()
    result=run(a.run,a.compare)
    if result['status']=='COMPARISON_FAILED_OR_PARTIAL':raise SystemExit(2)

if __name__=='__main__':main()
