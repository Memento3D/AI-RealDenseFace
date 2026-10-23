"""Download only the pinned official ONNX asset; never download user photographs."""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from memento3d_head.adapters.automatic_face_masks import MODEL_SHA256,MODEL_URL


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();path=args.output
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:raise SystemExit('Existing model has wrong checksum; choose a new output path')
        print('MODEL_READY: '+str(path.resolve()));return
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+uuid4().hex+'.download')
    try:
        print('Downloading official ResNet18 ONNX (~51 MiB)',flush=True)
        with urllib.request.urlopen(MODEL_URL,timeout=60) as response,temporary.open('xb') as dst:
            while chunk:=response.read(1024*1024):dst.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('Downloaded model checksum mismatch')
        if path.exists():raise FileExistsError('Output appeared during download')
        temporary.rename(path);print('MODEL_READY: '+str(path.resolve()))
    finally:
        if temporary.exists():temporary.unlink()


if __name__=='__main__':main()
