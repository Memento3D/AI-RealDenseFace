"""Standalone Python 3.10 worker; imports external MICA only after local pin verification.

This file contains integration code, not a redistributed copy of MICA. Run it by
absolute path with the separate model interpreter, never import Torch in the CLI.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def run(config_path, output, input_path):
    """Check the model environment, or execute batch-one geometry initialization."""
    config=json.loads(config_path.read_text(encoding='utf-8'))
    for item in config['assets'].values():
        digest=hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        if digest!=item['sha256']:
            raise RuntimeError('Asset checksum mismatch inside worker')
    import numpy as np
    import torch
    import onnxruntime as ort
    import cv2
    import trimesh
    from insightface import model_zoo
    from insightface.utils import face_align
    sys.path.insert(0,config['repo'])
    from configs.config import get_cfg_defaults
    from micalib.models.mica import MICA
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the separate MICA environment')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    probe=torch.ones((16,16),device='cuda')
    _=probe@probe
    torch.cuda.synchronize()
    del probe
    state={'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
           'gpu_name':torch.cuda.get_device_name(0),'onnxruntime_version':ort.__version__,
           'detector_device':'CPUExecutionProvider','device':'cuda:0',
           'upstream_commit':config['upstream_commit'],'inference_tested':False,
           'assets_sha256':{key:value['sha256'] for key,value in config['assets'].items()}}
    detector=model_zoo.get_model(config['assets']['detector']['path'],providers=['CPUExecutionProvider'])
    if detector is None or getattr(detector,'taskname',None)!='detection':
        raise RuntimeError('The ONNX asset must be the SCRFD face detector, not ArcFace recognition')
    detector.prepare(ctx_id=-1,input_size=(224,224),det_thresh=0.5)
    cfg=get_cfg_defaults()
    cfg.model.testing=True
    cfg.pretrained_model_path=config['assets']['data/pretrained/mica.tar']['path']
    # Guard against upstream silently accepting incomplete/untrained checkpoints.
    checkpoint=torch.load(cfg.pretrained_model_path,map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or not {'arcface','flameModel'}<=set(checkpoint):
        raise RuntimeError('MICA checkpoint is missing required state dictionaries')
    del checkpoint
    model=MICA(cfg,'cuda:0')
    model.eval()
    if input_path is None:
        state['status']='MODEL_LOADED'
    else:
        started=time.monotonic()
        image=cv2.imread(str(input_path))
        if image is None:
            raise RuntimeError('Cannot decode front input')
        boxes,keypoints=detector.detect(image,max_num=0,metric='default')
        if len(boxes)!=1 or keypoints is None:
            raise RuntimeError('Exactly one face with five detector landmarks is required')
        aligned=face_align.norm_crop(image,landmark=keypoints[0],image_size=112)
        blob=cv2.dnn.blobFromImage(aligned,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True)
        aligned224=face_align.norm_crop(image,landmark=keypoints[0],image_size=224)
        tensor=np.ascontiguousarray(aligned224[:,:,::-1].transpose(2,0,1)/255.0,dtype=np.float32)
        with torch.no_grad():
            code=model.encode(torch.from_numpy(tensor)[None].cuda(),torch.from_numpy(blob).cuda())
            decoded=model.decode(code)
            vertices=decoded['pred_canonical_shape_vertices'][0].detach().cpu().numpy()*1000.0
            faces=model.flameModel.generator.faces_tensor.detach().cpu().numpy()
            shape=decoded['pred_shape_code'][0].detach().cpu().numpy()
        if not np.isfinite(vertices).all() or not np.isfinite(shape).all():
            raise RuntimeError('Non-finite model output')
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        mesh.export(output/'head.obj')
        mesh.export(output/'head.ply')
        np.save(output/'identity.npy',shape,allow_pickle=False)
        cv2.imwrite(str(output/'front.aligned.jpg'),aligned224)
        torch.cuda.synchronize()
        state.update(status='GEOMETRY_EXPORTED',inference_tested=True,
                     inference_seconds=round(time.monotonic()-started,3),
                     vertices=int(len(vertices)),faces=int(len(faces)),units='millimeters',
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'worker_result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')


def main():
    """Run as a dedicated process so native/GPU memory is released on exit."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    try:
        run(args.config,args.output,args.input)
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        (args.output/'worker_result.json').write_text(json.dumps({'status':'FAILED','error':str(error)}),encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
