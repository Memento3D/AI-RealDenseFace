"""Export a local FLAME identity basis using the pinned CPU decoder, not MICA inference."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','identity','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    for item in config['assets'].values():
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Asset hash mismatch')
    import numpy as np
    import torch
    sys.path.insert(0,config['repo'])
    from configs.config import get_cfg_defaults
    from models.flame import FLAME
    identity=np.load(args.identity,allow_pickle=False)
    if identity.shape!=(300,) or not np.isfinite(identity).all():raise RuntimeError('Expected 300 FLAME identity coefficients')
    cfg=get_cfg_defaults().model
    cfg.flame_model_path=config['assets']['data/FLAME2020/generic_model.pkl']['path']
    cfg.flame_lmk_embedding_path=config['assets']['data/FLAME2020/landmark_embedding.npy']['path']
    cfg.n_shape=300
    torch.set_num_threads(2)
    decoder=FLAME(cfg).cpu().eval()
    beta=torch.from_numpy(identity.astype(np.float32))[None]
    with torch.no_grad():
        base=decoder(shape_params=beta)[0][0].numpy()*1000.
        basis=[]
        for i in range(8):
            changed=beta.clone();changed[0,i]+=1.
            basis.append(decoder(shape_params=changed)[0][0].numpy()*1000.-base)
    # Basis differences include the pinned decoder's canonical zero-pose behavior.
    np.savez(args.output,base=base,basis=np.stack(basis,axis=-1),
             faces=decoder.faces_tensor.numpy(),
             landmark_faces=decoder.full_lmk_faces_idx.numpy(),
             barycentric=decoder.full_lmk_bary_coords.numpy())
    print(json.dumps({'device':'cpu','torch_version':torch.__version__,'identity_dimensions':300,
                      'active_shape_dimensions':8,'units':'millimeters','mica_inference_run':False}))


if __name__=='__main__':main()
