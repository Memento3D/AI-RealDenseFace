"""Export pinned FLAME full and dynamic landmark tables in the isolated Python."""
import hashlib
import sys
import numpy as np

if __name__=='__main__':
    source,destination=sys.argv[1:]
    with open(source,'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest()!='8095348eeafce5a02f6bd8765146307f9567a3f03b316d788a2e47336d667954':
            raise ValueError('Unsupported FLAME embedding checksum')
    data=np.load(source,allow_pickle=True,encoding='latin1').item()
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
    np.savez(destination,full_faces=array(data['full_lmk_faces_idx']).reshape(68),
             full_weights=array(data['full_lmk_bary_coords']).reshape(68,3),
             dynamic_faces=array(data['dynamic_lmk_faces_idx']),
             dynamic_weights=array(data['dynamic_lmk_bary_coords']))
