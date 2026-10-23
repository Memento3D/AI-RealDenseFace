"""Read only the checksum-pinned upstream embedding in the isolated MICA environment."""
import hashlib
import sys
import numpy as np

if __name__ == '__main__':
    source, destination = sys.argv[1:]
    with open(source, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != '8095348eeafce5a02f6bd8765146307f9567a3f03b316d788a2e47336d667954':
            raise ValueError('Unsupported embedding checksum')
    data = np.load(source, allow_pickle=True, encoding='latin1').item()
    np.savez(destination, faces=np.asarray(data['full_lmk_faces_idx']),
             barycentric=np.asarray(data['full_lmk_bary_coords']))
