"""Load only the checksum-pinned FLAME semantic pickle in the isolated worker."""
import sys
import hashlib
import pickle
import numpy as np

if __name__=='__main__':
    source,expected,destination=sys.argv[1:]
    raw=open(source,'rb').read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Face mask checksum mismatch')
    data=pickle.loads(raw,encoding='latin1')
    exclude=np.unique(np.concatenate([data[key] for key in ('neck','left_ear','right_ear','scalp','left_eyeball','right_eyeball','eye_region','forehead','nose','lips')]))
    np.savez(destination,face=np.asarray(data['face'],dtype=np.int64),exclude=exclude.astype(np.int64))
