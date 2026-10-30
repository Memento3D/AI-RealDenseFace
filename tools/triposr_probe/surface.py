"""Explicit CPU marching-cubes backend for the upstream MarchingCubeHelper API."""
import numpy as np

def cpu_surface(volume,threshold):
 from skimage.measure import marching_cubes
 a=np.asarray(volume,dtype=np.float32)
 if a.ndim!=3 or min(a.shape)<2 or not np.isfinite(a).all() or not np.isfinite(threshold) or not a.min()<threshold<a.max():raise ValueError('No finite, bracketed density isosurface')
 # Grid axes in upstream are x,y,z. Its helper reverses returned z,y,x once.
 # Return z,y,x here so that the helper recovers x,y,z, without transposing data.
 v,f,_,_=marching_cubes(a,level=threshold,gradient_direction='ascent',allow_degenerate=False)
 return np.ascontiguousarray(v[:,[2,1,0]]),np.ascontiguousarray(f.astype(np.int64))

def register_backend():
 import torch,sys,types
 module=types.ModuleType('torchmcubes')
 def marching_cubes(volume,threshold):
  v,f=cpu_surface(volume.detach().cpu().numpy(),float(threshold))
  return torch.from_numpy(v.copy()),torch.from_numpy(f.copy())
 module.marching_cubes=marching_cubes
 # Scoped to this separate worker. Upstream source files are unmodified.
 sys.modules['torchmcubes']=module
 return 'scikit-image CPU marching cubes; explicit compatibility backend, not native torchmcubes'
