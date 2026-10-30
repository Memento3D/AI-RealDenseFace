"""Export freshly produced Pixel3DMM meshes/cameras to a numeric, pickle-free contract."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

def pixel_camera(frame,output_side=1024):
 """Match upstream project_points_screen_space, including its X-principal hack."""
 R=np.asarray(frame['flame']['R_rotation_matrix'])[0];t=np.asarray(frame['flame']['t']).reshape(-1,3)[0]
 base=np.eye(4);base[:3,:3]=np.asarray(frame['camera']['R_base_0'])[0];base[:3,3]=np.asarray(frame['camera']['t_base_0'])[0]
 pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=t
 extr=np.diag([1.,-1.,-1.,1.])@base@pose
 size=np.asarray(frame['img_size']).reshape(-1)
 if len(size)!=2 or size[0]!=size[1] or size[0]<=0:raise ValueError('Expected square Pixel crop')
 side=float(size[0]);K=np.eye(3);focal=float(np.asarray(frame['camera']['fl']).reshape(-1)[0]);pp=np.asarray(frame['camera']['pp']).reshape(-1,2)[0]
 K[0,0]=K[1,1]=focal*side;K[:2,2]=(pp+1)*(side/2+.5);K[0,2]-=1
 # PIL pixel-centre resize convention, rather than silently scaling image origin.
 factor=output_side/side;K[:2]*=factor;K[:2,2]+=(factor-1)/2
 if not np.isfinite(K).all() or not np.isfinite(extr).all() or focal<=0:raise ValueError('Invalid Pixel camera')
 return K,extr


def run(tracking,preprocessed,out):
 import torch,trimesh
 from .io import write,sha,SLOTS
 folders=[p for p in Path(tracking).iterdir() if p.is_dir() and (p/'mesh').is_dir() and (p/'checkpoint').is_dir()]
 if len(folders)!=1:raise ValueError('Expected exactly one fresh Pixel tracking result')
 root=folders[0];meshfiles=sorted(p for p in (root/'mesh').glob('*.ply') if p.stem.isdigit());frames=sorted((root/'checkpoint').glob('*.frame'))
 if len(meshfiles)!=3 or len(frames)!=3:raise ValueError('Pixel must produce exactly three ordered frame meshes and checkpoints')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);vertices=[];cameras=[];camera_points=[];intrinsics=[];faces=None;records={}
 for i,(meshpath,framepath) in enumerate(zip(meshfiles,frames)):
  # These checkpoints were produced by our immediately preceding owned subprocess,
  # never loaded from a user-supplied arbitrary pickle/URL.
  frame=torch.load(framepath,map_location='cpu',weights_only=False);mesh=trimesh.load(meshpath,force='mesh',process=False)
  if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite Pixel vertices')
  if faces is not None and not np.array_equal(mesh.faces,faces):raise ValueError('Pixel frame topology changed')
  faces=mesh.faces;K,extr=pixel_camera(frame)
  photo=Path(preprocessed)/'cropped'/(str(i).zfill(5)+'.jpg')
  if not photo.is_file():raise ValueError('Missing ordered Pixel crop '+str(photo))
  im=Image.open(photo).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS);im.save(out/(SLOTS[i]+'.png'))
  cam=mesh.vertices@extr[:3,:3].T+extr[:3,3]
  if (cam[:,2]<=0).any():raise ValueError('Pixel camera handedness/depth contract rejected; do not silently flip a mesh')
  vertices.append(mesh.vertices);cameras.append(K@extr[:3]);camera_points.append(cam);intrinsics.append(K);records[SLOTS[i]]={'mesh_sha256':sha(meshpath),'checkpoint_sha256':sha(framepath),'photo_sha256':sha(out/(SLOTS[i]+'.png'))}
 np.savez_compressed(out/'projection_bundle.npz',vertices=vertices[0],view_vertices=np.asarray(vertices),faces=faces,cameras=np.asarray(cameras),camera_points=np.asarray(camera_points),intrinsics=np.asarray(intrinsics))
 write(out/'native.report.json',{'status':'NUMERIC_CONTRACT_EXPORTED','native_inputs':records,'bundle_sha256':sha(out/'projection_bundle.npz'),'camera_convention':'OpenGL to CV row flip; independent image validation not yet passed','source_geometry_modified':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tracking',type=Path,required=True);p.add_argument('--preprocessed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.tracking,a.preprocessed,a.output)
