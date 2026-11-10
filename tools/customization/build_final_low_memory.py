"""Rebuild a head sidecar and stream it into existing cap/hair GLBs."""
from pathlib import Path
import sys,json,pickle,numpy as np,trimesh
from PIL import Image
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from tools.customization.review_glb import replace_head,crop_proxy
from tools.customization.lip_natural_trial import bounded_lip_candidate
from tools.customization.neck_natural_math import neck_base_mask,soften_neck_base
from tools.headapi.assembly_math import AXES,transform_points
from tools.headapi.assembly_cohesion_trial import smooth_seam_normals
from tools.headapi.service import sha,write

def main():
 out=R/'runs/neck-natural-review-20261105/final-low-memory-02';out.mkdir(parents=True,exist_ok=False)
 job=R/'runs/headapi-assembly/55b83418-5c1a-4608-9dda-429d786d5e51';neck=R/'runs/neck-natural-review-20261105/attempt-03';lip=R/'runs/neck-natural-review-20261105/lip-attempt-03'
 with np.load(job/'rdf/projection_bundle.npz',allow_pickle=False) as b:v=b['vertices'];f=b['faces']
 with np.load(R/'runs/prod-win-02/input04-production-02/color/embedding.npz',allow_pickle=False) as b:embedding={k:b[k] for k in b.files}
 with np.load(R/'vendor/RealDenseFace/weights/flame/flame_assets.npz',allow_pickle=False) as b:
  if not np.array_equal(b['faces'],f):raise ValueError('UV topology mismatch')
  uv=b['uvs'][b['uv_faces']]
 masks=pickle.loads((R/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl').read_bytes(),encoding='latin1')
 candidate,_,lip_report=bounded_lip_candidate(v,f,embedding,masks)
 atlas=np.asarray(Image.open(neck/'head-neck-atlas.png').convert('RGB')).copy()
 lip_atlas=np.asarray(Image.open(lip/'head-neck-mouth-atlas.png').convert('RGB'));mouth=np.asarray(Image.open(lip/'mouth-change-mask.png'))>0
 atlas[mouth]=lip_atlas[mouth];del lip_atlas
 M=np.array(json.loads((job/'render/assembly-report.json').read_text())['matrix_native_to_working'])
 native=transform_points(v,M);mask=neck_base_mask(native,masks['neck'],masks['boundary'],.795)
 world=transform_points(candidate,M)+[0,0,.018]
 conditioning=json.loads((neck/'restore-neck-tone/report.json').read_text())['neck_conditioning']
 world,_=soften_neck_base(world,mask,.765,target_center=conditioning['center_xy'],target_radii=conditioning['radii_xy'])
 material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),metallicFactor=0,roughnessFactor=1,doubleSided=True)
 head=trimesh.Trimesh(world[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),process=False,visual=trimesh.visual.TextureVisuals(uv=uv.reshape(-1,2),material=material))
 hv,hf,hu=trimesh.intersections.slice_faces_plane(head.vertices,head.faces,[0,0,1],[0,0,.765],uv=head.visual.uv)
 head=trimesh.Trimesh(hv,hf,vertex_normals=smooth_seam_normals(hv,hf),process=False,visual=trimesh.visual.TextureVisuals(uv=hu,material=material))
 scene=trimesh.Scene();scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head');scene.apply_transform(np.linalg.inv(AXES));scene.export(out/'head-only.glb')
 write(out/'report.json',{'policy':'neck_boundary_plus_guarded_lip_v1','lip_guard':lip_report,'sidecar_sha256':sha(out/'head-only.glb'),
  'neck_source_sha256':sha(neck/'head-neck-atlas.png'),'mouth_source_sha256':sha(lip/'head-neck-mouth-atlas.png'),
  'neck_base_boundary_vertices_included':int(mask.sum()-len(masks['neck'])),'source_head_expression_not_globally_refit':True,'api_integrated':False,'print_ready':False})
 del scene,head,material,atlas;__import__('gc').collect()
 sources=[('cap-neck-and-lip',lip/'scene.glb'),('hair-neck-and-lip',R/'runs/customization-prep-20261105/fit-review-06/hair-micket-side-part-glasses-03/scene.glb')]
 for name,source in sources:
  folder=out/name;folder.mkdir();merge=replace_head(source,out/'head-only.glb',folder/'scene.glb')
  proxy=crop_proxy(folder/'scene.glb',folder/'render-proxy.glb')
  write(folder/'report.json',{'case':name,'source_sha256':sha(source),'head_sha256':sha(out/'head-only.glb'),'stream_merge':merge,'render_proxy':proxy,
   'world_head_pose_changed':False,'neck_boundary_corrected':True,'lip_adjustment_world_max_mm':float(lip_report['guard']['maximum_displacement_native']*np.linalg.norm(M[:3,0])*1000),
   'api_integrated':False,'print_ready':False,'limitations':['Bounded local mouth adjustment, no independent likeness certification','Original cap/hand and hairstyle issues remain outside this change']})
  print(name,'ready',[(x['node'],x['proxy_triangles']) for x in proxy['components']],flush=True)
 # Also render an unchanged-face neck candidate in the same lightweight setup.
 folder=out/'neck-only';folder.mkdir();__import__('os').link(neck/'restore-neck-tone/scene.glb',folder/'scene.glb')
 proxy=crop_proxy(folder/'scene.glb',folder/'render-proxy.glb')
 write(folder/'report.json',{'case':'neck-only','source_sha256':sha(neck/'restore-neck-tone/scene.glb'),'face_geometry_and_texture_unchanged':True,'render_proxy':proxy,'api_integrated':False,'print_ready':False})
 print('Final low-memory build complete',flush=True)
if __name__=='__main__':main()
