"""Apply the same accessory policy to two existing RDF geometry bundles."""
import argparse,json,sys
from pathlib import Path
import numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.customization.fit_trial import world_parts,make_hair,fit_front_clearance,refine_temples,surface_samples,ear_support_anchors
from tools.customization.fit_math import glasses_transform,transform_points,profile_for_head
from tools.headapi.assembly_math import AXES,eye_alignment
from tools.headapi.assembly_cohesion_trial import smooth_seam_normals
from tools.headapi.service import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 catalog=ROOT/'assets/custom-library/prepared-20261105';profile=json.loads((ROOT/'tools/customization/fit-profiles.json').read_text())['glasses']['glasses-03']
 with np.load(ROOT/'runs/prod-win-02/input04-production-02/color/embedding.npz',allow_pickle=False) as e:embedding={k:e[k] for k in ['full_faces','full_weights']}
 ref=ROOT/'runs/experiments/api-demo-uv-01/case1/rdf/projection_bundle.npz'
 with np.load(ref,allow_pickle=False) as b:reference=b['vertices'].copy()
 pose=json.loads((ROOT/'runs/graduation-assembly-04/report.json').read_text())['cases']['balanced']['matrix_working']
 baseline=world_parts(a.baseline);crown=baseline['Personalized_Complete_Head'].vertices[:,2].max();cases=[]
 sources=[('existing-case2',ROOT/'runs/experiments/api-demo-uv-01/case2/rdf/projection_bundle.npz'),
          ('existing-capture02',ROOT/'runs/experiments/geometry-camera-batch-02/fit/capture02/camera_shape/projection_bundle.npz')]
 for name,source in sources:
  with np.load(source,allow_pickle=False) as b:vertices=b['vertices'].copy();faces=b['faces'].copy()
  M=eye_alignment(vertices,faces,embedding,reference,pose)
  head=trimesh.Trimesh(transform_points(vertices,M),faces,process=False)
  offset=crown-head.vertices[:,2].max();head.apply_translation([0,0,offset])
  lm=np.einsum('ij,ijk->ik',embedding['full_weights'],vertices[faces[embedding['full_faces']]])
  lm=transform_points(lm,M)+[0,0,offset];head.vertex_normals=smooth_seam_normals(head.vertices,head.faces)
  ears=ear_support_anchors(vertices,M,np.array([0,0,offset]),lm)
  head.visual.vertex_colors=np.tile([165,167,170,255],(len(head.vertices),1))
  glasses=world_parts(catalog/'glasses-03/asset.glb');subject_profile,width=profile_for_head(profile,ears,lm,np.vstack([m.vertices for m in glasses.values()]))
  G,fit=glasses_transform(lm,subject_profile)
  G,front=fit_front_clearance(glasses,head,lm,subject_profile,G);kept=[];arms=[]
  for mesh in glasses.values():
   if getattr(mesh.visual.material,'alphaMode',None)=='BLEND':continue
   source_v=mesh.vertices.copy();mesh.visual.material=trimesh.visual.material.PBRMaterial(baseColorFactor=[24,25,28,255],metallicFactor=.15,roughnessFactor=.4)
   mesh.apply_transform(G);arms.append(refine_temples(mesh,source_v,head,lm,subject_profile,ears));kept.append(mesh)
  clear=surface_samples(head,np.vstack([m.vertices for m in kept]));hair,hair_fit=make_hair(catalog/'micket-side-part/asset.glb',head,lm)
  for kind in ['cap','hair']:
   folder=a.output/(name+'-'+kind);folder.mkdir();scene=trimesh.Scene();scene.add_geometry(head.copy(include_cache=True),node_name='Geometry_Only_Head',geom_name='Geometry_Only_Head')
   wear={k:baseline[k] for k in ['Cap_Original','Tassel_Original']} if kind=='cap' else {f'Hair_{i}':m for i,m in enumerate(hair.values())}
   for n,m in wear.items():scene.add_geometry(m.copy(include_cache=True),node_name=n,geom_name=n)
   for i,m in enumerate(kept):scene.add_geometry(m.copy(include_cache=True),node_name=f'Glasses_{i}',geom_name=f'Glasses_{i}')
   scene.apply_transform(np.linalg.inv(AXES));scene.export(folder/'scene.glb')
   report={'case':folder.name,'source_bundle_sha256':sha(source),'native_geometry_sha256':__import__('hashlib').sha256(vertices.tobytes()+faces.tobytes()).hexdigest(),
           'status':'GEOMETRY_REVIEW_ONLY','headwear':kind,'fit':fit,'front_clearance':front,'arm_refinement':arms,
           'head_clearance':clear,'frame_width_adjustment':width,'ear_support_anchors':ears.tolist(),'hair_fit':hair_fit if kind=='hair' else None,'head_crown_translation_m':float(offset),
           'no_new_reconstruction_run':True,'texture_or_identity_validation':False,'api_integrated':False,'ready_for_machine':False}
   write(folder/'report.json',report);cases.append(report)
 write(a.output/'report.json',{'policy':'cross-existing-rdf-accessory-test-v1','scope':'Existing geometry bundles; independent subject identity not established','cases':cases})
 print([(r['case'],r['head_clearance']['samples_below_minus_1mm']) for r in cases],flush=True)
if __name__=='__main__':main()
