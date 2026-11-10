"""Compose verified UV head and existing cap without altering either geometry."""
import argparse,json,struct
from pathlib import Path
from tools.headapi.service import sha,write


def inspect_glb(path):
    raw=Path(path).read_bytes()
    if len(raw)<20:raise ValueError('Incomplete GLB')
    magic,version,length=struct.unpack_from('<4sII',raw)
    if magic!=b'glTF' or version!=2 or length!=len(raw):raise ValueError('Invalid GLB header')
    size,kind=struct.unpack_from('<II',raw,12)
    if kind!=0x4e4f534a or 20+size>len(raw):raise ValueError('Missing GLB JSON chunk')
    data=json.loads(raw[20:20+size])
    if any(item.get('uri') and not item['uri'].startswith('data:') for item in data.get('buffers',[])+data.get('images',[])):raise ValueError('Scene contains external resource URI')
    textured=any(m.get('pbrMetallicRoughness',{}).get('baseColorTexture') is not None for m in data.get('materials',[]))
    return {'embedded_base_color_texture':textured,'bytes':len(raw),'external_resources':False}


def run(job,output,presentation_policy='none'):
    import numpy as np
    import trimesh
    from tools.headapi.hair_cap_worker import export_render_scene
    job=Path(job).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','geometry_union_performed':False,'ready_for_machine':False,'input_sha256':{}}
    try:
        if presentation_policy not in ('none','harmonized_head_v1'):raise ValueError('Unsupported presentation policy')
        uv=job/'rdf_uv/report.json';cap=job/'hair/report.json';bundle=job/'rdf/projection_bundle.npz'
        face=job/'rdf_uv/final/face_textured.glb';hair=job/'hair/hair_cap_native.glb'
        face_report=json.loads(uv.read_text());hair_report=json.loads(cap.read_text())
        if face_report.get('status')!='UV_FACE_EXPORTED' or hair_report.get('status')!='HAIR_CAP_EXPORTED':raise ValueError('UV head and hair-cap stages must succeed')
        if face_report.get('head_sha256')!=sha(face):raise ValueError('Head checksum mismatch')
        if hair_report.get('rdf_bundle_sha256')!=sha(bundle):raise ValueError('Hair cap belongs to another head bundle')
        if hair_report.get('output_sha256',{}).get('hair_cap_native.glb')!=sha(hair):raise ValueError('Hair-cap checksum mismatch')
        with np.load(bundle,allow_pickle=False) as b:triangles=b['vertices'][b['faces']].astype(np.float32)
        inspect_glb(face)
        mesh=trimesh.load(face,force='mesh',process=False)
        if not np.array_equal(mesh.vertices[mesh.faces].astype(np.float32),triangles):raise ValueError('Render head does not preserve RDF geometry')
        report['input_sha256']={str(p):sha(p) for p in [uv,cap,bundle,face,hair]}
        render_face,render_hair=face,hair
        if presentation_policy=='harmonized_head_v1':
            from tools.headapi.demo_appearance import run as style
            appearance=style(job,output/'appearance')
            if appearance['status']!='DEMO_APPEARANCE_EXPORTED':raise ValueError('Presentation stage did not complete')
            render_face=output/'appearance/head.glb';render_hair=output/'appearance/hair.glb'
            report['appearance_report_sha256']=sha(output/'appearance/report.json')
        manifest=export_render_scene(render_face,render_hair,output/'scene.glb')
        scene=trimesh.load(output/'scene.glb',force='scene',process=False)
        for node,source in [('RDF_Head',face),('Hair_Cap',hair)]:
            transform,geometry=scene.graph[node];actual=scene.geometry[geometry];expected=trimesh.load(source,force='mesh',process=False)
            if not np.allclose(transform,np.eye(4)) or not np.array_equal(actual.vertices[actual.faces].astype(np.float32),expected.vertices[expected.faces].astype(np.float32)):raise ValueError('Scene component geometry changed: '+node)
        inspection=inspect_glb(output/'scene.glb')
        if not inspection['embedded_base_color_texture']:raise ValueError('Scene lost UV base-color texture')
        for p,digest in report['input_sha256'].items():
            if sha(p)!=digest:raise ValueError('Input changed during scene export')
        write(output/'scene.manifest.json',manifest)
        report.update(status='UV_SCENE_EXPORTED',scene_sha256=sha(output/'scene.glb'),scene_inspection=inspection,
                      head_geometry_preserved=True,hair_geometry_preserved=True,manufacturing_mesh=False,presentation_policy=presentation_policy)
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));raise
    finally:write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--presentation-policy',choices=['none','harmonized_head_v1'],default='none');a=p.parse_args();run(a.job,a.output,a.presentation_policy)
