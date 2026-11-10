"""Blender adapter: keep personalized face, simplify body for standard web GLB."""
import argparse,hashlib,json,sys
from pathlib import Path

def main():
    import bpy
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--body-ratio',type=float,default=.25);p.add_argument('--body-texture-size',type=int,default=1024)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not 0<a.body_ratio<=1 or not 256<=a.body_texture_size<=4096:raise ValueError('Invalid web export policy')
    a.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(a.source),load_ui=False,use_scripts=False)
    scene=bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    original=bpy.data.objects['node_0'];original.select_set(True);bpy.context.view_layer.objects.active=original
    bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.mesh.separate(type='MATERIAL');bpy.ops.object.mode_set(mode='OBJECT')
    meshes=[o for o in scene.objects if o.type=='MESH']
    def used_materials(obj):
        return [obj.data.materials[i] for i in {p.material_index for p in obj.data.polygons}]
    faces=[o for o in meshes if any(m and m.name.startswith('Khoa_Face_Photo') for m in used_materials(o))]
    if len(faces)!=1:raise ValueError('Expected exactly one personalized face object')
    face=faces[0];face.name='Personalized_Face_Master'
    # Snapshot local coordinates/UV; every face polygon remains untouched.
    before_vertices=[tuple(v.co) for v in face.data.vertices];before_uv=[tuple(x.uv) for x in face.data.uv_layers.active.data]
    bodies=[o for o in meshes if o is not face and len(o.data.polygons)>100000]
    if len(bodies)!=1:raise ValueError('Expected one retained body mesh for the pinned template')
    body=[]
    for obj in bodies:
        before=len(obj.data.polygons);bpy.context.view_layer.objects.active=obj
        mod=obj.modifiers.new('Web_Body_Decimation','DECIMATE');mod.ratio=a.body_ratio;mod.use_collapse_triangulate=True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        body.append({'name':obj.name,'triangles_before':before,'triangles_after':len(obj.data.polygons)})
    resized=[]
    face_images=set()
    for mat in used_materials(face):
        if mat and mat.use_nodes:
            face_images.update(n.image.name for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image)
    textures=a.output/'body-textures';textures.mkdir()
    for image in list(bpy.data.images):
        if image.name in face_images or image.source not in ('FILE','GENERATED') or not image.has_data:continue
        width,height=image.size
        if max(width,height)<=a.body_texture_size:continue
        scale=a.body_texture_size/max(width,height);image.scale(max(1,round(width*scale)),max(1,round(height*scale)))
        image.filepath_raw=str(textures/(hashlib.sha256(image.name.encode()).hexdigest()[:12]+'.png'));image.file_format='PNG';image.save();image.pack()
        resized.append({'name':image.name,'before':[width,height],'after':list(image.size)})
    if before_vertices!=[tuple(v.co) for v in face.data.vertices] or before_uv!=[tuple(x.uv) for x in face.data.uv_layers.active.data]:raise ValueError('Face geometry or UV changed')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene.objects:
        if obj.type=='MESH':obj.select_set(True)
    bpy.context.view_layer.objects.active=face
    bpy.ops.export_scene.gltf(filepath=str(a.output/'scene.glb'),export_format='GLB',use_selection=True,export_yup=True,export_cameras=False,export_lights=False,export_extras=True)
    report={'policy':'standard_web_figurine_v2','status':'WEB_GLB_EXPORTED_PENDING_AUDIT','source_blend_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'body_ratio':a.body_ratio,'body':body,'body_images_resized':resized,'personalized_face_geometry_unchanged':True,'personalized_face_uv_unchanged':True,'decoder_extension_required':False,'display_only':True,'ready_for_machine':False,'bytes':(a.output/'scene.glb').stat().st_size}
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
