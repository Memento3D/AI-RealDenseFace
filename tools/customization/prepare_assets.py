"""Blender offline normalization and thumbnails; originals remain untouched."""
import argparse,json,sys,hashlib,math
from pathlib import Path

def main():
    import bpy,bmesh
    from mathutils import Matrix,Vector
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    sources=json.loads((a.source/'manifest.json').read_text())['assets'];catalog=[]
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in sources:
        source=a.source/entry['original']
        if source.suffix=='.zip':source=next((source.parent/'unpacked').glob('*.blend'))
        bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
        meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
        meshes.sort(key=lambda o:int(''.join(c for c in o.name.split('.')[0] if c.isdigit()) or 0))
        scene=bpy.context.scene
        for obj in scene.objects:obj.hide_render=True
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=4
        scene.render.resolution_x=scene.render.resolution_y=240;scene.render.resolution_percentage=100
        world=bpy.data.worlds.new('Asset review neutral');world.use_nodes=True
        world.node_tree.nodes['Background'].inputs['Color'].default_value=(.22,.22,.22,1)
        world.node_tree.nodes['Background'].inputs['Strength'].default_value=.8;scene.world=world
        for name,pos,power in [('AssetKey',(-.3,-.4,.5),10),('AssetFill',(.4,-.2,.2),5)]:
            d=bpy.data.lights.new(name,'AREA');d.energy=power;d.size=.6
            o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=pos
            o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler()
        camera=bpy.data.objects.new('AssetCamera',bpy.data.cameras.new('AssetCamera'));scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='ORTHO'
        camera.location=(0,-.5,.18);camera.rotation_euler=(-camera.location).to_track_quat('-Z','Y').to_euler()
        for index,original in enumerate(meshes):
            kind='glasses' if entry['id']=='ipoly3d-glasses' else 'hair'
            asset_id=('glasses-'+original.name.split('.')[0].replace('glasses','').zfill(2)) if kind=='glasses' else entry['id']
            folder=a.output/asset_id;folder.mkdir()
            data=bpy.data.meshes.new_from_object(original.evaluated_get(bpy.context.evaluated_depsgraph_get()),preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
            data.transform(original.matrix_world)
            points=[v.co.copy() for v in data.vertices]
            lower=Vector([min(v[k] for v in points) for k in range(3)]);upper=Vector([max(v[k] for v in points) for k in range(3)])
            center=(lower+upper)/2;width=upper.x-lower.x
            if width<=0:raise ValueError('Degenerate asset bounds')
            scale=(.14 if kind=='glasses' else .18)/width
            for v in data.vertices:v.co=(v.co-center)*scale
            model=bpy.data.objects.new(asset_id,data);scene.collection.objects.link(model)
            if kind=='hair':
                mat=bpy.data.materials.new('Preview neutral dark brown');mat.use_nodes=True
                shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.025,.018,.014,1);shader.inputs['Roughness'].default_value=.65
                model.data.materials.clear();model.data.materials.append(mat)
                for poly in model.data.polygons:poly.material_index=0;poly.use_smooth=True
            bpy.ops.object.select_all(action='DESELECT');model.select_set(True);bpy.context.view_layer.objects.active=model
            bpy.ops.export_scene.gltf(filepath=str(folder/'asset.glb'),export_format='GLB',use_selection=True,export_yup=True,export_cameras=False,export_lights=False)
            bpy.context.view_layer.update();camera.data.ortho_scale=max(model.dimensions)*1.3
            scene.render.filepath=str(folder/'thumbnail.png');bpy.ops.render.render(write_still=True)
            bm=bmesh.new();bm.from_mesh(data)
            boundary=sum(len(e.link_faces)==1 for e in bm.edges);nonmanifold=sum(len(e.link_faces)>2 for e in bm.edges);bm.free()
            data.calc_loop_triangles()
            catalog.append({'id':asset_id,'kind':kind,'file':str((folder/'asset.glb').relative_to(a.output)),
                'thumbnail':str((folder/'thumbnail.png').relative_to(a.output)),'sha256':digest(folder/'asset.glb'),
                'source_id':entry['id'],'source_object':original.name,'source_sha256':digest(source),
                'author':entry['author'],'license':entry['license'],'source_page':entry['source_page'],
                'nominal_design_width_m':.14 if kind=='glasses' else .18,'source_center':list(center),'source_uniform_scale':scale,
                'triangles':len(data.loop_triangles),'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,
                'appearance':'original material' if kind=='glasses' else 'neutral preview brown; original retained separately',
                'status':'LOCAL_CANDIDATE_NOT_FITTED','print_ready':False})
            bpy.data.objects.remove(model,do_unlink=True);bpy.data.meshes.remove(data)
            print('Prepared',asset_id,flush=True)
    manifest={'schema_version':1,'scope':'Local customization candidates; no BE asset upload or public selection UI',
        'coordinate_system':'glTF Y-up; source Blender Z-up; centered bounds, uncalibrated design dimensions',
        'assets':catalog}
    (a.output/'catalog.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cards=[]
    for asset in catalog:
        cards.append('<article><img src="'+asset['thumbnail'].replace('\\','/')+'"><b>'+asset['id']+'</b><p>'+str(asset['triangles'])+' triangles · '+asset['license']+'</p><a href="'+asset['file'].replace('\\','/')+'">GLB</a> · <a href="'+asset['source_page']+'">Source</a></article>')
    (a.output/'review.html').write_text('<meta charset="utf-8"><title>Memento3D local asset candidates</title><style>body{background:#191d24;color:#eee;font:15px sans-serif;padding:24px}section{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}article{background:#292f39;padding:12px;border-radius:10px}img{width:100%}b{display:block}a{color:#9bf}</style><h1>Local glasses & hair candidates</h1><p>Not fitted or print-certified. Original downloads and license evidence are retained separately.</p><section>'+''.join(cards)+'</section>',encoding='utf-8')
    print('Prepared catalog',len(catalog),'assets',flush=True)
if __name__=='__main__':main()
