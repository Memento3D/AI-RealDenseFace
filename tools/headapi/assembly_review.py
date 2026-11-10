"""Blender CPU review of an existing GLB; does not run reconstruction."""
import argparse
from pathlib import Path
import sys


def main():
    import bpy
    from mathutils import Vector
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    scene.cycles.device='CPU';scene.cycles.samples=8
    scene.render.resolution_x=scene.render.resolution_y=600
    scene.render.resolution_percentage=100
    world=bpy.data.worlds.new('Neutral review world');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.18,.18,.18,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.7
    scene.world=world
    for name,pos,power in [('Key',(-1,-2,2),90),('Fill',(1,-1,1),45)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.size=2
        obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj)
        obj.location=pos;obj.rotation_euler=(Vector((0,0,.65))-obj.location).to_track_quat('-Z','Y').to_euler()
    camera=bpy.data.objects.new('Review camera',bpy.data.cameras.new('Review camera'))
    scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='ORTHO'
    for name,position,target,scale in [
        ('full',(0,-3,.65),(0,0,.51),1.18),
        ('face',(0,-2,.9),(-.02,0,.87),.36),
        ('quarter',(-1,-2,.95),(-.02,0,.87),.36),
        ('back',(0,2,.92),(-.02,0,.87),.36)]:
        camera.location=position
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale=scale;scene.render.filepath=str(args.output/(name+'.png'))
        bpy.ops.render.render(write_still=True)


if __name__=='__main__':main()
