"""Blender-only adapter: complete CC0 hand with editable rig and short mesh hair."""
import argparse,json,hashlib,math,sys
from pathlib import Path

def main():
    import bpy,bmesh,numpy as np
    from mathutils import Vector,Matrix
    from mathutils.bvhtree import BVHTree
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.output.resolve()
    if out.exists():raise ValueError('Choose a fresh attempt directory')
    base=root/'runs/hand-hair-assembly-01';source=base/'body-head-cap.glb';asset=root/'runs/hand-hair-research-01/hand-realistic.blend';structure=base/'hand-structure.json'
    def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
    pin=json.loads((root/'runs/hand-hair-research-01/candidate-manifest.json').read_text())['hand-realistic.blend']['export_sha256']
    if sha(asset)!=pin:raise ValueError('Hand asset pin changed')
    prep=json.loads((base/'body-preparation.json').read_text())
    if sha(source)!=prep['result_sha256']:raise ValueError('Prepared body changed')
    out.mkdir(parents=True);bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(source))
    head=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('Personalized_Complete_Head'));body=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('Body_Cuff_Prepared'))
    before_head=np.array([tuple(v.co) for v in head.data.vertices]);before_head_matrix=np.array(head.matrix_world)
    # A slightly broader, shallow mortarboard over the fitted hair envelope.
    cup=next(o for o in bpy.context.scene.objects if o.name.startswith('Cap_Cup'));board=next(o for o in bpy.context.scene.objects if o.name.startswith('Cap_Board'))
    for obj in (cup,board):
        for vertex in obj.data.vertices:vertex.co.z+=.003
    board_points=np.array([tuple(board.matrix_world@v.co) for v in board.data.vertices]);bc=board_points.mean(0)
    for vertex in board.data.vertices:
        q=board.matrix_world@vertex.co;q.x=bc[0]+(q.x-bc[0])*1.1875;q.y=bc[1]+(q.y-bc[1])*1.1875;vertex.co=board.matrix_world.inverted()@q
    cup_points=np.array([tuple(cup.matrix_world@v.co) for v in cup.data.vertices]);cc=cup_points.mean(0)
    for vertex in cup.data.vertices:
        q=cup.matrix_world@vertex.co;q.x=cc[0]+(q.x-cc[0])*1.045;q.y=cc[1]+(q.y-cc[1])*1.045;vertex.co=cup.matrix_world.inverted()@q
    for obj in (cup,board):
        bevel=obj.modifiers.new('Soft_fabric_edge','BEVEL');bevel.width=.0007;bevel.segments=2
    # Import the intact hand; source face sets define phalanx memberships.
    with bpy.data.libraries.load(str(asset),link=False) as (src,dst):dst.objects=['Hand  - Realistic']
    hand=dst.objects[0];bpy.context.scene.collection.objects.link(hand);hand.name='Complete_Hand_CC0'
    info=json.loads(structure.read_text());groups={s['id']:set(s['vertices']) for s in info['face_sets']};verts=np.array([tuple(v.co) for v in hand.data.vertices]);chains={'index':[19,16,3],'middle':[18,15,4],'ring':[17,14,5],'little':[13,12,11],'thumb':[23,22,21]}
    joints={};tips={}
    for name,ids in chains.items():
        points=[verts[list(groups[left]&groups[right])].mean(0) for left,right in zip([24]+ids,ids)]
        distal=verts[list(groups[ids[-1]])];direction=distal.mean(0)-points[-1];projection=distal@direction;tip=distal[projection>=np.percentile(projection,85)].mean(0);points.append(tip);joints[name]=points;tips[name]=distal.mean(0)
    rig_data=bpy.data.armatures.new('Five_Finger_Rig');rig=bpy.data.objects.new('Five_Finger_Rig',rig_data);bpy.context.scene.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT');palm=rig_data.edit_bones.new('palm');palm.head=(0,0,.04);palm.tail=(0,0,-.075)
    for name,points in joints.items():
        parent=palm
        for i in range(3):
            bone=rig_data.edit_bones.new(f'{name}.{i}');bone.head=points[i];bone.tail=points[i+1];bone.parent=parent;bone.use_connect=i>0;bone.align_roll(Vector((0,1,0)));parent=bone
    bpy.ops.object.mode_set(mode='OBJECT')
    vgroups={name:hand.vertex_groups.new(name=name) for name in rig_data.bones.keys()};unweighted=0
    for index in range(len(verts)):
        weights={}
        for name,ids in chains.items():
            present=[i for i,g in enumerate(ids) if index in groups[g]]
            if present:
                for i in present:weights[f'{name}.{i}']=1
                if index in groups[24]:weights['palm']=1
        if not weights and index in groups[10]:
            nearest=min(tips,key=lambda n:np.linalg.norm(verts[index]-tips[n]));weights[f'{nearest}.2']=1
        if not weights:weights={'palm':1}
        total=sum(weights.values())
        for name,w in weights.items():vgroups[name].add([index],w/total,'REPLACE')
    # Place the hand by a nominal pinch point, then solve thumb and index independently.
    rotation=np.array([[2**-.5,0,-2**-.5],[0,-1,0],[-2**-.5,0,-2**-.5]])
    scale=.62;board_points=np.array([tuple(board.matrix_world@v.co) for v in board.data.vertices]);lo,hi=board_points.min(0),board_points.max(0)
    anchor=np.array([lo[0]+.004,lo[1]+.025,(lo[2]+hi[2])/2]);nominal=np.array([.052,-.020,-.105]);translation=anchor-scale*rotation@nominal
    transform=np.eye(4);transform[:3,:3]=scale*rotation;transform[:3,3]=translation;rig.matrix_world=Matrix(transform);hand.matrix_world=Matrix(transform)
    # Fit only the forearm stub to the template cuff. Five fingers retain base topology.
    wrist=np.asarray(prep['wrist_cut_point']);local_wrist=np.linalg.solve(transform[:3,:3],wrist-translation)
    edge_count={}
    for poly in hand.data.polygons:
        ids=list(poly.vertices)
        for u,vv in zip(ids,ids[1:]+ids[:1]):key=tuple(sorted((u,vv)));edge_count[key]=edge_count.get(key,0)+1
    boundary_ids=sorted({i for edge,n in edge_count.items() if n==1 for i in edge if verts[i,2]>.03})
    wrist_reference='open boundary ring'
    if len(boundary_ids)<8:
        boundary_ids=np.flatnonzero(verts[:,2]>verts[:,2].max()-.009).tolist();wrist_reference='closed forearm-end band, flattened under cuff'
    if len(boundary_ids)<8:raise ValueError('Hand asset lacks a supported wrist reference')
    boundary_center=verts[boundary_ids].mean(0);max_z=boundary_center[2];basis=hand.shape_key_add(name='Basis');fit=hand.shape_key_add(name='Wrist_Fit')
    for i,v in enumerate(verts):
        t=min(1,max(0,v[2]/max_z));fit.data[i].co=Vector(v)+Vector(((local_wrist[0]-boundary_center[0])*t,(local_wrist[1]-boundary_center[1])*t,(local_wrist[2]-boundary_center[2])*t))
        if i in boundary_ids:fit.data[i].co.z=local_wrist[2]
    fit.value=1
    for mod in hand.modifiers:
        if mod.type=='MULTIRES':mod.levels=min(2,mod.total_levels);mod.render_levels=min(2,mod.total_levels)
    arm=hand.modifiers.new('Five_Finger_Deformation','ARMATURE');arm.object=rig
    for name in chains:
        for i in range(3):
            pb=rig.pose.bones[f'{name}.{i}'];pb.rotation_mode='XYZ';pb.rotation_euler.x=([-0.35,-.75,-.35] if name!='thumb' else [-.25,-.35,-.2])[i]
            if name=='thumb' and i==0:pb.rotation_euler.z=.65
    target_rows=[]
    for name,zsign in [('index',1),('thumb',-1)]:
        point=anchor+np.array([0,0,zsign*.005]);target=bpy.data.objects.new(name+'_Contact_Target',None);target.empty_display_size=.006;target.empty_display_type='SPHERE';target.location=point;bpy.context.scene.collection.objects.link(target)
        con=rig.pose.bones[f'{name}.2'].constraints.new('IK');con.target=target;con.chain_count=3;con.use_stretch=False;con.iterations=200
        target_rows.append((name,target))
    bpy.context.view_layer.update()
    errors={name:float(((rig.matrix_world@rig.pose.bones[f'{name}.2'].tail)-target.location).length) for name,target in target_rows}
    # A simple skin material for the reusable template hand, not recovered skin texture.
    skin=bpy.data.materials.new('Generic_Template_Hand_Skin');skin.use_nodes=True;bsdf=skin.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.58,.36,.24,1);bsdf.inputs['Roughness'].default_value=.72
    hand.data.materials.clear();hand.data.materials.append(skin)
    for face in hand.data.polygons:face.use_smooth=True;face.material_index=0
    # A reusable lining band covers the design-space wrist join under the sleeve.
    fitted=np.array([tuple(fit.data[i].co) for i in boundary_ids]);ring_world=fitted@transform[:3,:3].T+translation;ex=rotation[:,0];ey=rotation[:,1];forward=-rotation[:,2];wrist_center=wrist
    radius_x=max(abs((ring_world-wrist_center)@ex))+.0015;radius_y=max(abs((ring_world-wrist_center)@ey))+.0015
    cuff_v=[];segments=48
    for inside in (False,True):
        for row in (0,1):
            center0=wrist_center+forward*(row*.012-.004);rx=radius_x*(1.35-.35*row)-(.0012 if inside else 0);ry=radius_y*(1.35-.35*row)-(.0012 if inside else 0)
            for j in range(segments):angle=2*math.pi*j/segments;cuff_v.append(center0+ex*rx*math.cos(angle)+ey*ry*math.sin(angle))
    cuff_f=[]
    for layer in (0,1):
        offset=layer*2*segments
        for j in range(segments):a0=offset+j;b0=offset+(j+1)%segments;c0=a0+segments;d0=b0+segments;cuff_f.extend([(a0,b0,c0),(b0,d0,c0)] if layer==0 else [(a0,c0,b0),(b0,c0,d0)])
    for row in (0,1):
        for j in range(segments):a0=row*segments+j;b0=row*segments+(j+1)%segments;c0=a0+2*segments;d0=b0+2*segments;cuff_f.extend([(a0,c0,b0),(b0,c0,d0)] if row==0 else [(a0,b0,c0),(b0,d0,c0)])
    cuff_mesh=bpy.data.meshes.new('Wrist_Lining_Join');cuff_mesh.from_pydata([v.tolist() for v in cuff_v],[],cuff_f);cuff_mesh.update();cuff_obj=bpy.data.objects.new('Wrist_Lining_Join',cuff_mesh);bpy.context.scene.collection.objects.link(cuff_obj)
    cloth=bpy.data.materials.new('Cuff_Lining_Cream');cloth.use_nodes=True;cloth.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.62,.52,.34,1);cloth.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.9;cuff_mesh.materials.append(cloth)
    for poly in cuff_mesh.polygons:poly.use_smooth=True
    # Short-hair preset: scalp-following base plus volumetric grouped locks.
    hair=bpy.data.materials.new('Short_Hair_Dark');hair.use_nodes=True;hair.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.009,.006,.004,1);hair.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.8
    hverts=np.array([tuple(head.matrix_world@v.co) for v in head.data.vertices]);hfaces=[list(p.vertices) for p in head.data.polygons];theta=np.arctan2(hverts[:,1]-.045,hverts[:,0]+.020);front=np.maximum(0,-np.sin(theta));back=np.maximum(0,np.sin(theta));line=.883+.022*front-.022*back;mask=hverts[:,2]>=line
    selected=[p for p in hfaces if mask[p].all()];mesh=bpy.data.meshes.new('Scalp_Hair_Base');mesh.from_pydata(hverts.tolist(),[],selected);mesh.update();hair_base=bpy.data.objects.new('Hair_Short_Base',mesh);bpy.context.scene.collection.objects.link(hair_base);mesh.materials.append(hair)
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);loose=[v for v in bm.verts if not v.link_faces]
    if loose:bmesh.ops.delete(bm,geom=loose,context='VERTS')
    bm.to_mesh(mesh);bm.free();mesh.update()
    solid=hair_base.modifiers.new('Hair_Volume','SOLIDIFY');solid.thickness=.0022;solid.offset=1
    for face in mesh.polygons:face.use_smooth=True
    bvh=BVHTree.FromPolygons([Vector(v) for v in hverts],hfaces,all_triangles=True)
    def front_point(x,z,offset=.002):
        hit,normal,_,_=bvh.ray_cast(Vector((x,-.3,z)),Vector((0,1,0)))
        if hit is None:return None
        return np.array([hit.x,hit.y-offset,hit.z])
    def lock(name,points,width,depth):
        pts=np.asarray(points);n=len(pts);sides=12;vertices=[]
        for i,point in enumerate(pts):
            tangent=pts[min(i+1,n-1)]-pts[max(i-1,0)];tangent/=max(np.linalg.norm(tangent),1e-12);u=np.array([1.,0,0]);u-=tangent*np.dot(u,tangent);u/=max(np.linalg.norm(u),1e-12);w=np.cross(tangent,u);t=i/(n-1);taper=max(.12,math.sin(math.pi*t)**.55)
            for j in range(sides):angle=2*math.pi*j/sides;vertices.append(point+u*math.cos(angle)*width*taper+w*math.sin(angle)*depth*taper)
        faces=[]
        for i in range(n-1):
            for j in range(sides):a0=i*sides+j;b0=i*sides+(j+1)%sides;c0=a0+sides;d0=b0+sides;faces.extend([(a0,b0,c0),(b0,d0,c0)])
        vertices.extend([pts[0],pts[-1]])
        for j in range(sides):faces.append((n*sides,(j+1)%sides,j));faces.append((n*sides+1,(n-1)*sides+j,(n-1)*sides+(j+1)%sides))
        data=bpy.data.meshes.new(name);data.from_pydata([v.tolist() for v in vertices],[],faces);data.update();obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj);data.materials.append(hair)
        for f in data.polygons:f.use_smooth=True
        return obj
    locks=[]
    for i,x in enumerate(np.linspace(-.065,.025,14)):
        pts=[]
        for t in np.linspace(0,1,9):
            xx=x+.005*(1-t);zz=.930*(1-t)+(.901+.0018*math.sin(i*1.7))*t;p0=front_point(xx,zz)
            if p0 is not None:pts.append(p0)
        if len(pts)>3:locks.append(lock('Hair_Fringe_'+str(i),pts,.0045,.0015))
    for i,x in enumerate(np.linspace(-.06,.022,9)):
        pts=[q for q in [front_point(x,z,.0025) for z in np.linspace(.941,.917,10)] if q is not None]
        if len(pts)>3:locks.append(lock('Hair_Crown_Lock_'+str(i),pts,.006,.0017))
    if not locks:raise ValueError('No supported hair locks generated')
    if not np.array_equal(before_head,np.array([tuple(v.co) for v in head.data.vertices])) or not np.array_equal(before_head_matrix,np.array(head.matrix_world)):raise ValueError('Personalized head changed')
    # One new tassel, referenced by the finished cap rather than the old hand patch.
    gold=bpy.data.materials.new('Cap_Tassel_Gold');gold.use_nodes=True;gold.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.35,.19,.035,1);gold.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.65
    path=[Vector((bc[0],bc[1],hi[2]+.0015)),Vector((hi[0]-.012,lo[1]+.012,hi[2]+.0015)),Vector((hi[0]-.012,lo[1]+.012,hi[2]-.080))]
    curve=bpy.data.curves.new('Cap_Cord_Path','CURVE');curve.dimensions='3D';curve.bevel_depth=.0012;curve.bevel_resolution=2;spline=curve.splines.new('POLY');spline.points.add(len(path)-1)
    for point,co in zip(spline.points,path):point.co=(*co,1)
    cord=bpy.data.objects.new('Cap_Cord_New',curve);bpy.context.scene.collection.objects.link(cord);curve.materials.append(gold)
    bpy.ops.mesh.primitive_cone_add(vertices=20,radius1=.0035,radius2=.0018,depth=.018,location=path[-1]+Vector((0,0,-.009)));bpy.context.object.name='Cap_Tassel_New';bpy.context.object.data.materials.append(gold)
    scene=bpy.context.scene;scene['trial_policy']='complete_hand_capped_wrist_short_hair_v3';scene['ready_for_machine']=False
    report={'policy':'complete_hand_capped_wrist_short_hair_v3','status':'EXPORTED_NEEDS_VISUAL_REVIEW','input_sha256':{str(q):sha(q) for q in [source,asset,structure,base/'body-preparation.json']},'code_sha256':sha(Path(__file__)),'source_hand_license':'CC0','source_hand_base_vertices':len(verts),'rig_bones':len(rig_data.bones),'finger_chains':list(chains),'ik_tail_errors_working':errors,'wrist_target_local':local_wrist.tolist(),'wrist_reference_mode':wrist_reference,'wrist_reference_vertices':len(boundary_ids),'wrist_lining_join':True,'hand_scale_design':scale,'head_geometry_and_transform_unchanged':True,'hair_preset':'short crop with grouped mesh locks; not inferred from photos','hair_lock_count':len(locks),'old_hand_removed':True,'rig_pose_anatomically_validated':False,'hair_cap_collisions_checked':False,'ready_for_machine':False}
    (out/'report.json').write_text(json.dumps(report,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'assembly.blend'))
    depsgraph=bpy.context.evaluated_depsgraph_get();static_mesh=bpy.data.meshes.new_from_object(hand.evaluated_get(depsgraph),preserve_all_data_layers=True,depsgraph=depsgraph);static_hand=bpy.data.objects.new('Complete_Hand_Posed_Static',static_mesh);bpy.context.scene.collection.objects.link(static_hand);static_hand.matrix_world=hand.matrix_world.copy()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if (obj.type=='MESH' and obj!=hand) or obj.type=='CURVE':obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out/'assembly.glb'),export_format='GLB',use_selection=True,export_animations=False,export_apply=True,export_skins=False,export_cameras=False,export_lights=False)
    print('MODULAR_HAND_HAIR_SCENE_EXPORTED',json.dumps(errors),flush=True)

if __name__=='__main__':main()
