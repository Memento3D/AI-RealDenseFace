"""Stream GLB buffers for a low-memory review proxy and head replacement.

Only the standard, embedded, non-sparse GLB layout exported by these local
trimesh trials is supported. Unsupported layouts fail explicitly.
"""
import io,json,struct,shutil
from pathlib import Path
import numpy as np,trimesh
from PIL import Image


def header(path):
    with Path(path).open('rb') as stream:
        magic,version,total=struct.unpack('<III',stream.read(12))
        if magic!=0x46546C67 or version!=2:raise ValueError('Expected GLB2')
        length,kind=struct.unpack('<II',stream.read(8))
        if kind!=0x4E4F534A:raise ValueError('Expected JSON chunk')
        data=json.loads(stream.read(length));size,kind=struct.unpack('<II',stream.read(8));offset=stream.tell()
        if kind!=0x004E4942 or offset+size!=total:raise ValueError('Expected one embedded BIN chunk')
        if len(data.get('buffers',[]))!=1 or 'uri' in data['buffers'][0]:raise ValueError('Expected one embedded buffer')
        return data,offset,size


def matrices(data):
    result={}
    def walk(index,parent):
        node=data['nodes'][index]
        if 'matrix' in node:local=np.array(node['matrix']).reshape(4,4).T
        else:
            local=trimesh.transformations.quaternion_matrix(np.array(node.get('rotation',[0,0,0,1]))[[3,0,1,2]])
            local[:3,:3]*=np.asarray(node.get('scale',[1,1,1]))[None,:];local[:3,3]=node.get('translation',[0,0,0])
        result[index]=parent@local
        for child in node.get('children',[]):walk(child,result[index])
    for node in data['scenes'][data.get('scene',0)].get('nodes',[]):walk(node,np.eye(4))
    return result


def replace_head(base,head,output):
    data,base_offset,base_size=header(base);new,new_offset,new_size=header(head)
    old_nodes=[i for i,n in enumerate(data['nodes']) if n.get('name')=='Personalized_Complete_Head' and 'mesh' in n]
    new_nodes=[i for i,n in enumerate(new['nodes']) if n.get('name')=='Personalized_Complete_Head' and 'mesh' in n]
    if len(old_nodes)!=1 or len(new_nodes)!=1:raise ValueError('Expected one named head in each GLB')
    if not np.allclose(matrices(data)[old_nodes[0]],matrices(new)[new_nodes[0]],atol=1e-10):raise ValueError('Head node coordinate frames differ')
    offsets={key:len(data.get(key,[])) for key in ['bufferViews','accessors','images','samplers','textures','materials','meshes']}
    for view in new.get('bufferViews',[]):view['buffer']=0;view['byteOffset']=view.get('byteOffset',0)+base_size
    for accessor in new.get('accessors',[]):
        if 'sparse' in accessor:raise ValueError('Sparse accessor is unsupported')
        accessor['bufferView']+=offsets['bufferViews']
    for image in new.get('images',[]):image['bufferView']+=offsets['bufferViews']
    for texture in new.get('textures',[]):
        texture['source']+=offsets['images']
        if 'sampler' in texture:texture['sampler']+=offsets['samplers']
    def texture_indices(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.endswith('Texture') and isinstance(item,dict) and 'index' in item:item['index']+=offsets['textures']
                else:texture_indices(item)
        elif isinstance(value,list):
            for item in value:texture_indices(item)
    for material in new.get('materials',[]):texture_indices(material)
    for mesh in new.get('meshes',[]):
        for primitive in mesh['primitives']:
            primitive['attributes']={key:index+offsets['accessors'] for key,index in primitive['attributes'].items()}
            primitive['indices']+=offsets['accessors']
            if 'material' in primitive:primitive['material']+=offsets['materials']
            if primitive.get('targets') or primitive.get('extensions'):raise ValueError('Morph/compressed primitive unsupported')
    data['nodes'][old_nodes[0]]['mesh']=new['nodes'][new_nodes[0]]['mesh']+offsets['meshes']
    removed=[]
    for node in data['nodes']:
        if node.get('name')=='Neck_Transition' and 'mesh' in node:
            del node['mesh'];removed.append('Neck_Transition')
    for key in offsets:data.setdefault(key,[]).extend(new.get(key,[]))
    data['buffers'][0]['byteLength']=base_size+new_size
    encoded=json.dumps(data,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4)
    with Path(output).open('xb') as target:
        target.write(struct.pack('<III',0x46546C67,2,12+8+len(encoded)+8+base_size+new_size));target.write(struct.pack('<II',len(encoded),0x4E4F534A));target.write(encoded)
        target.write(struct.pack('<II',base_size+new_size,0x004E4942))
        for source,offset,size in [(base,base_offset,base_size),(head,new_offset,new_size)]:
            with Path(source).open('rb') as stream:
                stream.seek(offset)
                while size:
                    chunk=stream.read(min(size,1024*1024))
                    if not chunk:raise ValueError('Truncated source buffer')
                    target.write(chunk);size-=len(chunk)
    return {'non_head_non_connector_nodes_and_original_binary_preserved':True,'removed_obsolete_connector_nodes':removed,'appended_head_bytes':new_size}


def crop_proxy(source,output,minimum_z=.65):
    data,offset,_=header(source);world=matrices(data)
    axes=np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,0],[0,0,0,1.]])
    sizes={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4};dtypes={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}
    def view_bytes(index):
        view=data['bufferViews'][index]
        with Path(source).open('rb') as stream:stream.seek(offset+view.get('byteOffset',0));return stream.read(view['byteLength'])
    def accessor(index):
        a=data['accessors'][index]
        if 'sparse' in a or a['componentType'] not in dtypes:raise ValueError('Unsupported accessor')
        view=data['bufferViews'][a['bufferView']];dtype=np.dtype(dtypes[a['componentType']]);width=sizes[a['type']]
        raw=view_bytes(a['bufferView']);stride=view.get('byteStride',dtype.itemsize*width)
        array=np.ndarray((a['count'],width),dtype=dtype,buffer=raw,offset=a.get('byteOffset',0),strides=(stride,dtype.itemsize)).copy()
        return array
    image_cache={};material_cache={}
    def image(texture_index):
        index=data['textures'][texture_index]['source']
        if index not in image_cache:image_cache[index]=Image.open(io.BytesIO(view_bytes(data['images'][index]['bufferView']))).copy()
        return image_cache[index]
    def material(index):
        if index not in material_cache:
            m=data['materials'][index];p=m.get('pbrMetallicRoughness',{});args={'name':m.get('name'),'baseColorFactor':np.asarray(p.get('baseColorFactor',[1.,1.,1.,1.]),float),'metallicFactor':p.get('metallicFactor',1),'roughnessFactor':p.get('roughnessFactor',1),'doubleSided':m.get('doubleSided',False),'alphaMode':m.get('alphaMode','OPAQUE')}
            for key in ['baseColorTexture','metallicRoughnessTexture']:
                if key in p:args[key]=image(p[key]['index'])
            for key in ['normalTexture','occlusionTexture','emissiveTexture']:
                if key in m:args[key]=image(m[key]['index'])
            if 'emissiveFactor' in m:args['emissiveFactor']=m['emissiveFactor']
            material_cache[index]=trimesh.visual.material.PBRMaterial(**args)
        return material_cache[index]
    scene=trimesh.Scene();counts=[]
    for index,matrix in world.items():
        node=data['nodes'][index]
        if 'mesh' not in node:continue
        transform=axes@matrix
        for part,primitive in enumerate(data['meshes'][node['mesh']]['primitives']):
            if primitive.get('mode',4)!=4 or primitive.get('extensions'):raise ValueError('Expected uncompressed triangles')
            attr=primitive['attributes'];v=accessor(attr['POSITION']).astype(float);v=v@transform[:3,:3].T+transform[:3,3]
            f=accessor(primitive['indices']).reshape(-1,3).astype(int);keep=(v[f,2]>=minimum_z).any(1)
            if not keep.any():continue
            ids,inverse=np.unique(f[keep].ravel(),return_inverse=True)
            uv=accessor(attr['TEXCOORD_0'])[ids] if 'TEXCOORD_0' in attr else np.zeros((len(ids),2));uv[:,1]=1-uv[:,1]
            normals=accessor(attr['NORMAL'])[ids].astype(float) if 'NORMAL' in attr else None
            if normals is not None:
                normals=normals@np.linalg.inv(transform[:3,:3]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-30)
            name=node.get('name',str(index))+(f'_{part}' if part else '')
            if 'material' not in primitive and 'COLOR_0' in attr:
                visual=trimesh.visual.ColorVisuals(vertex_colors=accessor(attr['COLOR_0'])[ids])
            else:
                mat=material(primitive['material']) if 'material' in primitive else trimesh.visual.material.PBRMaterial(baseColorFactor=[1.,1.,1.,1.],metallicFactor=0,roughnessFactor=1)
                visual=trimesh.visual.TextureVisuals(uv=uv,material=mat)
            mesh=trimesh.Trimesh(v[ids],inverse.reshape(-1,3),vertex_normals=normals,process=False,visual=visual)
            scene.add_geometry(mesh,node_name=name,geom_name=name);counts.append({'node':name,'source_triangles':len(f),'proxy_triangles':int(keep.sum())})
    scene.apply_transform(np.linalg.inv(axes));scene.export(output)
    return {'minimum_working_z_m':minimum_z,'policy':'Omit lower-body triangles for close camera review; visible geometry is not decimated','components':counts}
