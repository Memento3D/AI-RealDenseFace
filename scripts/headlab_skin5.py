"""Remove source-photo hair-color spill from newly bare posterior skin.

Only appearance is edited. Original eye/mouth/front-face atlas cells are kept.
Fill is an approximate source skin color, not measured posterior albedo.
"""
import io,math
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from add_graduation_cap import load,save


def clean_skin_atlas(source,destination,shared,group):
    doc,blob=load(source);data=shared['data'];points=shared['source_mm'].reshape(-1,3,3).mean(1)
    nearest=cKDTree(group['work'].vertices).query(points)[1];bare=group['field'][nearest]<.25
    normals=np.asarray(data['normals']).reshape(-1,3,3).mean(1)
    bare&=normals[:,2]<.15
    bare&=shared['components']==0
    coordinates=np.asarray(data['coordinates']).reshape(3,-1,3,3).mean(2)
    weights=np.asarray(data['weights']).reshape(-1,3,3).mean(1)
    hair_photo=np.zeros(len(points),bool)
    for i in range(3):
        q=coordinates[i];height,width=shared['labels'][i].shape
        x=np.floor(q[:,0]/q[:,2]*width).astype(int);y=np.floor((1-q[:,1]/q[:,2])*height).astype(int)
        valid=(x>=0)&(x<width)&(y>=0)&(y<height)&(weights[:,i]>.05)
        ids=np.flatnonzero(valid);hair_photo[ids]|=shared['labels'][i][y[ids],x[ids]]==17
    selected=bare&hair_photo
    image=doc['images'][0];view=doc['bufferViews'][image['bufferView']];offset=view.get('byteOffset',0)
    atlas=np.asarray(Image.open(io.BytesIO(blob[offset:offset+view['byteLength']])).convert('RGB')).copy()
    grid=math.ceil(math.sqrt(len(points)));cell=atlas.shape[0]//grid
    skin=np.uint8(np.clip(np.asarray(data['skin']),0,1)*255)
    for i in np.flatnonzero(selected):
        y,x=divmod(int(i),grid);atlas[y*cell:(y+1)*cell,x*cell:(x+1)*cell]=skin
    buffer=io.BytesIO();Image.fromarray(atlas).save(buffer,format='PNG');raw=buffer.getvalue()
    while len(blob)%4:blob.append(0)
    doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(raw)});blob.extend(raw)
    image['bufferView']=len(doc['bufferViews'])-1;save(destination,doc,blob)
    return {'posterior_bare_skin_triangles_repainted':int(selected.sum()),'geometry_changed':False,'front_face_atlas_preserved':True,'fill':'approximate source skin color, not recovered rear albedo'}
