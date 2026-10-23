"""Dependency-light OBJ validation and orthographic inspection previews."""
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from memento3d_head.domain.contracts import ValidationError


def validate_and_preview(obj: Path, output: Path) -> dict:
    """Reject invalid triangles; render three views without claiming printability."""
    vertices=[]
    faces=[]
    for line in obj.read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if not parts:
            continue
        if parts[0]=='v':
            vertices.append([float(value) for value in parts[1:4]])
        if parts[0]=='f':
            if len(parts)!=4:
                raise ValidationError('INVALID_MESH','Expected triangular faces')
            faces.append([int(value.split('/')[0])-1 for value in parts[1:]])
    v=np.asarray(vertices,dtype=float)
    f=np.asarray(faces,dtype=int)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3
            or not np.isfinite(v).all() or len(v)<4 or len(f)<4
            or f.min()<0 or f.max()>=len(v)):
        raise ValidationError('INVALID_MESH','Invalid vertices or face indices')
    extent=np.ptp(v,axis=0)
    if np.any(extent<=0):
        raise ValidationError('INVALID_MESH','Degenerate bounds')
    triangles=v[f]
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    if np.any(areas<=1e-10):
        raise ValidationError('INVALID_MESH','Degenerate triangle')
    canvas=Image.new('RGB',(1200,460),'#e9edf2')
    draw=ImageDraw.Draw(canvas)
    centered=v-(v.min(axis=0)+v.max(axis=0))/2
    scale=330/max(extent)
    for column,angle in enumerate((0,-45,45)):
        theta=math.radians(angle)
        rotation=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        rotated=centered@rotation.T
        tri=rotated[f]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        shade=np.clip(.35+.65*np.abs(normals@np.array([.2,.3,.93])),0,1)
        for index in np.argsort(tri[:,:,2].mean(axis=1)):
            points=[(float(x*scale+200+column*400),float(-y*scale+245)) for x,y,_ in tri[index]]
            gray=int(55+180*shade[index])
            draw.polygon(points,fill=(gray,gray,gray))
        draw.text((column*400+120,20),f'Canonical yaw {angle} deg',fill='black')
    canvas.save(output)
    return {'vertices':len(v),'faces':len(f),'bounds_mm':extent.tolist(),
            'finite':True,'nondegenerate':True,'preview':'orthographic; not photo-registered'}
