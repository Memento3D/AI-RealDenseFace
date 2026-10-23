"""CPU smooth-normal rasterization with per-pixel depth; geometry stays unchanged."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from memento3d_head.domain.contracts import ValidationError

ANGLES=(0,-45,45,-90,90)
RENDERER_VERSION=1


def read_geometry(obj: Path) -> tuple[np.ndarray,np.ndarray]:
    """Load positive-index triangle OBJ from the MICA export boundary."""
    vertices=[]
    faces=[]
    if obj.stat().st_size>100_000_000:
        raise ValidationError('INVALID_MESH','Preview OBJ exceeds 100 MB')
    for line in obj.read_text(encoding='utf-8').splitlines():
        fields=line.split()
        if not fields:
            continue
        if fields[0]=='v':
            if len(fields)<4:
                raise ValidationError('INVALID_MESH','Invalid vertex record')
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0]=='f':
            if len(fields)!=4:
                raise ValidationError('INVALID_MESH','Expected triangular faces')
            faces.append([int(value.split('/')[0])-1 for value in fields[1:]])
    v=np.asarray(vertices,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3
            or len(v)<4 or len(f)<4 or not np.isfinite(v).all()
            or f.min()<0 or f.max()>=len(v)):
        raise ValidationError('INVALID_MESH','Invalid coordinates or triangle indices')
    extent=np.ptp(v,axis=0)
    triangles=v[f]
    area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    if np.any(extent<=0) or np.any(np.linalg.norm(area,axis=1)<=1e-10):
        raise ValidationError('INVALID_MESH','Degenerate geometry')
    return v,f


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals without moving vertices."""
    triangles=vertices[faces]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    result=np.zeros_like(vertices,dtype=np.float64)
    for corner in range(3):
        np.add.at(result,faces[:,corner],normal)
    return result/np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)


def rasterize(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
              width: int, height: int, scale: float) -> tuple[np.ndarray,np.ndarray]:
    """Render toward +Z using barycentric normals and a per-pixel z-buffer."""
    pixels=np.full((height,width,3),(233,237,242),dtype=np.uint8)
    depth=np.full((height,width),-np.inf,dtype=np.float64)
    points=np.column_stack((vertices[:,0]*scale+width/2,
                            -vertices[:,1]*scale+height/2,vertices[:,2]))
    light=np.array([.25,.35,.9]); light/=np.linalg.norm(light)
    fill=np.array([-.65,.15,.75]); fill/=np.linalg.norm(fill)
    for indices in faces:
        a,b,c=points[indices]
        x0=max(0,int(math.floor(min(a[0],b[0],c[0]))))
        x1=min(width-1,int(math.ceil(max(a[0],b[0],c[0]))))
        y0=max(0,int(math.floor(min(a[1],b[1],c[1]))))
        y1=min(height-1,int(math.ceil(max(a[1],b[1],c[1]))))
        if x0>x1 or y0>y1:
            continue
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-10:
            continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        xx=xx+.5; yy=yy+.5
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w=1-u-v
        z=u*a[2]+v*b[2]+w*c[2]
        region=depth[y0:y1+1,x0:x1+1]
        visible=(u>=-1e-8)&(v>=-1e-8)&(w>=-1e-8)&(z>region)
        if not visible.any():
            continue
        n=(u[visible,None]*normals[indices[0]]+v[visible,None]*normals[indices[1]]
           +w[visible,None]*normals[indices[2]])
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        # Two-sided lighting only; visibility is controlled by the depth buffer.
        n=np.where(n[:,2:3]<0,-n,n)
        intensity=.24+.58*np.maximum(n@light,0)+.18*np.maximum(n@fill,0)
        gray=np.clip(255*intensity,0,255).astype(np.uint8)
        pixels[y0:y1+1,x0:x1+1][visible]=gray[:,None]
        region[visible]=z[visible]
    return pixels,depth


def render_smooth(obj: Path, destination: Path) -> dict:
    """Render five canonical views; no sculpting, subdivision or texture changes."""
    vertices,faces=read_geometry(obj)
    normals=vertex_normals(vertices,faces)
    extent=np.ptp(vertices,axis=0)
    centered=vertices-(vertices.min(axis=0)+vertices.max(axis=0))/2
    # One scale across all five views, derived from maximum projected bounds.
    projections=[]
    for angle in ANGLES:
        theta=math.radians(angle)
        rotation=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        projections.append((centered@rotation.T,normals@rotation.T))
    scale=min(min(330/max(np.ptp(v[:,0]),1e-12),350/max(np.ptp(v[:,1]),1e-12)) for v,_ in projections)
    canvas=Image.new('RGB',(1200,940),'#e9edf2')
    draw=ImageDraw.Draw(canvas)
    for index,(angle,(rotated,rotated_normals)) in enumerate(zip(ANGLES,projections)):
        column,row=index%3,index//3
        pixels,_=rasterize(rotated,faces,rotated_normals,400,400,scale)
        canvas.paste(Image.fromarray(pixels),(column*400,row*460+40))
        draw.text((column*400+100,row*460+12),f'Canonical yaw {angle} deg',fill='black')
    draw.multiline_text((815,565),'Smooth vertex normals\nDepth-tested CPU preview\n\nSame vertices and triangles\nNo texture / no subdivision\nNot aligned to photo cameras\nNot a likeness score',fill='black',spacing=10)
    draw.text((15,920),'Display change only. Geometry, texture and identity are not improved by smoothing.',fill='black')
    canvas.save(destination)
    return {'renderer':'cpu_smooth_normals','renderer_version':RENDERER_VERSION,
            'angles_degrees':list(ANGLES),'vertices':int(len(vertices)),'faces':int(len(faces)),
            'bounds_mm':extent.tolist(),'geometry_modified':False,'projection':'orthographic',
            'photo_registered':False,'subdivision':False,'depth_test':True}
