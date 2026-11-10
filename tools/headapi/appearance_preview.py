"""Software scene review: photographic head colors and lit cap vertex colors."""
import numpy as np
import trimesh
from PIL import Image,ImageDraw


def preview_scene(head_path,hair_path,path,size=360,angles=(0,90,180,270),elevation=0,shade_head=False):
    head=trimesh.load(head_path,force='mesh',process=False)
    hair=trimesh.load(hair_path,force='mesh',process=False) if hair_path else trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),dtype=int),process=False)
    atlas=np.asarray(head.visual.material.baseColorTexture.convert('RGB'));uv=head.visual.uv[head.faces]
    vertices=np.concatenate([head.vertices,hair.vertices]);faces=np.r_[head.faces,hair.faces+len(head.vertices)]
    colors=np.asarray(hair.visual.vertex_colors)[:,:3].astype(float) if hair.visual.kind else np.full((len(hair.vertices),3),200.)
    # UV exports duplicate triangle vertices; rebuild smooth normals for review lighting.
    unique,inverse=np.unique(head.vertices,axis=0,return_inverse=True)
    smooth=trimesh.Trimesh(unique,inverse[head.faces],process=False).vertex_normals[inverse]
    normals=np.concatenate([smooth,hair.vertex_normals])[:,[2,0,1]];v=vertices[:,[2,0,1]].copy();v-=(v.min(0)+v.max(0))/2
    scale=.84*size/np.ptp(v,axis=0).max();sheet=Image.new('RGB',(size*len(angles),size+24),'#20252c');draw=ImageDraw.Draw(sheet)
    for index,angle in enumerate(angles):
        a=np.radians(angle);z=v[:,0]*np.cos(a)+v[:,1]*np.sin(a);x=-v[:,0]*np.sin(a)+v[:,1]*np.cos(a)
        e=np.radians(elevation);y=v[:,2]*np.cos(e)-z*np.sin(e);z=v[:,2]*np.sin(e)+z*np.cos(e)
        xy=np.c_[x*scale+size/2,-y*scale+size/2]
        nx=-normals[:,0]*np.sin(a)+normals[:,1]*np.cos(a);nz=normals[:,0]*np.cos(a)+normals[:,1]*np.sin(a)
        ny=normals[:,2]*np.cos(e)-nz*np.sin(e);nz=normals[:,2]*np.sin(e)+nz*np.cos(e)
        light=np.maximum((-.3*nx+.4*ny+nz)/np.sqrt(1.25),0)
        hair_colors=colors*(.35+.65*light[len(head.vertices):,None]);pixels=np.full((size,size,3),[35,40,47],np.uint8);depth=np.full((size,size),-np.inf)
        for j,face in enumerate(faces):
            tri=xy[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),size-1)
            if (hi<lo).any():continue
            p,q,r=tri;den=(q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1])
            if abs(den)<1e-10:continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
            b0=((q[1]-r[1])*(xx-r[0])+(r[0]-q[0])*(yy-r[1]))/den;b1=((r[1]-p[1])*(xx-r[0])+(p[0]-r[0])*(yy-r[1]))/den;b2=1-b0-b1
            values=b0*z[face[0]]+b1*z[face[1]]+b2*z[face[2]];section=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            keep=(b0>=0)&(b1>=0)&(b2>=0)&(values>section)
            if not keep.any():continue
            if j<len(head.faces):
                tex=b0[...,None]*uv[j,0]+b1[...,None]*uv[j,1]+b2[...,None]*uv[j,2]
                tx=np.clip((tex[...,0]*atlas.shape[1]).astype(int),0,atlas.shape[1]-1);ty=np.clip(((1-tex[...,1])*atlas.shape[0]).astype(int),0,atlas.shape[0]-1);rgb=atlas[ty,tx]
                if shade_head:
                    lighting=b0*light[face[0]]+b1*light[face[1]]+b2*light[face[2]]
                    rgb=np.clip(rgb*(.35+.65*lighting[...,None]),0,255).astype(np.uint8)
            else:
                ids=hair.faces[j-len(head.faces)];rgb=np.clip(b0[...,None]*hair_colors[ids[0]]+b1[...,None]*hair_colors[ids[1]]+b2[...,None]*hair_colors[ids[2]],0,255).astype(np.uint8)
            pixels[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=rgb[keep];section[keep]=values[keep]
        sheet.paste(Image.fromarray(pixels),(index*size,24));draw.text((index*size+8,5),str(angle)+' deg / demo appearance',fill='white')
    sheet.save(path)
