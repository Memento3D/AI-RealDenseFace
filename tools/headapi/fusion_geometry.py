"""CPU z-buffer and robust positive similarity, shared by the MVP assembly stages."""
import numpy as np


def raster(vertices,faces,colors,size=512,camera=None,angle=0):
    v=np.asarray(vertices,float);f=np.asarray(faces,int);colors=np.asarray(colors,float)[:,:3]
    if camera is None:
        a=np.deg2rad(angle);rot=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        p=v@rot.T;center=(p.min(0)+p.max(0))/2;scale=.86*size/max(np.ptp(p,axis=0).max(),1e-9)
        xy=np.column_stack(((p[:,0]-center[0])*scale+size/2,-(p[:,1]-center[1])*scale+size/2));depth=-p[:,2]
    else:
        q=np.column_stack((v,np.ones(len(v))))@np.asarray(camera).T
        depth=q[:,2];xy=q[:,:2]/q[:,2,None]
    image=np.full((size,size,3),[180,180,180],dtype=np.uint8)
    points=np.full((size,size,3),np.nan,dtype=np.float32);indices=np.full((size,size),-1,dtype=np.int32);buffer=np.full((size,size),np.inf)
    for fi,face in enumerate(f):
        if camera is not None and np.any(depth[face]<=1e-8):continue
        t=xy[face];lo=np.maximum(np.floor(t.min(0)).astype(int),0);hi=np.minimum(np.ceil(t.max(0)).astype(int),size-1)
        if np.any(lo>hi):continue
        x0,y0=t[0];x1,y1=t[1];x2,y2=t[2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-9:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1
        weights=np.stack((w0,w1,w2),axis=-1)
        if camera is not None:
            inv=weights/depth[face];total=inv.sum(-1);z=1/np.maximum(total,1e-20);weights=inv/np.maximum(total[...,None],1e-20)
        else:z=weights@depth[face]
        sub=buffer[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        mask=(w0>=-1e-7)&(w1>=-1e-7)&(w2>=-1e-7)&(z<sub)
        sub[mask]=z[mask]
        image[lo[1]:hi[1]+1,lo[0]:hi[0]+1][mask]=np.clip((weights@colors[face])[mask],0,255).astype(np.uint8)
        points[lo[1]:hi[1]+1,lo[0]:hi[0]+1][mask]=(weights@v[face])[mask]
        indices[lo[1]:hi[1]+1,lo[0]:hi[0]+1][mask]=fi
    return image,points,indices


def sample_surface(points,uv,radius=6):
    result=[];valid=[]
    for x,y in np.asarray(uv,float):
        x=int(round(x));y=int(round(y));lo=max(0,x-radius);hi=min(points.shape[1],x+radius+1);top=max(0,y-radius);bottom=min(points.shape[0],y+radius+1)
        rows,cols=np.where(np.isfinite(points[top:bottom,lo:hi,0]))
        if not len(rows):result.append([0,0,0]);valid.append(False);continue
        k=np.argmin((cols+lo-x)**2+(rows+top-y)**2)
        result.append(points[rows[k]+top,cols[k]+lo]);valid.append(True)
    return np.asarray(result,float),np.asarray(valid,bool)


def similarity(source,target,iterations=6):
    x=np.asarray(source,float);y=np.asarray(target,float)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<6 or not np.isfinite(np.r_[x,y]).all():raise ValueError('Insufficient finite 3D landmark pairs')
    if np.linalg.matrix_rank(x-x.mean(0),tol=1e-8)<2:raise ValueError('Degenerate registration controls')
    weights=np.ones(len(x))
    for _ in range(iterations):
        weights/=weights.sum();cx=(x*weights[:,None]).sum(0);cy=(y*weights[:,None]).sum(0);a=x-cx;b=y-cy
        u,s,vt=np.linalg.svd((b*weights[:,None]).T@a);d=np.ones(3);d[-1]=np.sign(np.linalg.det(u@vt));rotation=u@np.diag(d)@vt
        scale=float((s*d).sum()/max((weights*(a*a).sum(1)).sum(),1e-15))
        if not .001<scale<1000:raise ValueError('Unreasonable registration scale')
        translation=cy-scale*rotation@cx
        errors=np.linalg.norm(x@(scale*rotation).T+translation-y,axis=1)
        limit=max(float(np.median(errors)*1.5),1e-5);weights=np.minimum(1,limit/np.maximum(errors,1e-12))
    transform=np.eye(4);transform[:3,:3]=scale*rotation;transform[:3,3]=translation
    return transform,errors


def transform_points(vertices,matrix):return np.asarray(vertices)@matrix[:3,:3].T+matrix[:3,3]


def preview_assembly(mesh,path,gray=False,size=320):
    """Review lighting only; native RDF coordinates are not modified."""
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(size*4,size+24),(26,31,38));draw=ImageDraw.Draw(sheet)
    colors=np.full((len(mesh.vertices),3),185) if gray else np.asarray(mesh.visual.vertex_colors)[:,:3]
    light=np.array([-.3,.5,1.]);light/=np.linalg.norm(light)
    for panel,angle in enumerate([0,90,180,270]):
        rgb,_,indices=raster(mesh.vertices,mesh.faces,colors,size,angle=angle)
        a=np.deg2rad(angle);rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        normals=np.asarray(mesh.face_normals)@rotation.T;shade=.45+.55*np.maximum(normals@light,0)
        valid=indices>=0;rgb[valid]=np.clip(rgb[valid]*shade[indices[valid],None],0,255).astype('uint8');rgb[~valid]=[35,40,47]
        sheet.paste(Image.fromarray(rgb),(panel*size,24));draw.text((panel*size+8,5),str(angle)+' deg / MVP',fill='white')
    sheet.save(path)
