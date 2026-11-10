"""Exact subpixel triangle depth queries using conservative 2D bounding-box bins."""
import numpy as np


class ContinuousDepth:
    def __init__(self,vertices,faces,k,size,cell_size=16):
        vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
        if not np.isfinite(vertices).all() or np.any(vertices[:,2]<=0):raise ValueError('All mesh depths must be positive finite')
        self.width,self.height=size;self.cell_size=cell_size
        homogeneous=vertices@np.asarray(k,float).T
        xy=homogeneous[:,:2]/homogeneous[:,2,None]
        self.tri=xy[faces];self.z=vertices[faces,2]
        a,b,c=self.tri.transpose(1,0,2)
        self.den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        self.bins={}
        for index,triangle in enumerate(self.tri):
            if abs(self.den[index])<1e-12:continue
            low=np.maximum(np.floor(triangle.min(0)/cell_size).astype(int),0)
            high=np.minimum(np.floor(triangle.max(0)/cell_size).astype(int),[(self.width-1)//cell_size,(self.height-1)//cell_size])
            for y in range(low[1],high[1]+1):
                for x in range(low[0],high[0]+1):self.bins.setdefault((x,y),[]).append(index)
        self.bins={key:np.asarray(value,int) for key,value in self.bins.items()}

    def query(self,xy):
        xy=np.asarray(xy,float);out=np.full(len(xy),np.inf)
        valid=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<self.width)&(xy[:,1]>=0)&(xy[:,1]<self.height)
        ids=np.where(valid)[0]
        if not len(ids):return out
        cells=np.floor(xy[ids]/self.cell_size).astype(int)
        unique,inverse=np.unique(cells,axis=0,return_inverse=True)
        for group,cell in enumerate(unique):
            candidates=self.bins.get(tuple(cell))
            if candidates is None:continue
            indices=ids[inverse==group];a,b,c=self.tri[candidates].transpose(1,0,2)
            den=self.den[candidates];z=self.z[candidates]
            for start in range(0,len(indices),256):
                selected=indices[start:start+256];p=xy[selected]
                wa=((b[:,1]-c[:,1])*(p[:,0,None]-c[:,0])+(c[:,0]-b[:,0])*(p[:,1,None]-c[:,1]))/den
                wb=((c[:,1]-a[:,1])*(p[:,0,None]-c[:,0])+(a[:,0]-c[:,0])*(p[:,1,None]-c[:,1]))/den
                wc=1-wa-wb
                inside=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)
                inverse_depth=wa/z[:,0]+wb/z[:,1]+wc/z[:,2]
                depths=np.divide(1,inverse_depth,out=np.full_like(inverse_depth,np.inf),where=inside&(inverse_depth>0))
                out[selected]=depths.min(1)
        return out
