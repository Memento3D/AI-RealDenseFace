"""Overlapping-window landmark detection; all coordinates return to native pixels."""
import numpy as np
from PIL import Image
from head24_policy import InputRejected


def scan_windows(size):
    w,h=size;windows=[(0,0,w,h)]
    def starts(length,side):
        return sorted(set(list(range(0,max(1,length-side+1),max(1,side//2)))+[length-side]))
    for factor in (.65,.40):
        side=max(128,int(min(w,h)*factor))
        if side>=min(w,h):continue
        for y in starts(h,side):
            for x in starts(w,side):windows.append((x,y,x+side,y+side))
    if len(windows)>64:raise InputRejected('DETECTION_SCAN_BUDGET: unsupported aspect ratio')
    return list(dict.fromkeys(windows))


def map_detection(normalized,window,check_edges=True):
    p=np.asarray(normalized,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<400 or not np.isfinite(p).all():return None
    lo=p.min(0);hi=p.max(0)
    if (hi<=lo).any():return None
    if check_edges and ((lo<.015).any() or (hi>.985).any()):return None
    x0,y0,x1,y1=window
    points=p*np.array([x1-x0,y1-y0])+[x0,y0]
    return {'points':points,'box':np.r_[points.min(0),points.max(0)],'window':list(window),'effective_face_fraction':float(min(hi-lo))}


def same_face(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    area=lambda box:float(np.prod(np.maximum(box[2:]-box[:2],0)))
    intersect=area(np.r_[np.maximum(a[:2],b[:2]),np.minimum(a[2:],b[2:])])
    iou=intersect/max(area(a)+area(b)-intersect,1e-12)
    distance=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
    scale=min(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]))
    return bool(iou>=.35 and distance<=.3*scale)


def merge_detections(candidates):
    clusters=[]
    for c in sorted(candidates,key=lambda r:r['effective_face_fraction'],reverse=True):
        existing=next((r for r in clusters if same_face(c['box'],r['box'])),None)
        source={'window':c['window'],'variant':c['variant']}
        if existing is None:
            r=dict(c);r['observations']=[source];clusters.append(r)
        else:existing['observations'].append(source)
    return clusters


def scan_faces(image,detect):
    windows=scan_windows(image.size);candidates=[]
    stats={'method':'overlapping native-pixel windows; NMS-like duplicate grouping','windows':len(windows),'calls':0,'variants':['original','shadow_lift_detection_only'],'raw_face_observations':0,'invalid_or_tile_edge_observations':0,'full_frame_original_faces':None,'no_super_resolution':True,'source_colors_modified':False}
    lut=[round(255*(i/255.)**.6) for i in range(256)]
    for variant in stats['variants']:
        for index,window in enumerate(windows):
            tile=image.crop(window)
            if variant!='original':tile=tile.point(lut*3)
            result=detect(np.array(tile))
            if index==0 and variant=='original':stats['full_frame_original_faces']=len(result)
            stats['calls']+=1;stats['raw_face_observations']+=len(result)
            for p in result:
                candidate=map_detection(p,window,check_edges=index!=0)
                if candidate is None:stats['invalid_or_tile_edge_observations']+=1;continue
                candidate['variant']=variant;candidates.append(candidate)
            if stats['calls']%12==0:print('Detection windows:',stats['calls'],'/',len(windows)*2,flush=True)
    merged=merge_detections(candidates)
    stats['distinct_face_candidates']=[{'box_native':r['box'].tolist(),'native_width':float(r['box'][2]-r['box'][0]),'native_height':float(r['box'][3]-r['box'][1]),'observations':r['observations'],'distinct_window_support':len({tuple(o['window']) for o in r['observations']})} for r in merged]
    return merged,stats
