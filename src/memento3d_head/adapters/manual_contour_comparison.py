"""Diagnostic ordered silhouette arcs. Anatomical correspondence is not verified."""
import numpy as np
from PIL import Image, ImageDraw
from memento3d_head.domain.contracts import ValidationError


def resample(points, count=64):
    points=np.asarray(points,dtype=float)
    distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if distances[-1]<1e-6:raise ValidationError('DEGENERATE_ARC','Contour arc has zero length')
    at=np.linspace(0,distances[-1],count)
    return np.column_stack([np.interp(at,distances,points[:,i]) for i in range(2)])


def compare_arc(points, outlines):
    """Choose between both endpoint-bounded arcs, preserving contour order.

    The endpoints are nearest silhouette samples, not anatomical landmarks.
    Equal normalized arclength is a diagnostic pairing, not a fitting loss.
    """
    reference=resample(points)
    options=[]
    for outline in outlines:
        if len(outline)<3:continue
        dense=resample(list(outline)+[outline[0]],count=min(8192,max(128,len(outline)*4)))[:-1]
        start=int(np.argmin(np.linalg.norm(dense-reference[0],axis=1)))
        end=int(np.argmin(np.linalg.norm(dense-reference[-1],axis=1)))
        if start==end:continue
        for direction in (1,-1):
            indices=[start];current=start
            for _ in range(len(dense)):
                current=(current+direction)%len(dense);indices.append(current)
                if current==end:break
            arc=resample(dense[indices]);errors=np.linalg.norm(arc-reference,axis=1)
            options.append((float(errors.mean()),arc,errors))
    if not options:raise ValidationError('NO_COMPARABLE_ARC','No distinct silhouette endpoints found')
    options.sort(key=lambda item:item[0]);mean,arc,errors=options[0]
    return {'mean_ordered_distance_px':mean,'max_ordered_distance_px':float(errors.max()),
            'p95_ordered_distance_px':float(np.percentile(errors,95)),
            'endpoint_distances_px':[float(errors[0]),float(errors[-1])],
            'reference_samples_px':reference.tolist(),'mesh_samples_px':arc.tolist(),
            'candidate_arc_count':len(options),'correspondence_verified':False,
            'method':'nearest_endpoints_two_arcs_equal_normalized_arclength_64',
            'warning':'Whole-head silhouette may select ear/neck; review overlay before interpretation'}


def render_comparison(photo, destination, curves, results):
    with Image.open(photo) as source:image=source.convert('RGB')
    panels=[]
    for name,color in [('baseline',(0,210,230)),('candidate',(200,100,255))]:
        panel=image.copy();draw=ImageDraw.Draw(panel)
        draw.rectangle((0,0,panel.width,45),fill=(20,20,30))
        draw.text((10,10),name+' | green: manual; color: supported proposal; gray: unsupported anatomical proxy',fill='white')
        for curve,result in zip(curves,results):
            item=result[name]
            if 'mesh_samples_px' not in item:
                draw.line([tuple(p) for p in curve['points_px']],fill=(30,255,80),width=3)
                proxy=item.get('diagnostic_mesh_samples_px')
                if proxy:
                    draw.line([tuple(p) for p in proxy],fill=(180,180,180),width=3)
                draw.text(tuple(curve['points_px'][0]),'INSUFFICIENT SUPPORT' if proxy else 'NO SUPPORTED ARC',fill=(255,160,60))
                continue
            manual=item['reference_samples_px'];mesh=item['mesh_samples_px']
            draw.line([tuple(p) for p in manual],fill=(30,255,80),width=3)
            draw.line([tuple(p) for p in mesh],fill=color,width=3)
            for a,b in zip(manual[::8],mesh[::8]):draw.line([tuple(a),tuple(b)],fill=(255,180,60),width=1)
        panels.append(panel)
    canvas=Image.new('RGB',(image.width*2,image.height))
    for i,panel in enumerate(panels):canvas.paste(panel,(i*image.width,0))
    canvas.save(destination)
