"""Bound exterior movement separately from cavity fit; no ARAP claim."""
import numpy as np
from figurine22_hair import shell_fit

def protected_shell_fit(hair,head,eye_y,settings,correspondence=None):
    limit=settings['max_outer_radial_shift_mm']
    if not np.isfinite(limit) or not .4<=limit<=1.6:raise ValueError('Exterior radial shift budget outside range')
    if correspondence is None:correspondence={}
    compact,report=shell_fit(hair,head,eye_y,settings,correspondence)
    upper=head.vertices[head.vertices[:,1]>eye_y+.07*head.extents[1]];lo,hi=np.quantile(upper,[.005,.995],axis=0)
    origin=np.array([(lo[0]+hi[0])/2,eye_y-.04*head.extents[1],(lo[2]+hi[2])/2]);v=hair.vertices-origin;r=np.linalg.norm(v,axis=1);directions=v/r[:,None]
    old=np.linalg.norm(compact.vertices-origin,axis=1);wall=np.maximum(r-correspondence['inner'],0)
    weight=np.clip((wall-.08)/.35,0,1);weight=weight*weight*(3-2*weight)
    protected=np.maximum(old,r-limit)
    radius=old*(1-weight)+protected*weight
    output=hair.copy();output.vertices=origin+directions*radius[:,None]
    if not output.is_volume:raise ValueError('Protected exterior lost closed positive volume')
    outer=wall>=.43;shift=radius-r;max_retraction=float(np.maximum(-shift[outer],0).max())
    if max_retraction>limit+1e-8:raise ValueError('Outer radial shift budget violated')
    crown=float(output.bounds[1,1]-head.bounds[1,1])
    report.update(method='cavity contact plus independently bounded exterior radial retraction',
        max_outer_radial_shift_mm=limit,measured_outer_retraction_max_mm=max_retraction,
        outer_vertex_proxy_count=int(outer.sum()),outer_classification='source radial wall offset proxy; not exact semantic outer-surface partition',
        crown_added_height_mm=crown,crown_proxy_pass=bool(crown<=settings['max_crown_added_mm']),
        cage_volume_mm3=float(hair.volume),protected_volume_mm3=float(output.volume),v22_shell_volume_mm3=float(compact.volume),
        outer_budget_stage="before local sideburn customization",global_volume_preserving=False,arap_implemented=False)
    return output,report
