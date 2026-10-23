"""Validate explicit human-drawn visible curves; no SDK or automatic truth claim."""
import math
from memento3d_head.domain.contracts import ValidationError

KINDS={'jaw_left_image','jaw_right_image','visible_profile'}


def validate_annotations(data,manifest):
    if data.get('schema_version')!=1 or data.get('pipeline_type')!='manual_visible_contours':
        raise ValidationError('INVALID_ANNOTATIONS','Unsupported annotation schema')
    if data.get('workspace_id')!=manifest['workspace_id'] or data.get('source_binding')!=manifest['source_binding']:
        raise ValidationError('STALE_ANNOTATIONS','Annotations belong to another source/workspace')
    if data.get('visible_only_reviewed') is not True:
        raise ValidationError('REVIEW_REQUIRED','Confirm only visible face contours were marked')
    views=data.get('views',{})
    if set(views)!={'front','left','right'}:
        raise ValidationError('INVALID_ANNOTATIONS','Exactly three view entries required')
    counts={}
    for slot,view in views.items():
        spec=manifest['views'][slot]
        if view.get('image_size')!=spec['image_size'] or view.get('image_sha256')!=spec['image_sha256']:
            raise ValidationError('STALE_ANNOTATIONS','Image dimensions/checksum mismatch')
        curves=view.get('curves',[])
        if not isinstance(curves,list) or not 1<=len(curves)<=10:
            raise ValidationError('INVALID_CURVES',f'{slot}: require 1..10 visible curve segments')
        w,h=spec['image_size'];ids=[];point_count=0
        for curve in curves:
            if curve.get('kind') not in KINDS or not isinstance(curve.get('id'),str) or not curve['id']:
                raise ValidationError('INVALID_CURVES',f'{slot}: invalid curve identity/type')
            ids.append(curve['id']);points=curve.get('points_px')
            if not isinstance(points,list) or not 3<=len(points)<=500:
                raise ValidationError('INVALID_CURVES',f'{slot}: each segment needs 3..500 ordered points')
            previous=None;length=0.
            for point in points:
                if (not isinstance(point,list) or len(point)!=2
                        or any(type(x) not in (int,float) or not math.isfinite(x) for x in point)
                        or not (0<=point[0]<w and 0<=point[1]<h)):
                    raise ValidationError('INVALID_POINTS',f'{slot}: invalid/out-of-frame coordinates')
                if previous is not None:
                    step=math.hypot(point[0]-previous[0],point[1]-previous[1])
                    if step<.1:raise ValidationError('INVALID_POINTS',f'{slot}: repeated adjacent points')
                    length+=step
                previous=point
            if length<5:raise ValidationError('INVALID_CURVES',f'{slot}: segment too short')
            point_count+=len(points)
        if len(set(ids))!=len(ids):raise ValidationError('INVALID_CURVES',f'{slot}: duplicate segment identifiers')
        counts[slot]={'curves':len(curves),'points':point_count}
    return {'status':'ANNOTATIONS_STRUCTURALLY_VALID','views':counts,
            'human_review_is_declaration':True,'semantic_accuracy_verified':False,
            'ready_for_automatic_fitting':False}
