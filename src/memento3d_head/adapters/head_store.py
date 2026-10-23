"""Bind the front image to the prior, manually inspected photo-set report."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.head import HeadRequest


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class LocalHeadStore(LocalStore):
    """Snapshot the verified source rather than trusting edited normalized images."""
    def prepare(self, request: HeadRequest, output: Path) -> dict:
        """Require clean validation, explicit human review and exact front SHA256."""
        if not request.reviewed_by.strip():
            raise ValidationError('REVIEW_REQUIRED', 'Set reviewed-by after inspecting landmark images')
        report_path = request.validation_run / 'report.json'
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if (report.get('schema_version'), report.get('pipeline_type'), report.get('pipeline_version')) != (1,'photo_set_validation',1):
            raise ValidationError('INVALID_VALIDATION_REPORT', 'Unsupported photo validation report')
        assessment = report.get('assessment') or {}
        if report.get('status') != 'NEEDS_HUMAN_REVIEW' or report.get('error') or assessment.get('issues') != []:
            raise ValidationError('VALIDATION_NOT_PASSED', 'Resolve photo validation issues first')
        photos = assessment.get('photos', [])
        if len(photos) != 3 or {item.get('slot') for item in photos} != {'front','left','right'}:
            raise ValidationError('INVALID_VALIDATION_REPORT','Expected three measured photo slots')
        source_kind = report.get('provenance',{}).get('source_kind_declared')
        if source_kind != 'real_capture' and not request.allow_synthetic:
            raise ValidationError('NON_REAL_CAPTURE', 'Use real captures, or explicitly allow synthetic research inputs')
        front = next(item for item in photos if item['slot']=='front')
        if request.front.stat().st_size > 20_000_000:
            raise ValidationError('FILE_TOO_LARGE','Front input exceeds 20 MB')
        data = request.front.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != front.get('source_sha256'):
            raise ValidationError('STALE_FRONT_IMAGE','Front bytes differ from the validated input')
        with Image.open(BytesIO(data)) as original:
            if original.width*original.height > 24_000_000:
                raise ValidationError('IMAGE_TOO_LARGE','Front pixel limit exceeded')
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.save(output / 'front.input.png')
        self.write_json(output / 'validation.snapshot.json', report)
        return {'validation_report_sha256':hashlib.sha256(report_bytes).hexdigest(),
                'source_sha256':digest, 'normalized_sha256':sha256(output/'front.input.png'),
                'reviewed_by':request.reviewed_by.strip(), 'source_kind_declared':source_kind,
                'human_review_is_declaration':True,
                'views_used_for_geometry':['front'], 'views_reference_only':['left','right']}
