"""Local Im2Haircut fixes and ordered-strand export; no model imports at startup."""
import ast
import hashlib
import json
from pathlib import Path

PIN = '4afcc98c303b4108983252b0d4a671ffcee78480'
OLD_APP = '''        else:
            param_size = (self.resolution_upsample * self.resolution_upsample, self.num_points-1, 48)
            self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[interested_idxes].detach().contiguous().clone(), requires_grad=True)'''
NEW_APP = '''        elif self.optimize_appearance:
            # Match appearance rows to the selected base-resolution strands.
            param_size = (nonzero_idxs.numel(), self.num_points-1, 48)
            self.appearance = nn.Parameter(
                torch.ones(param_size, device=self.device),
                requires_grad=True,
            )'''
OLD_SDF = '''            interested_bs, interested_idxes = torch.where(baldness_mask.reshape(img.shape[0], -1) !=  0)
            dists = self.sdf.forward(pred_points_vis[interested_bs, interested_idxes].reshape(-1, 3), scale_pts=True).reshape(-1)'''
NEW_SDF = '''            # update_hairstyle already selected the valid strands.
            if pred_points_vis.numel() == 0:
                raise RuntimeError("No selected hair strands for SDF loss")
            sdf_points = pred_points_vis.reshape(-1, 3).clone()
            dists = self.sdf.forward(sdf_points, scale_pts=True).reshape(-1)'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_text(text):
    """Validate both blocks before changing either; reject unknown source."""
    result = text
    for old, new in ((OLD_APP, NEW_APP), (OLD_SDF, NEW_SDF)):
        if result.count(new) == 1 and old not in result:
            continue
        if result.count(old) != 1 or new in result:
            raise ValueError('Unknown trainer source; refusing partial/ambiguous patch')
        result = result.replace(old, new, 1)
    ast.parse(result)
    return result


def prepare_fixes(repo, apply=False):
    path = Path(repo) / 'trainers/trainer_single.py'
    before = path.read_text(encoding='utf-8')
    after = fixed_text(before)
    if after != before:
        if not apply:
            raise ValueError('Local fixes required; rerun with --apply-fixes')
        # Never overwrite any previous backup.
        backup = path.with_name(path.name + '.before-memento-' + sha256(path)[:12])
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(after, encoding='utf-8')
    return sha256(path)


def strand_array(vertices, points_per_strand):
    import numpy as np
    vertices = np.asarray(vertices)
    if (points_per_strand < 2 or vertices.ndim != 2 or vertices.shape[1] != 3
            or not len(vertices) or len(vertices) % points_per_strand
            or not np.isfinite(vertices).all()):
        raise ValueError('Invalid ordered strand point layout')
    return vertices.reshape(-1, points_per_strand, 3)


def export_strands(source, output, points_per_strand):
    import numpy as np
    import trimesh
    cloud = trimesh.load(str(source), process=False)
    # Meshed PLYs must not silently be treated as ordered strand point clouds.
    if not isinstance(cloud, trimesh.points.PointCloud):
        raise ValueError('Expected upstream point-cloud PLY, not a mesh')
    strands = strand_array(cloud.vertices, points_per_strand)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'hair.strands.npz', strands=strands.astype(np.float32))
    with (output / 'hair.strands.obj').open('w', encoding='utf-8') as stream:
        stream.write('# Ordered strand lines, no faces or recovered material\n')
        for x, y, z in strands.reshape(-1, 3):
            stream.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for index in range(len(strands)):
            start = index * points_per_strand + 1
            stream.write('l ' + ' '.join(map(str, range(start, start + points_per_strand))) + '\n')
    report = {
        'status': 'STRANDS_EXPORTED', 'source_ply': str(Path(source).resolve()),
        'source_sha256': sha256(source), 'strand_count': len(strands),
        'points_per_strand': points_per_strand,
        'bounds_native_units': np.ptp(strands.reshape(-1, 3), axis=0).tolist(),
        'units_calibrated': False, 'ordering': 'pinned_upstream_save_strands',
        'color_recovered': False, 'head_alignment_verified': False,
        'printability_verified': False,
    }
    (output / 'export.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export existing ordered Im2Haircut PLY without rerunning the model')
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Exact recording/config.conf from that run')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    from pyhocon import ConfigFactory
    config = ConfigFactory.parse_file(str(args.config))
    print(json.dumps(export_strands(args.ply, args.output, int(config['dataset']['num_points'])), indent=2))
