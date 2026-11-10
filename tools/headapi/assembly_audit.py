"""Independent decoded world-space audit after Blender web export."""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import trimesh
from .demo_scene_worker import inspect_glb


def corners(scene, node):
    transform, name = scene.graph[node]
    mesh = scene.geometry[name]
    points = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
    return points[mesh.faces], mesh


def compare_triangles(expected, actual, tolerance=1e-6):
    if expected.shape != actual.shape:
        raise ValueError('Protected component triangle count changed')
    source_tree, target_tree = cKDTree(expected.mean(1)), cKDTree(actual.mean(1))
    distance, index = target_tree.query(expected.mean(1))
    if distance.max() > tolerance:
        raise ValueError('Protected component triangle correspondence changed')
    if len(np.unique(index)) != len(index):
        # Tiny remeshed hair triangles can share a centroid. Require a complete
        # one-to-one triangle match by corner coordinates, not unique centroids.
        candidates = source_tree.sparse_distance_matrix(target_tree, tolerance, output_type='coo_matrix')
        before, after = expected[candidates.row], actual[candidates.col]
        error = np.minimum.reduce([
            np.max(np.abs(before-after[:, permutation]), axis=(1,2))
            for permutation in itertools.permutations(range(3))])
        valid = error <= tolerance
        graph = csr_matrix((np.ones(int(valid.sum()), dtype=np.int8),
            (candidates.row[valid], candidates.col[valid])), shape=(len(expected), len(actual)))
        index = maximum_bipartite_matching(graph, perm_type='column')
        if np.any(index < 0):
            raise ValueError('Protected component triangles cannot be matched one-to-one')
    matched = actual[index]
    errors = [np.max(np.abs(expected - matched[:, permutation]), axis=(1, 2))
              for permutation in itertools.permutations(range(3))]
    maximum = float(np.min(errors, axis=0).max())
    if maximum > tolerance:
        raise ValueError('Protected component geometry changed')
    return maximum


def audit(master_path, web_path, required):
    inspect_glb(master_path)
    inspect_glb(web_path)
    master = trimesh.load(master_path, force='scene', process=False)
    web = trimesh.load(web_path, force='scene', process=False)
    if set(web.graph.nodes_geometry) != set(required):
        raise ValueError('Missing or unexpected assembly components')
    checks = {}
    for node in required:
        before, source = corners(master, node)
        after, target = corners(web, node)
        if node == 'Body_Cuff_Prepared':
            if not len(after):
                raise ValueError('Body is empty')
            continue  # explicitly simplified for web transport
        error = compare_triangles(before, after)
        checks[node] = {'triangles': len(after), 'max_world_corner_error': error}
        if node == 'Personalized_Complete_Head':
            # Bind each UV to its world position, independent of exporter indexing.
            a = np.concatenate([before, source.visual.uv[source.faces]], axis=2)
            b = np.concatenate([after, target.visual.uv[target.faces]], axis=2)
            compare_triangles(a, b)
            original = np.asarray(source.visual.material.baseColorTexture)
            actual = np.asarray(target.visual.material.baseColorTexture)
            if not np.array_equal(original, actual):
                raise ValueError('Personalized head atlas changed in web export')
            checks[node]['atlas_pixels_identical'] = True
    return {'status': 'ASSEMBLY_WEB_AUDIT_PASSED', 'components': checks,
            'scope': 'Protected world corners, head UV and decoded head atlas; body simplified',
            'ready_for_machine': False}
