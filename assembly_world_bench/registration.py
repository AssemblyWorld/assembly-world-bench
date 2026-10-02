"""Shared deterministic proper-rigid registration and squared Chamfer distance."""

from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree

MAX_ITERATIONS = 100
IMPROVEMENT_TOLERANCE = 1e-9


def points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError("Expected nonempty (N, 3) points")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite points")
    return value


def chamfer(prediction, target):
    """Sum of directional means of squared Euclidean nearest-neighbor distances."""
    prediction, target = points(prediction), points(target)
    a = cKDTree(target).query(prediction)[0]
    b = cKDTree(prediction).query(target)[0]
    return float(np.mean(a * a) + np.mean(b * b))


def transform(cloud, rotation, translation):
    return cloud @ rotation.T + translation


def kabsch(source, target):
    """Fit a proper rigid transform to equal-weight corresponding points."""
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return rotation, b - rotation @ a


def align(prediction, target, pose_seeds=(), *, include_candidate_transforms=False):
    """Select minimum squared Chamfer across 24 PCA and optional rigid starts.

    Every candidate transforms the entire supplied point set. Bidirectional ICP is a local
    optimizer; multiple starts do not constitute a global-optimality guarantee.
    """
    from .geometry import pca_frame

    prediction, target = points(prediction), points(target)
    if len(prediction) != len(target):
        raise ValueError("Registration requires equal total point counts")
    pc, pb = pca_frame(prediction)
    tc, tb = pca_frame(target)
    seeds = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            axes = np.eye(3)[:, permutation] @ np.diag(signs)
            if np.linalg.det(axes) > 0:
                rotation = tb @ axes @ pb.T
                seeds.append((rotation, tc - rotation @ pc, f"pca-{len(seeds)}"))
    seeds.extend((r, t, f"part-{i}") for i, (r, t) in enumerate(pose_seeds))
    target_tree = cKDTree(target)
    source_tree = cKDTree(prediction)

    def nearest(rotation, translation):
        moved = transform(prediction, rotation, translation)
        a, forward = target_tree.query(moved)
        # Rigid distance invariance avoids rebuilding a tree each iteration.
        b, backward = source_tree.query((target - translation) @ rotation)
        return float(np.mean(a * a) + np.mean(b * b)), forward, backward

    best = None
    diagnostics = []
    for rotation, translation, name in seeds:
        rotation, translation = rotation.copy(), translation.copy()
        value, forward, backward = nearest(rotation, translation)
        candidate = (value, rotation.copy(), translation.copy(), 0)
        converged = False
        for iteration in range(1, MAX_ITERATIONS + 1):
            r, t = kabsch(
                np.concatenate((prediction, prediction[backward])),
                np.concatenate((target[forward], target)),
            )
            updated, forward, backward = nearest(r, t)
            if updated < candidate[0]:
                candidate = (updated, r.copy(), t.copy(), iteration)
            improvement = value - updated
            rotation, translation, value = r, t, updated
            if improvement < IMPROVEMENT_TOLERANCE:
                converged = improvement >= -1e-12
                break
        diagnostics.append(
            dict(seed=name, chamfer=candidate[0], iterations=iteration, converged=converged)
        )
        if include_candidate_transforms:
            diagnostics[-1].update(
                rotation=candidate[1].tolist(), translation=candidate[2].tolist()
            )
        # Strict comparison gives stable first-candidate tie breaks.
        if best is None or candidate[0] < best[0]:
            best = (*candidate, name, converged, iteration)
    value, rotation, translation, best_iteration, seed, converged, iterations = best
    return (
        rotation,
        translation,
        dict(
            rotation=rotation.tolist(),
            translation=translation.tolist(),
            chamfer=value,
            seed=seed,
            best_iteration=best_iteration,
            iterations=iterations,
            converged=converged,
            candidates=diagnostics,
        ),
    )
