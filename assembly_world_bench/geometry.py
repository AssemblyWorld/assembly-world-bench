"""Frozen rigid transforms and official part scores; no data preparation."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .registration import chamfer

THRESHOLD = 0.01


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def pca_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return origin and right-handed basis; smallest principal axis is local Z.

    Axis signs and repeated eigenspaces use ordered centered vertices only.
    Vertex order breaks exact symmetry ties; no source/world axes are consulted.
    """
    center = vertices.mean(axis=0)
    centered = vertices - vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    values, vectors = values[::-1], vectors[:, ::-1]
    tolerance = max(float(values[0]), np.finfo(float).tiny) * 1e-10
    axes = []
    start = 0
    while start < 3 and len(axes) < 2:
        end = start + 1
        while end < 3 and abs(values[end] - values[start]) <= tolerance:
            end += 1
        subspace = vectors[:, start:end]
        projector = subspace @ subspace.T
        chosen = []
        for axis in centered / max(np.linalg.norm(centered, axis=1).max(), np.finfo(float).tiny):
            candidate = projector @ axis
            for previous in chosen:
                candidate -= previous * np.dot(previous, candidate)
            length = np.linalg.norm(candidate)
            if length > 1e-8:
                candidate /= length
                chosen.append(candidate)
            if len(chosen) == end - start:
                break
        axes.extend(chosen)
        start = end
    if len(axes) < 2:
        raise ValueError("Degenerate part: fewer than two geometric axes")
    # Projection cancellation can leave tiny cross-axis residuals after normalization.
    first = axes[0] / np.linalg.norm(axes[0])
    second = axes[1] - first * np.dot(first, axes[1])
    second /= np.linalg.norm(second)
    basis = np.column_stack([first, second, np.cross(first, second)])
    return center, basis


def metrics_from_errors(errors, shape_chamfer):
    errors = np.asarray(errors, dtype=float)
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(shape_chamfer):
        raise ValueError("Metrics require finite, nonempty errors")
    correct = errors <= THRESHOLD
    return dict(SCD=float(shape_chamfer * 1000), PA=float(correct.mean()), SR=int(correct.all()))


def score_parts(prediction, target, groups, part_ids, shape_chamfer):
    """Match only within supplied equivalence groups, without clipping costs."""
    if sorted(i for group in groups for i in group) != list(range(len(part_ids))):
        raise ValueError("Equivalence groups must partition the parts")
    records = []
    for group in groups:
        costs = np.array([[chamfer(prediction[i], target[j]) for j in group] for i in group])
        rows, cols = linear_sum_assignment(costs)
        for a, b in zip(rows, cols):
            records.append(
                dict(
                    part_id=part_ids[group[a]],
                    target_part_id=part_ids[group[b]],
                    chamfer=float(costs[a, b]),
                    correct=bool(costs[a, b] <= THRESHOLD),
                )
            )
    records.sort(key=lambda record: record["part_id"])
    return metrics_from_errors([r["chamfer"] for r in records], shape_chamfer), records
