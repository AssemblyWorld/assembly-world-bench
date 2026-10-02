import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from assembly_world_bench.geometry import metrics_from_errors, score_parts
from assembly_world_bench.registration import align, chamfer, transform


def test_chamfer_formula_and_threshold():
    a = np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 1.0, 0]])
    b = np.array([[1.0, 0, 0], [4.0, 1, 0]])
    distances = cdist(a, b, metric="sqeuclidean")
    assert chamfer(a, b) == pytest.approx(distances.min(0).mean() + distances.min(1).mean())
    assert metrics_from_errors([0.01, 0], 0.002) == dict(SCD=2.0, PA=1.0, SR=1)
    assert metrics_from_errors([np.nextafter(0.01, np.inf), 0], 0.002) == dict(
        SCD=2.0, PA=0.5, SR=0
    )


def clouds():
    rng = np.random.default_rng(52)
    return [
        rng.normal(size=(50, 3)) * [0.3, 0.15, 0.07] + center
        for center in ([0, 0, 0], [2, 0, 0], [0.7, 1, 0.2])
    ]


def test_global_alignment_and_local_error():
    target = clouds()
    rotation = Rotation.from_euler("xyz", [73, -41, 127], degrees=True).as_matrix()
    translation = np.array([12.0, -5.0, 3.0])
    prediction = [transform(p, rotation, translation) for p in target]
    r, t, info = align(np.concatenate(prediction), np.concatenate(target))
    assert info["chamfer"] < 1e-20
    assert np.linalg.det(r) == pytest.approx(1)
    score, _ = score_parts(
        [transform(p, r, t) for p in prediction],
        target,
        [[0], [1], [2]],
        ["a", "b", "c"],
        info["chamfer"],
    )
    assert score["PA"] == score["SR"] == 1
    prediction[1] += [0, 1.1, 0]
    r, t, info = align(np.concatenate(prediction), np.concatenate(target))
    score, _ = score_parts(
        [transform(p, r, t) for p in prediction],
        target,
        [[0], [1], [2]],
        ["a", "b", "c"],
        info["chamfer"],
    )
    assert score["SCD"] > 1 and score["PA"] < 1 and score["SR"] == 0


def test_equivalent_exchange_and_unique_exchange():
    a = clouds()[0]
    target = [a, a + [3, 0, 0]]
    prediction = target[::-1]
    matched, records = score_parts(prediction, target, [[0, 1]], ["a", "b"], 0)
    assert matched == dict(SCD=0.0, PA=1.0, SR=1)
    assert records[0]["target_part_id"] == "b"
    unique, _ = score_parts(prediction, target, [[0], [1]], ["a", "b"], 0)
    assert unique["PA"] == unique["SR"] == 0


def test_no_scale_or_reflection_fitting():
    target = np.concatenate(clouds())
    for prediction in (target * 3, target * [-1, 1, 1]):
        r, _, info = align(prediction, target)
        assert np.linalg.det(r) == pytest.approx(1)
        assert info["chamfer"] > 0.001


def test_incorrect_relative_orientation():
    rng = np.random.default_rng(74)
    target = [
        rng.normal(size=(60, 3)) * [0.7, 0.04, 0.03] + center
        for center in ([0, 0, 0], [2, 0, 0], [0, 2, 0])
    ]
    prediction = [p.copy() for p in target]
    prediction[0] = target[0] @ Rotation.from_euler("z", 90, degrees=True).as_matrix().T
    rotation, translation, info = align(np.concatenate(prediction), np.concatenate(target))
    score, _ = score_parts(
        [transform(p, rotation, translation) for p in prediction],
        target,
        [[0], [1], [2]],
        ["0", "1", "2"],
        info["chamfer"],
    )
    assert score["SR"] == 0 and score["PA"] < 1
