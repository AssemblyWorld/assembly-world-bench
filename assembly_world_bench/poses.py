"""Restore recorded states against the benchmark initial world."""

from xml.etree import ElementTree as ET

import numpy as np

from .common import ENGINE
from .episode_model import load_model, validate_compiled_model


def _restore_poses(model, episode, names):
    import mujoco as mj

    manifest = episode["manifest"]
    if mj.mj_stateSize(model, manifest["stateSpec"]) != manifest["stateSize"]:
        raise ValueError("Episode state size differs from native model")
    frame = episode["states"][max(episode["states"])]
    data = mj.MjData(model)
    mj.mj_setState(model, data, np.array(frame["integration"]), manifest["stateSpec"])
    mj.mj_forward(model, data)
    poses = []
    for name in names:
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError("Missing episode body")
        poses.append((data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy()))
    return poses, frame["index"]


def final_poses_against_initial(episode, initial, part_ids):
    """Restore body poses of a final episode validated against its initial episode.

    Used when source geometry comes from the evaluation cache: the final episode
    must carry the initial episode's object catalog and world files (byte-identical
    for XML worlds, structurally identical compiled models for MJB worlds).
    """
    import mujoco as mj

    if mj.__version__ != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    ids = list(part_ids)
    if sorted(ids) != ids or len(set(ids)) != len(ids):
        raise ValueError("Part IDs must be sorted and unique")
    names = [f"part-{i + 1:04d}" for i in range(len(ids))]
    expected_objects = [
        dict(id=name, name=f"Part {pid}", type="rigid", collisionEnabled=False)
        for name, pid in zip(names, ids)
    ]
    manifest, files = episode["manifest"], episode["files"]
    reference, reference_files = initial["manifest"], initial["files"]
    if reference["objects"] != expected_objects:
        raise ValueError("Initial episode object catalog differs from cached parts")
    if manifest["objects"] != expected_objects:
        raise ValueError("Episode object catalog differs from reconstructed parts")
    for key in ("id", "producer", "contract", "engine", "stateSpec", "stateSize"):
        if manifest[key] != reference[key]:
            raise ValueError(f"Episode {key} differs from the initial episode")
    if manifest.get("model") != reference.get("model"):
        raise ValueError("Episode model declaration differs from the initial episode")
    world = {k: v for k, v in files.items() if k.startswith("world/")}
    reference_world = {k: v for k, v in reference_files.items() if k.startswith("world/")}
    if manifest.get("model"):
        if set(world) != set(reference_world):
            raise ValueError("Episode world inventory differs from the initial episode")
        model = load_model(episode)
        expected = load_model(initial)
        validate_compiled_model(model, expected)
        del expected
    else:
        if world != reference_world:
            raise ValueError("Episode world files differ from the initial episode")
        xml = ET.fromstring(files["world/model.xml"])
        if any("world/" + e.attrib["file"] not in files for e in xml.findall("./asset/mesh")):
            raise ValueError("Missing episode mesh asset")
        model = mj.MjModel.from_xml_string(
            files["world/model.xml"].decode(),
            assets={k[6:]: v for k, v in world.items() if k != "world/model.xml"},
        )
    return _restore_poses(model, episode, names)
