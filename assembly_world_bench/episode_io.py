"""Validated public episode reading shared by visualization and evaluation."""

import json
import zipfile
from pathlib import PurePosixPath

import numpy as np

from .common import CONTRACT_DIRECTORY, ENGINE, sha256


def read_episode(path, *, verify_hashes=True):
    """Read a public archive with structural validation and optional resource hashes."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe archive member")
        files = {name: archive.read(name) for name in names}
    manifest = json.loads(files["manifest.json"])
    if manifest["format"] != "3dwebagent-episode" or manifest["version"] != 1:
        raise ValueError("Unsupported episode format")
    if manifest["engine"] != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    if set(files) != {"manifest.json", *manifest["hashes"]}:
        raise ValueError("Archive hash inventory differs")
    if verify_hashes:
        for name, digest in manifest["hashes"].items():
            if sha256(files[name]) != digest:
                raise ValueError(f"Archive checksum mismatch: {name}")
    rows = [json.loads(line) for line in files["frames.jsonl"].splitlines()]
    size = manifest["stateSize"]
    if size <= 0 or len(files["frames.bin"]) != len(rows) * size * 8:
        raise ValueError("Invalid binary state size")
    states = np.frombuffer(files["frames.bin"], dtype="<f8").reshape(len(rows), size)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite recorded state")
    initial, snapshots, traces = [], {}, []
    for row, state in zip(rows, states):
        frame = {k: v for k, v in row.items() if k != "kind"}
        frame["integration"] = state.tolist()
        if row["kind"] == "initial":
            initial.append(frame)
        elif row["kind"] == "state":
            if frame["index"] in snapshots:
                raise ValueError("Duplicate state index")
            snapshots[frame["index"]] = frame
        elif row["kind"] == "trace":
            traces.append(frame)
        else:
            raise ValueError("Unknown frame kind")
    if len(initial) != 1:
        raise ValueError("Expected exactly one initial frame")
    calls = [json.loads(line) for line in files["calls.jsonl"].splitlines()]
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    logical = dict(
        manifest=manifest,
        initial=initial[0],
        states=list(snapshots.values()),
        calls=calls,
        events=events,
        trajectory=traces,
        revision=0,
    )
    import jsonschema

    directory = CONTRACT_DIRECTORY
    if "model" in manifest:
        directory = CONTRACT_DIRECTORY / "mjb"
    jsonschema.validate(logical, json.loads((directory / "episode.schema.json").read_text()))
    model_path = manifest.get("model", {}).get("path", "model.xml")
    if "world/" + model_path not in files:
        raise ValueError("Missing episode model")
    snapshots[0] = initial[0]
    for call in calls:
        if call["state_index"] not in snapshots or call["before_index"] not in snapshots:
            raise ValueError("Call references a missing state")
        if not 0 <= call["trace_start"] <= call["trace_end"] <= len(traces):
            raise ValueError("Call references an invalid trace range")
    return dict(manifest=manifest, files=files, states=snapshots, traces=traces, calls=calls)
