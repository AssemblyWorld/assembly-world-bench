"""Synthetic contract fixtures. No licensed source geometry or dataset imports."""

import io
import json
import zipfile

import mujoco as mj
import numpy as np
import pytest

from assembly_world_bench.common import ENGINE, PROTOCOL, sha256, write_json
from assembly_world_bench.data import BLOCKS, Package


def make_episode(mjb=False):
    xml = b'<mujoco><worldbody><body name="part-0001"><freejoint/><geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>'
    model = mj.MjModel.from_xml_string(xml.decode())
    data = mj.MjData(model)
    spec = int(mj.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mj.mj_stateSize(model, spec))
    mj.mj_getState(model, data, state, spec)
    files = {
        "world/model.xml": xml,
        "frames.bin": state.astype("<f8").tobytes(),
        "frames.jsonl": json.dumps(
            dict(
                kind="initial",
                index=0,
                groups=[],
                groupCounter=0,
                camera={"position": [3, -4, 3], "target": [0, 0, 0]},
            )
        ).encode(),
        "calls.jsonl": b"",
        "events.jsonl": b"",
    }
    manifest = dict(
        format="3dwebagent-episode",
        version=1,
        id="fixture",
        name="Synthetic fixture",
        contract="3dwebagent-runtime-1",
        engine=ENGINE,
        lifecycle="setup",
        originTime=0,
        units="normalized",
        task="Assemble the synthetic fixture.",
        producer=dict(
            backend="native",
            build="0" * 64,
            engine=ENGINE,
            implementation="synthetic-fixture",
            language="python",
        ),
        requiredCapabilities=["state"],
        stateSpec=spec,
        stateSize=len(state),
        objects=[dict(id="part-0001", name="Part p", type="rigid", collisionEnabled=False)],
        runtime=dict(
            enabledTools=[],
            physics=dict(
                enabled=False,
                detection=False,
                response=False,
                gravity=[0, 0, -9.81],
                strategy="fixed",
                duration=0.25,
                maxDuration=5,
                quietDuration=0.1,
                linearThreshold=0.01,
                angularThreshold=0.01,
            ),
        ),
    )
    if mjb:
        buf = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
        mj.mj_saveModel(model, buffer=buf)
        files["world/model.mjb"] = buf.tobytes()
        del files["world/model.xml"]
        manifest["model"] = {"format": "mjb", "path": "model.mjb"}
    manifest["hashes"] = {k: sha256(v) for k, v in files.items()}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, value in files.items():
            archive.writestr(name, value)
    return out.getvalue()


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    episode = make_episode()
    cloud = np.random.default_rng(12).normal(size=(1000, 3)).tolist()
    blocks = []
    for index, name in enumerate(BLOCKS):
        repo = f"AssemblyWorld/source-{min(index, 1) if index < 2 else index}"
        identity = dict(dataset=repo, revision="a" * 40, protocol="assembly-preparation-v1")
        samples = {
            f"sample-{i:02d}": dict(
                episode=f"sample-{i:02d}.episode.zip",
                episode_id="fixture",
                sha256=sha256(episode),
                parts=1,
            )
            for i in range(20)
        }
        block = dict(
            name=name,
            source="partnet" if index < 2 else name,
            repo_id=repo,
            revision="a" * 40,
            config_id="fixture",
            data=name,
            samples=samples,
            reference_mode="none",
            prompt_file=f"{name}/task.txt",
            prompt_sha256=sha256(b"Assemble."),
        )
        blocks.append(block)
        directory = root / name / repo.split("/")[-1] / "fixture"
        write_json(
            directory / "config.json",
            dict(
                version=1,
                identity=identity,
                config_id="fixture",
                samples=samples,
            ),
        )
        (root / name / "task.txt").write_text("Assemble.")
        for sid, expected in samples.items():
            (directory / expected["episode"]).write_bytes(episode)
            similarity = dict(policy="geometry", threshold=0.0001)
            write_json(
                directory / "cache" / sid / "evaluation.json",
                dict(
                    version=1,
                    key=dict(
                        sample_id=sid,
                        identity=identity,
                        initial_sha256=expected["sha256"],
                        parts=1,
                        preparation_protocol="assembly-preparation-v1",
                        evaluation_protocol=PROTOCOL,
                        similarity=similarity,
                    ),
                    part_ids=["p"],
                    points={"p": cloud},
                    gt_poses={"p": [0, 0, 0, 1, 0, 0, 0]},
                    scale_divisor=1,
                    equivalence=dict(groups=[["p"]], protocol=similarity),
                ),
            )
    write_json(
        root / "benchmark.json",
        dict(
            version=1,
            blocks=blocks,
            sources=["partnet", *BLOCKS[2:]],
            evaluation=dict(
                protocol=PROTOCOL, similarity_policy="geometry", similarity_threshold=0.0001
            ),
        ),
    )
    from assembly_world_bench import data
    from assembly_world_bench.common import file_hash

    monkeypatch.setattr(
        data,
        "FROZEN_HASHES",
        {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*.json")},
    )
    return Package(root)
