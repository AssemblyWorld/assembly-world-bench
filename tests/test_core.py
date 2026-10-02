import asyncio
import json
import shutil
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest
from conftest import make_episode

from assembly_world_bench import runs
from assembly_world_bench.cli import main
from assembly_world_bench.common import file_hash, inside, read_json, sample_name, write_json
from assembly_world_bench.data import BLOCKS, Package
from assembly_world_bench.episode_io import read_episode
from assembly_world_bench.experiments import agents
from assembly_world_bench.experiments.serving import EpisodeServer
from assembly_world_bench.experiments.transport import image_references, images
from assembly_world_bench.scoring import (
    evaluation_inputs,
    score_episode,
    selected_attempts,
    summarize,
)


def make_run(package, tmp_path, count=3, source=None):
    selection = package.select([BLOCKS[0]], [f"sample-{i:02d}" for i in range(count)])
    return runs.create_run(
        dict(logs=str(tmp_path / "runs"), concurrency=2, agent="codex", model="fixture"),
        package,
        selection,
        source_run=source,
    )


def finish(run, sid, status="completed", exported=True):
    directory = run / "blocks" / BLOCKS[0] / "samples" / sample_name(sid)
    inputs = read_json(directory / "input.json")
    result = dict(status=status, execution={"status": status}, archive={"status": "not_saved"})
    if exported:
        shutil.copyfile(inputs["initial_path"], directory / "final.episode.zip")
        result["archive"] = dict(status="saved", sha256=file_hash(directory / "final.episode.zip"))
    write_json(directory / "result.json", result)


def test_local_only_and_corruption(package, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("Local package attempted network"),
    )
    package.download(package.select([BLOCKS[0]], ["sample-00"]))
    entry = package.entry(BLOCKS[0], "sample-00", scoring=True)
    entry["initial"].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        package.entry(BLOCKS[0], "sample-00")
    with pytest.raises(ValueError, match="Unknown sample"):
        package.select([BLOCKS[0]], ["unknown"])


def test_legacy_protocol_and_missing_local(package):
    path = package.root / "benchmark.json"
    value = read_json(path)
    value["evaluation"]["protocol"] = "assembly-evaluation-v2"
    write_json(path, value)
    with pytest.raises(ValueError, match="v1"):
        Package(package.root)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        Package(package.root)


@pytest.mark.parametrize("shared", [False, True])
def test_local_hf_snapshot_blob_layouts_and_escape_guard(tmp_path, shared):
    hub = tmp_path / "hub"
    repository = hub / "datasets--AssemblyWorld--AssemblyWorldBench"
    snapshot = repository / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    blobs = (hub if shared else repository) / "blobs"
    blobs.mkdir()
    payload = blobs / "verified-content"
    payload.write_bytes(b"content")
    (snapshot / "episode.zip").symlink_to(payload)
    assert inside(snapshot, "episode.zip").read_bytes() == b"content"
    outside = tmp_path / "private"
    outside.write_bytes(b"must not be served")
    (snapshot / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        inside(snapshot, "escape")
    with pytest.raises(ValueError, match="escapes"):
        inside(snapshot, "../episode.zip")
    local = tmp_path / "ordinary-package"
    local.mkdir()
    (local / "episode.zip").symlink_to(payload)
    with pytest.raises(ValueError, match="escapes"):
        inside(local, "episode.zip")


@pytest.mark.parametrize("mjb", [False, True])
def test_xml_mjb_and_official_scoring(package, tmp_path, mjb):
    entry = package.entry(BLOCKS[0], "sample-00", scoring=True)
    if mjb:
        entry["initial"].write_bytes(make_episode(True))
        entry["expected"]["sha256"] = file_hash(entry["initial"])
        value = read_json(entry["evaluation"])
        value["key"]["initial_sha256"] = entry["expected"]["sha256"]
        write_json(entry["evaluation"], value)
        entry["evaluation_sha256"] = file_hash(entry["evaluation"])
    assert read_episode(entry["initial"])["manifest"]["id"] == "fixture"
    row = score_episode(entry, entry["initial"])
    assert row["PA"] == row["SR"] == 1
    assert row["SCD"] < 1e-20
    value = read_json(entry["evaluation"])
    value["points"]["p"][0][0] = float("inf")
    # Use direct JSON to deliberately create an invalid fixture.
    entry["evaluation"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluation_inputs(entry)


def test_chains_latest_failed_and_unrelated_duplicates(package, tmp_path):
    original = make_run(package, tmp_path, 1)
    finish(original, "sample-00")
    resumed = make_run(package, tmp_path, 1, original)
    finish(resumed, "sample-00", "failed", False)
    selected = selected_attempts([resumed, original], package)
    assert selected[(BLOCKS[0], "sample-00")]["run"] == resumed
    other = make_run(package, tmp_path, 1)
    finish(other, "sample-00")
    with pytest.raises(ValueError, match="unrelated"):
        selected_attempts([original, other], package)
    sibling = make_run(package, tmp_path, 1, original)
    finish(sibling, "sample-00")
    with pytest.raises(ValueError, match="sibling"):
        selected_attempts([resumed, sibling], package)


def test_resume_and_no_mutation(package, tmp_path):
    original = make_run(package, tmp_path, 3)
    finish(original, "sample-00")
    finish(original, "sample-01", "failed", False)
    before = {p: p.read_bytes() for p in original.rglob("*") if p.is_file()}
    _, selection = runs.resume_selection(original)
    assert selection == {BLOCKS[0]: ["sample-02"]}
    _, selection = runs.resume_selection(original, True)
    assert selection == {BLOCKS[0]: ["sample-01", "sample-02"]}
    make_run(package, tmp_path, 1, original)
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("quota", [False, True])
def test_failure_drain_and_concurrency(package, tmp_path, quota):
    run = make_run(package, tmp_path, 10)
    meta = read_json(run / "run.json")
    entries = {
        (name, sid): package.entry(name, sid)
        for name, ids in meta["selection"].items()
        for sid in ids
    }
    active = peak = 0
    visited = []

    async def execute(directory, name, sid, meta, entry, url):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        visited.append(sid)
        await asyncio.sleep(0.01)
        active -= 1
        result = dict(
            status="failed",
            execution=dict(
                status="failed", error="usage limit exceeded" if quota else "Chrome startup failed"
            ),
            archive={"status": "not_saved"},
        )
        write_json(
            directory / "blocks" / name / "samples" / sample_name(sid) / "result.json", result
        )
        return result

    assert asyncio.run(runs.schedule(run, entries, execute=execute)) == 1
    assert peak == 2 and active == 0
    assert len(visited) <= (2 if quota else 4)
    assert runs.status(run)["status"] == "paused"


def test_partial_complete_and_equal_source_weights(package):
    rows = []
    for i, name in enumerate(BLOCKS):
        for sid in package.blocks[name]["samples"]:
            rows.append(
                dict(
                    block=name,
                    sample_id=sid,
                    status="scored",
                    SCD=float(i),
                    PA=float(i) / 4,
                    SR=i % 2,
                )
            )
    assert summarize(rows[:20], package)["official_overall"] is None
    full = summarize(rows, package)
    assert full["attempted"] == 100
    assert full["official_overall"]["SCD"] == np.mean([0.5, 2, 3, 4])


def test_restricted_service(package):
    episode = package.entry(BLOCKS[0], "sample-00")["initial"]
    with EpisodeServer("https://example.test") as server:
        url = server.add(episode)
        assert urllib.request.urlopen(url).read() == episode.read_bytes()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url.rsplit("/", 2)[0])
        request = urllib.request.Request(url, headers={"Origin": "https://other.test"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url, timeout=1)


def test_cli_no_implicit_full_run():
    with pytest.raises(SystemExit):
        main(["run", "--agent", "codex", "--model", "fixture"])
    with pytest.raises(SystemExit):
        main(["run", "--all", "--agent", "codex", "--model", "fixture", "--prompt-file", "x"])


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_harness_command_and_privacy(tmp_path, agent):
    args = agents.command(dict(agent=agent, model="test"), tmp_path, tmp_path / "bridge")
    assert args[0] == agent
    assert "assembly_world_bench.experiments.transport" in (
        " ".join(args) if agent == "codex" else (tmp_path / "mcp.json").read_text()
    )
    assert (
        agents.normalize(dict(type="item.completed", item=dict(type="reasoning", text="secret")))
        is None
    )
    assert agents.public(dict(role="developer", content="secret")) is None
    with pytest.raises(ValueError):
        agents.command(
            dict(agent="codex", model="test", codex_config=["features.shell_tool=true"]),
            tmp_path,
            tmp_path / "bridge",
        )
    image = {"type": "image", "data": "aA==", "mimeType": "image/png"}
    assert list(images(json.dumps(image))) == [image]
    assert "aA==" not in str(image_references(image))


@pytest.mark.parametrize("invalid", [False, True])
def test_real_subprocess_streaming_without_model(tmp_path, monkeypatch, invalid):
    script = tmp_path / "fake_cli.py"
    script.write_text(
        "import json,sys\nsys.stdin.read()\n"
        + ("print('invalid event')\n" if invalid else "")
        + "print(json.dumps({'type':'item.completed','item':"
        "{'type':'agent_message','text':'final reply'}}))\n"
        "print('diagnostic',file=sys.stderr)\n"
    )
    monkeypatch.setattr(agents, "command", lambda *args: [sys.executable, str(script)])
    result = asyncio.run(
        agents.run_agent(
            {},
            tmp_path,
            None,
            "test",
            tmp_path / "conversation.jsonl",
        )
    )
    assert result["status"] == ("failed" if invalid else "completed")
    assert result["final_answer"] == "final reply"
    assert result["usage"] is None


@pytest.mark.parametrize("failure", ["agent", "export", "timeout", "cancel", None])
def test_export_salvage_and_cleanup(package, tmp_path, failure):
    run = make_run(package, tmp_path, 1)
    entry = package.entry(BLOCKS[0], "sample-00")
    meta = read_json(run / "run.json")
    meta["options"]["timeout_seconds"] = 0.01 if failure == "timeout" else None
    closed = []

    class FakeBrowser:
        loaded = False

        def __init__(self, *args):
            pass

        async def start(self, *args):
            self.loaded = True

        def mcp_command(self):
            return ["unused"]

        async def export(self, path, inputs):
            if failure == "export":
                raise RuntimeError("export failed")
            shutil.copyfile(entry["initial"], path)
            return {"status": "saved", "sha256": file_hash(path)}

        async def close(self):
            closed.append(True)

    async def fake_agent(*args, **kwargs):
        if failure == "agent":
            raise RuntimeError("agent failed")
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "timeout":
            await asyncio.sleep(1)
        return {"status": "completed", "final_answer": '{"status":"partial"}'}

    result = asyncio.run(
        runs.execute_sample(
            run,
            BLOCKS[0],
            "sample-00",
            meta,
            entry,
            "http://127.0.0.1/episode",
            browser_factory=FakeBrowser,
            agent=fake_agent,
        )
    )
    assert closed == [True]
    assert result["status"] == ("completed" if failure is None else "failed")
    assert (result["archive"]["status"] == "saved") == (failure != "export")
    if failure is None:
        assert result["agent_outcome"]["status"] == "partial"


def test_remote_download_is_pinned(package, monkeypatch):
    import huggingface_hub

    import assembly_world_bench.data as data

    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(package.root / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    remote = Package(revision="b" * 40)
    remote.download(remote.select([BLOCKS[0]], ["sample-00"]))
    assert calls and all(
        repo == data.REPO and kw["revision"] == "b" * 40 and kw["repo_type"] == "dataset"
        for repo, _, kw in calls
    )


def test_reference_order_and_integrity(package, monkeypatch):
    from PIL import Image

    name, sid = BLOCKS[0], "sample-00"
    block = package.blocks[name]
    block["reference_mode"] = "manualbook"
    directory, cfg, _ = package.configuration(name)
    refdir = package.root / directory / "cache" / sid / "reference/manualbook"
    refdir.mkdir(parents=True)
    pages = []
    for i, color in enumerate(["red", "blue"], 1):
        p = refdir / f"{i}.png"
        Image.new("RGB", (4, 4), color).save(p)
        pages.append(dict(page=i, file=p.name, source_sha256=file_hash(p)))
    write_json(
        refdir / "pages.json",
        dict(
            dataset=block["repo_id"],
            revision=block["revision"],
            reference_mode="manualbook",
            pages=pages,
        ),
    )
    entry = package.entry(name, sid)
    assert [p.name for p in entry["pages"]] == ["1.png", "2.png"]
    assert len(agents.prompt_content("task", entry["pages"])) == 3
    pages[0]["page"] = 2
    write_json(
        refdir / "pages.json",
        dict(
            dataset=block["repo_id"],
            revision=block["revision"],
            reference_mode="manualbook",
            pages=pages,
        ),
    )
    with pytest.raises(ValueError, match="order"):
        package.entry(name, sid)


def test_no_overall_until_all_attempted_and_no_cost_estimation(package):
    rows = [
        dict(
            block=BLOCKS[0],
            sample_id="sample-00",
            status="error",
            PA=0,
            SR=0,
            SCD=None,
            cost_usd=None,
            usage={"input_tokens": 50},
        )
    ]
    summary = summarize(rows, package)
    assert summary["official_overall"] is None
    assert summary["resources"]["cost_usd"]["mean"] is None
    assert summary["token_usage"]["input_tokens"]["mean"] == 50


def test_unfinished_latest_attempt_never_falls_back(package, tmp_path):
    from assembly_world_bench.scoring import _score_job

    original = make_run(package, tmp_path, 1)
    finish(original, "sample-00")
    newer = make_run(package, tmp_path, 1, original)
    directory = newer / "blocks" / BLOCKS[0] / "samples" / "sample-00"
    write_json(directory / "result.json", {"status": "running", "started_at": "fixture"})
    (directory / "conversation.jsonl").write_text(
        json.dumps({"type": "tool_call", "name": "get_scene"})
        + "\n"
        + json.dumps({"type": "tool_result"})
        + "\n"
    )
    selected = selected_attempts([newer], package)
    record = selected[(BLOCKS[0], "sample-00")]
    assert record["run"] == newer
    row = _score_job((package.entry(BLOCKS[0], "sample-00", scoring=True), record))
    assert row["PA"] == row["SR"] == 0
    assert row["tool_calls"] == 1 and not row["attempt_finished"]
    assert summarize([row], package)["official_overall"] is None


def test_private_mcp_broker_reconnect_and_restrictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    import stat

    from assembly_world_bench.experiments.socket_proxy import Broker
    from assembly_world_bench.experiments.transport import Client

    class Upstream:
        initialization = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "offline-fixture", "version": "1"},
        }
        active = 0
        calls = 0

        async def request(self, method, params):
            assert self.active == 0
            self.active += 1
            await asyncio.sleep(0.01)
            self.calls += 1
            self.active -= 1
            return {"content": [{"type": "text", "text": str(self.calls)}]}

    async def check():
        upstream = Upstream()
        broker = await Broker(upstream).start()
        assert stat.S_IMODE(broker.path.stat().st_mode) == 0o600
        command = [
            sys.executable,
            "-m",
            "assembly_world_bench.experiments.socket_proxy",
            str(broker.path),
        ]
        clients = []
        try:
            clients = [await Client().start(command), await Client().start(command)]
            results = await asyncio.gather(
                *[
                    client.request("tools/call", {"name": "list_pages", "arguments": {}})
                    for client in clients
                ]
            )
            assert sorted(r["content"][0]["text"] for r in results) == ["1", "2"]
            with pytest.raises(RuntimeError, match="not allowed"):
                await clients[0].request("tools/call", {"name": "evaluate_script", "arguments": {}})
            assert upstream.calls == 2
        finally:
            for client in clients:
                await client.close()
            await broker.close()
        assert not broker.path.exists()

    asyncio.run(check())


def test_resume_revisits_failures_in_earlier_ancestors(package, tmp_path):
    root = make_run(package, tmp_path, 3)
    finish(root, "sample-00", "failed", False)
    finish(root, "sample-01")
    child = runs.create_run(
        dict(logs=str(tmp_path / "runs"), agent="codex", model="fixture"),
        package,
        {BLOCKS[0]: ["sample-02"]},
        source_run=root,
    )
    finish(child, "sample-02")
    assert runs.resume_selection(child)[1] == {}
    assert runs.resume_selection(child, True)[1] == {BLOCKS[0]: ["sample-00"]}


def test_native_webmcp_early_response_and_page_boundary(tmp_path):
    from assembly_world_bench.experiments.browser import Browser

    async def check():
        browser = Browser(tmp_path, {})
        browser.page_id = 7
        browser.registered = set()
        browser.tool_definitions = {}
        browser.tool_responses = {}
        browser.tool_response_event = asyncio.Event()
        browser._tools({"tools": [{"name": "get_scene", "frameId": "frame", "inputSchema": {}}]})

        class Session:
            async def send(self, method, params):
                assert method == "WebMCP.invokeTool" and params["frameId"] == "frame"
                browser._response({"invocationId": "early", "status": "success", "output": "{}"})
                return {"invocationId": "early"}

        browser.cdp = Session()
        result = await browser.webmcp_request(
            {
                "name": "execute_webmcp_tool",
                "arguments": {
                    "pageId": 7,
                    "toolName": "get_scene",
                    "input": "{}",
                },
            }
        )
        assert json.loads(result["content"][0]["text"])["status"] == "success"
        assert browser.tool_responses == {}
        with pytest.raises(ValueError, match="loaded episode"):
            await browser.webmcp_request({"name": "list_webmcp_tools", "arguments": {"pageId": 8}})
        browser._removed({"tools": [{"name": "get_scene"}]})
        assert not browser.registered and not browser.tool_definitions

    asyncio.run(check())
