"""Frozen block execution, draining failures and explicit fresh-run recovery."""

import asyncio
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .common import file_hash, now, read_json, sample_name, write_json
from .episode_io import read_episode
from .experiments.agents import prompt_content, run_agent
from .experiments.browser import DEFAULT_ENVIRONMENT, Browser, doctor
from .experiments.serving import EpisodeServer
from .experiments.transport import append

PROTOCOL = """
The scheduler has loaded your episode in a dedicated browser.
Use list_pages, list_webmcp_tools and execute_webmcp_tool. All scene observations,
captures and transformations must use these page WebMCP tools.
Do not navigate, reload, close or reset the page. Do not read source code, local
geometry, ground truth, dataset annotations or other experiments. Do not spawn agents.
Preserve the existing physics and tool settings.
Reference images, if supplied, are attached in page order. You may reread only
these images with read_manual_page (one-based page).
The scheduler exports the full episode after your final answer; do not export or
write files yourself. End with a JSON object containing status (completed, partial
or unable), summary, checked_connections and uncertainties.
"""
QUOTA = re.compile(
    r"usage.limit|usage_limit|quota.exceed|insufficient.quota|you.ve hit.*limit|"
    r"weekly.limit|5.hour.limit|five.hour.limit|credits?.*(?:exhaust|deplet)",
    re.I,
)


def validate_options(options):
    if options.get("concurrency", 1) < 1:
        raise ValueError("concurrency must be positive")
    if options.get("timeout_seconds") is not None and options["timeout_seconds"] <= 0:
        raise ValueError("timeout must be positive")
    # Provider overrides may not re-enable local tools or change benchmark instructions.
    for override in options.get("codex_config") or []:
        key = override.split("=", 1)[0].strip()
        if "=" not in override or not (
            key in {"model_provider", "model_context_window"} or key.startswith("model_providers.")
        ):
            raise ValueError("Codex overrides are limited to model provider/context configuration")


def outcome(text):
    decoder = __import__("json").JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict) and value.get("status") in {
                    "completed",
                    "partial",
                    "unable",
                }:
                    return value
            except ValueError:
                pass
    return None


def create_run(options, package, selection, *, source_run=None, entries=None, versions=None):
    validate_options(options)
    entries = entries or {
        (name, sid): package.entry(name, sid) for name, ids in selection.items() for sid in ids
    }
    directory = Path(options.get("logs", "logs")).resolve() / (
        now().replace(":", "") + "-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True)
    try:
        code = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        code = None
    meta = dict(
        version=1,
        kind="assembly-world-bench",
        status="running",
        started_at=now(),
        source_run=str(Path(source_run).resolve()) if source_run else None,
        options=options,
        selection=selection,
        benchmark=dict(
            revision=package.revision, local=str(package.local) if package.local else None
        ),
        tasks={name: package.configuration(name)[2] for name in selection},
        reference_modes={name: package.blocks[name]["reference_mode"] for name in selection},
        code_commit=code,
        versions=versions or {},
    )
    write_json(directory / "run.json", meta)
    for (name, sid), entry in entries.items():
        sample = directory / "blocks" / name / "samples" / sample_name(sid)
        episode = read_episode(entry["initial"])
        write_json(
            sample / "input.json",
            dict(
                **entry["expected"],
                block=name,
                sample_id=sid,
                initial_path=str(entry["initial"]),
                initial_calls=len(episode["calls"]),
                identity=entry["config"]["identity"],
                reference=entry["reference"],
            ),
        )
        write_json(sample / "result.json", {"status": "pending"})
        (sample / "prompt.txt").write_text(entry["task"] + "\n" + PROTOCOL)
        (sample / "conversation.jsonl").touch()
    return directory


async def execute_sample(
    directory, name, sid, meta, entry, episode_url, *, browser_factory=Browser, agent=run_agent
):
    sample = directory / "blocks" / name / "samples" / sample_name(sid)
    inputs = read_json(sample / "input.json")
    result = dict(
        status="running",
        started_at=now(),
        execution={"status": "pending"},
        archive={"status": "not_saved"},
        agent_outcome=None,
    )
    write_json(sample / "result.json", result)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="awb-sample-") as temporary:
        root = Path(temporary)
        options = {
            **meta["options"],
            "expected_tools": read_episode(entry["initial"])["manifest"]
            .get("runtime", {})
            .get("enabledTools"),
        }
        browser = browser_factory(root, options)
        try:
            if file_hash(entry["initial"]) != inputs["sha256"]:
                raise ValueError("Input changed since run creation")
            pages = []
            # Only reference bytes are copied to the independent CLI workspace.
            for i, source in enumerate(entry["pages"], 1):
                target = root / f"reference-{i:04d}{source.suffix}"
                shutil.copyfile(source, target)
                if file_hash(target) != entry["reference"]["pages"][i - 1]["source_sha256"]:
                    raise ValueError("Reference changed since preflight")
                pages.append(str(target))
            prompt = (sample / "prompt.txt").read_text()
            append(
                sample / "conversation.jsonl",
                "message",
                role="user",
                content=prompt_content(prompt, pages) if pages else prompt,
            )
            await browser.start(episode_url)
            bridge = root / "bridge.json"
            write_json(
                bridge,
                dict(
                    command=browser.mcp_command(),
                    manual=pages,
                    conversation=str(sample / "conversation.jsonl"),
                ),
            )
            async with asyncio.timeout(meta["options"].get("timeout_seconds")):
                result["execution"] = await agent(
                    meta["options"],
                    root,
                    bridge,
                    prompt,
                    sample / "conversation.jsonl",
                    images=pages,
                )
            result["agent_outcome"] = outcome(result["execution"].get("final_answer", ""))
        except TimeoutError:
            result["execution"] = {"status": "timeout"}
        except asyncio.CancelledError:
            result["execution"] = {"status": "interrupted"}
        except Exception as exc:
            result["execution"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                if browser.loaded:
                    result["archive"] = await browser.export(sample / "final.episode.zip", inputs)
            except Exception as exc:
                result["archive"] = {"status": "failed", "error": str(exc)}
            try:
                await browser.close()
            except Exception as exc:
                result["cleanup_error"] = str(exc)
            result["finished_at"] = now()
            result["duration_seconds"] = time.monotonic() - started
            from .scoring import count_tool_calls

            result["tool_calls"] = count_tool_calls(sample / "conversation.jsonl")
            result["status"] = (
                "completed"
                if (
                    result["execution"]["status"] == "completed"
                    and result["archive"]["status"] == "saved"
                )
                else "failed"
            )
            write_json(sample / "result.json", result)
    return result


def infrastructure_failure(result):
    execution = result.get("execution", {})
    if execution.get("status") == "timeout" and result.get("archive", {}).get("status") == "saved":
        return False
    return (
        execution.get("status") != "completed" or result.get("archive", {}).get("status") != "saved"
    )


async def schedule(directory, entries, *, execute=execute_sample):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    stopping, failures, errors = None, 0, []
    tasks = []

    def stop():
        nonlocal stopping
        stopping = "Operator interruption"
        for task in tasks:
            task.cancel()

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    try:
        with EpisodeServer(meta["options"].get("environment_url", DEFAULT_ENVIRONMENT)) as server:
            for name, ids in meta["selection"].items():
                if stopping:
                    break
                queue = asyncio.Queue()
                for sid in ids:
                    queue.put_nowait(sid)

                async def worker():
                    nonlocal stopping, failures
                    while not stopping:
                        try:
                            sid = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        entry = entries[(name, sid)]
                        try:
                            result = await execute(
                                directory,
                                name,
                                sid,
                                meta,
                                entry,
                                server.add(entry["initial"]),
                            )
                        except Exception as exc:
                            result = dict(
                                status="failed",
                                execution={"status": "failed", "error": str(exc)},
                                archive={"status": "not_saved"},
                            )
                            write_json(
                                directory
                                / "blocks"
                                / name
                                / "samples"
                                / sample_name(sid)
                                / "result.json",
                                result,
                            )
                        text = __import__("json").dumps(result.get("execution", {}))
                        failures = failures + 1 if infrastructure_failure(result) else 0
                        if QUOTA.search(text):
                            stopping = "Account quota exhausted"
                        elif failures >= 3:
                            stopping = "Three consecutive infrastructure failures"
                        print(f"{name}/{sid}: {result['status']}", flush=True)

                tasks = [
                    asyncio.create_task(worker())
                    for _ in range(min(meta["options"].get("concurrency", 1), len(ids)))
                ]
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                errors.extend(str(x) for x in outcomes if isinstance(x, Exception))
        states = status(directory)["counts"]
        meta.update(
            status="paused"
            if stopping
            else "completed"
            if set(states) == {"completed"}
            else "failed",
            pause_reason=stopping,
            errors=errors,
            finished_at=now(),
        )
        write_json(directory / "run.json", meta)
        return 0 if meta["status"] == "completed" else 1
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def status(directory):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    states = {
        f"{name}/{sid}": read_json(
            directory / "blocks" / name / "samples" / sample_name(sid) / "result.json"
        )["status"]
        for name, ids in meta["selection"].items()
        for sid in ids
    }
    return dict(
        run=str(directory),
        status=meta["status"],
        samples=states,
        counts={
            state: list(states.values()).count(state) for state in sorted(set(states.values()))
        },
    )


def resume_selection(source, retry_failed=False):
    source = Path(source).resolve()
    meta = read_json(source / "run.json")
    chain, seen = [], set()
    current = source
    while current:
        if current in seen:
            raise ValueError("Cycle in source_run chain")
        seen.add(current)
        ancestor = read_json(current / "run.json")
        if ancestor.get("kind") != "assembly-world-bench" or ancestor.get("version") != 1:
            raise ValueError("Expected an assembly-world-bench run")
        chain.append((current, ancestor))
        current = Path(ancestor["source_run"]).resolve() if ancestor.get("source_run") else None
    latest = {}
    for directory, ancestor in reversed(chain):
        for name, ids in ancestor["selection"].items():
            for sid in ids:
                latest[(name, sid)] = directory / "blocks" / name / "samples" / sample_name(sid)
    selection = {}
    for (name, sid), sample in latest.items():
        result = read_json(sample / "result.json")
        interrupted = result.get("execution", {}).get("status") == "interrupted"
        saved = result.get("archive", {})
        damaged = result["status"] == "completed" and (
            not (sample / "final.episode.zip").is_file()
            or file_hash(sample / "final.episode.zip") != saved.get("sha256")
        )
        if (
            result["status"] in {"pending", "running"}
            or interrupted
            or damaged
            or (retry_failed and result["status"] == "failed")
        ):
            selection.setdefault(name, []).append(sid)
    return meta, selection


async def launch(options, package, selection, *, source=None):
    validate_options(options)
    # Finish all data validation before invoking even a doctor version probe.
    entries = {
        (name, sid): package.entry(name, sid) for name, ids in selection.items() for sid in ids
    }
    versions = await doctor(options)
    directory = create_run(
        options,
        package,
        selection,
        source_run=source,
        entries=entries,
        versions=versions,
    )
    print(f"Run: {directory}", flush=True)
    return await schedule(directory, entries)
