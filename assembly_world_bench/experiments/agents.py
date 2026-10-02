"""Non-interactive agent subprocesses and exportable conversation normalization."""

import asyncio
import base64
import json
import sys
from pathlib import Path

from ..common import source_environment, write_json
from .browser import terminate
from .transport import append


def public(value):
    """Drop private reasoning and hidden instruction records recursively."""
    if isinstance(value, dict):
        kind = value.get("type", "")
        if (
            kind in {"reasoning", "thinking", "redacted_thinking"}
            or value.get("role") in {"system", "developer"}
            or value.get("channel") in {"analysis", "justify", "confidence"}
        ):
            return None
        return {
            k: cleaned
            for k, v in value.items()
            if k not in {"signature", "encrypted_content", "system_prompt", "instructions"}
            and (cleaned := public(v)) is not None
        }
    if isinstance(value, list):
        return [cleaned for v in value if (cleaned := public(v)) is not None]
    return value


def normalize(row):
    kind = row.get("type", "unknown")
    if kind == "system":
        # Initialization payloads can contain inherited instructions/configuration.
        return {"type": "session", **{k: row[k] for k in ("session_id", "model") if k in row}}
    item = row.get("item", {})
    if item.get("type") in {"reasoning", "mcp_tool_call"}:
        return None  # MCP calls/results are recorded in full by the transport.
    cleaned = public(row)
    if not cleaned:
        return None
    if kind in {"assistant", "user"}:
        message = cleaned.get("message", {})
        content = message.get("content", [])
        if isinstance(content, list):
            content = [v for v in content if v.get("type") not in {"tool_use", "tool_result"}]
        if not content:
            return None
        return {"type": "message", "role": kind, "content": content}
    if kind == "item.completed" and item.get("type") == "agent_message":
        return {"type": "message", "role": "assistant", "content": item.get("text", "")}
    return {"type": "event", "event": cleaned}


def prompt_content(prompt, images=()):
    """Build the same ordered multimodal user content for input and public logging."""
    from PIL import Image

    content = [{"type": "text", "text": prompt}]
    for path in images:
        with Image.open(path) as image:
            media_type = Image.MIME[image.format]
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(Path(path).read_bytes()).decode(),
                },
            }
        )
    return content


def command(options, root, bridge, images=()):
    from ..runs import validate_options

    validate_options(options)
    server = {
        "command": sys.executable,
        "args": ["-m", "assembly_world_bench.experiments.transport", str(bridge)],
    }
    model = options["model"]
    if options["agent"] == "codex":
        args = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "-C",
            str(root),
            "-c",
            'approval_policy="never"',
            "-c",
            "features.shell_tool=false",
            "-c",
            "features.multi_agent=false",
            "-c",
            "features.memories=false",
            "-c",
            'web_search="disabled"',
        ]
        for key, value in server.items():
            args += ["-c", f"mcp_servers.assembly.{key}={json.dumps(value)}"]
        args += [
            "-c",
            "mcp_servers.assembly.required=true",
            "-c",
            'mcp_servers.assembly.default_tools_approval_mode="approve"',
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            "mcp_servers.assembly.tool_timeout_sec=300",
        ]
        if options.get("effort"):
            args += ["-c", "model_reasoning_effort=" + json.dumps(options["effort"])]
        # User overrides (for example a local OpenAI-compatible provider) come last and
        # are recorded verbatim in run.json through the options.
        for override in options.get("codex_config") or []:
            args += ["-c", override]
        for path in images:
            args += ["--image", str(path)]
        return [*args, "--", "-"]
    config = root / "mcp.json"
    write_json(config, {"mcpServers": {"assembly": server}})
    args = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--mcp-config",
        str(config),
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "mcp__assembly__*",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--no-chrome",
    ]
    if options.get("effort"):
        args += ["--effort", options["effort"]]
    if images:
        args += ["--input-format", "stream-json"]
    return args


async def run_agent(options, root, bridge, prompt, conversation, *, images=()):
    args = command(options, Path(root), bridge, images)
    env = source_environment()
    # Do not accidentally attach this independent process to the invoking desktop task.
    for key in list(env):
        if key.startswith("CODEX_THREAD") or key in {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"}:
            env.pop(key)
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=root,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        limit=128 * 1024 * 1024,
    )
    result = {"status": "failed", "final_answer": "", "usage": None, "cost_usd": None}
    errors = bytearray()

    async def stderr():
        while chunk := await process.stderr.read(4096):
            errors.extend(chunk)
            del errors[:-8192]

    async def stdout():
        while line := await process.stdout.readline():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("CLI event must be an object")
            except ValueError:
                append(conversation, "parse_error", text=line.decode(errors="replace"))
                result["parse_errors"] = result.get("parse_errors", 0) + 1
                continue
            record = normalize(row)
            if record:
                append(conversation, record.pop("type"), **record)
            item = row.get("item", {})
            if row.get("type") == "item.completed" and item.get("type") == "agent_message":
                result["final_answer"] = item.get("text", "")
            if row.get("type") == "turn.completed":
                result["usage"] = row.get("usage")
            if row.get("type") == "system" and row.get("model"):
                result["model"] = row["model"]
            if row.get("type") == "result":
                result["final_answer"] = row.get("result", "")
                result["usage"] = row.get("usage")
                result["cost_usd"] = row.get("total_cost_usd")
                if row.get("is_error"):
                    result["agent_error"] = row.get("errors", row.get("subtype"))
            if row.get("type") in {"error", "turn.failed"}:
                result["agent_error"] = public(row)

    readers = [asyncio.create_task(stdout()), asyncio.create_task(stderr())]
    try:
        if options.get("agent") == "claude" and images:
            payload = (
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": prompt_content(prompt, images)},
                        "parent_tool_use_id": None,
                    }
                )
                + "\n"
            )
        else:
            payload = prompt
        process.stdin.write(payload.encode())
        await process.stdin.drain()
        process.stdin.close()
        await asyncio.gather(process.wait(), *readers)
        result["exit_code"] = process.returncode
        if (
            process.returncode == 0
            and result["final_answer"]
            and not result.get("agent_error")
            and not result.get("parse_errors")
        ):
            result["status"] = "completed"
        if result["status"] != "completed":
            result["error"] = errors.decode(errors="replace")
        return result
    finally:
        await terminate(process)
        for task in readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
