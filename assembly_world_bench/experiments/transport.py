"""Restricted stdio forwarding for Chrome DevTools MCP (not a scene runtime).

Promote serialized WebMCP image blocks to MCP images so both CLIs can see captures.
The upstream browser remains the sole owner of scene tools and episode recording.
"""

import asyncio
import fcntl
import json
import re
import sys
from pathlib import Path

from ..common import now, source_environment

TOOLS = {"list_pages", "list_webmcp_tools", "execute_webmcp_tool"}


def append(path, kind, **values):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(dict(timestamp=now(), type=kind, **values)) + "\n")
        stream.flush()


def images(value):
    """Find image content in nested WebMCP transport envelopes."""
    if isinstance(value, str):
        try:
            yield from images(json.loads(value))
        except (ValueError, RecursionError):
            return
    elif isinstance(value, list):
        for item in value:
            yield from images(item)
    elif isinstance(value, dict):
        if value.get("type") == "image" and value.get("data") and value.get("mimeType"):
            yield {k: value[k] for k in ("type", "data", "mimeType")}
        else:
            for item in value.values():
                yield from images(item)


def image_references(value):
    """Avoid sending image base64 a second time as language-model text tokens."""
    if isinstance(value, str):
        try:
            return json.dumps(image_references(json.loads(value)))
        except (ValueError, RecursionError):
            return value
    if isinstance(value, list):
        return [image_references(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image" and "data" in value:
            return {"type": "text", "text": "Image attached as MCP image content."}
        return {key: image_references(item) for key, item in value.items()}
    return value


class Client:
    """Sequential JSON-RPC client; no scene operations outside the upstream MCP."""

    async def start(self, command):
        self.process = await asyncio.create_subprocess_exec(
            *command,
            env=source_environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=128 * 1024 * 1024,
        )
        self.sequence = 0
        try:
            async with asyncio.timeout(30):
                self.initialization = await self.request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "assembly-world-bench", "version": "1"},
                    },
                )
                await self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            await self.close()
            raise
        return self

    async def send(self, row):
        self.process.stdin.write((json.dumps(row) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, method, params):
        self.sequence += 1
        ident = self.sequence
        await self.send(dict(jsonrpc="2.0", id=ident, method=method, params=params))
        while line := await self.process.stdout.readline():
            row = json.loads(line)
            if row.get("id") == ident:
                if "error" in row:
                    raise RuntimeError(str(row["error"]))
                return row["result"]
        raise RuntimeError("Chrome DevTools MCP exited before responding")

    async def close(self):
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def page_id(self, url):
        result = await self.request("tools/call", {"name": "list_pages", "arguments": {}})
        text = "\n".join(c.get("text", "") for c in result.get("content", []))
        matches = [
            int(match[1])
            for line in text.splitlines()
            if (match := re.match(r"^(\d+): (.+)", line)) and url in match[2]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely locate the loaded episode page: {text}")
        return matches[0]

    async def webmcp_tools(self, page_id):
        """Wait for tool registration in this new CDP session, without scene actions."""
        async with asyncio.timeout(30):
            while True:
                result = await self.request(
                    "tools/call", {"name": "list_webmcp_tools", "arguments": {"pageId": page_id}}
                )
                if "No WebMCP tools available" not in json.dumps(result):
                    return result
                await asyncio.sleep(0.1)


async def serve(configuration):
    config = json.loads(Path(configuration).read_text())
    upstream = await Client().start(config["command"])
    reader = asyncio.StreamReader(limit=128 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    try:
        while line := await reader.readline():
            row = json.loads(line)
            if "id" not in row:
                continue
            method, params = row["method"], row.get("params", {})
            try:
                if method == "initialize":
                    result = {
                        "protocolVersion": params["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "assembly-browser", "version": "1"},
                    }
                elif method == "tools/list":
                    result = await upstream.request(method, params)
                    result["tools"] = [t for t in result["tools"] if t["name"] in TOOLS]
                    result["tools"].append(
                        {
                            "name": "read_manual_page",
                            "description": "Read a supplied manual page.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"page": {"type": "integer", "minimum": 1}},
                                "required": ["page"],
                                "additionalProperties": False,
                            },
                        }
                    )
                elif method == "tools/call":
                    name = params["name"]
                    if name not in TOOLS | {"read_manual_page"}:
                        raise ValueError("Tool is not allowed")
                    append(config["conversation"], "tool_call", id=row["id"], **params)
                    if name == "read_manual_page":
                        import base64

                        page = params.get("arguments", {}).get("page")
                        if type(page) is not int or not 1 <= page <= len(config["manual"]):
                            raise ValueError("Manual page is out of range")
                        data = Path(config["manual"][page - 1]).read_bytes()
                        from PIL import Image

                        with Image.open(config["manual"][page - 1]) as picture:
                            mime_type = Image.MIME[picture.format]
                        result = {
                            "content": [
                                {
                                    "type": "image",
                                    "mimeType": mime_type,
                                    "data": base64.b64encode(data).decode(),
                                }
                            ]
                        }
                    else:
                        result = (
                            await upstream.webmcp_tools(params["arguments"]["pageId"])
                            if name == "list_webmcp_tools"
                            else await upstream.request(method, params)
                        )
                        found = list(images(result.get("content", [])))
                        # Upstream 1.8 serializes WebMCP outputs as text, including images.
                        native = [c for c in result.get("content", []) if c.get("type") == "image"]
                        if found and not native:
                            result["content"] = image_references(result["content"])
                            result["content"].extend(found)
                    append(config["conversation"], "tool_result", id=row["id"], result=result)
                elif method == "ping":
                    result = {}
                else:
                    raise ValueError(f"Unsupported method: {method}")
                response = dict(jsonrpc="2.0", id=row["id"], result=result)
            except Exception as error:
                if method == "tools/call":
                    result = {"isError": True, "content": [{"type": "text", "text": str(error)}]}
                    append(config["conversation"], "tool_result", id=row["id"], result=result)
                    response = dict(jsonrpc="2.0", id=row["id"], result=result)
                else:
                    response = dict(
                        jsonrpc="2.0", id=row["id"], error={"code": -32603, "message": str(error)}
                    )
            print(json.dumps(response), flush=True)
    finally:
        await upstream.close()


if __name__ == "__main__":
    asyncio.run(serve(sys.argv[1]))
