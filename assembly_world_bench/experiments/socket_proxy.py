"""Private Unix-socket relay to one already initialized Chrome MCP process.

The browser owns the upstream process for its entire sample. Connecting agent CLIs
never reconnect CDP, reload the episode, or race initial tool registration.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from .transport import TOOLS


class Broker:
    def __init__(self, upstream, webmcp=None):
        self.upstream = upstream
        self.webmcp = webmcp
        self.lock = asyncio.Lock()
        self.temporary = tempfile.TemporaryDirectory(prefix="awb-mcp-", dir="/tmp")
        self.path = Path(self.temporary.name) / "rpc.sock"
        self.server = None
        self.connections = set()
        self.handlers = set()

    async def start(self):
        self.server = await asyncio.start_unix_server(
            self.handle, path=str(self.path), limit=128 * 1024 * 1024
        )
        os.chmod(self.path, 0o600)
        return self

    async def handle(self, reader, writer):
        self.connections.add(writer)
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            while line := await reader.readline():
                row = json.loads(line)
                if "id" not in row:
                    continue
                method, params = row["method"], row.get("params", {})
                try:
                    if method == "initialize":
                        result = self.upstream.initialization
                    elif method == "ping":
                        result = {}
                    elif method in {"tools/list", "tools/call"}:
                        if method == "tools/call" and params.get("name") not in TOOLS:
                            raise ValueError("Browser transport tool is not allowed")
                        async with self.lock:
                            result = (
                                await self.webmcp(params)
                                if self.webmcp
                                and method == "tools/call"
                                and params["name"] in {"list_webmcp_tools", "execute_webmcp_tool"}
                                else await self.upstream.request(method, params)
                            )
                    else:
                        raise ValueError("Browser transport method is not allowed")
                    response = dict(jsonrpc="2.0", id=row["id"], result=result)
                except Exception as exc:
                    response = dict(
                        jsonrpc="2.0",
                        id=row["id"],
                        error={"code": -32603, "message": str(exc)},
                    )
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
        finally:
            self.connections.discard(writer)
            self.handlers.discard(task)
            writer.close()
            await writer.wait_closed()

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        for writer in list(self.connections):
            writer.close()
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.temporary.cleanup()


async def relay(path):
    remote, writer = await asyncio.open_unix_connection(path, limit=128 * 1024 * 1024)
    stdin = asyncio.StreamReader(limit=128 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin
    )

    async def send():
        while line := await stdin.readline():
            writer.write(line)
            await writer.drain()
        writer.close()

    async def receive():
        while line := await remote.readline():
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    try:
        await asyncio.gather(send(), receive())
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(relay(sys.argv[1]))
