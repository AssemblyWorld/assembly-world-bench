"""Dedicated Chrome lifecycle, episode URL loading and public UI export."""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from ..common import sha256
from ..episode_io import read_episode
from .serving import environment_url

MCP_VERSION = "1.8.0"
DEFAULT_ENVIRONMENT = "https://assemblyworld.github.io/3DWebAgent/"
LOGGER = logging.getLogger(__name__)


def chrome_path(explicit=None):
    candidates = [
        explicit,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("Chrome not found; supply --chrome-path")


async def terminate(process):
    if process is None:
        return
    # Kill the owned process group even when its leader exited, to reap MCP children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        if process.returncode is None:
            raise
        return  # An exited Chrome can leave OS-protected service helpers.
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.returncode is None:
            raise
    await process.wait()


class Browser:
    def __init__(self, root, options):
        self.root, self.options = Path(root), options
        self.process = self.playwright = self.browser = None
        self.page = None
        self.loaded = False
        self.stderr_path = self.root / "chrome.stderr.log"
        self.upstream = self.broker = None

    async def start(self, episode_url=None):
        from playwright.async_api import async_playwright

        profile = self.root / "chrome"
        profile.mkdir()
        with self.stderr_path.open("wb") as stderr:
            self.process = await asyncio.create_subprocess_exec(
                chrome_path(self.options.get("chrome_path")),
                f"--user-data-dir={profile}",
                "--remote-debugging-port=0",
                "--remote-debugging-address=127.0.0.1",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1440,1000",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-background-timer-throttling",
                *(
                    ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
                    if sys.platform.startswith("linux")
                    else []
                ),
                "--enable-features=WebMCP",
                "--enable-blink-features=WebMCP,WebMCPTesting",
                *(["--headless"] if self.options.get("headless", False) else []),
                "about:blank",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr,
                start_new_session=True,
            )
        # Chrome creates DevToolsActivePort before writing the port line; poll until the
        # file holds a port number, since several browsers may start at the same time.
        async with asyncio.timeout(30):
            while True:
                if self.process.returncode is not None:
                    detail = self.stderr_path.read_text(errors="replace")[-4000:]
                    raise RuntimeError(f"Chrome exited during startup: {detail}")
                lines = []
                if (profile / "DevToolsActivePort").exists():
                    lines = (profile / "DevToolsActivePort").read_text().splitlines()
                if lines and lines[0].strip().isdigit():
                    break
                await asyncio.sleep(0.1)
        port = lines[0].strip()
        self.url = f"http://127.0.0.1:{port}"
        LOGGER.info("Dedicated Chrome started: %s", self.root)
        self.playwright = await async_playwright().start()
        try:
            self.browser = await self.playwright.chromium.connect_over_cdp(self.url, timeout=30000)
        except Exception as error:
            detail = self.stderr_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"Chrome CDP connection failed: {error}\n{detail}") from error
        context = self.browser.contexts[0]
        if episode_url:
            environment = urlsplit(self.options["environment_url"])
            await context.grant_permissions(
                ["local-network-access"], origin=f"{environment.scheme}://{environment.netloc}"
            )
        self.page = context.pages[0] if context.pages else await context.new_page()
        self.page_errors = []
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.set_default_timeout(60000)
        await self.page.set_viewport_size({"width": 1440, "height": 900})
        self.cdp = await context.new_cdp_session(self.page)
        self.registered = set()
        self.tool_definitions = {}
        self.tool_responses = {}
        self.tool_response_event = asyncio.Event()
        self.cdp.on("WebMCP.toolsAdded", self._tools)
        self.cdp.on("WebMCP.toolsRemoved", self._removed)
        self.cdp.on("WebMCP.toolResponded", self._response)
        await self.cdp.send("WebMCP.enable")
        from .socket_proxy import Broker
        from .transport import Client

        self.upstream = await Client().start(self.upstream_command())
        LOGGER.info("MCP initialized: %s", self.root)
        async with asyncio.timeout(30):
            await self.upstream.request("tools/call", {"name": "list_pages", "arguments": {}})
        LOGGER.info("MCP page attachment ready: %s", self.root)
        self.broker = await Broker(self.upstream, self.webmcp_request).start()
        target = self.options["environment_url"]
        if episode_url:
            target = environment_url(target, episode_url)
        await self.page.goto(target, wait_until="domcontentloaded")
        LOGGER.info("Episode document loaded: %s", self.root)
        await self.page.bring_to_front()
        expected_tools = set(
            self.options.get("expected_tools")
            or {
                "start_episode",
                "get_scene",
                "capture_scene",
            }
        )
        stable_since = None
        try:
            async with asyncio.timeout(90):
                while True:
                    if self.page_errors:
                        raise RuntimeError(
                            f"Episode page crashed: {'; '.join(self.page_errors[-5:])}"
                        )
                    if expected_tools <= self.registered:
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= 0.3:
                            break
                    else:
                        stable_since = None
                    await asyncio.sleep(0.1)
        except Exception as error:
            if self.page_errors:
                raise RuntimeError(
                    f"Episode page crashed: {'; '.join(self.page_errors[-5:])}"
                ) from error
            text = await self.page.locator("body").inner_text(timeout=2000)
            detail = self.stderr_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"Episode UI did not become ready at {self.page.url}: {error}\n{text[-4000:]}\n{detail}"
            ) from error
        errors = [text.strip() for text in await self.page.get_by_role("alert").all_text_contents()]
        if any(errors):
            raise RuntimeError(f"Episode page failed: {'; '.join(filter(None, errors))}")
        self.loaded = episode_url is not None
        self.page_id = await self.upstream.page_id(self.page.url)
        LOGGER.info("WebMCP registration ready: %s", self.root)
        return self

    def _tools(self, event):
        for tool in event.get("tools", []):
            self.registered.add(tool["name"])
            self.tool_definitions[tool["name"]] = tool

    def _removed(self, event):
        for tool in event.get("tools", []):
            self.registered.discard(tool["name"])
            self.tool_definitions.pop(tool["name"], None)

    def _response(self, event):
        self.tool_responses[event["invocationId"]] = event
        self.tool_response_event.set()

    async def webmcp_request(self, params):
        """Use native Chrome WebMCP events, avoiding an upstream frame-cache race."""
        arguments = params.get("arguments", {})
        if arguments.get("pageId") != self.page_id:
            raise ValueError("WebMCP calls must target the loaded episode page")
        if params["name"] == "list_webmcp_tools":
            definitions = [
                {
                    key: tool[key]
                    for key in ("name", "description", "inputSchema", "annotations")
                    if key in tool
                }
                for tool in self.tool_definitions.values()
            ]
            return {"content": [{"type": "text", "text": json.dumps(definitions)}]}
        tool = self.tool_definitions.get(arguments["toolName"])
        if tool is None:
            raise ValueError("Requested WebMCP tool is not registered on the episode page")
        raw = arguments.get("input", "{}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("WebMCP input must be a JSON object")
        async with asyncio.timeout(300):
            invocation = await self.cdp.send(
                "WebMCP.invokeTool",
                {
                    "frameId": tool["frameId"],
                    "toolName": tool["name"],
                    "input": parsed,
                },
            )
            ident = invocation["invocationId"]
            while ident not in self.tool_responses:
                self.tool_response_event.clear()
                await self.tool_response_event.wait()
            event = self.tool_responses.pop(ident)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {key: event.get(key) for key in ("status", "output", "errorText")}
                    ),
                }
            ]
        }

    def mcp_command(self):
        return [
            sys.executable,
            "-m",
            "assembly_world_bench.experiments.socket_proxy",
            str(self.broker.path),
        ]

    def upstream_command(self):
        return [
            self.options.get("mcp_command", "chrome-devtools-mcp"),
            "--browserUrl",
            self.url,
            "--categoryExperimentalWebmcp",
            "--no-usage-statistics",
            "--no-performance-crux",
        ]

    async def export(self, target, expected):
        if not self.loaded:
            raise RuntimeError("Input episode was not loaded")
        # MCP can reconnect after a transport interruption while Playwright still
        # holds a closed page handle. Reattach only to the same existing episode.
        if not self.browser.is_connected() or self.page.is_closed():
            expected_url = self.page.url
            self.browser = await self.playwright.chromium.connect_over_cdp(self.url)
            candidates = [
                page
                for context in self.browser.contexts
                for page in context.pages
                if not page.is_closed() and page.url == expected_url
            ]
            if len(candidates) != 1:
                raise RuntimeError("Cannot uniquely recover the existing episode page for export")
            self.page = candidates[0]
            self.page.set_default_timeout(60000)
        target = Path(target)
        temporary = self.root / "export.episode.zip"
        async with asyncio.timeout(60):
            async with self.page.expect_download() as pending:
                await self.page.get_by_text("File", exact=True).click()
                await self.page.get_by_role(
                    "button", name="Export full episode…", exact=True
                ).click()
            await (await pending.value).save_as(temporary)
        episode = await asyncio.to_thread(read_episode, temporary)
        if episode["manifest"]["id"] != expected["episode_id"]:
            raise ValueError("Exported episode identity differs from input")
        if len(episode["calls"]) < expected["initial_calls"]:
            raise ValueError("Export lost input call history")
        checksum = sha256(temporary.read_bytes())
        # Use a same-filesystem temporary file for the atomic replacement.
        staging = target.with_suffix(".tmp")
        try:
            shutil.copyfile(temporary, staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        return {"status": "saved", "sha256": checksum, "calls": len(episode["calls"])}

    async def close(self):
        try:
            if self.broker:
                await self.broker.close()
        finally:
            try:
                if self.upstream:
                    await self.upstream.close()
            finally:
                try:
                    if self.browser:
                        await asyncio.wait_for(self.browser.close(), 10)
                finally:
                    try:
                        if self.playwright:
                            await asyncio.wait_for(self.playwright.stop(), 10)
                    finally:
                        await terminate(self.process)


async def doctor(options):
    """Check installed versions and actual WebMCP discovery without running a model."""
    import tempfile
    from importlib.metadata import version

    from .transport import Client

    executable = shutil.which(options["mcp_command"])
    if executable is None:
        raise FileNotFoundError(f"Install chrome-devtools-mcp@{MCP_VERSION} or use --mcp-command")
    options["mcp_command"] = str(Path(executable).resolve())
    versions = {"playwright": version("playwright")}
    commands = {
        "agent": [options["agent"], "--version"],
        "chrome": [chrome_path(options.get("chrome_path")), "--version"],
        "mcp": [options["mcp_command"], "--version"],
    }
    for name, command in commands.items():
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"{name} version check timed out") from None
        if process.returncode:
            raise RuntimeError(f"{name} version check failed: {stderr.decode()[-1000:]}")
        versions[name] = stdout.decode().strip()
    if versions["mcp"] != MCP_VERSION:
        raise ValueError(f"Install chrome-devtools-mcp@{MCP_VERSION}; found {versions['mcp']}")
    with tempfile.TemporaryDirectory(prefix="awb-doctor-") as root:
        browser = Browser(root, options)
        client = None
        try:
            await browser.start()
            client = await asyncio.wait_for(Client().start(browser.mcp_command()), 30)
            async with asyncio.timeout(30):
                listed = await client.request("tools/list", {})
                required = {"list_pages", "list_webmcp_tools", "execute_webmcp_tool"}
                if not required <= {t["name"] for t in listed["tools"]}:
                    raise RuntimeError("Chrome DevTools MCP lacks WebMCP tools")
                pages = await client.webmcp_tools(await client.page_id(browser.page.url))
                if pages.get("isError") or "capture_scene" not in json.dumps(pages):
                    raise RuntimeError(f"Actual WebMCP discovery failed: {json.dumps(pages)}")
            versions["webmcp_tools"] = sorted(browser.registered)
        finally:
            if client:
                await client.close()
            await browser.close()
    return versions
