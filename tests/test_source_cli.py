"""Source execution and credential forwarding without installed project or models."""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from assembly_world_bench.experiments import agents

ROOT = Path(__file__).resolve().parent.parent


def test_cli_outside_checkout(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "download,doctor,run,status,resume,eval" in result.stdout


@pytest.mark.parametrize("exported", [False, True])
def test_entrypoint_loads_checkout_dotenv_with_environment_precedence(tmp_path, exported):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copyfile(ROOT / "bench.py", checkout / "bench.py")
    module = checkout / "assembly_world_bench"
    module.mkdir()
    (module / "__init__.py").touch()
    (module / "cli.py").write_text(
        "import os\n"
        "def main():\n"
        "    assert os.environ['CODEX_API_KEY'] == os.environ['EXPECTED_KEY']\n"
        "    assert os.environ['ANTHROPIC_API_KEY'] == 'fixture-claude-key'\n"
        "    return 0\n"
    )
    (checkout / ".env").write_text(
        "CODEX_API_KEY=fixture-file-key\nANTHROPIC_API_KEY=fixture-claude-key\n"
    )
    # The caller's .env must not be loaded instead of the checkout's file.
    (tmp_path / ".env").write_text("CODEX_API_KEY=wrong-cwd-key\n")
    env = os.environ.copy()
    for key in ("CODEX_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH"):
        env.pop(key, None)
    if exported:
        env["CODEX_API_KEY"] = "fixture-exported-key"
    env["EXPECTED_KEY"] = "fixture-exported-key" if exported else "fixture-file-key"
    result = subprocess.run(
        [sys.executable, str(checkout / "bench.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize(
    ("agent", "key"), [("codex", "CODEX_API_KEY"), ("claude", "ANTHROPIC_API_KEY")]
)
def test_agent_forwards_credentials_and_source_imports_without_logging_them(
    tmp_path, monkeypatch, agent, key
):
    fake = tmp_path / "fake_agent.py"
    fake.write_text(
        "import json, os, subprocess, sys\n"
        f"assert os.environ[{key!r}] == 'fixture-secret-value'\n"
        "assert not any(k.startswith('CODEX_THREAD') for k in os.environ)\n"
        "assert 'CLAUDECODE' not in os.environ\n"
        "subprocess.run([sys.executable, '-c', "
        "'import assembly_world_bench.experiments.transport'], check=True)\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type': 'result', 'result': 'fixture answer'}))\n"
    )
    workspace = tmp_path / "sample"
    workspace.mkdir()
    monkeypatch.setenv(key, "fixture-secret-value")
    monkeypatch.setenv("CODEX_THREAD_ID", "invoking-thread")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(agents, "command", lambda *a: [sys.executable, str(fake)])
    conversation = workspace / "conversation.jsonl"
    result = asyncio.run(
        agents.run_agent(
            {"agent": agent, "model": "fixture"},
            workspace,
            workspace / "bridge.json",
            "fixture prompt",
            conversation,
        )
    )
    assert result["status"] == "completed"
    assert "fixture-secret-value" not in conversation.read_text()
    assert "fixture-secret-value" not in str(result)
