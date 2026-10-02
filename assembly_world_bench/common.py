"""Small artifact primitives independent of source datasets."""

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ENGINE = "3.12.0"
PROTOCOL = "assembly-evaluation-v1"
CONTRACT_DIRECTORY = Path(__file__).parent / "contracts"


def source_environment():
    """Make this checkout importable by MCP subprocesses in sample workspaces."""
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *paths])
    return env


def now():
    return datetime.now(UTC).isoformat(timespec="microseconds")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sample_name(sid):
    if (
        not isinstance(sid, str)
        or not sid
        or any(
            not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."}
            for part in sid.split("/")
        )
    ):
        raise ValueError(f"Unsafe sample ID: {sid!r}")
    return sid.replace("/", "--")


def inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Path escapes package: {relative}")
    path = root / rel
    resolved = path.resolve()
    # Hub versions use either per-repository or shared content-addressed blobs.
    allowed = [root]
    repository = root.parent.parent
    if (
        root.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-f]{40}", root.name)
        and repository.name.startswith("datasets--")
    ):
        allowed.extend([(repository / "blobs").resolve(), (repository.parent / "blobs").resolve()])
    if not any(resolved.is_relative_to(directory) for directory in allowed):
        raise ValueError(f"Path escapes package: {relative}")
    return path
