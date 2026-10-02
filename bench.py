"""Run AssemblyWorldBench directly from this checkout with uv run bench.py."""

from pathlib import Path

from dotenv import load_dotenv

from assembly_world_bench.cli import main

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    raise SystemExit(main())
