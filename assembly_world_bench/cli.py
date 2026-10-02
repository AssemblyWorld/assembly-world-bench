"""Six public commands; fixed task/reference conditions and no model defaults."""

import argparse
import asyncio
import json
from pathlib import Path

from .common import now
from .data import BLOCKS, REVISION, Package
from .experiments.browser import DEFAULT_ENVIRONMENT


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def data_options(parser):
    parser.add_argument("--benchmark", type=Path, help="Local-only benchmark package root")
    parser.add_argument("--revision", default=REVISION, help="Pinned HF commit")
    parser.add_argument("--cache-dir", type=Path)


def selection_options(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--block", action="append", choices=BLOCKS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--sample-id", action="append")


def browser_options(parser):
    parser.add_argument("--environment-url", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--chrome-path")
    parser.add_argument("--mcp-command", default="chrome-devtools-mcp")
    parser.add_argument("--headless", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch selected benchmark inputs")
    data_options(download)
    selection_options(download)
    check = commands.add_parser("doctor", help="Check real WebMCP without model calls")
    check.add_argument("--agent", choices=("codex", "claude"), required=True)
    browser_options(check)
    run = commands.add_parser("run", help="Run frozen benchmark conditions")
    data_options(run)
    selection_options(run)
    browser_options(run)
    run.add_argument("--agent", choices=("codex", "claude"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--effort")
    run.add_argument("--codex-config", action="append", default=[])
    run.add_argument("--concurrency", type=positive, default=1)
    run.add_argument("--timeout-seconds", type=positive)
    run.add_argument("--logs", type=Path, default=Path("logs"))
    state = commands.add_parser("status", help="Read execution status")
    state.add_argument("run", type=Path)
    resume = commands.add_parser("resume", help="Continue into a new linked run")
    resume.add_argument("run", type=Path)
    resume.add_argument("--retry-failed", action="store_true")
    resume.add_argument("--concurrency", type=positive)
    resume.add_argument("--logs", type=Path)
    resume.add_argument("--benchmark", type=Path, help="Relocated local-only package")
    evaluate = commands.add_parser("eval", help="Score final attempts in explicit chains")
    data_options(evaluate)
    evaluate.add_argument("runs", type=Path, nargs="+")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--workers", type=positive, default=4)
    args = parser.parse_args(argv)
    options = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    try:
        if args.command == "doctor":
            from .experiments.browser import doctor

            print(json.dumps(asyncio.run(doctor(options)), indent=2))
            return 0
        from . import runs

        if args.command == "status":
            print(json.dumps(runs.status(args.run), indent=2))
            return 0
        if args.command == "resume":
            previous, selection = runs.resume_selection(args.run, args.retry_failed)
            if not selection:
                print("No samples need resuming.")
                return 0
            options = {
                **previous["options"],
                **{
                    key: value
                    for key, value in options.items()
                    if key in {"concurrency", "logs"} and value is not None
                },
            }
            local = args.benchmark or previous["benchmark"]["local"]
            package = Package(
                local,
                revision=previous["benchmark"]["revision"],
                cache_dir=options.get("cache_dir"),
            )
            return asyncio.run(runs.launch(options, package, selection, source=args.run))
        package = Package(args.benchmark, revision=args.revision, cache_dir=args.cache_dir)
        if args.command == "eval":
            from .scoring import evaluate as score

            output = args.output or Path("logs/evaluation") / now().replace(":", "")
            summary = score(args.runs, package, output, workers=args.workers)
            print(json.dumps(summary, indent=2))
            return 0 if summary["scored"] == summary["attempted"] else 1
        selection = package.select(BLOCKS if args.all else args.block, args.sample_id)
        if args.command == "download":
            print(json.dumps(package.download(selection), indent=2))
            return 0
        return asyncio.run(runs.launch(options, package, selection))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
