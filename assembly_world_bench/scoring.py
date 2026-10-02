"""Official cache-only scoring. No source loading, reconstruction or regrouping."""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import PROTOCOL, file_hash, read_json, sample_name, write_json
from .episode_io import read_episode
from .geometry import rotation_matrix, score_parts
from .poses import final_poses_against_initial
from .registration import align, transform


def evaluation_inputs(entry):
    value = read_json(entry["evaluation"])
    key, cfg, expected = value["key"], entry["config"], entry["expected"]
    if (
        value["version"] != 1
        or key["evaluation_protocol"] != PROTOCOL
        or key["sample_id"] != entry["sample_id"]
        or key["identity"] != cfg["identity"]
        or key["initial_sha256"] != expected["sha256"]
        or key["parts"] != expected["parts"]
        or key["preparation_protocol"] != cfg["identity"]["protocol"]
        or key["similarity"]["policy"] != "geometry"
        or key["similarity"]["threshold"] != 0.0001
        or value["equivalence"]["protocol"] != key["similarity"]
    ):
        raise ValueError("GT identity or frozen protocol differs")
    ids = value["part_ids"]
    if sorted(set(ids)) != ids or len(ids) != expected["parts"]:
        raise ValueError("Invalid GT part IDs")
    groups = value["equivalence"]["groups"]
    if sorted(pid for group in groups for pid in group) != ids or any(not g for g in groups):
        raise ValueError("Invalid equivalence partition")
    divisor = float(value["scale_divisor"])
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("Invalid GT scale")
    clouds, poses = [], []
    for pid in ids:
        cloud = np.asarray(value["points"][pid], dtype=float)
        pose = np.asarray(value["gt_poses"][pid], dtype=float)
        if cloud.shape != (1000, 3) or pose.shape != (7,):
            raise ValueError("Invalid GT dimensions")
        if not np.isfinite(cloud).all() or not np.isfinite(pose).all():
            raise ValueError("Nonfinite GT")
        rotation_matrix(pose[3:])
        clouds.append(cloud)
        poses.append(pose)
    return ids, clouds, poses, divisor, [[ids.index(pid) for pid in g] for g in groups]


def score_episode(entry, final, result=None):
    """Restore the latest recorded state and run unchanged frozen mathematics."""
    ids, clouds, gt, divisor, groups = evaluation_inputs(entry)
    expected = entry["expected"]
    if file_hash(entry["initial"]) != expected["sha256"]:
        raise ValueError("Initial checksum differs")
    if file_hash(entry["evaluation"]) != entry["evaluation_sha256"]:
        raise ValueError("GT file changed after validation")
    checksum = file_hash(final)
    if result and (saved := result.get("archive", {}).get("sha256")) and saved != checksum:
        raise ValueError("Final episode checksum differs from execution record")
    episode = read_episode(final)
    poses, state_index = final_poses_against_initial(
        episode,
        read_episode(entry["initial"]),
        ids,
    )
    prediction, target, seeds = [], [], []
    for cloud, pose, (rotation, translation) in zip(clouds, gt, poses):
        gt_rotation = rotation_matrix(pose[3:])
        prediction.append(transform(cloud, rotation, translation) / divisor)
        target.append((cloud @ gt_rotation.T + pose[:3]) / divisor)
        r = gt_rotation @ rotation.T
        seeds.append((r, (pose[:3] - r @ translation) / divisor))
    r, t, alignment = align(np.concatenate(prediction), np.concatenate(target), seeds)
    prediction = [transform(p, r, t) for p in prediction]
    scores, records = score_parts(prediction, target, groups, ids, alignment["chamfer"])
    if file_hash(final) != checksum:
        raise ValueError("Final episode changed during evaluation")
    return dict(
        block=entry["block"],
        sample_id=entry["sample_id"],
        status="scored",
        **scores,
        protocol_version=PROTOCOL,
        episode_sha256=checksum,
        state_index=state_index,
        parts=records,
        alignment=alignment,
        scale_divisor=divisor,
    )


def selected_attempts(runs, package):
    """Merge explicit chains only; never prefer an older successful export."""
    roots, records, visited = {}, {}, set()

    def visit(path, stack=()):
        path = Path(path).resolve()
        if path in stack:
            raise ValueError("Cycle in source_run chain")
        if path in visited:
            return roots[path]
        meta = read_json(path / "run.json")
        if meta.get("kind") != "assembly-world-bench" or meta.get("version") != 1:
            raise ValueError("Expected an assembly-world-bench run")
        root = visit(meta["source_run"], (*stack, path)) if meta.get("source_run") else path
        for name, ids in meta["selection"].items():
            block = package.blocks[name]
            if meta["tasks"][name] != package.configuration(name)[2]:
                raise ValueError("Run task differs from official task")
            if meta["reference_modes"][name] != block["reference_mode"]:
                raise ValueError("Run reference condition differs")
            for sid in ids:
                if sid not in block["samples"]:
                    raise ValueError("Run contains a sample outside the frozen block")
                sample = path / "blocks" / name / "samples" / sample_name(sid)
                inputs = read_json(sample / "input.json")
                expected = package.configuration(name)[1]["samples"][sid]
                if (
                    inputs["sample_id"] != sid
                    or inputs["block"] != name
                    or inputs["identity"] != package.configuration(name)[1]["identity"]
                    or any(inputs[k] != expected[k] for k in ("sha256", "episode_id", "parts"))
                ):
                    raise ValueError("Run input identity differs from frozen benchmark")
                result = read_json(sample / "result.json")
                if result["status"] == "pending":
                    continue
                key = name, sid
                if key in records:
                    previous = records[key]
                    if previous["root"] != root:
                        raise ValueError(f"Duplicate sample in unrelated runs: {key}")
                    if previous["run"] not in stack and previous["run"] != path:
                        # Two siblings are alternative attempts, not an ordered chain.
                        ancestor = path
                        ancestors = set()
                        while ancestor:
                            ancestors.add(ancestor)
                            source = read_json(ancestor / "run.json").get("source_run")
                            ancestor = Path(source).resolve() if source else None
                        if previous["run"] not in ancestors:
                            raise ValueError(f"Ambiguous sibling attempts: {key}")
                records[key] = dict(run=path, root=root, directory=sample, result=result)
        visited.add(path)
        roots[path] = root
        return root

    # Process ancestors before children, regardless of user argument order.
    def depth(p, seen=()):
        p = Path(p).resolve()
        if p in seen:
            raise ValueError("Cycle in source_run chain")
        source = read_json(p / "run.json").get("source_run")
        return 1 + depth(source, (*seen, p)) if source else 0

    for run in sorted(set(map(Path, runs)), key=depth):
        visit(run)
    return records


def _score_job(job):
    entry, record = job
    result = record["result"]
    final = record["directory"] / "final.episode.zip"
    base = dict(
        block=entry["block"],
        sample_id=entry["sample_id"],
        run=str(record["run"]),
        duration_seconds=result.get("duration_seconds"),
        usage=result.get("execution", {}).get("usage"),
        cost_usd=result.get("execution", {}).get("cost_usd"),
        tool_calls=result.get(
            "tool_calls", count_tool_calls(record["directory"] / "conversation.jsonl")
        ),
        attempt_finished=result["status"] != "running",
    )
    try:
        if result["status"] == "running":
            raise RuntimeError("Latest attempt is unfinished; no fallback to an earlier export")
        if not final.is_file() or result.get("archive", {}).get("status") != "saved":
            raise FileNotFoundError("Latest attempted execution has no saved final episode")
        return {**base, **score_episode(entry, final, result)}
    except Exception as exc:
        return dict(
            **base,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            SCD=None,
            PA=0.0,
            SR=0,
            protocol_version=PROTOCOL,
        )


def count_tool_calls(path):
    """Count recorded MCP invocations, including read-only and failed calls."""
    if not Path(path).is_file():
        return None
    try:
        with Path(path).open() as stream:
            return sum(json.loads(line).get("type") == "tool_call" for line in stream)
    except (ValueError, AttributeError):
        return None


def summarize(rows, package):
    blocks = {}
    for name, block in package.blocks.items():
        selected = [r for r in rows if r["block"] == name]
        if not selected:
            continue
        scored = [r for r in selected if r["status"] == "scored"]
        blocks[name] = dict(
            attempted=len(selected),
            expected=len(block["samples"]),
            scored=len(scored),
            SCD=float(np.mean([r["SCD"] for r in scored])) if scored else None,
            PA=float(np.mean([r["PA"] for r in selected])),
            SR=float(np.mean([r["SR"] for r in selected])),
        )
    expected = {(b["name"], sid) for b in package.benchmark["blocks"] for sid in b["samples"]}
    complete = {(r["block"], r["sample_id"]) for r in rows} == expected and all(
        r.get("attempt_finished", True) for r in rows
    )
    summary = dict(
        protocol=PROTOCOL,
        status="complete" if complete else "partial",
        attempted=len(rows),
        expected=100,
        scored=sum(r["status"] == "scored" for r in rows),
        blocks=blocks,
        official_overall=None,
        resource_accounting="Selected final attempt only; unknown values remain null",
    )
    if complete:
        source_scores = {}
        for source in package.benchmark["sources"]:
            names = [b["name"] for b in package.blocks.values() if b["source"] == source]
            source_scores[source] = {
                metric: float(np.mean([blocks[n][metric] for n in names]))
                if all(blocks[n][metric] is not None for n in names)
                else None
                for metric in ("SCD", "PA", "SR")
            }
        summary["sources"] = source_scores
        summary["official_overall"] = {
            metric: float(np.mean([v[metric] for v in source_scores.values()]))
            if all(v[metric] is not None for v in source_scores.values())
            else None
            for metric in ("SCD", "PA", "SR")
        }
    summary["resources"] = {
        name: dict(
            reported_samples=sum(r.get(name) is not None for r in rows),
            mean=float(np.mean([r[name] for r in rows if r.get(name) is not None]))
            if any(r.get(name) is not None for r in rows)
            else None,
        )
        for name in ("duration_seconds", "cost_usd", "tool_calls")
    }
    usage_keys = sorted(
        {k for row in rows for k in (row.get("usage") or {}) if k.endswith("_tokens")}
    )
    summary["token_usage"] = {
        key: dict(
            reported_samples=sum((row.get("usage") or {}).get(key) is not None for row in rows),
            mean=float(
                np.mean(
                    [
                        (row.get("usage") or {})[key]
                        for row in rows
                        if (row.get("usage") or {}).get(key) is not None
                    ]
                )
            ),
        )
        for key in usage_keys
        if any((row.get("usage") or {}).get(key) is not None for row in rows)
    }
    return summary


def evaluate(runs, package, output, *, workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output).resolve()
    for run in runs:
        if output.is_relative_to(Path(run).resolve()):
            raise ValueError("Evaluation output must be outside archived runs")
    if output.is_relative_to(package.root.resolve()):
        raise ValueError("Evaluation output must be outside the benchmark package")
    if output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    records = selected_attempts(runs, package)
    if not records:
        raise ValueError("No final execution attempts to evaluate")
    jobs = [(package.entry(*key, scoring=True), record) for key, record in records.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_score_job, jobs))
    rows.sort(key=lambda r: (r["block"], r["sample_id"]))
    summary = summarize(rows, package)
    output.mkdir(parents=True)
    (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(output / "summary.json", summary)
    return summary
