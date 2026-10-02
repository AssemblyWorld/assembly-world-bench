"""Pinned benchmark files, with explicit local-only operation."""

import re
from pathlib import Path

from .common import PROTOCOL, file_hash, inside, read_json, sample_name

REPO = "AssemblyWorld/AssemblyWorldBench"
REVISION = "f351f0f6e9f8a314ec7593e847d07a45b776481e"
BLOCKS = (
    "partnet-none",
    "partnet-final-image",
    "ikea-manualbook",
    "assemblybench-manualbook",
    "fantastic-breaks-none",
)


FROZEN_HASHES = read_json(Path(__file__).with_name("frozen.json"))


class Package:
    """Access a pinned HF snapshot or an explicitly selected local package."""

    def __init__(self, local=None, *, revision=REVISION, cache_dir=None):
        self.local = Path(local).resolve() if local is not None else None
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A full HF commit is required")
        self.revision, self.cache_dir = revision, cache_dir
        self.root = self.local
        self.benchmark = read_json(self.get("benchmark.json"))
        b = self.benchmark
        if b.get("version") != 1 or b.get("evaluation") != {
            "protocol": PROTOCOL,
            "similarity_policy": "geometry",
            "similarity_threshold": 0.0001,
        }:
            raise ValueError("Expected the migrated assembly-evaluation-v1 benchmark package")
        if tuple(x["name"] for x in b["blocks"]) != BLOCKS:
            raise ValueError("Expected the five frozen benchmark blocks")
        if sum(len(x["samples"]) for x in b["blocks"]) != 100:
            raise ValueError("Expected 100 frozen evaluations")
        self.blocks = {x["name"]: x for x in b["blocks"]}
        self.configs = {}
        self.entries = {}

    def get(self, relative):
        if self.local is not None:
            path = inside(self.local, relative)
            if not path.is_file():
                raise FileNotFoundError(f"Missing local benchmark file: {relative}")
            return self._verified(relative, path)
        # Validate the remote filename before handing it to the Hub client.
        inside(Path("/benchmark"), relative)
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                REPO,
                relative,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        )
        if self.root is None:
            self.root = path.parent
        return self._verified(relative, path)

    def _verified(self, relative, path):
        if relative in FROZEN_HASHES and file_hash(path) != FROZEN_HASHES[relative]:
            raise ValueError(f"Expected frozen v1 metadata checksum: {relative}")
        return path

    def configuration(self, name):
        if name not in self.blocks:
            raise ValueError(f"Unknown block: {name}")
        if name not in self.configs:
            block = self.blocks[name]
            directory = f"{block['data']}/{block['repo_id'].split('/')[-1]}/{block['config_id']}"
            cfg = read_json(self.get(f"{directory}/config.json"))
            if (
                cfg["config_id"] != block["config_id"]
                or cfg["identity"]["dataset"] != block["repo_id"]
                or cfg["identity"]["revision"] != block["revision"]
                or set(cfg["samples"]) != set(block["samples"])
                or any(
                    cfg["samples"][sid][key] != block["samples"][sid][key]
                    for sid in cfg["samples"]
                    for key in ("sha256", "parts", "episode_id")
                )
            ):
                raise ValueError(f"Configuration differs from frozen block {name}")
            task = self.get(block["prompt_file"]).read_bytes()
            from .common import sha256

            if sha256(task) != block["prompt_sha256"]:
                raise ValueError(f"Task checksum differs: {name}")
            self.configs[name] = (directory, cfg, task.decode())
        return self.configs[name]

    def select(self, blocks, sample_ids=None):
        blocks = list(blocks)
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("Select distinct blocks or use --all")
        if sample_ids and len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Duplicate sample selection")
        selection = {}
        found = set()
        for name in blocks:
            _, cfg, _ = self.configuration(name)
            ids = [sid for sid in cfg["samples"] if not sample_ids or sid in sample_ids]
            if ids:
                selection[name] = ids
                found.update(ids)
        if sample_ids and found != set(sample_ids):
            raise ValueError(
                f"Unknown sample IDs in selected blocks: {sorted(set(sample_ids) - found)}"
            )
        if not selection:
            raise ValueError("Empty selection")
        return selection

    def entry(self, name, sid, *, scoring=False):
        directory, cfg, task = self.configuration(name)
        expected = cfg["samples"][sid]
        slug = sample_name(sid)
        initial = self.get(f"{directory}/{expected['episode']}")
        if file_hash(initial) != expected["sha256"]:
            raise ValueError(f"Initial episode checksum differs: {name}/{sid}")
        block = self.blocks[name]
        reference, pages = None, []
        if block["reference_mode"] != "none":
            refdir = f"{directory}/cache/{slug}/reference/{block['reference_mode']}"
            reference = read_json(self.get(f"{refdir}/pages.json"))
            if (
                reference["dataset"] != block["repo_id"]
                or reference["revision"] != block["revision"]
                or reference["reference_mode"] != block["reference_mode"]
                or not reference["pages"]
            ):
                raise ValueError("Reference identity differs")
            for index, page in enumerate(reference["pages"], 1):
                path = self.get(f"{refdir}/{page['file']}")
                if page["page"] != index or file_hash(path) != page["source_sha256"]:
                    raise ValueError("Reference checksum/order differs")
                pages.append(path)
        entry = dict(
            block=name,
            sample_id=sid,
            config=cfg,
            task=task,
            expected=expected,
            initial=initial,
            reference=reference,
            pages=pages,
        )
        if scoring:
            entry["evaluation"] = self.get(f"{directory}/cache/{slug}/evaluation.json")
            entry["evaluation_sha256"] = file_hash(entry["evaluation"])
        self.entries[(name, sid)] = entry
        return entry

    def download(self, selection):
        for name, ids in selection.items():
            for sid in ids:
                self.entry(name, sid, scoring=True)
        return dict(repo=REPO, revision=self.revision, root=str(self.root), selection=selection)
