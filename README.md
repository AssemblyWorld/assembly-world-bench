# AssemblyWorldBench

Download the frozen benchmark, run Codex or Claude Code through browser WebMCP,
and score the exported assemblies with official SCD, PA and SR.

[Benchmark](https://huggingface.co/datasets/AssemblyWorld/AssemblyWorldBench) ·
[Paper](https://arxiv.org/abs/2609.40353) ·
[Research website](https://assemblyworld.github.io/) ·
[Recorded paper results](https://huggingface.co/datasets/AssemblyWorld/AssemblyWorldBench-Results)

This is a source repository with a CLI, maintained for macOS and Linux.
`uv` manages its Python dependencies; the project itself is not installed as a
Python distribution. The separate results dataset is linked for reference only.

## Quick start

Install these prerequisites once:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python 3.12+ is managed by uv).
- Google Chrome 150+ with WebMCP support.
- Node.js 24+ and `chrome-devtools-mcp@1.8.0` on your `PATH`.
- Either [Codex CLI](https://developers.openai.com/codex/cli/) or
  [Claude Code](https://code.claude.com/docs/en/setup), on your `PATH`.

```bash
npm install --global chrome-devtools-mcp@1.8.0
# Install one of the agent CLIs if you do not already have it:
npm install --global @openai/codex
# Or follow the Claude Code installation guide linked above.

git clone https://github.com/AssemblyWorld/assembly-world-bench
cd assembly-world-bench
cp .env.example .env
```

Edit `.env` and set the key for the agent you want to run:

```dotenv
# Codex: OpenAI API key
CODEX_API_KEY=your-key
# Claude Code: Anthropic API key
ANTHROPIC_API_KEY=your-key
```

Only the selected agent's key is needed. `bench.py` automatically reads the
repository's `.env`; already exported environment variables take precedence.
`.env` is ignored by Git. Keys are passed through the subprocess environment,
not written into run configuration. API-key authentication is supported directly
by [Codex exec](https://developers.openai.com/codex/noninteractive#authenticate-in-automation)
and [Claude Code](https://code.claude.com/docs/en/authentication).

Then run a sample, replacing `MODEL_ID` with an available model ID:

```bash
uv run bench.py run --block ikea-manualbook --sample-id Chair/reidar \
  --agent codex --model MODEL_ID --headless
# Or use Claude Code:
uv run bench.py run --block ikea-manualbook --sample-id Chair/reidar \
  --agent claude --model MODEL_ID --headless
```

The first invocation installs locked Python dependencies automatically, downloads
only the required benchmark files and starts an independent browser and agent.
No Python package installation or separate data preparation is needed.

To check Chrome, MCP and the chosen CLI before spending model credits:

```bash
uv run bench.py doctor --agent codex --headless
```

`doctor` checks installed versions and real browser/tool discovery without calling
a model. Use `--chrome-path` or `--mcp-command` for nonstandard installations.
Linux uses software WebGL and supports both headed and headless execution.

## Commands and selections

```bash
uv run bench.py --help
uv run bench.py run --help
```

| Command | Purpose |
| --- | --- |
| `download` | Prefetch selected input and scoring files |
| `doctor` | Check browser/MCP and agent CLI prerequisites without a model |
| `run` | Execute selected frozen benchmark tasks |
| `status` | Inspect a run's progress |
| `resume` | Start a new run linked to an interrupted or failed run |
| `eval` | Score final attempts and write a separate report |

There are 100 tasks over 80 shapes, with 20 tasks per block:

| Block | Source | Reference |
| --- | --- | --- |
| `partnet-none` | PartNet / Manual-PA | None |
| `partnet-final-image` | Same 20 PartNet shapes | Final image |
| `ikea-manualbook` | IKEA-Manual | Full manual |
| `assemblybench-manualbook` | AssemblyBench | Ordered diagrams |
| `fantastic-breaks-none` | Fantastic Breaks | None |

```bash
uv run bench.py download --block ikea-manualbook
uv run bench.py run --block partnet-none --agent codex --model MODEL_ID --headless
uv run bench.py run --all --agent claude --model MODEL_ID --concurrency 2 --headless
```

Repeat `--block` and `--sample-id` to select tasks. `--all` explicitly runs all
five blocks in frozen order; each block finishes before the next begins.
Concurrency defaults to 1. The model must be specified explicitly. Agent stages
have no default time limit; use `--effort` and `--timeout-seconds` as needed.
Task text and reference conditions are fixed by the benchmark.

## Data and custom providers

Without `--benchmark`, input and scoring resources are downloaded on demand from
HF commit `f351f0f6e9f8a314ec7593e847d07a45b776481e` using standard Hub caches.
`download --all` prefetches the full benchmark. `--cache-dir` relocates the cache;
`--revision` accepts another full commit compatible with the frozen contracts.

For a local snapshot, add `--benchmark /absolute/path/to/package` to `download`,
`run` or `eval`. Local mode never fetches missing files: missing, corrupted or
legacy v2 inputs fail explicitly. The required evaluation protocol is
`assembly-evaluation-v1`; no source adapters or GT reconstruction are provided.

Codex can use a custom provider through repeated `--codex-config KEY=VALUE`.
For example, put `PROVIDER_API_KEY=your-key` in `.env`, then configure your service:

```bash
uv run bench.py run --block partnet-none --agent codex --model MODEL_ID --headless \
  --codex-config 'model_provider="custom"' \
  --codex-config 'model_providers.custom.name="Custom service"' \
  --codex-config 'model_providers.custom.base_url="https://YOUR_SERVICE/v1"' \
  --codex-config 'model_providers.custom.env_key="PROVIDER_API_KEY"' \
  --codex-config 'model_providers.custom.wire_api="responses"'
```

The service must support the Responses API used by Codex. Only `model_provider`,
`model_context_window` and `model_providers.*` overrides are accepted.
Use environment-variable references for secrets: literal override values are
recorded in `run.json`. Benchmark tool restrictions remain fixed.

Each sample has its own Chrome profile and CLI workspace. Only selected input
episodes are served over a run-scoped loopback service. Reference images are
attached in the frozen page order. GT never enters prompts, MCP tools or HTTP.
The [3DWebAgent deployment](https://assemblyworld.github.io/3DWebAgent/) owns the scene and
XML/MJB episode contracts; `--environment-url` selects a compatible deployment.

## Records and recovery

Each execution writes a new `logs/<run-id>/`:

```text
run.json
blocks/<block>/samples/<sample-slug>/
  input.json
  prompt.txt
  conversation.jsonl
  result.json
  final.episode.zip
```

Records preserve the source commit, CLI versions, source identities, input hashes,
HF revision, public conversation and actual execution events. Unknown usage/cost
stays null, and private reasoning is excluded. `--logs` selects another output root.

```bash
uv run bench.py status logs/RUN_ID
uv run bench.py resume logs/RUN_ID --concurrency 2
uv run bench.py resume logs/RUN_ID --retry-failed
```

Quota exhaustion or three consecutive infrastructure failures stops new
dispatches and drains current workers. There is no automatic retry or quota reset.
Resume creates a new `source_run`-linked directory and leaves earlier runs intact.
It resumes pending/interrupted tasks across the chain by default;
`--retry-failed` also includes failures from earlier runs.
A valid completed export remains available for geometric scoring even if the
agent reports a partial outcome.

## Official evaluation

```bash
uv run bench.py eval logs/RUN_ID --output logs/evaluation/NEW_ID
uv run bench.py eval logs/RESUMED_ID --benchmark /absolute/path/to/package --workers 4
```

Evaluation follows explicit recovery chains and selects each sample's last actual
attempt. A newer failure without a valid export never falls back to an older
success. Unrelated overlapping runs and ambiguous sibling retries are errors.
`metrics.jsonl` and `summary.json` go into a new directory; archived runs and
input data are never overwritten.

The frozen `assembly-evaluation-v1` protocol uses shared SE(3) alignment with
24 proper PCA starts, same-ID part pose starts and symmetric ICP. Matching uses
fixed cached geometry equivalence groups and unclipped Hungarian Chamfer costs.
SCD sums directional mean squared nearest-neighbor distances ×1000; PA is the
fraction of parts with Chamfer distance <=0.01; SR requires all parts correct.
There is no scale or reflection fitting, and geometric correctness does not
establish physical validity.

Partial selections report sample/block scores and coverage. Official Overall
appears only when all 100 tasks have explicit final attempt records. Unscorable
attempts have PA=SR=0; SCD averages successfully scored attempts. Refusals remain
in the denominator, and Agent-reported status does not replace geometry scores.
Overall first averages blocks within each source, then the four sources equally.
Resource summaries use only the selected attempt's time, tokens, tool calls and
CLI-reported cost; earlier costs remain in their logs. Costs are never estimated.

## Development and provenance

```bash
uv run ruff check .
uv run ruff format --check .
HF_HUB_OFFLINE=1 uv run pytest -q
```

CI validates dependency setup, direct source CLI execution, synthetic offline
tests and real Chrome/WebMCP roundtrips on Ubuntu and macOS. Browser jobs also
check pinned downloads from every block and local-only snapshots. No model calls
are made in CI. For local browser tests, set `AWB_BROWSER_EPISODE` to an input
archive and `AWB_MCP_COMMAND` to the installed MCP executable, then run
`uv run pytest -q -s tests/test_browser_live.py`.

The numerical runtime is pinned in `uv.lock`. Validation of the 100 existing
GPT-6 Astra exports matched the former evaluator: PA/SR exactly,
SCD within atol=1e-8 / rtol=1e-6 (observed maximum difference 0).
The HF v1 migration changed protocol markers and documentation only, preserving
all data assets and GT numeric values. Scoring mathematics came from
`assembly-world-agent` commit `c363cd9`; retained schemas record the exact
3DWebAgent provenance commits and hashes under `assembly_world_bench/contracts/`.
No sibling source repository or source dataset library is needed at runtime.

Software is [MIT](LICENSE). Downloaded geometry, images and GT retain the terms in
the [benchmark license](https://huggingface.co/datasets/AssemblyWorld/AssemblyWorldBench/blob/main/LICENSE.md).
The software license does not relicense those assets.
