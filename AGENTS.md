# AssemblyWorldBench

This independent uv project only downloads, runs and scores the frozen HF benchmark.
Do not add source adapters, preparation, GT reconstruction, paper analysis or visualization.
Keep maintained documentation in README.md. Code and documentation are English.
Never expose GT through the agent workspace, prompts, MCP or HTTP. Read data without
modifying it, use pinned revisions, and preserve every execution attempt in its own run.
Only explicit source_run chains may merge; score the last actual attempt, including failures.
Only a complete 100-item selection may report official Overall. Preserve frozen mathematics.
No model calls in automated validation. Real browser integration is opt-in, without models.
Ignore data, logs, credentials and build artifacts. Keep fixtures synthetic.
