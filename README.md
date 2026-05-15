# Learning In Context

Code for the paper *Minimal neural circuits for contextual learning in humans and recurrent networks*.

This repository is the consolidated pipeline for the paper's neural and behavioral analyses. It supersedes the working state reached by the earlier `In-Context-CPD` project, porting in the validated portions and leaving behind unfinished work.

## Quickstart

```bash
# install uv if you don't have it: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
uv run pytest
```

## Status

First-pass port from `In-Context-CPD` in progress. See the plan in `dissertation/decisions/` for the porting roadmap and open decisions.

## License

MIT — see `LICENSE`.
