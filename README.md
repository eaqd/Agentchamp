# VRP-Agents

> Build a small, honest experiment that asks: can borrowed cognitive styles, held by LLM agents, outperform classical heuristics on a well-defined combinatorial problem — and what do we learn either way?

Multi-agent LLM reasoning for the Capacitated Vehicle Routing Problem (CVRP), benchmarked against classical heuristics on the Augerat Set A instances from CVRPLIB.

## Status

Phase 1 scaffolding: data models, CVRPLIB parser, and a Set A download script. No solvers yet.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env  # then add your ANTHROPIC_API_KEY when Phase 2 starts
```

## Download benchmark data

```bash
uv run python scripts/download_data.py                   # all 27 Set A instances
uv run python scripts/download_data.py --instances A-n32-k5,A-n45-k6
```

Files land in `data/raw/` (and solutions in `data/solutions/`).

## Run tests

```bash
uv run pytest
```

The parser test skips if `A-n32-k5.vrp` hasn't been downloaded yet.

## Layout

```
src/
  models.py       # Pydantic models: Instance, Customer, Route, Solution
  parser.py       # CVRPLIB .vrp parser
  solvers/        # (Phase 1+) classical, single-agent, multi-agent
  agents/         # (Phase 3) personas + orchestrator
  evaluation/     # (Phase 4) benchmark runner + metrics
scripts/
  download_data.py
tests/
  test_parser.py
data/
  raw/            # .vrp instances (gitignored)
  solutions/      # .sol files (gitignored)
```

## License

MIT.
