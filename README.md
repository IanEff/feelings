# feelings

A vector rendering of tenderness

## Setup

direnv does it: `cd` in and `.envrc` runs `uv sync --frozen` and activates `.venv`
(and loads `.env` if there is one). Without direnv:

```bash
uv sync
```

`just` lists every recipe.

## Structure

```
feelings/
├── src/feelings/
├── tests/
└── pyproject.toml
```

## Running

```bash
uv run feelings           # console script
uv run python -m feelings
```

## Checks

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv check                   # type check: uv runs ty, at the version pinned in uv.lock
uv run pytest              # tests
```

## Dependencies

```bash
uv add httpx               # runtime
uv add --dev hypothesis    # dev group
```

Commit `uv.lock`.

## Zed

Global Zed settings run ty (with `experimental.useUv`) and Ruff for Python.
`.zed/tasks.json` has test, check and format tasks (`task: spawn`); `.zed/debug.json`
has debugpy configs (F4).

## Python version

Pinned in `.python-version` (3.15).

---

Ian Furst (ian.furst@gmail.com)
