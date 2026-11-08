# feelings

Five poems loaded into pandas, one stanza-numbered row per line. The vectors in
"a vector rendering of tenderness" haven't been written yet.

## What's here

| Path | What it is |
|---|---|
| `main.py` | The code. Edit this one. |
| `main.ipynb` | Generated from `main.py` by jupytext; gitignored. |
| `data/poems/*.txt` | The poems, one stanza per blank-line-separated block. |
| `data/poems/SOURCES.tsv` | Title, stanza count and source URL for each poem. |

## The notebook

`main.py` is a jupytext percent-format script: each `# %%` starts a cell. Kelly
works in Jupyter, so `task nb` syncs and opens `main.ipynb` in Jupyter Lab.
jupytext is a dev dependency, so its Jupyter extension saves both files at once.
Edits made in Zed reach the notebook on the next `task sync`, or when Jupyter
reloads the file.

`task sync` runs `jupytext --sync main.py`. Whichever file was saved last wins,
so sync before switching editors.

## Tasks

```
task          # list them
task setup    # uv sync
task sync     # main.py <-> main.ipynb
task nb       # sync, then Jupyter Lab
task fmt      # ruff fix + format
task ci       # lock-check, fmt-check, lint, typecheck; CI runs this
```

ruff and ty aren't in `.venv`. The Taskfile runs them with `uvx` at the versions
pinned at its top, so local runs and CI use the same binaries. ty still reads
packages from `.venv`, so `typecheck` syncs it first.

On Linux, torch comes from PyTorch's CPU index (`[tool.uv.sources]`), so CI
downloads about 200 MB of torch instead of 3 GB of CUDA. CI doesn't run
`main.py`.

direnv runs `uv sync --frozen` on `cd` (`.envrc`), so `task setup` is only
needed without it. Zed has the same tasks in `.zed/tasks.json`.
