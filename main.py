# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %%
from pathlib import Path

import pandas as pd


# %%
def load_poem(path: Path) -> pd.DataFrame:
    df = pd.DataFrame({"line": path.read_text(encoding="utf-8").splitlines()})
    df["stanza"] = df["line"].eq("").cumsum()
    return df[df["line"].ne("")].reset_index(drop=True)


poems = {p.stem: load_poem(p) for p in sorted(Path("data/poems").glob("*.txt"))}

# %%
poems["how-am-i"]
