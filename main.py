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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer("all-MiniLM-L6-v2")


# %%
def load_poem(path: Path) -> pd.DataFrame:
    df = pd.DataFrame({"line": path.read_text(encoding="utf-8").splitlines()})
    df["stanza"] = df["line"].eq("").cumsum()
    return df[df["line"].ne("")].reset_index(drop=True)


poems = {p.stem: load_poem(p) for p in sorted(Path("data/poems").glob("*.txt"))}


# %%
def chunk_poem(df: pd.DataFrame, lines_per_chunk: int = 2) -> list[str]:
    return [
        " ".join(batch)
        for _, stanza in df.groupby("stanza")
        for batch in batched(stanza["line"], lines_per_chunk, strict=False)
    ]


poem_chunks: dict[str, list[str]] = {name: chunk_poem(df) for name, df in poems.items()}

poem_vectors: dict[str, np.ndarray] = {
    name: model.encode(chunks, normalize_embeddings=True) for name, chunks in poem_chunks.items()
}

# %%
# Squash into 2d array with reference words for plotting.  Robot said to do this for the whole
# thing, otherwise the coords are all relative.


reference_words = ["tenderness", "anxiety", "grief", "water", "death", "love"]
word_vectors = model.encode(reference_words, normalize_embeddings=True)

rows = [
    {"group": name, "text": text, "vector": vec}
    for name in poem_vectors
    for text, vec in zip(poem_chunks[name], poem_vectors[name], strict=True)
] + [
    {"group": "reference_word", "text": word, "vector": vec}
    for word, vec in zip(reference_words, word_vectors, strict=True)
]

plot_df = pd.DataFrame(rows)

# Squoooosh!
matrix_384d = np.vstack(plot_df["vector"].to_list())
coords_2d = PCA(n_components=2).fit_transform(matrix_384d)
plot_df[["x", "y"]] = coords_2d

# %%
# Let's display this mess..

plot_df["label"] = plot_df["text"].where(plot_df["group"].eq("reference_word"), "")

fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="group",
    text="label",
    hover_data=["text"],
)
fig.update_traces(textposition="top center")
# fig.show()
out_dir = Path("out")
out_dir.mkdir(exist_ok=True)
fig.write_image(out_dir / "stanzas.png", width=1200, height=800, scale=2)
fig.write_html(out_dir / "stanzas.html")
fig
