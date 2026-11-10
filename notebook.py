import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import networkx as nx
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import spacy
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE


@app.cell
def _():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return (model,)


@app.function
def load_poem(path) -> pd.DataFrame:
    df = pd.DataFrame({"line": path.read_text(encoding="utf-8").splitlines()})
    df["stanza"] = df["line"].eq("").cumsum()
    return df[df["line"].ne("")].reset_index(drop=True)


@app.cell
def _():
    poems = {p.stem: load_poem(p) for p in sorted(Path("data/poems").glob("*.txt"))}
    return (poems,)


@app.cell
def _(model, poems):
    # Get vector embeddings from stanzas.
    stanzas = pd.DataFrame(
        [
            {"poem": name, "stanza": number, "text": " ".join(lines["line"])}
            for name, df in poems.items()
            for number, lines in df.groupby("stanza")
        ]
    )

    poem_vectors = model.encode(stanzas["text"].tolist(), normalize_embeddings=True)
    return poem_vectors, stanzas


@app.cell
def _(poem_vectors):
    similarity = poem_vectors @ poem_vectors.T

    pairs = similarity[np.triu_indices(len(similarity), k=1)]
    px.histogram(pairs, nbins=20)
    return (similarity,)


@app.cell
def _(similarity, stanzas):
    # Create graph from neighbors.
    k = 3
    floor = 0.2

    neighbors = np.argsort(-similarity, axis=1)[:, 1 : k + 1]

    G = nx.Graph()
    G.add_nodes_from(
        (i, {"poem": row.poem, "stanza": row.stanza, "text": row.text})
        for i, row in enumerate(stanzas.itertuples())
    )
    G.add_edges_from(
        (i, int(j), {"weight": float(similarity[i, j])})
        for i, row in enumerate(neighbors)
        for j in row
        if similarity[i, j] >= floor
    )
    return (G,)


@app.cell
def _():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return (nlp,)


@app.cell
def _(nlp, stanzas):
    # Lemmatization!!!
    lemmas = [
        {
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "VERB"} and not token.is_stop
        }
        for doc in nlp.pipe(stanzas["text"])
    ]

    # Calculate how much our stanzas overlap by verb/noun
    overlap = np.array(
        [[len(a & b) / len(a | b) if a | b else 0.0 for b in lemmas] for a in lemmas]
    )
    return (overlap,)


@app.cell
def _(overlap, similarity):
    alpha = 2.0

    distance = (1 - similarity) + alpha * overlap
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    pos = TSNE(metric="precomputed", init="random", random_state=42).fit_transform(distance)
    return (pos,)


@app.cell
def _(G, pos, stanzas):
    # pos = nx.spring_layout(G, weight="weight", seed=42)
    placed = stanzas.assign(x=pos[:, 0], y=pos[:, 1])

    fig = px.scatter(placed, x="x", y="y", color="poem", hover_data=["stanza", "text"])

    edge_x = [c for u, v in G.edges for c in (pos[u][0], pos[v][0], None)]
    edge_y = [c for u, v in G.edges for c in (pos[u][1], pos[v][1], None)]
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "lightgray", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.data = fig.data[-1:] + fig.data[:-1]
    fig.update_layout(showlegend=False)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    fig.write_image(out_dir / "graph.png")
    fig.write_html(out_dir / "graph.html")
    fig
    return


@app.cell
def _(G):
    sum(G.nodes[u]["poem"] != G.nodes[v]["poem"] for u, v in G.edges)
    return


if __name__ == "__main__":
    app.run()
