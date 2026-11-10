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
import urllib.request
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from transformers import pipeline
from umap import UMAP

model = SentenceTransformer("all-MiniLM-L6-v2")


# %%
def load_poem(path: Path) -> pd.DataFrame:
    """Return one row per text line of the poem at path, whitespace intact.

    blank_before counts the blank or whitespace-only lines above each line.
    Most poems start a new stanza at any blank line. A double-spaced poem
    puts a blank line between nearly every pair of lines; there one blank
    line is vertical spacing, and only a run of two or more starts a stanza.
    """
    rows, run = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append({"line": line, "blank_before": run})
            run = 0
        else:
            run += 1
    df = pd.DataFrame(rows)

    double_spaced = df["blank_before"].iloc[1:].gt(0).mean() > 0.9
    breaks = df["blank_before"].ge(2 if double_spaced else 1)
    df["stanza"] = (breaks & (df.index > 0)).cumsum()
    return df


# One row per line, across all poems.
lines = pd.concat(
    [load_poem(p).assign(poem=p.stem) for p in sorted(Path("data/poems").glob("*.txt"))],
    ignore_index=True,
)


# %%
# ENRICHMENT (form): internal gaps per line, runs of 3+ spaces or any tab.
lines["n_gaps"] = lines["line"].str.strip().str.count(r"\s{3,}|\t")

# One row per stanza.
stanzas = (
    lines.groupby(["poem", "stanza"])
    .agg(
        text=("line", " ".join),
        # ENRICHMENT (form): line count and gap count per stanza.
        n_lines=("line", "size"),
        n_gaps=("n_gaps", "sum"),
    )
    .reset_index()
)

# Get vector embeddings from stanzas.
poem_vectors = model.encode(stanzas["text"].tolist(), normalize_embeddings=True)

# %%
similarity = poem_vectors @ poem_vectors.T

pairs = similarity[np.triu_indices(len(similarity), k=1)]
px.histogram(pairs, nbins=20)

# %%
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

# %%
# Lemmatization!!!
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
docs = list(nlp.pipe(stanzas["text"]))

stanzas["lemmas"] = [
    {token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "VERB"} and not token.is_stop}
    for doc in docs
]

# Calculate how much our stanzas overlap by verb/noun
overlap = np.array(
    [
        [len(a & b) / len(a | b) if a | b else 0.0 for b in stanzas["lemmas"]]
        for a in stanzas["lemmas"]
    ]
)

# %%
# ENRICHMENT (metadata): book, date, place and series, from SOURCES.tsv and file names.
sources = pd.read_csv("data/poems/SOURCES.tsv", sep="\t")
book = sources.set_index(sources["file"].str.removesuffix(".txt"))["source"]
book = book.str.removesuffix(" (Lauren Ireland)").where(~book.str.startswith("http"), "uncollected")
stanzas["book"] = stanzas["poem"].map(book)

# Dated titles look like april-30-2013-seattle or october-10-2008-queens-1.
dated = stanzas["poem"].str.extract(
    r"^(?P<month>january|february|march|april|may|june|july|august|september|october"
    r"|november|december)-(?P<day>\d{1,2})-(?P<year>\d{4})-(?P<place>.+?)(?:-\d)?$"
)
stanzas["date"] = pd.to_datetime(dated["month"] + " " + dated["day"] + " " + dated["year"])
stanzas["year"] = stanzas["date"].dt.year
stanzas["place"] = dated["place"].str.replace("-", " ")
stanzas["is_ritual"] = stanzas["poem"].str.startswith("ritual-for")
stanzas["n_stanzas"] = stanzas.groupby("poem")["stanza"].transform("size")


# %%
# Lexicons, downloaded once into data/lexicons.
def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


lexicons = Path("data/lexicons")

# Brysbaert et al. (2014): concreteness of ~40k English lemmas, 1 (abstract) to 5 (concrete).
concreteness_path = fetch(
    "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/"
    "Concreteness_ratings_Brysbaert_et_al_BRM.txt",
    lexicons / "brysbaert-concreteness.txt",
)
concreteness = pd.read_csv(concreteness_path, sep="\t", index_col="Word")["Conc.M"]

# Warriner, Kuperman & Brysbaert (2013): valence, arousal, dominance of ~14k English lemmas,
# 1 to 9. NRC VAD was the first choice, but saifmohammad.com stopped resolving.
vad_path = fetch(
    "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv",
    lexicons / "warriner-vad.csv",
)
vad = pd.read_csv(vad_path, index_col="Word")[["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]].set_axis(
    ["valence", "arousal", "dominance"], axis=1
)


def lexicon_mean(doc, scores: pd.Series) -> float:
    """Mean score over the stanza's content words found in the lexicon, by lemma then word."""
    hits = [
        scores.get(t.lemma_.lower(), scores.get(t.lower_))
        for t in doc
        if t.is_alpha and not t.is_stop
    ]
    hits = [h for h in hits if h is not None]
    return float(np.mean(hits)) if hits else np.nan


# ENRICHMENT (lexicon): mean concreteness, valence, arousal and dominance per stanza.
stanzas["concreteness"] = [lexicon_mean(doc, concreteness) for doc in docs]
for dim in ["valence", "arousal", "dominance"]:
    stanzas[dim] = [lexicon_mean(doc, vad[dim]) for doc in docs]

# ENRICHMENT (lexicon): share of words that are first-person, second-person, plural pronouns.
pronouns = {
    "pron_i": {"i", "me", "my", "mine", "myself"},
    "pron_you": {"you", "your", "yours", "yourself", "yourselves"},
    "pron_we": {"we", "us", "our", "ours", "ourselves"},
}
for col, words in pronouns.items():
    stanzas[col] = [
        sum(t.lower_ in words for t in doc) / max(sum(t.is_alpha for t in doc), 1) for doc in docs
    ]

# %%
# ENRICHMENT (novelty): cosine distance from the previous stanza of the same poem.
# Stanzas are sorted by poem then stanza, so the previous row is the previous stanza.
step = 1 - (poem_vectors[1:] * poem_vectors[:-1]).sum(axis=1)
stanzas["novelty"] = np.r_[np.nan, step]
stanzas.loc[stanzas["stanza"].eq(0), "novelty"] = np.nan

# %%
# ENRICHMENT (emotion): GoEmotions' 28 labels, from a RoBERTa model trained on Reddit comments.
classify = pipeline(
    "text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, truncation=True
)
# With top_k=None the pipeline returns every label's score for each stanza.
results = cast(list[list[dict]], classify(stanzas["text"].tolist()))
emotions = pd.DataFrame([{s["label"]: s["score"] for s in scores} for scores in results])
stanzas["emotion"] = emotions.idxmax(axis=1)
stanzas[[f"emo_{label}" for label in emotions.columns]] = emotions.to_numpy()

# %%
# ENRICHMENT (topic): BERTopic over the stanza embeddings. Its default HDBSCAN put 471 of
# 492 stanzas in one topic; KMeans assigns every stanza, and n_topics is the knob.
n_topics = 12
topic_model = BERTopic(
    embedding_model=model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
    hdbscan_model=KMeans(n_clusters=n_topics, random_state=42, n_init=10),
    vectorizer_model=CountVectorizer(stop_words="english"),
)
stanzas["topic"], _ = topic_model.fit_transform(stanzas["text"].tolist(), poem_vectors)
topics = topic_model.get_topic_info()
stanzas["topic_name"] = stanzas["topic"].map(topics.set_index("Topic")["Name"])
topics[["Topic", "Count", "Name"]]

# %%

alpha = 2.0

distance = (1 - similarity) + alpha * overlap
distance = (distance + distance.T) / 2
np.fill_diagonal(distance, 0)

pos = TSNE(metric="precomputed", init="random", random_state=42).fit_transform(distance)


# %%
# pos = nx.spring_layout(G, weight="weight", seed=42)
stanzas[["x", "y"]] = [pos[i] for i in stanzas.index]

fig = px.scatter(stanzas, x="x", y="y", color="poem", hover_data=["stanza", "text"])

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

# %%
sum(G.nodes[u]["poem"] != G.nodes[v]["poem"] for u, v in G.edges)

# %%
