"""Cluster the net20k corpus by Jina clustering-task embeddings, label each
cluster with its top c-TF-IDF terms, and rank clusters by mean internal-LoC
probability from the decider-2b (short rubric) scores.

Inputs:
  training/data/dataset_raw_net20k.jsonl              text, uri
  training/data/scores_loc_decider_ft_net20k.jsonl     ts_class_probs, ts_class, ts_score
  research/topic_model/embeddings_net20k_jina_clustering.npz   uri-aligned X (built by
      embed_net20k_jina.py, jina-embeddings-v5-text-small, task=clustering)

Output: research/topic_model/topic_model_results.json (full per-cluster table)
and a printed ranking.

Run: ../../tmp_venv/bin/python topic_model_loc.py
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW_FILE = os.path.join(ROOT, "training", "data", "dataset_raw_net20k.jsonl")
SCORES_FILE = os.path.join(ROOT, "training", "data", "scores_loc_decider_ft_net20k.jsonl")
DATA_DIR = os.path.join(ROOT, "research", "data", "topic_model")
EMB_FILE = os.path.join(DATA_DIR, "embeddings_net20k_jina_clustering.npz")
OUT_FILE = os.path.join(DATA_DIR, "topic_model_results.json")

N_CLUSTERS = int(os.environ.get("LOC_TOPIC_K", "40"))
MIN_CLUSTER_N = int(os.environ.get("LOC_TOPIC_MIN_N", "40"))
SEED = 42

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@[\w.\-]+")
TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜßàâçéèêëîïôûùüÿñæœ]+")

STOPWORDS = set("""
the a an and or but if then else so of to in on at for with by from as is are
was were be been being this that these those it its it's i you he she we they
them his her their our your my me him us not no nor do does did doing have
has had having will would shall should can could may might must there here
what which who whom whose when where why how all any both each few more most
other some such only own same than too very s t just don now
""".split()) | set("""
der die das ein eine einer eines einem einen und oder aber wenn dann sonst
von zu in an auf für mit bei aus als ist sind war waren sein wird werden
nicht kein keine nur auch noch schon wie was wer wo wenn dass ob sich mich
dich uns euch ihnen ihr ihre ihrer ihres euer meine mein dein sein er sie es
wir ihr sie man den dem des im am zum zur vom beim über unter durch gegen
ohne um so mehr sehr immer heute schon jetzt hier da
""".split()) | set(
    "rt amp https http www com net org de en co th nd rd para que los las el "
    "de la en y un una con por que se lo al del este esta esto".split()
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_tokens(text):
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def main():
    raw = {r["uri"]: r for r in load_jsonl(RAW_FILE) if r.get("text", "").strip()}
    scores = {r["uri"]: r for r in load_jsonl(SCORES_FILE)}

    data = np.load(EMB_FILE, allow_pickle=True)
    X, uris = data["X"], list(data["uris"])
    rows = []
    keep_idx = []
    for i, u in enumerate(uris):
        r = raw.get(u)
        s = scores.get(u)
        if r is None or s is None:
            continue
        keep_idx.append(i)
        rows.append({
            "uri": u,
            "text": r["text"],
            "langs": r.get("langs") or [],
            "p_internal": s["ts_class_probs"]["internal"],
            "p_external": s["ts_class_probs"]["external"],
            "p_neutral": s["ts_class_probs"]["neutral"],
            "ts_class": s["ts_class"],
            "ts_score": s["ts_score"],
        })
    X = X[keep_idx].astype(np.float32)
    print(f"{len(rows)} rows with embedding + text + score")

    # L2-normalize so k-means (euclidean) behaves like cosine k-means.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-12, None)

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10, batch_size=1024)
    labels = km.fit_predict(Xn)

    from sklearn.feature_extraction.text import TfidfVectorizer

    # c-TF-IDF: treat each cluster's concatenated, tokenized text as one
    # "document" and rank terms within it against the other clusters' documents.
    cluster_docs = []
    for c in range(N_CLUSTERS):
        toks = []
        for i in np.where(labels == c)[0]:
            toks.extend(clean_tokens(rows[i]["text"]))
        cluster_docs.append(" ".join(toks))

    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True, min_df=1)
    tfidf = vec.fit_transform(cluster_docs)
    terms = np.array(vec.get_feature_names_out())

    clusters = []
    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]
        n = len(idx)
        if n == 0:
            continue
        top_term_idx = tfidf[c].toarray().ravel().argsort()[::-1][:10]
        top_terms = [t for t in terms[top_term_idx] if t]

        p_int = np.array([rows[i]["p_internal"] for i in idx])
        p_ext = np.array([rows[i]["p_external"] for i in idx])
        p_neu = np.array([rows[i]["p_neutral"] for i in idx])
        ts_score = np.array([rows[i]["ts_score"] for i in idx])
        cls = [rows[i]["ts_class"] for i in idx]

        # Example posts closest to the cluster centroid, most internal first.
        centroid = km.cluster_centers_[c]
        sims = Xn[idx] @ centroid
        order_by_internal = idx[np.argsort(-p_int)]
        examples = []
        for i in order_by_internal[:3]:
            t = rows[i]["text"].replace("\n", " ")
            examples.append({
                "text": t[:220],
                "p_internal": round(float(rows[i]["p_internal"]), 3),
                "ts_class": rows[i]["ts_class"],
            })

        clusters.append({
            "cluster": int(c),
            "n": int(n),
            "top_terms": top_terms,
            "mean_p_internal": float(p_int.mean()),
            "mean_p_external": float(p_ext.mean()),
            "mean_p_neutral": float(p_neu.mean()),
            "mean_ts_score": float(ts_score.mean()),
            "frac_class_internal": float(np.mean([x == "internal" for x in cls])),
            "frac_class_external": float(np.mean([x == "external" for x in cls])),
            "examples": examples,
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"n_clusters": N_CLUSTERS, "min_cluster_n": MIN_CLUSTER_N,
                    "clusters": clusters}, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_FILE}")

    ranked = sorted([c for c in clusters if c["n"] >= MIN_CLUSTER_N],
                     key=lambda c: -c["mean_p_internal"])
    print(f"\n{'rank':>4} {'n':>5} {'P(int)':>7} {'P(ext)':>7} {'%int-cls':>8}  top terms")
    for r, c in enumerate(ranked, 1):
        print(f"{r:>4} {c['n']:>5} {c['mean_p_internal']:>7.3f} {c['mean_p_external']:>7.3f} "
              f"{c['frac_class_internal']*100:>7.1f}%  {', '.join(c['top_terms'][:8])}")

    print("\n--- most EXTERNAL clusters (for contrast) ---")
    for c in ranked[::-1][:8]:
        print(f"n={c['n']:>5} P(int)={c['mean_p_internal']:.3f} P(ext)={c['mean_p_external']:.3f}  "
              f"{', '.join(c['top_terms'][:8])}")


if __name__ == "__main__":
    main()
