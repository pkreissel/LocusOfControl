"""Distill a single query vector for "internal" posts from LM Studio soft labels.

Trains a Linear Ridge Regression against a continuous Target Score. With v4
teacher annotations the target is the graded 0-9 loc_score (expected value over
the teacher's digit logprobs) — a genuinely rankable locus-of-control axis. For
older data without loc_score it falls back to the legacy P(Internal)*Expected_Level,
rescaled to the same [0,9] axis. See get_target_score / get_expected_score.

Outputs internal_vector.npy (best model) and results_internal_vector.json.

Run: uv run --with numpy --with scipy --with scikit-learn python train_internal_vector.py
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

# Overridable to train on a different teacher dataset (see README):
LMSTUDIO_FILE = os.environ.get("LOC_TEACHER_FILE", "dataset_annotated_lmstudio_soft.jsonl")
GEMINI_FILE = "dataset_validation_gemini.jsonl"  # defines validation posts + embedding row order
VAL_LABELS_FILE = os.environ.get("LOC_VAL_LABELS_FILE")  # optional uri->loc_label override
INTERACTIONS_FILE = "dataset_interactions.jsonl"
CACHE_LMSTUDIO_EMBS = os.environ.get("LOC_EMB_CACHE", "embeddings_lmstudio_soft_v3_jina.npz") # Updated cache name for v3
CACHE_INTERACTIONS_EMBS = "embeddings_interactions_jina.npz"
CACHE_GEMINI_EMBS = "embeddings_gemini_jina.npz"
VECTOR_OUT = os.environ.get("LOC_VECTOR_OUT", "internal_vector.npy")
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"

# Determine label set from prompt version
_PROMPT_VER = os.environ.get("LOC_PROMPT_VERSION", "v1")
if _PROMPT_VER == "assist":
    VALID_LABELS = ["helpful", "inert", "neutral"]
    LABEL_MAP = {"helpful": 0, "inert": 1, "neutral": 2}
    POSITIVE_LABEL = "helpful"
    NEGATIVE_LABEL = "inert"
else:
    VALID_LABELS = ["internal", "external", "neutral"]
    LABEL_MAP = {"internal": 0, "external": 1, "neutral": 2}
    POSITIVE_LABEL = "internal"
    NEGATIVE_LABEL = "external"

CACHE_TEMPERATURE = 2.0
REFERENCE_T = 1.0
TEMPERATURES = [0.5, 1.0, 2.0, 3.0, 5.0]
L2_REGS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SYNTH_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DEV_FRAC = 0.15
SEED = 42




def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def softmax_positive(logprobs, temperature):
    """Softmax over the positive/neutral/negative labels based on prompt version."""
    logits = np.array(
        [logprobs.get(POSITIVE_LABEL, -20.0), logprobs.get(NEGATIVE_LABEL, -20.0), logprobs.get("neutral", -20.0)]
    ) / temperature
    e = np.exp(logits - logits.max())
    return float(e[0] / e.sum())


def get_p_positive(r, temperature):
    lp = r.get("raw_logprobs")
    if not lp:
        return 0.0
    return softmax_positive(lp, temperature)


def get_expected_level(r, temperature):
    level_lps = r.get("level_logprobs")
    if not level_lps:
        return 1.0

    logits = np.array([level_lps.get(str(i), -20.0) for i in (1, 2, 3)]) / temperature
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    return float(probs[0]*1 + probs[1]*2 + probs[2]*3)


def get_expected_score(r, temperature):
    """v4 graded 0-9 locus-of-control target: expected value over the teacher's
    digit distribution, re-softmaxed at `temperature` (so the sweep can still
    sharpen/soften). Returns None when the row has no graded score."""
    lp = r.get("loc_score_logprobs")
    if not lp:
        return None
    logits = np.array([lp.get(str(d), -20.0) for d in range(10)]) / temperature
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float((probs * np.arange(10)).sum())


# Target modes (LOC_TARGET_MODE):
#   band   (default) - explicit 3-band ordering external(0) < neutral(NEU) < internal(6-9).
#            This is what EXCLUDES external: the legacy target lumps external and neutral
#            both at 0, so Ridge never learns external is worse than neutral and external
#            leaks into the top. Internal intensity = smooth v4 loc_score if present, else
#            derived from the concreteness level. Works on existing (pre-v4) data too.
#   graded - raw v4 loc_score (0-9), neutral in the middle. Best teacher-fidelity but lets
#            the neutral bulk crowd the internal head (worse internal detection).
#   legacy - old P(internal)*E[level], rescaled to [0,9].
TARGET_MODE = os.environ.get("LOC_TARGET_MODE", "band")
NEU_BAND = float(os.environ.get("LOC_NEU_BAND", "2.0"))
# Model-selection metric (a key in evaluate()'s dict). Default to the graded
# ranking objective; set LOC_SELECT_METRIC=val_auc_internal for the old behavior.
SELECT_METRIC = os.environ.get("LOC_SELECT_METRIC", "val_ndcg_graded")
# Back-compat: LOC_USE_LOC_SCORE=0 forces the legacy target.
if os.environ.get("LOC_USE_LOC_SCORE", "1") == "0":
    TARGET_MODE = "legacy"


def get_class(r, temperature):
    """internal / external / neutral for a row: hard loc_label for interactions,
    else argmax of the teacher's raw_logprobs."""
    if r.get("is_interaction") and r.get("loc_label") in VALID_LABELS:
        return r["loc_label"]
    lp = r.get("raw_logprobs")
    if lp:
        a = np.array([lp.get(l, -20.0) for l in VALID_LABELS]) / temperature
        return VALID_LABELS[int(a.argmax())]
    return r.get("loc_label") if r.get("loc_label") in VALID_LABELS else "neutral"


def get_positive_intensity(r, temperature):
    """0-9 intensity for an internal post: smooth v4 loc_score when available,
    else mapped from the 3-level concreteness rubric (1->6, 2->7.5, 3->9)."""
    s = get_expected_score(r, temperature)
    if s is not None:
        return s
    if r.get("loc_score") is not None:
        return float(r["loc_score"])
    return 6.0 + 1.5 * (get_expected_level(r, temperature) - 1.0)


def get_target_score(r, temperature):
    if TARGET_MODE == "graded":
        s = get_expected_score(r, temperature)
        if s is not None:
            return s
        if r.get("loc_score") is not None:
            return float(r["loc_score"])
        return get_p_positive(r, temperature) * get_expected_level(r, temperature) * 3.0
    if TARGET_MODE == "legacy":
        return get_p_positive(r, temperature) * get_expected_level(r, temperature) * 3.0
    # band (default): NEGATIVE_LABEL -> 0, neutral -> NEU_BAND, POSITIVE_LABEL -> intensity (6-9)
    c = get_class(r, temperature)
    if c == POSITIVE_LABEL:
        return get_positive_intensity(r, temperature)
    if c == "neutral":
        return NEU_BAND
    return 0.0


def load_train_rows():
    """Reproduce the row filtering used when the embedding cache was built."""
    val_uris = {item["uri"] for item in load_jsonl(GEMINI_FILE)}
    rows = []
    for item in load_jsonl(LMSTUDIO_FILE):
        if item.get("uri") in val_uris:
            continue
        if not item.get("text"):
            continue
        lp = item.get("raw_logprobs")
        if not lp or POSITIVE_LABEL not in lp:
            continue
        item["is_interaction"] = False
        rows.append(item)
    return rows

def load_interaction_rows():
    val_uris = {item["uri"] for item in load_jsonl(GEMINI_FILE)}
    rows = []
    for item in load_jsonl(INTERACTIONS_FILE):
        if item.get("uri") in val_uris:
            continue
        if not item.get("text"):
            continue
        if not item.get("loc_label"):
            continue
        item["is_interaction"] = True
        rows.append(item)
    return rows


def load_val_rows():
    rows = []
    for item in load_jsonl(GEMINI_FILE):
        if not item.get("text"):
            continue
        if item.get("loc_label") in VALID_LABELS:
            rows.append(item)
    return rows


def train_ridge(X, y_target, l2_reg, sample_weight=None):
    """Linear Ridge Regression to directly fit the continuous target score."""
    model = Ridge(alpha=l2_reg, fit_intercept=True)
    model.fit(X, y_target, sample_weight=sample_weight)
    return model.coef_, model.intercept_


def rocchio_vector(X, y_target, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y_target))
    # Weighted average where weight is the target score
    pos = ((y_target * sample_weight)[:, None] * X).sum(axis=0) / ((y_target * sample_weight).sum() + 1e-9)
    # Negative examples (score near 0); scale-agnostic so it works for the
    # legacy [0,3] target and the v4 [0,9] loc_score alike.
    neg_weights = (y_target.max() - y_target) * sample_weight
    neg = (neg_weights[:, None] * X).sum(axis=0) / (neg_weights.sum() + 1e-9)
    v = pos - neg
    return v / (np.linalg.norm(v) + 1e-9)


def precision_at_k(scores, is_relevant, k):
    top = np.argsort(-scores)[:k]
    return float(is_relevant[top].mean())


def evaluate(v, X_dev, p_dev, X_val, val_internal, p_val_teacher, teacher_mask,
             val_rel=None, val_external=None):
    m = {}
    s_dev = X_dev @ v
    m["dev_spearman_vs_teacher"] = float(spearmanr(s_dev, p_dev).statistic)
    m["dev_ndcg_teacher_rel"] = float(ndcg_score(p_dev[None, :], s_dev[None, :]))

    # Calculate AUC against teacher's thresholded labels on dev set.
    # Target is on the [0,9] loc_score axis (internal >= 6); 5.5 splits
    # internal from neutral/external. (Legacy fallback is rescaled to [0,9] too.)
    dev_binary = (p_dev > 5.5).astype(int)
    if len(np.unique(dev_binary)) > 1:
        m["dev_auc_teacher"] = float(roc_auc_score(dev_binary, s_dev))
    else:
        m["dev_auc_teacher"] = 0.0

    s_val = X_val @ v
    m["val_auc_internal"] = float(roc_auc_score(val_internal, s_val))
    m["val_avg_precision"] = float(average_precision_score(val_internal, s_val))
    for k in (10, 25, 50):
        m[f"val_precision@{k}"] = precision_at_k(s_val, val_internal, k)
    if teacher_mask.sum() > 10:
        m["val_spearman_vs_teacher"] = float(
            spearmanr(s_val[teacher_mask], p_val_teacher[teacher_mask]).statistic
        )

    # Graded objective: internal high, external low. val_rel = {external:0,neutral:1,
    # internal:2}. nDCG rewards internal-at-top; ext_leak_auc is AUC(external-as-positive)
    # -> LOWER is better (external pushed down); ext_in_top25 counts leaked doomer posts.
    if val_rel is not None:
        m["val_ndcg_graded"] = float(ndcg_score(val_rel[None, :], s_val[None, :]))
        m["val_spearman_graded"] = float(spearmanr(s_val, val_rel).statistic)
    if val_external is not None and 0 < val_external.sum() < len(val_external):
        m["val_ext_leak_auc"] = float(roc_auc_score(val_external, s_val))
        top = np.argsort(-s_val)[:25]
        m["val_ext_in_top25"] = int(val_external[top].sum())
    return m


def compute_embeddings(texts, batch_size=64):
    import torch
    from transformers import AutoModel

    torch.manual_seed(SEED)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing {len(texts)} embeddings with {EMBEDDING_MODEL} on {device}...")
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(device).eval()
    out = []
    from tqdm import tqdm
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            emb = model.encode([t[:2000] for t in texts[i:i + batch_size]],
                               task="classification")
            if isinstance(emb, torch.Tensor):
                emb = emb.float().cpu().numpy()
            elif isinstance(emb, list):
                emb = np.stack([
                    e.float().cpu().numpy() if isinstance(e, torch.Tensor)
                    else np.array(e, dtype=np.float32) for e in emb
                ])
            out.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(out)


def main():
    rng = np.random.default_rng(SEED)

    rows = load_train_rows()
    if not os.path.exists(CACHE_LMSTUDIO_EMBS):
        X = compute_embeddings([r["text"] for r in rows])
        # Cache both components so we don't have to recompute
        y = np.array([[get_p_positive(r, CACHE_TEMPERATURE),
                       get_expected_level(r, CACHE_TEMPERATURE)] for r in rows])
        np.savez_compressed(CACHE_LMSTUDIO_EMBS, X=X, y=y)
        
    train_data = np.load(CACHE_LMSTUDIO_EMBS, allow_pickle=True)
    X, y_cached = train_data["X"], train_data["y"]
    X = np.nan_to_num(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert len(rows) == len(X), (
        f"Row filter mismatch: {len(rows)} jsonl rows vs {len(X)} cached embeddings"
    )

    # Load Interactions
    int_rows = load_interaction_rows()
    if int_rows:
        if os.path.exists(CACHE_INTERACTIONS_EMBS):
            int_data = np.load(CACHE_INTERACTIONS_EMBS, allow_pickle=True)
            X_int = int_data["X"]
        if not os.path.exists(CACHE_INTERACTIONS_EMBS) or len(int_rows) != len(X_int):
            print(f"Interaction embeddings cache invalid or missing. Computing {len(int_rows)} embeddings...")
            X_int = compute_embeddings([r["text"] for r in int_rows])
            np.savez_compressed(CACHE_INTERACTIONS_EMBS, X=X_int)
            
        X_int = np.nan_to_num(X_int)
        X_int /= np.linalg.norm(X_int, axis=1, keepdims=True)
        assert len(int_rows) == len(X_int), "Interaction embeddings row count mismatch"
        rows.extend(int_rows)
        X = np.vstack([X, X_int])
        print(f"Added {len(X_int)} real interactions to training set.")

    val_data = np.load(CACHE_GEMINI_EMBS, allow_pickle=True)
    X_val, y_val = val_data["X"], val_data["y"]
    X_val = np.nan_to_num(X_val)
    X_val /= np.linalg.norm(X_val, axis=1, keepdims=True)
    if isinstance(y_val[0], str):
        y_val = np.array([LABEL_MAP.get(l, 2) for l in y_val])
    val_rows = load_val_rows()
    assert len(val_rows) == len(X_val), (
        f"Val row mismatch: {len(val_rows)} jsonl rows vs {len(X_val)} cached embeddings"
    )
    if VAL_LABELS_FILE:
        by_uri = {item["uri"]: item["loc_label"] for item in load_jsonl(VAL_LABELS_FILE)}
        keep = np.array([r["uri"] in by_uri for r in val_rows])
        y_val = np.array([LABEL_MAP[by_uri[r["uri"]]] for r, k in zip(val_rows, keep) if k])
        X_val = X_val[keep]
        val_rows = [r for r, k in zip(val_rows, keep) if k]
        print(f"Using validation labels from {VAL_LABELS_FILE}: {len(val_rows)} posts")
    val_internal = (y_val == LABEL_MAP[POSITIVE_LABEL]).astype(int)
    val_external = (y_val == LABEL_MAP[NEGATIVE_LABEL]).astype(int)
    # Graded relevance for the "internal high, external low" objective.
    _rel_by_class = {LABEL_MAP[POSITIVE_LABEL]: 2, LABEL_MAP["neutral"]: 1, LABEL_MAP[NEGATIVE_LABEL]: 0}
    val_rel = np.array([_rel_by_class[int(y)] for y in y_val])
    print(f"Validation: {len(X_val)} posts, {val_internal.sum()} internal / "
          f"{val_external.sum()} external ({val_internal.mean():.1%} / {val_external.mean():.1%})")

    # Teacher soft targets for validation posts
    teacher_by_uri = {
        item["uri"]: item
        for item in load_jsonl(LMSTUDIO_FILE)
        if item.get("raw_logprobs") and POSITIVE_LABEL in item.get("raw_logprobs", {})
    }
    teacher_mask = np.array([r["uri"] in teacher_by_uri for r in val_rows])
    print(f"Validation posts with teacher soft targets: {teacher_mask.sum()}/{len(val_rows)}")

    # Fidelity targets: the teacher's actual Target Score (raw T=1),
    p_ref = np.array([get_target_score(r, REFERENCE_T) for r in rows])
    p_val_teacher = np.array([
        get_target_score(teacher_by_uri[r["uri"]], REFERENCE_T)
        if r["uri"] in teacher_by_uri else np.nan
        for r in val_rows
    ])

    # Train/dev split of the LM Studio data
    n_lmstudio = len(X) - (len(int_rows) if 'int_rows' in locals() and int_rows else 0)
    perm = rng.permutation(n_lmstudio)
    n_dev = int(n_lmstudio * DEV_FRAC)
    dev_idx, train_idx = perm[:n_dev], perm[n_dev:]
    
    if 'int_rows' in locals() and int_rows:
        int_idx = np.arange(n_lmstudio, len(X))
        train_idx = np.concatenate([train_idx, int_idx])

    all_results = {}
    best = None
    for T in TEMPERATURES:
        p_all = np.array([get_target_score(r, T) for r in rows])
        for W in SYNTH_WEIGHTS:
            # calculate sample weights
            w_all = np.ones(len(rows))
            for i, r in enumerate(rows):
                if "#synth" in r["uri"]:
                    w_all[i] = W
                    
            configs = [("rocchio", None, lambda Xt, pt, wt=w_all[train_idx]: (rocchio_vector(Xt, pt, wt), 0.0))]
            configs += [(f"ridge_l2{l2:g}", l2,
                         lambda Xt, pt, l2=l2, wt=w_all[train_idx]: train_ridge(Xt, pt, l2, wt))
                        for l2 in L2_REGS]
            for name, l2, fit in configs:
                w, _ = fit(X[train_idx], p_all[train_idx])
                v = w / (np.linalg.norm(w) + 1e-9)
                m = evaluate(v, X[dev_idx], p_ref[dev_idx], X_val, val_internal,
                             p_val_teacher, teacher_mask, val_rel, val_external)
                key = f"{name}_T{T}_W{W}"
                all_results[key] = m
                print(f"{key:26s} valAUC_int={m['val_auc_internal']:.4f} "
                      f"extLeak={m.get('val_ext_leak_auc', 0.0):.4f} "
                      f"gnDCG={m.get('val_ndcg_graded', 0.0):.4f} "
                      f"gSpear={m.get('val_spearman_graded', 0.0):.4f} "
                      f"ext@top25={m.get('val_ext_in_top25', -1)}")
                # Select on the graded objective (internal high + external low).
                if l2 is not None and (best is None or
                                       m.get(SELECT_METRIC, 0.0) > best[4].get(SELECT_METRIC, 0.0)):
                    best = (key, T, W, l2, m)

    best_key, best_T, best_W, best_l2, _ = best
    print(f"\nBest Ridge by {SELECT_METRIC}: {best_key} — refitting on all LM Studio data")
    p_best = np.array([get_target_score(r, best_T) for r in rows])
    
    w_best_all = np.ones(len(rows))
    for i, r in enumerate(rows):
        if "#synth" in r["uri"]:
            w_best_all[i] = best_W
            
    w, b = train_ridge(X, p_best, best_l2, w_best_all)
    v = w / (np.linalg.norm(w) + 1e-9)
    final = evaluate(v, X[dev_idx], p_ref[dev_idx], X_val, val_internal,
                     p_val_teacher, teacher_mask, val_rel, val_external)
    print("Final model (trained on full data; dev metrics no longer held out):")
    for k_, v_ in final.items():
        print(f"  {k_}: {v_:.4f}")

    np.save(VECTOR_OUT, v)
    stem = VECTOR_OUT[:-4] if VECTOR_OUT.endswith(".npy") else VECTOR_OUT
    with open(f"{stem}_meta.json", "w") as f:
        json.dump({"scale": float(np.linalg.norm(w)), "bias": float(b),
                   "temperature": best_T, "l2_reg": best_l2,
                   "embedding_model": EMBEDDING_MODEL,
                   "embedding_task": "classification"}, f, indent=2)
    with open(f"results_{stem}.json", "w") as f:
        json.dump({"sweep": all_results, "best": best_key,
                   "final_full_data": final}, f, indent=2)
    print(f"\nSaved {VECTOR_OUT} (1024-dim, unit norm), {stem}_meta.json "
          f"and results_{stem}.json")


if __name__ == "__main__":
    main()
