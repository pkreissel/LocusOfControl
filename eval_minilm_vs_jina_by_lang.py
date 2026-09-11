"""MiniLM (Green Earth's free hydrated space) vs Jina, split by language.

REPORT_ge_embeddings.md established that the two spaces are equally good overall
and better fused. That was measured on a corpus that is ~68% English. The open
question it never asked: all-MiniLM-L12-v2 is an *English* model, while
jina-embeddings-v5-small is multilingual — so an aggregate tie could hide a
German deficit that is simply outvoted by the English majority.

Three measurements, weakest-powered first:

  A  gold      the 500 Gemini-labelled held-out posts, scored by the two
                deployed-style vectors (identical harness, identical splits,
                identical sweep — only the embedding cache differs).
  B  out-of-fold  5-fold OOF Ridge over the full 2822-row teacher corpus, both
                spaces, same hyperparameters, graded band target. ~5x the German
                rows of A and no selection leakage.
  C  geometry  what the embedding spaces do to German text irrespective of any
                probe: anisotropy, and how far the two rankings diverge.

Language is the post's declared `langs[0]`; untagged posts (26% of the val set,
31% of the corpus) are additionally resolved with langdetect and reported as a
second grouping, because dropping them silently re-weights the comparison.

Run: HF_HOME=/Volumes/Daten/huggingface \
     LOC_TEACHER_FILE=dataset_annotated_bonsai_v2.jsonl \
     tmp_venv/bin/python eval_minilm_vs_jina_by_lang.py
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("LOC_TEACHER_FILE", "dataset_annotated_bonsai_v2.jsonl")
import train_internal_vector as T  # noqa: E402

VAL_FILE = "dataset_validation_gemini.jsonl"
# looksGerman() over each corpus, written by bunny_feed/dump_lang_flags.ts.
FLAGS_VAL = "lang_flags_gemini.json"
FLAGS_TRAIN = "lang_flags_bonsai_v2.json"
FLAGS_INT = "lang_flags_interactions.json"
JINA = {
    "name": "jina",
    "vector": "internal_vector_bonsai_v2.npy",
    "meta": "internal_vector_bonsai_v2_meta.json",
    "val_emb": "embeddings_val_loc_jina.npz",
    "train_emb": "embeddings_bonsai_v2_jina.npz",
    "int_emb": "embeddings_interactions_jina.npz",
}
MINILM = {
    "name": "minilm",
    "vector": "internal_vector_bonsai_v2_minilm.npy",
    "meta": "internal_vector_bonsai_v2_minilm_meta.json",
    "val_emb": "embeddings_gemini_minilm_v2.npz",
    "train_emb": "embeddings_bonsai_v2_minilm.npz",
    "int_emb": "embeddings_interactions_minilm.npz",
}
SPACES = [JINA, MINILM]
# Both metas selected these, so the OOF comparison can hold them fixed and still
# be the sweep's own choice rather than an arbitrary one.
TEMPERATURE = 5.0
L2 = 10.0
FOLDS = 5
SEED = 20260911
N_BOOT = 4000


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lang_of(row):
    ls = row.get("langs") or []
    return ls[0].split("-")[0] if ls else "(untagged)"


def detected_langs(rows):
    """Declared tag where present, langdetect otherwise."""
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    out = []
    for r in rows:
        d = lang_of(r)
        if d == "(untagged)":
            try:
                d = detect_langs(r["text"])[0].lang
            except Exception:
                d = "?"
        out.append(d)
    return np.array(out)


def detector_flags(rows, *files):
    """looksGerman() for each row, as computed by the SHIPPED detector.

    Produced by bunny_feed/dump_lang_flags.ts rather than re-implemented here: the
    routing groups these numbers describe are formed by that code, and a Python
    copy of it would be a second detector free to drift from the one that ships.
    """
    flags = {}
    for f in files:
        with open(f) as fh:
            flags.update(json.load(fh))
    missing = [r["uri"] for r in rows if r["uri"] not in flags]
    assert not missing, f"{len(missing)} rows have no flag; re-run dump_lang_flags.ts"
    return np.array([flags[r["uri"]] for r in rows], dtype=bool)


def unit(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def boot_auc(y, s, rng, n=N_BOOT):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    idx, vals = np.arange(len(y)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            vals.append(roc_auc_score(y[b], s[b]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_paired_auc_diff(y, s_a, s_b, rng, n=N_BOOT):
    """AUC(a) - AUC(b) on the same posts: CI and two-sided bootstrap p."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    if len(np.unique(y)) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    obs = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    idx, vals = np.arange(len(y)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            vals.append(roc_auc_score(y[b], s_a[b]) - roc_auc_score(y[b], s_b[b]))
    vals = np.array(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(obs), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), float(p)


def boot_interaction(y_de, a_de, b_de, y_en, a_en, b_en, rng, n=N_BOOT):
    """(jina-minilm on German) - (jina-minilm on English): does the gap differ by language?

    This is the question the marginals cannot answer. Each language is resampled
    independently; a CI that straddles 0 means the aggregate result transfers.
    """
    def gap(y, a, b):
        return roc_auc_score(y, a) - roc_auc_score(y, b)
    obs = gap(y_de, a_de, b_de) - gap(y_en, a_en, b_en)
    ide, ien, vals = np.arange(len(y_de)), np.arange(len(y_en)), []
    for _ in range(n):
        bd = rng.choice(ide, len(ide), replace=True)
        be = rng.choice(ien, len(ien), replace=True)
        if len(np.unique(y_de[bd])) < 2 or len(np.unique(y_en[be])) < 2:
            continue
        vals.append(gap(y_de[bd], a_de[bd], b_de[bd]) - gap(y_en[be], a_en[be], b_en[be]))
    vals = np.array(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(obs), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), float(p)


def rank_norm(s):
    from scipy.stats import rankdata
    return rankdata(s) / len(s)


def groups_for(langs, detected, flagged=None):
    """Three ways to split the same rows.

    `de`/`en` are the declared tags, `+det` adds langdetect over the untagged
    quarter — and `routed`/`kept` is the only split that describes what the feed
    does, because looksGerman() is what decides it. The first two are the honest
    measurement of the language effect; the third is the measurement of the
    policy, false positives and missed posts included.
    """
    groups = [("all", np.ones(len(langs), bool)),
              ("en", langs == "en"), ("de", langs == "de"),
              ("(untagged)", langs == "(untagged)"),
              ("en+det", detected == "en"), ("de+det", detected == "de")]
    if flagged is not None:
        groups += [("routed(de)", flagged), ("kept(rest)", ~flagged)]
    return groups


# --------------------------------------------------------------------------- A
def part_a(report, rng):
    val = load_jsonl(VAL_FILE)
    rows = [r for r in val if r.get("text") and r.get("loc_label") in T.VALID_LABELS]
    langs = np.array([lang_of(r) for r in rows])
    detected = detected_langs(rows)
    flagged = detector_flags(rows, FLAGS_VAL)
    gold = np.array([r["loc_label"] for r in rows])
    print(f"\n=== A. Gold standard: {len(rows)} Gemini-labelled held-out posts "
          f"({flagged.sum()} flagged German by the shipped detector) ===")

    probs = {}
    for sp in SPACES:
        d = np.load(sp["val_emb"], allow_pickle=True)
        X = d["X"]
        assert len(X) == len(rows), f"{sp['val_emb']}: {len(X)} rows, need {len(rows)}"
        if "y" in d.files and isinstance(d["y"][0], str):
            # The MiniLM cache stores its own labels; a row-order slip would
            # otherwise silently compare two different orderings of the set.
            assert (d["y"] == gold).all(), f"{sp['val_emb']} row order does not match {VAL_FILE}"
        v = np.load(sp["vector"]).astype(np.float64)
        meta = json.load(open(sp["meta"]))
        cos = unit(X) @ (v / np.linalg.norm(v))
        probs[sp["name"]] = 1.0 / (1.0 + np.exp(-(meta["scale"] * cos + meta["bias"])))
    probs["fusion"] = (rank_norm(probs["jina"]) + rank_norm(probs["minilm"])) / 2

    out = []
    print(f"{'':12}{'n':>5}{'int':>5}  {'jina AUC':>22}  {'minilm AUC':>22}  "
          f"{'jina-minilm':>22}  {'fusion':>7}")
    print("-" * 104)
    for name, m in groups_for(langs, detected, flagged):
        y = (gold[m] == "internal").astype(int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        ja, ma = roc_auc_score(y, probs["jina"][m]), roc_auc_score(y, probs["minilm"][m])
        jlo, jhi = boot_auc(y, probs["jina"][m], rng)
        mlo, mhi = boot_auc(y, probs["minilm"][m], rng)
        d, (dlo, dhi), p = boot_paired_auc_diff(y, probs["jina"][m], probs["minilm"][m], rng)
        fa = roc_auc_score(y, probs["fusion"][m])
        print(f"{name:<12}{m.sum():5d}{y.sum():5d}  {ja:6.3f} [{jlo:.3f}, {jhi:.3f}]  "
              f"{ma:6.3f} [{mlo:.3f}, {mhi:.3f}]  {d:+6.3f} [{dlo:+.3f}, {dhi:+.3f}]  {fa:7.3f}"
              f"  p={p:.2f}")
        out.append({"group": name, "n": int(m.sum()), "n_internal": int(y.sum()),
                    "jina_auc": ja, "jina_auc_ci": [jlo, jhi],
                    "minilm_auc": ma, "minilm_auc_ci": [mlo, mhi],
                    "jina_minus_minilm": d, "diff_ci": [dlo, dhi], "diff_p": p,
                    "fusion_auc": fa})
    report["A_gold"] = {"n": len(rows), "groups": out}

    # internal vs external only: the contrast the feed acts on
    print(f"\n  internal vs external only")
    print(f"  {'':10}{'n':>5}  {'jina':>7}  {'minilm':>7}  {'fusion':>7}")
    for name, m in groups_for(langs, detected, flagged):
        mm = m & np.isin(gold, ["internal", "external"])
        y = (gold[mm] == "internal").astype(int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        vals = {k: roc_auc_score(y, probs[k][mm]) for k in ("jina", "minilm", "fusion")}
        print(f"  {name:<10}{mm.sum():5d}  {vals['jina']:7.3f}  {vals['minilm']:7.3f}  {vals['fusion']:7.3f}")
        for g in out:
            if g["group"] == name:
                g["ie_n"] = int(mm.sum())
                g.update({f"ie_{k}_auc": v for k, v in vals.items()})

    de, en = langs == "de", langs == "en"
    yde = (gold[de] == "internal").astype(int)
    yen = (gold[en] == "internal").astype(int)
    obs, (lo, hi), p = boot_interaction(yde, probs["jina"][de], probs["minilm"][de],
                                        yen, probs["jina"][en], probs["minilm"][en], rng)
    print(f"\n  interaction (jina-minilm on de) - (on en): {obs:+.3f} "
          f"CI [{lo:+.3f}, {hi:+.3f}] p={p:.2f}")
    report["A_gold"]["interaction"] = {"obs": obs, "ci": [lo, hi], "p": p}
    return probs, langs, detected, gold, flagged


# --------------------------------------------------------------------------- B
def part_b(report, rng):
    rows = T.load_train_rows()
    int_rows = T.load_interaction_rows()
    print(f"\n=== B. Out-of-fold CV: {len(rows)} teacher + {len(int_rows)} interaction rows ===")

    Xs = {}
    for sp in SPACES:
        X = np.load(sp["train_emb"], allow_pickle=True)["X"]
        Xi = np.load(sp["int_emb"], allow_pickle=True)["X"]
        assert len(X) == len(rows) and len(Xi) == len(int_rows), f"{sp['name']} cache mismatch"
        Xs[sp["name"]] = unit(np.vstack([X, Xi]))

    all_rows = rows + int_rows
    langs = np.array([lang_of(r) for r in all_rows])
    detected = detected_langs(all_rows)
    flagged = detector_flags(all_rows, FLAGS_TRAIN, FLAGS_INT)
    target = np.array([T.get_target_score(r, TEMPERATURE) for r in all_rows])
    cls = np.array([T.get_class(r, TEMPERATURE) for r in all_rows])
    y = (target >= T.POSITIVE_BAND[0]).astype(int)
    print(f"  {y.sum()} band-positives; language mix: "
          f"en {(langs == 'en').sum()}, de {(langs == 'de').sum()}, "
          f"untagged {(langs == '(untagged)').sum()} "
          f"(detected de {(detected == 'de').sum()}); "
          f"{flagged.sum()} flagged German by the shipped detector")

    # One fold assignment, shared by both spaces: the comparison is paired.
    fold = np.arange(len(all_rows)) % FOLDS
    fold = np.random.default_rng(SEED).permutation(fold)
    oof = {}
    for name, X in Xs.items():
        s = np.zeros(len(all_rows))
        for k in range(FOLDS):
            tr, te = fold != k, fold == k
            model = Ridge(alpha=L2, fit_intercept=True).fit(X[tr], target[tr])
            v = model.coef_ / np.linalg.norm(model.coef_)
            s[te] = X[te] @ v
        oof[name] = s
    oof["fusion"] = (rank_norm(oof["jina"]) + rank_norm(oof["minilm"])) / 2

    out = []
    print(f"\n{'':12}{'n':>5}{'pos':>5}  {'jina AUC':>22}  {'minilm AUC':>22}  "
          f"{'jina-minilm':>22}  {'fusion':>7}  {'jina AP':>8}{'mlm AP':>8}")
    print("-" * 124)
    for name, m in groups_for(langs, detected, flagged):
        yy = y[m]
        if yy.sum() < 3 or (1 - yy).sum() < 3:
            continue
        ja, ma = roc_auc_score(yy, oof["jina"][m]), roc_auc_score(yy, oof["minilm"][m])
        jlo, jhi = boot_auc(yy, oof["jina"][m], rng)
        mlo, mhi = boot_auc(yy, oof["minilm"][m], rng)
        d, (dlo, dhi), p = boot_paired_auc_diff(yy, oof["jina"][m], oof["minilm"][m], rng)
        fa = roc_auc_score(yy, oof["fusion"][m])
        jap = average_precision_score(yy, oof["jina"][m])
        map_ = average_precision_score(yy, oof["minilm"][m])
        print(f"{name:<12}{m.sum():5d}{yy.sum():5d}  {ja:6.3f} [{jlo:.3f}, {jhi:.3f}]  "
              f"{ma:6.3f} [{mlo:.3f}, {mhi:.3f}]  {d:+6.3f} [{dlo:+.3f}, {dhi:+.3f}]  "
              f"{fa:7.3f}  {jap:8.3f}{map_:8.3f}  p={p:.2f}")
        # Fidelity to the graded target, and how much external leaks up the ranking.
        rho_j = spearmanr(oof["jina"][m], target[m]).statistic
        rho_m = spearmanr(oof["minilm"][m], target[m]).statistic
        ext = cls[m] == "external"
        leak = {}
        if ext.sum() >= 3 and (~ext).sum() >= 3:
            leak = {"jina_ext_leak_auc": roc_auc_score(ext.astype(int), oof["jina"][m]),
                    "minilm_ext_leak_auc": roc_auc_score(ext.astype(int), oof["minilm"][m])}
        out.append({"group": name, "n": int(m.sum()), "n_positive": int(yy.sum()),
                    "jina_auc": ja, "jina_auc_ci": [jlo, jhi],
                    "minilm_auc": ma, "minilm_auc_ci": [mlo, mhi],
                    "jina_minus_minilm": d, "diff_ci": [dlo, dhi], "diff_p": p,
                    "fusion_auc": fa, "jina_ap": jap, "minilm_ap": map_,
                    "jina_rho_target": float(rho_j), "minilm_rho_target": float(rho_m),
                    **leak})
    report["B_oof"] = {"n": len(all_rows), "n_positive": int(y.sum()), "folds": FOLDS,
                       "temperature": TEMPERATURE, "l2": L2, "groups": out}

    print(f"\n  {'':10}{'rho vs graded target':>24}{'external-leak AUC (lower better)':>36}")
    for g in out:
        leak = (f"  jina {g['jina_ext_leak_auc']:.3f}  minilm {g['minilm_ext_leak_auc']:.3f}"
                if "jina_ext_leak_auc" in g else "  (too few external)")
        print(f"  {g['group']:<10}  jina {g['jina_rho_target']:.3f}  "
              f"minilm {g['minilm_rho_target']:.3f}   {leak}")

    de, en = langs == "de", langs == "en"
    obs, (lo, hi), p = boot_interaction(y[de], oof["jina"][de], oof["minilm"][de],
                                        y[en], oof["jina"][en], oof["minilm"][en], rng)
    print(f"\n  interaction (jina-minilm on de) - (on en): {obs:+.3f} "
          f"CI [{lo:+.3f}, {hi:+.3f}] p={p:.2f}")
    report["B_oof"]["interaction"] = {"obs": obs, "ci": [lo, hi], "p": p}

    # German-only probe: does a vector trained on a 68%-English corpus transfer,
    # or is the German signal simply not in the space? Train on English rows,
    # test on German, and vice versa.
    print("\n  transfer (train on one language, test on the other)")
    for name, X in Xs.items():
        m_en, m_de = langs == "en", langs == "de"
        rows_ = []
        for tr_m, te_m, tag in ((m_en, m_de, "en->de"), (m_de, m_en, "de->en"),
                                (m_de, m_de, "de->de (oof)")):
            if tag.startswith("de->de"):
                s = oof[name][m_de]  # already out-of-fold
                a = roc_auc_score(y[m_de], s)
            else:
                model = Ridge(alpha=L2, fit_intercept=True).fit(X[tr_m], target[tr_m])
                v = model.coef_ / np.linalg.norm(model.coef_)
                a = roc_auc_score(y[te_m], X[te_m] @ v)
            rows_.append((tag, a))
        print(f"    {name:<8}" + "  ".join(f"{t} {a:.3f}" for t, a in rows_))
        report["B_oof"].setdefault("transfer", {})[name] = {t: a for t, a in rows_}
    return oof, langs, detected, target, all_rows, flagged


# --------------------------------------------------------------------------- C
def part_c(report, langs_a, probs_a, oof_b, langs_b):
    print("\n=== C. Geometry: what the spaces do to German text ===")
    rows = [r for r in load_jsonl(VAL_FILE) if r.get("text") and r.get("loc_label") in T.VALID_LABELS]
    stats = {}
    for sp in SPACES:
        X = unit(np.load(sp["val_emb"], allow_pickle=True)["X"])
        for lang in ("en", "de"):
            m = langs_a == lang
            S = X[m] @ X[m].T
            iu = np.triu_indices(m.sum(), 1)
            stats[(sp["name"], lang)] = (float(S[iu].mean()), float(S[iu].std()))
    print(f"  mean pairwise cosine within a language (anisotropy: higher = more collapsed)")
    print(f"  {'':10}{'en':>16}{'de':>16}{'de - en':>12}")
    for sp in SPACES:
        e, d = stats[(sp["name"], "en")], stats[(sp["name"], "de")]
        print(f"  {sp['name']:<10}{e[0]:8.3f}±{e[1]:.3f}{d[0]:8.3f}±{d[1]:.3f}{d[0]-e[0]:+12.3f}")
    report["C_geometry"] = {"pairwise_cosine": {f"{k[0]}_{k[1]}": v for k, v in stats.items()}}

    print("\n  Spearman between the two rankings (how much fusion has to work with)")
    for tag, langs, scores in (("gold set", langs_a, probs_a), ("corpus", langs_b, oof_b)):
        line = []
        for lang in ("all", "en", "de"):
            m = np.ones(len(langs), bool) if lang == "all" else langs == lang
            line.append(f"{lang} {spearmanr(scores['jina'][m], scores['minilm'][m]).statistic:.3f}")
        print(f"    {tag:<10}" + "  ".join(line))
        report["C_geometry"].setdefault("cross_space_spearman", {})[tag] = {
            lang: float(spearmanr(scores["jina"][np.ones(len(langs), bool) if lang == "all" else langs == lang],
                                  scores["minilm"][np.ones(len(langs), bool) if lang == "all" else langs == lang]).statistic)
            for lang in ("all", "en", "de")}



# --------------------------------------------------------------------------- D
def part_d(report, oof, langs, detected, target, rows, flagged):
    """The shipping question: a global MiniLM prefilter, judged per language.

    REPORT_ge_embeddings.md's architecture scores 100% of candidates with MiniLM
    and sends only the top ~20% to Jina. A global cut is applied to a mixed-
    language pool, so two things can go wrong for German independently: German
    posts can be kept at the wrong *rate*, and the German posts that are kept can
    be the wrong *ones*.
    """
    print("\n=== D. A global MiniLM prefilter, per language ===")
    y = (target >= T.POSITIVE_BAND[0]).astype(int)
    groups = [("en", langs == "en"), ("de", langs == "de"), ("de+det", detected == "de"),
              ("routed(de)", flagged)]
    report["D_prefilter"] = {}
    for keep_frac in (0.10, 0.20, 0.30):
        print(f"\n  keep top {keep_frac:.0%} of the pool by score")
        print(f"  {'':10}{'space':>8}{'share of lang kept':>20}{'lang band-pos recall':>22}")
        for space in ("minilm", "jina", "fusion"):
            thr = np.quantile(oof[space], 1 - keep_frac)
            keep = oof[space] >= thr
            for name, m in groups:
                share = float(keep[m].mean())
                rec = float(keep[m & (y == 1)].mean()) if (m & (y == 1)).sum() else float("nan")
                print(f"  {name:<10}{space:>8}{share:19.1%}{rec:21.1%}")
                report["D_prefilter"].setdefault(f"keep_{int(keep_frac*100)}", {}) \
                    .setdefault(space, {})[name] = {"share_kept": share, "positive_recall": rec}

    # The shipped policy, end to end, against the one it replaces.
    #
    # Budgets come from the live config as fractions of the pool it was sized
    # against (shortlist 150 and bypassLimit 50 of ~706 candidates past minWords).
    # `paid` is shown on every row because the arms do NOT cost the same: the
    # bypass is spent on top of the shortlist, which is what happens in production
    # too (150 + 10 = 160 on the live pool). This corpus is 22.6% flagged against
    # the live pool's 1.4%, so it is the stress case — the one where the cap binds.
    POOL, SHORTLIST, BYPASS_CAP = 706, 150, 50
    n = len(target)
    shortlist_n = round(n * SHORTLIST / POOL)
    bypass_n = round(n * BYPASS_CAP / POOL)
    print(f"\n  the routing rule vs the global cut "
          f"(pool {n}, {flagged.sum()} flagged, shortlist {shortlist_n}, cap {bypass_n})")

    def recall(sel):
        out = {"paid": int(sel.sum()), "all": float((y & sel).sum() / y.sum())}
        for tag, m in (("de", langs == "de"), ("routed", flagged), ("en", langs == "en")):
            pos = m & (y == 1)
            out[tag] = float(sel[pos].mean()) if pos.sum() else float("nan")
        return out

    order = np.argsort(-oof["minilm"])
    idx_flagged = np.flatnonzero(flagged)
    # Within the flagged group, MiniLM's order is still worth something even
    # though its scores cannot be compared against English ones — the routed group
    # scores AUC 0.618, well clear of the 0.500 an arbitrary order gets. So how the
    # cap picks its 200 is a real decision, and these two rows are it.
    by_mini = idx_flagged[np.argsort(-oof["minilm"][idx_flagged])]

    def routed_sel(shortlist_k, bypass_order, bypass_k):
        sel = np.zeros(n, bool)
        sel[order[~flagged[order]][:shortlist_k]] = True
        sel[bypass_order[:bypass_k]] = True
        return sel

    global_cut = np.zeros(n, bool)
    global_cut[order[:shortlist_n]] = True
    jina_only = np.zeros(n, bool)
    jina_only[np.argsort(-oof["jina"])[:shortlist_n]] = True

    arms = [
        ("global MiniLM cut", global_cut),
        ("routed, pool order", routed_sel(shortlist_n, idx_flagged, bypass_n)),
        ("routed, MiniLM order", routed_sel(shortlist_n, by_mini, bypass_n)),
        ("routed, budget-matched", routed_sel(shortlist_n - bypass_n, by_mini, bypass_n)),
        ("Jina on everything", jina_only),
    ]
    print(f"  {'':24}{'paid':>6}{'band-pos recall':>17}{'de':>8}{'routed':>9}{'en':>8}")
    for tag, sel in arms:
        r = recall(sel)
        print(f"  {tag:<24}{r['paid']:6d}{r['all']:17.1%}{r['de']:8.1%}{r['routed']:9.1%}{r['en']:8.1%}")
        report["D_prefilter"].setdefault("policy", {})[tag] = r

    # Length confound: German posts are not a random sample of the corpus, and
    # both spaces truncate (MiniLM at 128 tokens, Jina at 2000 chars). If the
    # German rows are simply longer, that — not the language — could be the gap.
    L = np.array([len(r["text"]) for r in rows])
    print(f"\n  median chars: en {np.median(L[langs == 'en']):.0f}, "
          f"de {np.median(L[langs == 'de']):.0f}; "
          f"p90 en {np.percentile(L[langs == 'en'], 90):.0f}, "
          f"de {np.percentile(L[langs == 'de'], 90):.0f}")
    band = (L >= np.percentile(L[langs == "de"], 10)) & (L <= np.percentile(L[langs == "de"], 90))
    for name, m in (("en", langs == "en"), ("de", langs == "de")):
        mm = m & band
        if y[mm].sum() < 3:
            continue
        print(f"  length-matched {name:<4} n={mm.sum():5d} pos={y[mm].sum():3d}  "
              f"jina {roc_auc_score(y[mm], oof['jina'][mm]):.3f}  "
              f"minilm {roc_auc_score(y[mm], oof['minilm'][mm]):.3f}")
        report["D_prefilter"].setdefault("length_matched", {})[name] = {
            "n": int(mm.sum()), "n_positive": int(y[mm].sum()),
            "jina_auc": roc_auc_score(y[mm], oof["jina"][mm]),
            "minilm_auc": roc_auc_score(y[mm], oof["minilm"][mm])}


def main():
    rng = np.random.default_rng(SEED)
    report = {"seed": SEED, "n_boot": N_BOOT}
    probs_a, langs_a, det_a, gold, flag_a = part_a(report, rng)
    oof_b, langs_b, det_b, target, rows_b, flag_b = part_b(report, rng)
    part_c(report, langs_a, probs_a, oof_b, langs_b)
    part_d(report, oof_b, langs_b, det_b, target, rows_b, flag_b)
    with open("REPORT_minilm_vs_jina_by_lang.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote REPORT_minilm_vs_jina_by_lang.json")


if __name__ == "__main__":
    main()
