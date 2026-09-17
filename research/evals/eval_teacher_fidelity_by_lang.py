"""Does the distillation lose German? Measured on the full teacher set.

The gold-labelled set has only 9 German internal posts, which is why
eval_vector_teacher_by_lang.py cannot answer anything about German. The teacher
set is 11x bigger and carries 665 German posts (36 teacher-internal) — but no
gold labels, so it answers a different question:

    gold set     is the vector RIGHT?      (accuracy vs Gemini)
    teacher set  is the vector FAITHFUL?   (agreement with its own teacher)

Fidelity is the half of the German question that is actually answerable at this
sample size, and it is the half the band-target hypothesis lives in: if the
teacher's German signal is strong and the vector's is not, the loss is in the
distillation, not in the teacher.

Scoring the training rows with the deployed vector would be in-sample and
inflated, so fidelity is measured out-of-fold: 5-fold CV refitting the same
Ridge recipe, each row scored only by a vector that never saw it. The deployed
vector's in-sample number is printed alongside to show the size of that gap.

Run: tmp_venv/bin/python eval_teacher_fidelity_by_lang.py
"""

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import train_internal_vector as T

VECTOR_JSON = "bunny_feed/src/vector.json"
TRAIN_EMB = "embeddings_lmstudio_soft_v3_jina.npz"
INT_EMB = "embeddings_interactions_jina.npz"
VAL_EMB = "embeddings_val_loc_jina.npz"
VAL_FILE = "dataset_validation_gemini.jsonl"
TEACHER_FILE = "dataset_annotated_lmstudio_soft.jsonl"
# The deployed vector's own hyperparameters (internal_vector_meta.json).
TEMPERATURE = 5.0
L2 = 10.0
FOLDS = 5
SEED = 42
N_BOOT = 2000


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lang_of(row):
    ls = row.get("langs") or []
    return ls[0].split("-")[0] if ls else "(untagged)"


def unit(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def boot_ci(fn, rng, *arrays, n=N_BOOT):
    arrays = [np.asarray(a) for a in arrays]
    idx = np.arange(len(arrays[0]))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try:
            v = fn(*[a[b] for a in arrays])
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def group_diff(fn, rng, a, b, n=N_BOOT):
    """CI and p for stat(group a) - stat(group b), each resampled independently."""
    ia, ib = np.arange(len(a[0])), np.arange(len(b[0]))
    vals = []
    for _ in range(n):
        ba, bb = rng.choice(ia, len(ia), replace=True), rng.choice(ib, len(ib), replace=True)
        try:
            v = fn(*[x[ba] for x in a]) - fn(*[x[bb] for x in b])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if not len(vals):
        return (float("nan"), float("nan")), float("nan")
    return ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))),
            float(2 * min((vals <= 0).mean(), (vals >= 0).mean())))


def main():
    rng = np.random.default_rng(SEED)

    rows = T.load_train_rows()
    X = unit(np.load(TRAIN_EMB, allow_pickle=True)["X"])
    assert len(rows) == len(X), f"{len(rows)} rows vs {len(X)} embeddings"

    # Interaction rows ride along in every training fold as extra supervision,
    # exactly as they do in the real fit, but are never scored.
    int_rows = T.load_interaction_rows()
    X_int = unit(np.load(INT_EMB, allow_pickle=True)["X"]) if int_rows else np.zeros((0, X.shape[1]))
    y_int = np.array([T.get_target_score(r, TEMPERATURE) for r in int_rows]) if int_rows else np.zeros(0)
    print(f"{len(rows)} teacher rows + {len(int_rows)} interaction rows")

    y = np.array([T.get_target_score(r, TEMPERATURE) for r in rows])
    t_class = np.array([T.get_class(r, TEMPERATURE) for r in rows])
    langs = np.array([lang_of(r) for r in rows])

    with open(VECTOR_JSON) as f:
        vec = json.load(f)
    v_deployed = np.asarray(vec["vector"], dtype=np.float64)

    # Can this recipe reproduce the deployed vector at all? If the refit on all
    # data does not land close to vector.json, the CV below is measuring some
    # other model and the whole comparison is void.
    full = Ridge(alpha=L2, fit_intercept=True).fit(np.vstack([X, X_int]), np.concatenate([y, y_int]))
    v_full = full.coef_
    cos_check = float(v_full @ v_deployed / (np.linalg.norm(v_full) * np.linalg.norm(v_deployed)))
    print(f"refit-on-everything vs deployed vector: cosine {cos_check:.3f}")

    # Out-of-fold scores: every row scored by a vector that never saw it.
    oof = np.zeros(len(rows))
    for tr, te in KFold(FOLDS, shuffle=True, random_state=SEED).split(X):
        m = Ridge(alpha=L2, fit_intercept=True).fit(
            np.vstack([X[tr], X_int]), np.concatenate([y[tr], y_int]))
        oof[te] = X[te] @ m.coef_
    in_sample = X @ v_deployed

    # The 500 gold posts are out-of-sample for the deployed vector and carry
    # teacher labels, so they give a second, unbiased fidelity read.
    val_rows = [r for r in load_jsonl(VAL_FILE) if r.get("text")]
    teacher = {r["uri"]: r for r in load_jsonl(TEACHER_FILE)}
    vX = unit(np.load(VAL_EMB)["X"])
    v_keep = [i for i, r in enumerate(val_rows) if r["uri"] in teacher]
    v_score = (vX @ v_deployed)[v_keep]
    v_y = np.array([T.get_target_score(teacher[val_rows[i]["uri"]], TEMPERATURE) for i in v_keep])
    v_class = np.array([T.get_class(teacher[val_rows[i]["uri"]], TEMPERATURE) for i in v_keep])
    v_lang = np.array([lang_of(val_rows[i]) for i in v_keep])

    report = {"n_teacher_rows": len(rows), "folds": FOLDS, "temperature": TEMPERATURE,
              "l2": L2, "recipe_cosine_vs_deployed": cos_check, "groups": []}

    print(f"\nfidelity to the teacher, out-of-fold (row scored by a vector that never saw it)")
    print(f"{'':12}{'n':>6}{'t.int':>7}  {'AUC t-internal (95% CI)':>26}"
          f"  {'rho vs target (95% CI)':>26}  {'in-sample AUC':>14}")
    print("-" * 100)
    groups = [("all", np.ones(len(rows), bool)), ("en", langs == "en"),
              ("de", langs == "de"), ("(untagged)", langs == "(untagged)")]
    for name, m in groups:
        yi = (t_class[m] == "internal").astype(int)
        if yi.sum() < 3:
            continue
        auc = roc_auc_score(yi, oof[m])
        lo, hi = boot_ci(lambda a, b: roc_auc_score(a, b), rng, yi, oof[m])
        rho = float(spearmanr(oof[m], y[m]).statistic)
        rlo, rhi = boot_ci(lambda a, b: spearmanr(a, b).statistic, rng, oof[m], y[m])
        ins = roc_auc_score(yi, in_sample[m])
        print(f"{name:<12}{m.sum():6d}{yi.sum():7d}  {auc:8.3f} [{lo:.3f}, {hi:.3f}]"
              f"  {rho:8.3f} [{rlo:.3f}, {rhi:.3f}]  {ins:14.3f}")
        report["groups"].append({
            "group": name, "n": int(m.sum()), "n_teacher_internal": int(yi.sum()),
            "oof_auc": auc, "oof_auc_ci": [lo, hi], "oof_spearman": rho,
            "oof_spearman_ci": [rlo, rhi], "in_sample_auc": float(ins)})

    print(f"\nsame thing on the 500 gold posts, which the deployed vector never trained on")
    print(f"{'':12}{'n':>6}{'t.int':>7}  {'AUC t-internal':>15}  {'rho vs target':>14}")
    print("-" * 60)
    for name, m in [("all", np.ones(len(v_keep), bool)), ("en", v_lang == "en"),
                    ("de", v_lang == "de"), ("(untagged)", v_lang == "(untagged)")]:
        yi = (v_class[m] == "internal").astype(int)
        if yi.sum() < 3:
            print(f"{name:<12}{m.sum():6d}{yi.sum():7d}  {'too few':>15}")
            continue
        print(f"{name:<12}{m.sum():6d}{yi.sum():7d}  {roc_auc_score(yi, v_score[m]):15.3f}"
              f"  {spearmanr(v_score[m], v_y[m]).statistic:14.3f}")

    de, en = langs == "de", langs == "en"
    yde = (t_class[de] == "internal").astype(int)
    yen = (t_class[en] == "internal").astype(int)
    (alo, ahi), ap = group_diff(lambda a, b: roc_auc_score(a, b), rng,
                                [yde, oof[de]], [yen, oof[en]])
    (slo, shi), sp = group_diff(lambda a, b: spearmanr(a, b).statistic, rng,
                                [oof[de], y[de]], [oof[en], y[en]])
    print(f"\nGerman vs English fidelity, tested directly "
          f"(de n={de.sum()} / {yde.sum()} internal, en n={en.sum()} / {yen.sum()} internal)")
    print(f"  AUC        de-en {roc_auc_score(yde, oof[de]) - roc_auc_score(yen, oof[en]):+.3f}"
          f"  CI [{alo:+.3f}, {ahi:+.3f}]  p={ap:.3f}")
    print(f"  Spearman   de-en "
          f"{spearmanr(oof[de], y[de]).statistic - spearmanr(oof[en], y[en]).statistic:+.3f}"
          f"  CI [{slo:+.3f}, {shi:+.3f}]  p={sp:.3f}")
    report["de_vs_en"] = {"auc_diff_ci": [alo, ahi], "auc_p": ap,
                          "spearman_diff_ci": [slo, shi], "spearman_p": sp,
                          "n_de": int(de.sum()), "n_de_internal": int(yde.sum()),
                          "n_en": int(en.sum()), "n_en_internal": int(yen.sum())}

    with open("REPORT_teacher_fidelity_by_lang.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote REPORT_teacher_fidelity_by_lang.json")


if __name__ == "__main__":
    main()
