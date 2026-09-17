"""German vs English: how good is the distilled vector, and how good is its teacher?

The vector is a 1024-dim query vector distilled from LM Studio teacher labels;
the teacher is the 27B model that produced them. Both are graded here against
the same gold standard — the 500 Gemini-labelled held-out posts — split by the
language tag on the post, so the question "does distillation lose more on German
than on English" gets a direct answer.

Three scores per post:
  gold     Gemini's internal / external / neutral label
  teacher  P(internal) from the LM Studio logprobs, plus its own argmax label
  vector   the deployed vector's probability, reproduced exactly as the edge
           worker computes it (L2-normalize, dot, sigmoid(scale*cos + bias))

The val embedding cache in the repo was built for a different axis (176 rows),
so the 500 texts are re-embedded here with the production embedding path and
cached separately.

Run: HF_HOME=/Volumes/Daten/huggingface tmp_venv/bin/python eval_vector_teacher_by_lang.py
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import train_internal_vector as T

VAL_FILE = "dataset_validation_gemini.jsonl"
TEACHER_FILE = "dataset_annotated_lmstudio_soft.jsonl"
VECTOR_JSON = "bunny_feed/src/vector.json"
EMB_CACHE = "embeddings_val_loc_jina.npz"
THRESHOLD = 0.88
SEED = 20260910
N_BOOT = 4000


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def lang_of(row):
    ls = row.get("langs") or []
    if not ls:
        return "(untagged)"
    return ls[0].split("-")[0]


def detect_lang(text):
    """Language of a post that carries no `langs` tag.

    A quarter of the validation set is untagged, which is enough to distort the
    per-language split if it is simply dropped. langdetect agrees with the
    declared tag on 87% of the de/en posts that do have one, so treat detected
    labels as noisier than declared ones and report both groupings.
    """
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    try:
        return detect_langs(text)[0].lang
    except Exception:
        return "?"


def boot_auc(y, s, rng, n=N_BOOT):
    """Percentile bootstrap CI for AUC, resampling posts."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_stat(fn, rng, *arrays, n=N_BOOT):
    """Percentile CI for any statistic computed over paired per-post arrays."""
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


def boot_group_diff(fn, rng, a_arrays, b_arrays, n=N_BOOT):
    """CI and p for stat(group A) - stat(group B), resampling each group."""
    a_arrays = [np.asarray(x) for x in a_arrays]
    b_arrays = [np.asarray(x) for x in b_arrays]
    ia, ib = np.arange(len(a_arrays[0])), np.arange(len(b_arrays[0]))
    vals = []
    for _ in range(n):
        ba = rng.choice(ia, len(ia), replace=True)
        bb = rng.choice(ib, len(ib), replace=True)
        try:
            v = fn(*[x[ba] for x in a_arrays]) - fn(*[x[bb] for x in b_arrays])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if not len(vals):
        return (float("nan"), float("nan")), float("nan")
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), float(p)


def boot_diff(y, s1, s2, rng, n=N_BOOT):
    """CI and sign test for AUC(s1) - AUC(s2) on the same posts (paired)."""
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    if len(np.unique(y)) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    obs = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    vals, idx = [], np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s1[b]) - roc_auc_score(y[b], s2[b]))
    vals = np.array(vals)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(obs), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), float(p)


def main():
    rng = np.random.default_rng(SEED)
    val = load_jsonl(VAL_FILE)
    teacher = {r["uri"]: r for r in load_jsonl(TEACHER_FILE)}
    rows = [r for r in val if r.get("text") and r["uri"] in teacher
            and r.get("loc_label") in ("internal", "external", "neutral")]
    print(f"{len(rows)} validation posts with both a gold label and a teacher label")

    texts = [r["text"] for r in rows]
    if os.path.exists(EMB_CACHE):
        X = np.load(EMB_CACHE)["X"]
        assert len(X) == len(rows), f"cache has {len(X)} rows, need {len(rows)}"
    else:
        X = T.compute_embeddings(texts)
        np.savez_compressed(EMB_CACHE, X=X)
    X = np.nan_to_num(X).astype(np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    with open(VECTOR_JSON) as f:
        vec = json.load(f)
    cos = X @ np.asarray(vec["vector"], dtype=np.float64)
    v_prob = 1.0 / (1.0 + np.exp(-(vec["scale"] * cos + vec["bias"])))

    # Teacher: P(internal) at T=1 as its ranking score, argmax as its hard label.
    t_prob = np.array([T.softmax_positive(teacher[r["uri"]]["raw_logprobs"], 1.0) for r in rows])
    t_label = np.array([T.get_class(teacher[r["uri"]], 1.0) for r in rows])
    # The band target the vector was actually trained to reproduce.
    t_band = np.array([T.get_target_score(teacher[r["uri"]], 2.0) for r in rows])

    gold = np.array([r["loc_label"] for r in rows])
    langs = np.array([lang_of(r) for r in rows])

    detected = np.array([detect_lang(r["text"]) if lang_of(r) == "(untagged)"
                         else lang_of(r) for r in rows])
    import collections as _c
    print("untagged posts resolve to:",
          _c.Counter(detected[langs == "(untagged)"]).most_common(5))

    groups = [("all", np.ones(len(rows), bool)),
              ("en", langs == "en"), ("de", langs == "de"),
              ("(untagged)", langs == "(untagged)"),
              ("en+detected", detected == "en"), ("de+detected", detected == "de")]

    report = {"n": len(rows), "threshold": THRESHOLD, "seed": SEED, "groups": []}

    print(f"\n{'':12}{'n':>5}{'int':>5}  "
          f"{'vector AUC (95% CI)':>26}  {'teacher P(int) (95% CI)':>26}  {'gap':>15}"
          f"  {'teacher band target':>22}")
    print("-" * 126)
    for name, m in groups:
        y = (gold[m] == "internal").astype(int)
        if y.sum() == 0:
            continue
        va, ta = roc_auc_score(y, v_prob[m]), roc_auc_score(y, t_prob[m])
        vlo, vhi = boot_auc(y, v_prob[m], rng)
        tlo, thi = boot_auc(y, t_prob[m], rng)
        d, (dlo, dhi), dp = boot_diff(y, v_prob[m], t_prob[m], rng)
        # Two readouts of the same teacher: the bare label probability, and the
        # band target that was actually handed to Ridge as the training signal.
        # The vector beats the first and not the second, so which one you call
        # "the teacher" decides whether distillation looks like a gain.
        ba = roc_auc_score(y, t_band[m])
        blo, bhi = boot_auc(y, t_band[m], rng)
        print(f"{name:<12}{m.sum():5d}{y.sum():5d}  "
              f"{va:8.3f} [{vlo:.3f}, {vhi:.3f}]  {ta:8.3f} [{tlo:.3f}, {thi:.3f}]  "
              f"{d:+6.3f} p={dp:.2f}  {ba:6.3f} [{blo:.3f}, {bhi:.3f}]")
        report["groups"].append({
            "group": name, "n": int(m.sum()), "n_internal": int(y.sum()),
            "vector_auc": va, "vector_auc_ci": [vlo, vhi],
            "teacher_auc": ta, "teacher_auc_ci": [tlo, thi],
            "vector_minus_teacher": d, "diff_ci": [dlo, dhi], "diff_p": dp,
            "teacher_band_auc": ba, "teacher_band_auc_ci": [blo, bhi],
        })

    # internal vs external only — the contrast the feed actually acts on
    print(f"\ninternal vs external only (neutrals dropped)")
    print(f"{'':12}{'n':>5}  {'vector':>8}  {'teacher':>8}  {'band target':>12}")
    print("-" * 52)
    for name, m in groups:
        mm = m & np.isin(gold, ["internal", "external"])
        y = (gold[mm] == "internal").astype(int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        va = roc_auc_score(y, v_prob[mm])
        ta = roc_auc_score(y, t_prob[mm])
        ba = roc_auc_score(y, t_band[mm])
        print(f"{name:<12}{mm.sum():5d}  {va:8.3f}  {ta:8.3f}  {ba:12.3f}")
        for g in report["groups"]:
            if g["group"] == name:
                g["ie_n"] = int(mm.sum())
                g["ie_vector_auc"] = va
                g["ie_teacher_auc"] = ta
                g["ie_band_auc"] = ba

    # Hard-label agreement with gold, and what the live gate does per language
    print(f"\nteacher label vs gold, and the {THRESHOLD} gate, per language")
    print(f"{'':12}{'t.acc':>7}{'t.kappa':>9}{'t.int-recall':>13}"
          f"{'gate keeps':>12}{'keeps int':>11}{'keeps ext':>11}{'rho v-t':>9}")
    print("-" * 84)
    for name, m in groups:
        acc = float((t_label[m] == gold[m]).mean())
        kap = float(cohen_kappa_score(gold[m], t_label[m],
                                      labels=["internal", "external", "neutral"]))
        gi = gold[m] == "internal"
        t_rec = float((t_label[m][gi] == "internal").mean()) if gi.sum() else float("nan")
        keep = v_prob[m] >= THRESHOLD
        k_all = float(keep.mean())
        k_int = float(keep[gi].mean()) if gi.sum() else float("nan")
        ge = gold[m] == "external"
        k_ext = float(keep[ge].mean()) if ge.sum() else float("nan")
        rho = float(spearmanr(v_prob[m], t_prob[m]).statistic)
        klo, khi = boot_stat(
            lambda g, t: cohen_kappa_score(g, t, labels=["internal", "external", "neutral"]),
            rng, gold[m], t_label[m])
        rlo, rhi = boot_stat(lambda a, b: spearmanr(a, b).statistic, rng, v_prob[m], t_prob[m])
        print(f"{name:<12}{acc:7.3f}{kap:9.3f}{t_rec:13.3f}"
              f"{k_all:12.1%}{k_int:11.1%}{k_ext:11.1%}{rho:9.3f}"
              f"   kappa CI [{klo:.3f}, {khi:.3f}]  rho CI [{rlo:.3f}, {rhi:.3f}]")
        for g in report["groups"]:
            if g["group"] == name:
                g.update({"teacher_kappa_ci": [klo, khi], "spearman_ci": [rlo, rhi],
                          "teacher_acc": acc, "teacher_kappa": kap,
                          "teacher_internal_recall": t_rec, "gate_keep_all": k_all,
                          "gate_keep_internal": k_int, "gate_keep_external": k_ext,
                          "spearman_vector_teacher": rho})

    # The actual question: is the German gap real, or is it 9 positives of noise?
    print("\nGerman vs English, tested directly")
    for tag, sel in (("declared tags", langs), ("declared or detected", detected)):
        de, en = sel == "de", sel == "en"
        kfn = lambda g, t: cohen_kappa_score(g, t, labels=["internal", "external", "neutral"])
        (klo, khi), kp = boot_group_diff(kfn, rng, [gold[de], t_label[de]], [gold[en], t_label[en]])
        afn = lambda y, s: roc_auc_score(y, s)
        yde, yen = (gold[de] == "internal").astype(int), (gold[en] == "internal").astype(int)
        (vlo, vhi), vp = boot_group_diff(afn, rng, [yde, v_prob[de]], [yen, v_prob[en]])
        (tlo, thi), tp = boot_group_diff(afn, rng, [yde, t_prob[de]], [yen, t_prob[en]])
        print(f"  {tag} (de n={de.sum()}, {yde.sum()} internal; en n={en.sum()}, {yen.sum()} internal)")
        print(f"    teacher kappa  de-en {kfn(gold[de], t_label[de]) - kfn(gold[en], t_label[en]):+.3f}"
              f"  CI [{klo:+.3f}, {khi:+.3f}]  p={kp:.2f}")
        print(f"    vector AUC     de-en {roc_auc_score(yde, v_prob[de]) - roc_auc_score(yen, v_prob[en]):+.3f}"
              f"  CI [{vlo:+.3f}, {vhi:+.3f}]  p={vp:.2f}")
        print(f"    teacher AUC    de-en {roc_auc_score(yde, t_prob[de]) - roc_auc_score(yen, t_prob[en]):+.3f}"
              f"  CI [{tlo:+.3f}, {thi:+.3f}]  p={tp:.2f}")
        report.setdefault("de_vs_en", {})[tag] = {
            "n_de": int(de.sum()), "n_de_internal": int(yde.sum()),
            "n_en": int(en.sum()), "n_en_internal": int(yen.sum()),
            "teacher_kappa_diff_ci": [klo, khi], "teacher_kappa_p": kp,
            "vector_auc_diff_ci": [vlo, vhi], "vector_auc_p": vp,
            "teacher_auc_diff_ci": [tlo, thi], "teacher_auc_p": tp,
        }

    with open("REPORT_vector_teacher_by_lang.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote REPORT_vector_teacher_by_lang.json")


if __name__ == "__main__":
    main()
