# Overnight Runbook v4 Pilot - Report

**Phase Reached:** Phase 8 (Completed all phases)
**Stop Reason:** Reached end of runbook. (Phase 7 skipped due to Phase 6 Regression).

### Phase 4 Signal Stats
- **n_with_score**: 1500
- **mean**: 4.14
- **std**: 1.56
- **distinct(round1)**: 10
- **mean_loc_score (internal)**: 7.10 (n=92)
- **mean_loc_score (neutral)**: 4.96 (n=890)
- **mean_loc_score (external)**: 2.19 (n=518)

### Phase 6 Metrics Table
| Metric | Baseline (Deployed) | B (Legacy @ 1500) | A (Graded @ 1500) |
| :--- | :--- | :--- | :--- |
| **val_auc_internal** | 0.760 | 0.7661 | 0.6727 |
| **val_avg_precision** | 0.452 | 0.3700 | 0.2427 |
| **dev_spearman_vs_teacher** | 0.141 | 0.2052 | 0.5182 |

*(Noisy footnote: P@10 / P@25 / P@50 for A = 0.10 / 0.20 / 0.28. For B = 0.60 / 0.44 / 0.40)*

### Phase 6 Verdict & Recommendation
**Verdict:** **Regression** (A < baseline and A < B)
**Recommendation:** Do not proceed with full scale annotation or reranker; the graded target (A) severely degraded validation AUC and AP compared to the legacy binary target (B) at 1500 posts, despite showing better correlation with the teacher (`dev_spearman_vs_teacher`).

### New Files Created
- `pilot_raw_1500.jsonl`
- `pilot_smoke.jsonl`
- `pilot_smoke_v4.jsonl`
- `dataset_annotated_pilot_v4.jsonl`
- `embeddings_pilot_v4_jina.npz`
- `internal_vector_v4.npy`
- `internal_vector_v4_meta.json`
- `results_internal_vector_v4.json`
- `train_v4_A.log`
- `internal_vector_ctrl.npy`
- `internal_vector_ctrl_meta.json`
- `results_internal_vector_ctrl.json`
- `train_v4_B.log`
- `REPORT_v4_pilot.md` (this file)
