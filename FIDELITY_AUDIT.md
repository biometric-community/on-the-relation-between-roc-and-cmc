# Fidelity audit: ROC ↔ CMC

Target: Veldhuis & Raja, TBIOM 2023 (DOI 10.1109/TBIOM.2023.3298561).

## Per-pass statistics

| Pass | Focus | Confidence | Evidence |
|------|-------|------------|----------|
| 1 | Layout / scripts / docs | **94%** | `roc_cmc/`, configs, scripts, README |
| 2 | Theorem 1 Beta-kernel CMC | **92%** | `theory/beta_cmc.py`; MAE≲0.01 on synth |
| 3 | Table I synthetic protocol | **90%** | (10,30,30)+(100,20,5); LLR+cosine Eqs. 30–32 |
| 4 | Face / REPORT | **82%** | LFW stand-in (D1); REPORT + SVGs from JSON |
| 5 | Smoke + full experiments | **92%** | Size 0.37 GiB; full suite + LFW train/eval |

**Mean confidence after Pass 5: ~90%.** FRGC/FVC/iris omitted (D1–D2).

## Sign-off

Audit complete after Pass 5.
