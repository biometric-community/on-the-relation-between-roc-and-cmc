# Implementation plan: ROC ↔ CMC

Paper: Veldhuis & Raja, TBIOM 2023, DOI 10.1109/TBIOM.2023.3298561.

## Scope

1. Theorem-1 CMC predictor (Beta kernel integral).
2. Synthetic Table I experiments (LLR + cosine).
3. LFW face stand-in (FRGC blocked).

## Figures

| Paper | Ours |
|-------|------|
| Figs 3–6 synthetic ROC/CMC | `roc_cmc_overlay.svg`, `synthetic_mae.svg` |
| Figs 7–8 face | `lfw_roc_cmc.svg` |

## Size gate

LFW ≈ 0.4 GiB → full_train / full experiment suite.
