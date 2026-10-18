# Report: On the Relation Between ROC and CMC

Rebuild of Veldhuis & Raja, TBIOM 2023 (DOI [10.1109/TBIOM.2023.3298561](https://doi.org/10.1109/TBIOM.2023.3298561)).

## Setup

- Synthetic Gaussian features (Eq. 30–32), Table I `(n,m,mg)`
- Dataset size gate: full_train (0.3659 GiB)
- Face: LFW ResNet-18 embeddings as FRGC-V2 stand-in (see `DEVIATIONS.md`)

## Synthetic MAE (predicted vs empirical CMC)

| n | classifier | MAE |
|--:|--|--:|
| 10 | llr | 0.0061 |
| 10 | cosine | 0.0099 |
| 100 | llr | 0.0052 |
| 100 | cosine | 0.0105 |

## LFW face stand-in

- n=10, MAE pred vs emp CMC = **0.0037**

## Figures

### `outputs/figures/synthetic_mae.svg`

![](outputs/figures/synthetic_mae.svg)

### `outputs/figures/roc_cmc_overlay.svg`

![](outputs/figures/roc_cmc_overlay.svg)

### `outputs/figures/lfw_roc_cmc.svg`

![](outputs/figures/lfw_roc_cmc.svg)

## Regenerate

```bash
bash scripts/report.sh
```
