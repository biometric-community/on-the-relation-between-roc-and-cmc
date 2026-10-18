# On the Relation Between ROC and CMC

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch/NumPy rebuild of Veldhuis & Raja (TBIOM 2023): probabilistic **ROC→CMC** relation (Theorem 1) with synthetic Gaussian biometric scores and an LFW face stand-in.

**Repository:** https://github.com/biometric-community/on-the-relation-between-roc-and-cmc

**Paper:** [10.1109/TBIOM.2023.3298561](https://doi.org/10.1109/TBIOM.2023.3298561)

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/report.sh
```

## Results (this rebuild)

Synthetic Table I — MAE of Theorem-1 predicted CMC vs empirical CMC:

| n | classifier | MAE |
|--:|--|--:|
| 10 | llr | 0.0061 |
| 10 | cosine | 0.0099 |
| 100 | llr | 0.0052 |
| 100 | cosine | 0.0105 |

LFW face stand-in (n=10): MAE **0.0037**. See `REPORT.md`.

## Citation

```
@article{veldhuis2023roc,
  title={On the Relation Between ROC and CMC},
  author={Veldhuis, Raymond N. J. and Raja, Kiran},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2023},
  doi={10.1109/TBIOM.2023.3298561}
}
```
