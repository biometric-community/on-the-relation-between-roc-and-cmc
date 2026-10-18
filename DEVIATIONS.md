# Deviations from the paper

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | FRGC-V2 + ElasticFace | LFW + ResNet-18 ID embeddings | FRGC-V2 / ElasticFace weights not registered |
| D2 | FVC2006 fingerprints; HKPU iris | Omitted | Datasets not under `projects/datasets/` |
| D3 | Exact Table I `mg` gallery Monte Carlo | Implemented; seed fixed | Reproducibility |
| D4 | AgeDB / CFP-FP / CALFW / CPLFW listed in digest | LFW primary for face stand-in | Sufficient for ROC↔CMC illustration |

## Taken from paper

- Theorem 1: `cmc(r)=∫ ρ(α) Beta(α;r,n−r) dα`
- Synthetic σ²=(0.5,0.8,0.85,0.9), LLR + cosine scores (Eq. 30–32)
- Table I: synth (10,30,30), (100,20,5)
