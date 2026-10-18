"""Predict CMC from an empirical ROC CSV (alpha,rho columns)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roc_cmc.theory import fractional_ranks, predict_cmc_from_roc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--roc-csv", required=True, help="CSV with columns alpha,rho")
    p.add_argument("--n", type=int, required=True, help="Gallery size")
    p.add_argument("--out", default="outputs/predictions/predicted_cmc.json")
    args = p.parse_args(argv)
    data = np.loadtxt(args.roc_csv, delimiter=",", skiprows=1)
    alpha, rho = data[:, 0], data[:, 1]
    cmc = predict_cmc_from_roc(alpha, rho, args.n)
    payload = {
        "n": args.n,
        "fractional_rank": fractional_ranks(args.n).tolist(),
        "cmc": cmc.tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
