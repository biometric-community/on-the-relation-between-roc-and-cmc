"""Evaluate ROC→CMC prediction; optional LFW face scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import models

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roc_cmc.data.lfw import LFWIdentityDataset, list_identities, resolve_lfw_root
from roc_cmc.metrics import empirical_cmc_closed_set, predict_and_compare
from roc_cmc.theory import fractional_ranks


@torch.no_grad()
def extract_lfw_features(ckpt_path: Path, cfg, device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    names = ckpt["names"]
    root = resolve_lfw_root(cfg)
    id_map = list_identities(root, min_images=int(cfg["lfw"].get("min_images", 20)))
    id_map = {k: id_map[k] for k in names if k in id_map}
    m = int(cfg["lfw"].get("m_samples", 20))
    for k in list(id_map):
        id_map[k] = id_map[k][:m]
    ds = LFWIdentityDataset(id_map, max_per_id=m)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, ckpt["num_classes"])
    net.load_state_dict(ckpt["model"])
    net.fc = nn.Identity()
    net = net.to(device).eval()
    # build (n, m, D) array aligned with names order
    feats_by_lab = {i: [] for i in range(len(names))}
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    for batch in loader:
        f = net(batch["image"].to(device)).cpu().numpy()
        for vec, lab in zip(f, batch["label"].numpy()):
            if len(feats_by_lab[int(lab)]) < m:
                feats_by_lab[int(lab)].append(vec)
    n = len(names)
    D = next(iter(feats_by_lab.values()))[0].shape[0]
    X = np.zeros((n, m, D), dtype=np.float32)
    for i in range(n):
        arr = feats_by_lab[i]
        for j in range(min(m, len(arr))):
            X[i, j] = arr[j]
        if len(arr) < m:
            # pad by repeating
            for j in range(len(arr), m):
                X[i, j] = arr[j % max(len(arr), 1)] if arr else 0
    return X


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    log_dir = Path(cfg["paths"]["log_dir"])
    synth = json.loads((log_dir / "synthetic_results.json").read_text())
    # summary MAE table
    rows = []
    for r in synth["synthetic"]:
        rows.append({
            "n": r["n"], "classifier": r["classifier"],
            "mae": r["mae_pred_vs_emp_cmc"],
            "frac_rank_corr": float(np.corrcoef(
                fractional_ranks(r["n"]), np.asarray(r["emp_cmc"])
            )[0, 1]) if r["n"] > 1 else 0.0,
        })
    out = {"synthetic_mae": rows}

    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "lfw_resnet18.pt"
    if ckpt.exists() and cfg.get("lfw", {}).get("enabled", True):
        device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        n = int(cfg["lfw"].get("eval_n", 10))
        m = int(cfg["lfw"].get("m_samples", 20))
        mg = int(cfg["lfw"].get("mg", 30))
        X_all = extract_lfw_features(ckpt, cfg, device)
        # take first n identities with full samples
        X = X_all[:n]
        from roc_cmc.data.synthetic import pair_scores

        mated, nonmated = pair_scores(X, "cosine")
        emp_cmc, _ = empirical_cmc_closed_set(X, "cosine", mg=mg, rng=np.random.default_rng(0))
        face = predict_and_compare(mated, nonmated, emp_cmc, n)
        face["note"] = "LFW stand-in for FRGC-V2 (DEVIATIONS D1)"
        out["lfw_face"] = {
            "n": n, "m": m, "mg": mg,
            "mae_pred_vs_emp_cmc": face["mae_pred_vs_emp_cmc"],
            "pred_cmc": face["pred_cmc"],
            "emp_cmc": face["emp_cmc"],
            "avg_roc_alpha": face["avg_roc_alpha"],
            "avg_roc_rho": face["avg_roc_rho"],
        }
        print(f"[lfw] MAE={face['mae_pred_vs_emp_cmc']:.4f}", flush=True)

    (log_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"synthetic_mae": rows, "has_lfw": "lfw_face" in out}, indent=2))


if __name__ == "__main__":
    main()
