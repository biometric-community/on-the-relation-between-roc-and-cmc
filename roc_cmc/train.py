"""Run synthetic (Table I) + optional LFW embedding training."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from roc_cmc.data.synthetic import pair_scores, sample_features
from roc_cmc.metrics import empirical_cmc_closed_set, predict_and_compare


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_synthetic_case(n: int, m: int, mg: int, classifier: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X = sample_features(n, m, rng)
    score_fn = "llr" if classifier == "llr" else "cosine"
    mated, nonmated = pair_scores(X, score_fn)
    emp_cmc, _ = empirical_cmc_closed_set(X, score_fn, mg=mg, rng=rng)
    out = predict_and_compare(mated, nonmated, emp_cmc, n)
    out.update({"classifier": classifier, "m": m, "mg": mg, "seed": seed})
    return out


def train_lfw_embedding(cfg, device) -> Path:
    """Train a small ResNet-18 ID head on LFW identities with ≥m images."""
    root = resolve_lfw_root(cfg)
    min_imgs = int(cfg["lfw"].get("min_images", 20))
    id_map = list_identities(root, min_images=min_imgs)
    max_ids = cfg["lfw"].get("max_identities")
    names = sorted(id_map.keys())
    if max_ids is not None:
        names = names[: int(max_ids)]
        id_map = {k: id_map[k] for k in names}
    ds = LFWIdentityDataset(id_map, max_per_id=cfg["lfw"].get("max_per_id"))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["lfw"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        drop_last=True,
    )
    net = models.resnet18(weights="DEFAULT")
    feat_dim = net.fc.in_features
    net.fc = nn.Linear(feat_dim, len(id_map))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lfw"].get("lr", 1e-3)))
    ce = nn.CrossEntropyLoss()
    epochs = int(cfg["lfw"].get("epochs", 5))
    max_steps = cfg["lfw"].get("max_steps")
    step = 0
    hist = []
    for ep in range(epochs):
        net.train()
        running, n = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(net(x), y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n += 1
            step += 1
            if max_steps is not None and step >= int(max_steps):
                break
        row = {"epoch": ep + 1, "loss": running / max(n, 1)}
        hist.append(row)
        print(f"[lfw] epoch={ep+1} loss={row['loss']:.4f}", flush=True)
        if max_steps is not None and step >= int(max_steps):
            break
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "lfw_resnet18.pt"
    torch.save({"model": net.state_dict(), "num_classes": len(id_map), "names": names, "history": hist}, path)
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["log_dir"]) / "lfw_train_history.json").write_text(json.dumps(hist, indent=2))
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-lfw", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    for k in ("checkpoint_dir", "log_dir", "pred_dir", "figure_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    results = []
    for case in cfg["synthetic"]["cases"]:
        for clf in cfg["synthetic"]["classifiers"]:
            print(f"[synth] n={case['n']} m={case['m']} mg={case['mg']} clf={clf}", flush=True)
            r = run_synthetic_case(
                n=int(case["n"]), m=int(case["m"]), mg=int(case["mg"]),
                classifier=clf, seed=int(cfg["synthetic"].get("seed", 42)) + int(case["n"]),
            )
            print(f"  MAE pred vs emp CMC = {r['mae_pred_vs_emp_cmc']:.4f}", flush=True)
            results.append(r)
    out = {
        "synthetic": results,
        "sec": time.time() - t0,
        "paper_table_i": True,
    }
    log_dir = Path(cfg["paths"]["log_dir"])
    (log_dir / "synthetic_results.json").write_text(json.dumps(out, indent=2))
    if not args.skip_lfw and cfg.get("lfw", {}).get("enabled", True):
        ckpt = train_lfw_embedding(cfg, device)
        out["lfw_checkpoint"] = str(ckpt)
        (log_dir / "train_summary.json").write_text(json.dumps(out, indent=2))
    else:
        (log_dir / "train_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote synthetic_results.json ({len(results)} cases)", flush=True)


if __name__ == "__main__":
    main()
