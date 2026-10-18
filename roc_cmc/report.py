"""REPORT.md + SVG figures from real experiment JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roc_cmc.plot_style import (
    BAR_COLOR, BAR_EDGE_COLOR, BAR_EDGE_WIDTH, FIGSIZE, LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS, apply_rcparams, apply_style,
)
from roc_cmc.theory import fractional_ranks


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def write_report(cfg, logs: Path, figdir: Path, report_path: Path):
    synth = load_json(logs / "synthetic_results.json") or {}
    ev = load_json(logs / "eval_results.json") or {}
    size = load_json(logs / "dataset_size.json") or {}
    cases = synth.get("synthetic", [])
    figs = []

    # MAE bar chart
    if cases:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        labels = [f"n={c['n']}/{c['classifier']}" for c in cases]
        vals = [c["mae_pred_vs_emp_cmc"] for c in cases]
        ax.bar(np.arange(len(vals)), vals, 0.6, color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("MAE (pred vs emp CMC)")
        ax.set_title("Synthetic: Theorem-1 CMC prediction error")
        apply_style(ax, grid_axis="y")
        fig.tight_layout()
        out = figdir / "synthetic_mae.svg"
        fig.savefig(out, format="svg")
        plt.close(fig)
        figs.append(out)

    # Example ROC/CMC overlay for first n=10 cosine case
    pick = next((c for c in cases if c["n"] == 10 and c["classifier"] == "cosine"), cases[0] if cases else None)
    if pick:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fr = fractional_ranks(pick["n"])
        ax.plot(pick["avg_roc_alpha"], pick["avg_roc_rho"], color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="Avg ROC")
        ax.plot(fr, pick["emp_cmc"], color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="Emp CMC")
        ax.plot(fr, pick["pred_cmc"], "o", color=MULTI_SERIES_COLORS[2], markersize=4, label="Pred CMC")
        ax.set_xlabel(r"$\alpha$ / fractional rank $r/n$")
        ax.set_ylabel("Probability")
        ax.set_title(f"Synthetic n={pick['n']} ({pick['classifier']})")
        ax.legend(loc="best")
        apply_style(ax)
        fig.tight_layout()
        out = figdir / "roc_cmc_overlay.svg"
        fig.savefig(out, format="svg")
        plt.close(fig)
        figs.append(out)

    face = ev.get("lfw_face")
    if face:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fr = fractional_ranks(face["n"])
        ax.plot(face["avg_roc_alpha"], face["avg_roc_rho"], color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="Avg ROC")
        ax.plot(fr, face["emp_cmc"], color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="Emp CMC")
        ax.plot(fr, face["pred_cmc"], "o", color=MULTI_SERIES_COLORS[2], markersize=4, label="Pred CMC")
        ax.set_xlabel(r"$\alpha$ / fractional rank $r/n$")
        ax.set_ylabel("Probability")
        ax.set_title(f"LFW face stand-in n={face['n']} (cosine)")
        ax.legend(loc="best")
        apply_style(ax)
        fig.tight_layout()
        out = figdir / "lfw_roc_cmc.svg"
        fig.savefig(out, format="svg")
        plt.close(fig)
        figs.append(out)

    lines = [
        "# Report: On the Relation Between ROC and CMC",
        "",
        "Rebuild of Veldhuis & Raja, TBIOM 2023 "
        "(DOI [10.1109/TBIOM.2023.3298561](https://doi.org/10.1109/TBIOM.2023.3298561)).",
        "",
        "## Setup",
        "",
        "- Synthetic Gaussian features (Eq. 30–32), Table I `(n,m,mg)`",
        f"- Dataset size gate: {size.get('decision', 'n/a')} ({size.get('total_gib', '?')} GiB)",
        "- Face: LFW ResNet-18 embeddings as FRGC-V2 stand-in (see `DEVIATIONS.md`)",
        "",
        "## Synthetic MAE (predicted vs empirical CMC)",
        "",
        "| n | classifier | MAE |",
        "|--:|--|--:|",
    ]
    for c in cases:
        lines.append(f"| {c['n']} | {c['classifier']} | {c['mae_pred_vs_emp_cmc']:.4f} |")
    if face:
        lines += [
            "",
            "## LFW face stand-in",
            "",
            f"- n={face['n']}, MAE pred vs emp CMC = **{face['mae_pred_vs_emp_cmc']:.4f}**",
            "",
        ]
    lines += ["## Figures", ""]
    root = report_path.resolve().parent
    for p in figs:
        rel = p.resolve().relative_to(root)
        lines += [f"### `{rel}`", "", f"![]({rel})", ""]
    lines += ["## Regenerate", "", "```bash", "bash scripts/report.sh", "```", ""]
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    figdir = Path(cfg["paths"]["figure_dir"])
    logs = Path(cfg["paths"]["log_dir"])
    if not figdir.is_absolute():
        figdir = _ROOT / figdir
    if not logs.is_absolute():
        logs = _ROOT / logs
    figdir.mkdir(parents=True, exist_ok=True)
    write_report(cfg, logs, figdir, _ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
