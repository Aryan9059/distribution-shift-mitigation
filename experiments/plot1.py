"""Phase 1 figures from results/phase1.csv.  Run: python -m experiments.plot1 [csv]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHIFTS = ["covariate", "noise", "drift", "label", "concept", "combo"]
COLORS = {"LR": "#1f77b4", "RF": "#2ca02c", "GBT": "#d62728", "MLP": "#9467bd"}


def curves(df, dataset, out):
    d = df[df.dataset == dataset]
    ident = d[d["shift"] == "identity"]  # severity-0 point shared by every panel
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, shift in zip(axes.ravel(), SHIFTS):
        sub = pd.concat([ident, d[d["shift"] == shift]])
        for m, g in sub.groupby("model"):
            for col, ls in (("acc_base", "-"), ("acc_upper", "--")):
                s = g.groupby("severity")[col].agg(["mean", "std"])
                ax.errorbar(s.index, s["mean"], yerr=s["std"], color=COLORS[m], ls=ls,
                            marker="o", ms=3, capsize=2, label=m if ls == "-" else None)
        ax.set_title(shift)
        ax.set_xlabel("severity")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("accuracy on target-test")
    axes[0, 0].legend(title="solid = source model\ndashed = upper bound", fontsize=8)
    fig.suptitle(f"{dataset}: accuracy vs shift severity (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out / f"phase1_{dataset}.png", dpi=150)
    plt.close(fig)


def acs_bars(df, out):
    d = df[df.dataset.str.startswith("acs")].copy()
    d["target"] = d.dataset.str.split(">").str[1]
    models = sorted(d.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    cols = ["acc_source_test", "acc_base", "acc_upper"]
    for ax, m in zip(axes, models):
        g = d[d.model == m].groupby("target")[cols].agg(["mean", "std"])
        x = np.arange(len(g))
        for k, c in enumerate(cols):
            ax.bar(x + (k - 1) * 0.27, g[c]["mean"], 0.27, yerr=g[c]["std"], capsize=2, label=c)
        ax.set_xticks(x, g.index)
        ax.set_title(m)
        ax.set_ylim(0.6, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("ACS Income, trained on CA: real shift to other states")
    fig.tight_layout()
    fig.savefig(out / "phase1_acs.png", dpi=150)
    plt.close(fig)


def main():
    csv = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1.csv")
    df = pd.read_csv(csv)
    out = csv.parent
    for ds in df.dataset.unique():
        if not ds.startswith("acs"):
            curves(df, ds, out)
    if df.dataset.str.startswith("acs").any():
        acs_bars(df, out)
    print("figures saved in", out)


if __name__ == "__main__":
    main()
