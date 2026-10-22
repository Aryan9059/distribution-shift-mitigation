"""Turn results/*.csv into the report's figures and LaTeX tables (report/figures/) and a
plain-text summary of every number used in the report (results/summary.md).

Run from the project root after run_all:  python -m experiments.make_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adapt import TAU
from src.data import load_acs_state, load_adult, load_covertype
from src.detect import ALPHA
from src.models import MODEL_NAMES
from src.shifts import SHIFT_TYPE

RES, FIG = Path("results"), Path("report/figures")
SIM = ["adult", "covertype"]
FAMILIES = ["covariate", "noise", "drift", "label", "concept", "combo"]
METHODS = ["none", "bbse", "iw", "coral", "d2a", "oracle"]
METHOD_NAME = {"none": "No adapt", "bbse": "BBSE", "iw": "IW", "coral": "CORAL", "d2a": "D2A (ours)",
               "oracle": "Oracle"}
# Categorical slots in fixed order (validated default palette); grey is reserved for references.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "pdf.fonttype": 42})
COL_W, PAGE_W = 3.5, 7.16  # IEEE column and page widths in inches

summary = []


def load(name):
    """Results of the final run. Impact, detection and malignancy do not depend on D2A's design, so
    they also include the development run (seed 0, results/dev_seed0); adaptation does not."""
    files = sorted(RES.glob(f"{name}_*.csv"))
    if name != "adaptation":
        files += sorted((RES / "dev_seed0").glob(f"{name}_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["shift_type"] = df["shift"].map(SHIFT_TYPE)
    return df


def nice(dataset):
    return {"adult": "Adult", "covertype": "Covertype"}.get(dataset, dataset.replace("acs:", "ACS ").replace(">", "$\\to$"))


def note(title, df):
    summary.append(f"## {title}\n\n{df.round(3).to_string()}\n")


def to_latex(df, path, caption_free_fmt="{:.3f}", bold_max_cols=None):
    """Write a tabular body (no float environment) so main.tex controls captions and placement.
    bold_max_cols: columns among which the row maximum is bolded."""
    lines = []
    for idx, row in df.iterrows():
        cells = []
        best = row[bold_max_cols].max() if bold_max_cols else None
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = "--" if np.isnan(v) else caption_free_fmt.format(v)
                if bold_max_cols and c in bold_max_cols and np.isclose(v, best):
                    s = f"\\textbf{{{s}}}"
            else:
                s = str(v)
            cells.append(s)
        label = idx if isinstance(idx, str) else " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        lines.append(" & ".join([label] + cells) + r" \\")
    (FIG / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 1) impact of shift
def fig_impact(imp):
    sim = imp[imp.dataset.isin(SIM)]
    fig, axes = plt.subplots(2, 6, figsize=(PAGE_W, 2.6), sharey="row")
    for r, d in enumerate(SIM):
        part = sim[sim.dataset == d]
        ident = part[part["shift"] == "identity"].groupby("model")[["acc_base", "acc_oracle"]].mean()
        for c, fam in enumerate(FAMILIES):
            ax = axes[r, c]
            g = part[part["shift"] == fam].groupby(["model", "severity"])[["acc_base", "acc_oracle"]].mean()
            for k, m in enumerate(MODEL_NAMES):
                s = g.loc[m]
                x = np.r_[0, s.index.to_numpy()]
                ax.plot(x, np.r_[ident.loc[m, "acc_base"], s["acc_base"]], color=SLOTS[k], marker="o",
                        ms=2.5, label=m)
            o = g.groupby("severity")["acc_oracle"].mean()
            ax.plot(np.r_[0, o.index], np.r_[ident["acc_oracle"].mean(), o], color=GREY, ls="--",
                    label="Oracle (mean)")
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(f"{nice(d)}\naccuracy")
            if r == 1:
                ax.set_xlabel("severity")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4, rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig_impact.pdf")
    plt.close(fig)

    # numbers for the text: drop of the source model and gap to oracle at the strongest severity
    strong = sim[sim["shift"] != "identity"]
    strong = strong[strong.severity == strong.groupby(["dataset", "shift"]).severity.transform("max")]
    t = strong.groupby(["dataset", "shift", "model"])[["acc_source_test", "acc_base", "acc_oracle"]].mean()
    note("Impact at the strongest severity (mean over seeds)", t)
    ident = sim[sim["shift"] == "identity"].groupby(["dataset", "model"])[["acc_source_test", "acc_base", "acc_oracle"]].mean()
    note("No shift (identity)", ident)
    acs = imp[~imp.dataset.isin(SIM)].groupby(["dataset", "model"])[["acc_source_test", "acc_base", "acc_oracle"]].mean()
    note("ACS real shift", acs)


# ---------------------------------------------------------------- 2) detection (base paper)
DET_ROWS = [("NoRed", "KS"), ("PCA", "KS"), ("SRP", "KS"), ("BBSDs", "KS"), ("BBSDh", "Chi2"),
            ("Classif", "Binomial"), ("NoRed", "MMD"), ("PCA", "MMD"), ("SRP", "MMD"), ("BBSDs", "MMD")]
REP_COLOR = {r: SLOTS[i] for i, r in enumerate(["NoRed", "PCA", "SRP", "BBSDs", "BBSDh", "Classif"])}


def detection_tables(det):
    det = det[det.dataset.isin(SIM)].copy()
    det["hit"] = det.p < ALPHA
    shifted = det[det["shift"] != "identity"]
    power = shifted.groupby(["rep", "test", "n"]).hit.mean().unstack("n")
    power = power.reindex(DET_ROWS)
    power.index = [f"{r} ({t})" for r, t in power.index]
    to_latex(power, "table_detection.tex", "{:.2f}")
    note("Detection rate over all shifted scenarios (Adult + Covertype)", power)

    fpr = det[det["shift"] == "identity"].groupby(["rep", "test", "n"]).hit.mean().unstack("n").reindex(DET_ROWS)
    fpr.index = [f"{r} ({t})" for r, t in fpr.index]
    note("False-alarm rate on the no-shift control", fpr)

    best = det[(det.rep == "BBSDs") & (det.test == "KS")]
    by_shift = best.groupby(["shift", "n"]).hit.mean().unstack("n").reindex(["identity"] + FAMILIES)
    to_latex(by_shift, "table_detect_shift.tex", "{:.2f}")
    note("BBSDs+KS detection rate per shift (identity = false alarms)", by_shift)

    by_sev = best[best["shift"] != "identity"].groupby(["shift", "severity"]).hit.mean()
    note("BBSDs+KS detection rate per shift and severity (all n)", by_sev.unstack())

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    for ax, kind in zip(axes, ["univariate", "multivariate"]):
        rows = DET_ROWS[:6] if kind == "univariate" else DET_ROWS[6:]
        for rep, test in rows:
            s = shifted[(shifted.rep == rep) & (shifted.test == test)].groupby("n").hit.mean()
            ax.plot(s.index, s.values, marker="o", ms=2.5, color=REP_COLOR[rep], label=f"{rep}")
        ax.set_xscale("log")
        ax.set_xlabel("target samples $n$")
        ax.set_title("KS / $\\chi^2$ / binomial" if kind == "univariate" else "MMD")
    axes[0].set_ylabel("detection rate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False,
               columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig_detection.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 3) malignancy
def malignancy(mal):
    mal = mal.copy()
    mal["drop_true"] = mal.acc_source_val - mal.acc_true
    est = {"ATC (0 labels)": "acc_atc", "Paper top-$k$ (100 labels)": "acc_topk", "Random (100 labels)": "acc_randk"}
    harmful = mal.drop_true > TAU
    rows = {}
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.45), sharex=True, sharey=True)
    lim = [mal.drop_true.min() - 0.02, mal.drop_true.max() + 0.02]
    for k, (ax, (name, col)) in enumerate(zip(axes, est.items())):
        d = mal.acc_source_val - mal[col]
        rows[name] = {"MAE of accuracy": (mal[col] - mal.acc_true).abs().mean(),
                      "harmful? accuracy": ((d > TAU) == harmful).mean(),
                      "harmful recall": ((d > TAU) & harmful).sum() / max(harmful.sum(), 1),
                      "benign recall": ((d <= TAU) & ~harmful).sum() / max((~harmful).sum(), 1)}
        ax.scatter(mal.drop_true, d, s=3, color=SLOTS[k], alpha=0.5, linewidths=0)
        ax.plot(lim, lim, color=GREY, lw=0.8, ls="--")
        ax.set_title(name.split(" (")[0], fontsize=6.5)
        ax.set_xlabel("true drop")
    axes[0].set_ylabel("estimated drop")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_malignancy.pdf")
    plt.close(fig)
    t = pd.DataFrame(rows).T
    to_latex(t, "table_malignancy.tex", "{:.3f}")
    note(f"Malignancy estimates (GBT, all datasets, harmful = true drop > {TAU})", t)


# ---------------------------------------------------------------- 4) adaptation (ours)
def adaptation(ad):
    gbt = ad[ad.model == "GBT"]
    acc_cols = [f"acc_{m}" for m in METHODS]
    rename = {f"acc_{m}": METHOD_NAME[m] for m in METHODS}
    non_oracle = [METHOD_NAME[m] for m in METHODS[:-1]]

    rows = []
    for d in SIM:
        part = gbt[gbt.dataset == d]
        for fam in ["identity"] + FAMILIES:
            r = part[part["shift"] == fam][acc_cols].mean().rename(rename)
            r.name = (nice(d), fam)
            rows.append(r)
    for d in sorted(gbt.dataset.unique()):
        if d not in SIM:
            r = gbt[gbt.dataset == d][acc_cols].mean().rename(rename)
            r.name = ("ACS", d.split(">")[1])
            rows.append(r)
    t = pd.DataFrame(rows)
    t.loc[("All", "mean"), :] = t.mean()
    to_latex(t, "table_adapt.tex", "{:.3f}", bold_max_cols=non_oracle)
    note("Adaptation accuracy, GBT (mean over seeds and severities)", t)

    # generality across models: mean accuracy over every scenario + D2A's worst loss vs no-adapt
    g = ad.groupby("model")[acc_cols + ["acc_d2a_gated"]].mean().rename(columns=rename).reindex(MODEL_NAMES)
    g = g.rename(columns={"acc_d2a_gated": "D2A+gate"})[non_oracle + ["D2A+gate", "Oracle"]]
    to_latex(g, "table_models.tex", "{:.3f}", bold_max_cols=non_oracle + ["D2A+gate"])
    note("All models, mean accuracy over all scenarios", g)
    gain = {m: (ad[f"acc_{m}"] - ad.acc_none) for m in ["bbse", "iw", "coral", "d2a", "d2a_gated"]}
    note("Gain vs no-adapt over all models and scenarios: share losing > 1 pt / share gaining > 1 pt / worst loss",
         pd.DataFrame({m: {"lose>1pt": (v < -0.01).mean(), "gain>1pt": (v > 0.01).mean(), "worst": v.min(),
                           "mean": v.mean()} for m, v in gain.items()}).T)
    correct = {"identity": "none", "covariate": "iw", "label": "bbse", "noise": "coral", "drift": "coral"}
    sim_rows = ad[ad["shift"].isin(correct)]
    note("Diagnosis accuracy per shift (all models; covariate->IW, label->BBSE, noise/drift->CORAL, identity->none)",
         (sim_rows.action == sim_rows["shift"].map(correct)).groupby(sim_rows["shift"]).mean())

    # diagnosis: which action did D2A take for each true shift (GBT, all datasets)
    order = ["identity"] + FAMILIES + ["real"]
    diag = pd.crosstab(gbt["shift"], gbt["action"], normalize="index").reindex(order).reindex(columns=["none", "bbse", "iw", "coral"]).fillna(0)
    diag.columns = ["none", "BBSE", "IW", "CORAL"]
    to_latex(diag * 100, "table_diagnosis.tex", "{:.0f}")
    note("D2A action per true shift (% of runs, GBT)", diag)
    harm = (gbt.acc_source_test if "acc_source_test" in gbt else gbt.acc_val) - gbt.acc_none > TAU
    note("D2A harmful flag vs true harm (GBT)", pd.crosstab(harm.rename("truly harmful"), gbt.harmful))

    # figure: accuracy gain over no-adapt per shift family (GBT, simulated datasets)
    sim = gbt[gbt.dataset.isin(SIM)]
    fams = ["identity"] + FAMILIES
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    show = ["bbse", "iw", "coral", "d2a", "oracle"]
    width = 0.8 / len(show)
    for k, m in enumerate(show):
        gain = [(sim[sim["shift"] == f][f"acc_{m}"] - sim[sim["shift"] == f].acc_none).mean() for f in fams]
        ax.bar(np.arange(len(fams)) + (k - 2) * width, gain, width * 0.9, label=METHOD_NAME[m],
               color=GREY if m == "oracle" else SLOTS[k])
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xticks(np.arange(len(fams)), fams)
    ax.set_ylabel("accuracy gain vs no adapt")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), columnspacing=0.8,
              handlelength=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_adapt.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 5) datasets
def datasets():
    adult_X, adult_y = load_adult()
    cov_X, cov_y = load_covertype()
    acs_X, acs_y = load_acs_state("CA")
    rows = []
    for name, X, y, shift in [("Adult", adult_X, adult_y, "simulated"), ("Covertype", cov_X, cov_y, "simulated"),
                              ("ACS Income (CA)", acs_X, acs_y, "real (state)")]:
        num = X.select_dtypes(include="number").shape[1]
        rows.append(f"{name} & {len(X):,} & {num} & {X.shape[1] - num} & {len(np.unique(y))} & {shift} \\\\")
    (FIG / "table_datasets.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cols = ["age", "education", "occupation", "hours-per-week", "sex"]
    sample = adult_X.loc[[0, 1, 2, 3], cols].assign(label=np.where(adult_y[[0, 1, 2, 3]] == 1, ">50K", "<=50K"))
    tex = lambda v: str(v).replace("_", " ").replace("&", "\\&").replace("<=", "$\\le$").replace(">", "$>$")
    lines = [" & ".join(tex(v) for v in r) + r" \\" for r in sample.itertuples(index=False)]
    (FIG / "table_sample.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    note("Adult sample rows", sample)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    note("Seeds per dataset", pd.DataFrame({t: load(t).groupby("dataset").seed.unique().map(sorted)
                                            for t in ["impact", "detection", "malignancy", "adaptation"]}))
    fig_impact(load("impact"))
    detection_tables(load("detection"))
    malignancy(load("malignancy"))
    adaptation(load("adaptation"))
    datasets()
    (RES / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("wrote", FIG, "and", RES / "summary.md")


if __name__ == "__main__":
    main()
