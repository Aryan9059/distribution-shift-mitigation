"""The whole study in one loop: simulate shift -> measure the harm -> detect it (base paper)
-> handle it (our D2A model).

Run from the project root:
    python -m experiments.run_all --quick                 # adult, 1 seed, small grid (minutes)
    python -m experiments.run_all --datasets adult        # one dataset, all seeds
    python -m experiments.run_all                         # everything, 3 seeds (about an hour)
    python -m experiments.run_all --full                  # 5 seeds, 8000-row pools, n up to 5000 (several hours)

For every (dataset, seed) block it writes rows to four tables in results/ (one file per dataset):
  impact_*.csv      accuracy of each model on source-test, shifted target-test, and the oracle
  detection_*.csv   p-value of each (representation, test) of Failing Loudly per target sample size
  malignancy_*.csv  true accuracy vs three estimates (ATC label-free, paper's top-k labeled, random-k labeled)
  adaptation_*.csv  accuracy of No-adapt / Always-BBSE / Always-IW / Always-CORAL / D2A / Oracle

Splits (no leakage): source -> train 50% / val 25% / test 25%; target -> pool A / pool B.
The shift is applied to both pools. Pool A gives the unlabeled adaptation sample (and, with
labels, the oracle's training set); pool B is where everything is scored.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from src.adapt import D2A
from src.data import make_domains, split_pools, tilt_cols
from src.detect import FailingLoudly
from src.models import MODEL_NAMES
from src.shifts import SHIFT_TYPE, scenarios

DATASETS = ["adult", "covertype", "acs:CA>WY", "acs:CA>MS", "acs:CA>NY", "acs:CA>TX"]
N_TARGET = 4000   # rows per shifted target pool (--full: 8000)
N_ADAPT = 1000    # unlabeled target rows D2A may look at
N_GRID = [10, 20, 50, 100, 200, 500, 1000]  # --full adds 5000
DETECT_MODEL = "GBT"  # the black box whose softmax BBSDs uses


def fit_oracle(model, X, y):
    """Retrain on labeled shifted target data (the upper bound). A shifted pool can lose a
    class entirely, which XGBoost rejects, so fit on compacted labels and map back."""
    classes, y_c = np.unique(y, return_inverse=True)
    fitted = clone(model).fit(X, y_c)
    return lambda Z: classes[fitted.predict(Z)]


def scores(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def run_block(dataset, seed, models, n_grid, n_target):
    t0 = time.time()
    Xs, ys, Xt, yt = make_domains(dataset, seed)
    Xs_tr, ys_tr, Xs_rest, ys_rest = split_pools(Xs, ys, seed)
    Xs_val, ys_val, Xs_te, ys_te = split_pools(Xs_rest, ys_rest, seed + 100)
    Xt_A, yt_A, Xt_B, yt_B = split_pools(Xt, yt, seed)

    d2a = {m: D2A(m, seed).fit(Xs_tr, ys_tr, Xs_val, ys_val) for m in models}
    acc_src = {m: accuracy_score(ys_te, d2a[m].model.predict(Xs_te)) for m in models}
    fl = FailingLoudly(d2a[DETECT_MODEL].model, Xs_tr, Xs_val, seed)

    if dataset.startswith("acs"):  # real shift: no dial, one scenario
        scen = [("real", 0.0, lambda X, y, rng, n: (X.iloc[:n].reset_index(drop=True), y[:n]))]
    else:
        scen = scenarios(tilt_cols(dataset))

    tables = {"impact": [], "detection": [], "malignancy": [], "adaptation": []}
    for i, (shift, sev, fn) in enumerate(scen):
        rng = np.random.default_rng((seed, i))
        n = min(n_target, len(Xt_A), len(Xt_B))
        XA, yA = fn(Xt_A, yt_A, rng, n)
        XB, yB = fn(Xt_B, yt_B, rng, n)
        X_adapt = XA.iloc[:N_ADAPT]
        key = dict(dataset=dataset, seed=seed, shift=shift, severity=sev, shift_type=SHIFT_TYPE[shift])

        # 1) impact of the shift on each model + oracle; 4) adaptation
        for m in models:
            base = scores(yB, d2a[m].model.predict(XB))
            oracle = scores(yB, fit_oracle(d2a[m].model, XA, yA)(XB))
            tables["impact"].append({**key, "model": m, "acc_source_test": acc_src[m],
                                     "acc_base": base[0], "f1_base": base[1],
                                     "acc_oracle": oracle[0], "f1_oracle": oracle[1]})
            info = d2a[m].adapt(X_adapt)
            row = {**key, "model": m, "acc_val": d2a[m].acc_val, **info,
                   "acc_none": base[0], "acc_oracle": oracle[0]}
            for action in ("bbse", "iw", "coral"):
                row[f"acc_{action}"] = accuracy_score(yB, d2a[m].fixed(action)(XB))
            row["acc_d2a"] = row[f"acc_{d2a[m].action}"]
            row["acc_d2a_gated"] = row[f"acc_{info['action_gated']}"]  # ablation: fix only if ATC says harmful
            tables["adaptation"].append(row)

        # 2) detection with the paper's methods (black box = GBT), growing target sample sizes
        drng = np.random.default_rng((seed, i, 1))
        for nn in [x for x in n_grid if x <= len(XB)]:
            for (rep, test), p in fl.run(XB.iloc[:nn], drng).items():
                tables["detection"].append({**key, "n": nn, "rep": rep, "test": test, "p": p})

        # 3) malignancy: true target accuracy vs estimates from 1000 target rows
        m = DETECT_MODEL
        Xm, ym = XB.iloc[:N_ADAPT], yB[:N_ADAPT]
        rand = drng.choice(len(ym), size=100, replace=False)
        tables["malignancy"].append({
            **key, "acc_source_val": d2a[m].acc_val,
            "acc_true": tables["impact"][-len(models) + models.index(m)]["acc_base"],
            "acc_atc": d2a[m].info["est_acc"],  # from adapt() above, on the same kind of sample
            "acc_topk": fl.malignancy_labeled(Xm, ym, drng, k=100),
            "acc_randk": float((d2a[m].model.predict(Xm.iloc[rand]) == ym[rand]).mean())})
    print(f"  {dataset} seed={seed}: {len(scen)} scenarios in {time.time() - t0:.0f}s", flush=True)
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="adult, 1 seed, small pools")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="larger setting: 5 seeds, 8000-row pools, n up to 5000")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    a = ap.parse_args()
    if DETECT_MODEL not in a.models:
        a.models.append(DETECT_MODEL)

    datasets = a.datasets or (["adult"] if a.quick else DATASETS)
    if a.full:
        a.seeds = 5
    # Seed 0 was the development run used to design D2A's decision rules, so the reported
    # results use fresh seeds 1, 2, ...
    seeds = [0] if a.quick else list(range(1, a.seeds + 1))
    n_grid = N_GRID + [5000] if a.full else N_GRID
    n_target = 2000 if a.quick else 8000 if a.full else N_TARGET
    out = Path("results/quick" if a.quick else "results")
    out.mkdir(parents=True, exist_ok=True)

    for d in datasets:
        tag = d.replace(":", "_").replace(">", "-")
        tables = {}
        for s in seeds:
            for name, rows in run_block(d, s, a.models, n_grid, n_target).items():
                tables.setdefault(name, []).extend(rows)
            for name, rows in tables.items():  # save as we go
                pd.DataFrame(rows).to_csv(out / f"{name}_{tag}.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
