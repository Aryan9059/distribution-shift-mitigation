"""Phase 1: how much does each shift hurt each model, and how far are we from the upper bound?

Run from the project root:  python -m experiments.phase1 --debug
Full run (slow, overnight):  python -m experiments.phase1

baseline 1 = source model scored on target-test.
baseline 2 (upper bound) = same model type retrained on target-train, scored on the SAME target-test.
The gap between them is the most any adaptation method could hope to recover.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from src.data import make_domains, split_pools, tilt_cols
from src.models import MODEL_NAMES, make_model
from src.shifts import scenarios

FULL_DATASETS = ["adult", "covertype", "acs:CA>WY", "acs:CA>MS", "acs:CA>NY", "acs:CA>TX"]
N_TARGET = 8000  # rows per target pool (A and B) after shifting


def score(model, X, y):
    pred = model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def fit_upper(model, X, y):
    """Fit on target-train. A shifted pool can lose a class entirely, which XGBoost rejects
    (labels must be 0..K-1 without gaps), so we fit on compacted labels and map back."""
    classes, y_c = np.unique(y, return_inverse=True)
    fitted = clone(model).fit(X, y_c)

    class Mapped:
        def predict(self, Z):
            return classes[fitted.predict(Z)]
    return Mapped()


def run_block(dataset, seed, n_target, models):
    t0 = time.time()
    Xs, ys, Xt, yt = make_domains(dataset, seed)
    Xs_tr, ys_tr, Xs_te, ys_te = split_pools(Xs, ys, seed)
    Xt_A, yt_A, Xt_B, yt_B = split_pools(Xt, yt, seed)  # disjoint target pools
    fitted = {m: make_model(m, Xs_tr, seed).fit(Xs_tr, ys_tr) for m in models}
    acc_src = {m: score(fitted[m], Xs_te, ys_te)[0] for m in models}

    if dataset.startswith("acs"):
        # real shift: no dial, one scenario
        scen = [("real", 0.0, lambda X, y, rng, n: (X.iloc[:n].reset_index(drop=True), y[:n]))]
    else:
        scen = scenarios(tilt_cols(dataset))

    rows = []
    for i, (shift, sev, fn) in enumerate(scen):
        rng = np.random.default_rng((seed, i))  # same (seed, scenario) -> same data in Phase 2
        n = min(n_target, len(Xt_A), len(Xt_B))
        XA, yA = fn(Xt_A, yt_A, rng, n)
        XB, yB = fn(Xt_B, yt_B, rng, n)
        for m in models:
            acc_b, f1_b = score(fitted[m], XB, yB)
            upper = fit_upper(make_model(m, Xs_tr, seed), XA, yA)
            pred = upper.predict(XB)
            rows.append(dict(dataset=dataset, seed=seed, shift=shift, severity=sev, model=m,
                             acc_source_test=acc_src[m], acc_base=acc_b, f1_base=f1_b,
                             acc_upper=accuracy_score(yB, pred),
                             f1_upper=f1_score(yB, pred, average="macro")))
    print(f"  {dataset} seed={seed}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="1 seed, small pools, adult only")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--out", default="results/phase1.csv")
    a = ap.parse_args()

    datasets = a.datasets or (["adult"] if a.debug else FULL_DATASETS)
    seeds = [0] if a.debug else list(range(a.seeds))
    n_target = 2000 if a.debug else N_TARGET
    out = Path("results/phase1_debug.csv" if a.debug else a.out)
    out.parent.mkdir(exist_ok=True)

    rows = []
    for d in datasets:
        for s in seeds:
            rows += run_block(d, s, n_target, a.models)
            pd.DataFrame(rows).to_csv(out, index=False)  # save as we go: overnight-safe
    print("wrote", out)


if __name__ == "__main__":
    main()
