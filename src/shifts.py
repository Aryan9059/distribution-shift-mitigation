"""Shift simulator.

Every function has the signature (X, y, rng, n, ...) and returns (X_shifted, y_shifted)
with exactly n rows. X is a DataFrame, y is an integer array with labels 0..K-1.

WHY a simulator: with simulated shifts we know the ground truth (which kind of shift,
how strong), so we can later check whether detectors and adaptation methods behave.
"""
import numpy as np
import pandas as pd


def _numeric_cols(X):
    return list(X.select_dtypes(include="number").columns)


def _reset(X, y):
    return X.reset_index(drop=True), np.asarray(y)


def identity(X, y, rng, n):
    """No shift: plain resample. Needed as the control for false-alarm checks."""
    idx = rng.choice(len(X), size=n, replace=len(X) < n)
    return _reset(X.iloc[idx], y[idx])


def covariate(X, y, rng, n, cols, s):
    """Covariate shift by exponential tilting.

    Each row is sampled with probability proportional to exp(s * z), where z is the sum of
    the standardised tilt columns. Larger s = stronger push toward high values of those
    columns. P(y|x) is untouched, so this is pure covariate shift.
    """
    z = sum((X[c] - X[c].mean()) / (X[c].std() + 1e-12) for c in cols).to_numpy()
    z = z / np.sqrt(len(cols))  # keep the scale comparable whatever the number of columns
    p = np.exp(s * (z - z.max()))
    p /= p.sum()
    idx = rng.choice(len(X), size=n, replace=True, p=p)
    return _reset(X.iloc[idx], y[idx])


def noise(X, y, rng, n, s, delta=0.5):
    """Gaussian noise (std = s * column std) added to numeric columns of a fraction delta of rows."""
    Xs, ys = identity(X, y, rng, n)
    cols = _numeric_cols(Xs)
    Xs[cols] = Xs[cols].astype(float)  # int columns cannot hold the noisy values
    rows = rng.random(n) < delta
    for c in cols:
        sd = Xs[c].std()
        Xs.loc[rows, c] = Xs.loc[rows, c].astype(float) + rng.normal(0, s * sd, rows.sum())
    return Xs, ys


def drift(X, y, rng, n, s, delta=0.5):
    """Mean shift of s column-stds on numeric columns of a fraction delta of rows."""
    Xs, ys = identity(X, y, rng, n)
    Xs[_numeric_cols(Xs)] = Xs[_numeric_cols(Xs)].astype(float)
    rows = rng.random(n) < delta
    for c in _numeric_cols(Xs):
        Xs.loc[rows, c] = Xs.loc[rows, c].astype(float) + s * Xs[c].std()
    return Xs, ys


def label(X, y, rng, n, s):
    """Label shift: new class priors q = (1-s)*p + s/K, then sample rows per class.

    s=0 keeps the priors, s=1 makes the classes balanced. P(x|y) is untouched.
    Sampling is with replacement because rare classes may not have enough rows.
    """
    classes = np.unique(y)
    K = len(classes)
    p = np.array([(y == k).mean() for k in classes])
    q = (1 - s) * p + s / K
    counts = rng.multinomial(n, q)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == k), size=c, replace=True)
        for k, c in zip(classes, counts) if c > 0
    ])
    rng.shuffle(idx)
    return _reset(X.iloc[idx], y[idx])


def concept(X, y, rng, n, col, s):
    """Concept drift: P(y|x) changes. Rows above the median of `col` get their label moved
    to a different class with probability s. P(x) is unchanged."""
    Xs, ys = identity(X, y, rng, n)
    ys = ys.copy()
    classes = np.unique(y)
    hit = (Xs[col] > Xs[col].median()).to_numpy() & (rng.random(n) < s)
    for i in np.flatnonzero(hit):
        ys[i] = rng.choice(classes[classes != ys[i]])
    return Xs, ys


def combo(X, y, rng, n, cols, s_cov, s_lab):
    """Covariate shift followed by label shift."""
    Xc, yc = covariate(X, y, rng, n, cols, s_cov)
    return label(Xc, yc, rng, n, s_lab)


def scenarios(cols):
    """Fixed, ordered list of (shift, severity, fn). The list index is part of the rng seed,
    so Phase 2 gets exactly the same scenarios as Phase 1. Only ever append to this list."""
    out = [("identity", 0.0, lambda X, y, rng, n: identity(X, y, rng, n))]
    for s in (0.5, 1, 1.5, 2, 3):
        out.append(("covariate", s, lambda X, y, rng, n, s=s: covariate(X, y, rng, n, cols, s)))
    for s in (0.25, 0.5, 1, 2):
        out.append(("noise", s, lambda X, y, rng, n, s=s: noise(X, y, rng, n, s)))
    for s in (0.25, 0.5, 1, 2):
        out.append(("drift", s, lambda X, y, rng, n, s=s: drift(X, y, rng, n, s)))
    for s in (0.25, 0.5, 0.75, 1.0):
        out.append(("label", s, lambda X, y, rng, n, s=s: label(X, y, rng, n, s)))
    for s in (0.1, 0.2, 0.4):
        out.append(("concept", s, lambda X, y, rng, n, s=s: concept(X, y, rng, n, cols[0], s)))
    for s in (1, 2):
        out.append(("combo", s, lambda X, y, rng, n, s=s: combo(X, y, rng, n, cols, s, 0.5)))
    return out
