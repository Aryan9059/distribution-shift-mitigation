"""Dataset loading and the leak-free source/target split.

load_dataset(name) returns (source_X, source_y, target_X, target_y, tilt_cols), or for the
simulated datasets one table that make_domains() splits into two halves.
Names: "adult", "covertype", "acs:CA>WY" (source state > target state).
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split

TILT_COLS = {
    "adult": ["age", "education-num"],
    "covertype": ["Elevation", "Horizontal_Distance_To_Roadways"],
    "acs": ["AGEP", "WKHP"],
}
# ACS columns that are codes, not quantities: "OCCP=4720" is not 4720 times "OCCP=1".
ACS_CATEGORICAL = ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
ACS_MAX_ROWS = 40_000


def load_adult():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    X = X.drop(columns=["fnlwgt"])
    for c in X.select_dtypes(include=["category", "object"]).columns:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X.reset_index(drop=True), (y.astype(str) == ">50K").to_numpy().astype(int)


def load_covertype(n=40_000, seed=0):
    X, y = fetch_covtype(as_frame=True, return_X_y=True)
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    # Labels are 1..7 originally; XGBoost needs 0..6.
    return X.iloc[idx].reset_index(drop=True), y.to_numpy()[idx].astype(int) - 1


def load_acs_state(state, year="2018"):
    from folktables import ACSDataSource, ACSIncome
    src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person", root_dir="data/acs")
    raw = src.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(raw)
    for c in ACS_CATEGORICAL:
        X[c] = X[c].astype(int).astype(str)
    X = X.reset_index(drop=True)
    y = y.iloc[:, 0].to_numpy().astype(int)
    if len(X) > ACS_MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(X), size=ACS_MAX_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y


def tilt_cols(name):
    return TILT_COLS["acs" if name.startswith("acs") else name]


def make_domains(name, seed):
    """Return (source_X, source_y, target_X, target_y).

    Adult/Covertype: two random stratified halves of the same table (the simulated shift is
    applied on top of the target half later). ACS: two different states.
    """
    if name.startswith("acs"):
        src_state, tgt_state = name.split(":")[1].split(">")
        return (*load_acs_state(src_state), *load_acs_state(tgt_state))
    X, y = load_adult() if name == "adult" else load_covertype()
    Xs, Xt, ys, yt = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    return (Xs.reset_index(drop=True), ys, Xt.reset_index(drop=True), yt)


def split_pools(X, y, seed):
    """Stratified 50/50 split into two disjoint parts (no row ends up in both)."""
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    return Xa.reset_index(drop=True), ya, Xb.reset_index(drop=True), yb
