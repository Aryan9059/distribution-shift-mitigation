"""The four models, each a Pipeline = preprocessing + classifier.

Putting preprocessing INSIDE the pipeline means .fit() learns scaling/encoding only from the
data it is given. We only ever fit on source-train, so nothing leaks from the target.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ["LR", "RF", "GBT", "MLP"]


def make_preprocessor(X):
    num = list(X.select_dtypes(include="number").columns)
    cat = [c for c in X.columns if c not in num]
    # infrequent_if_exist: categories unseen (or rare) at fit time go into an "infrequent"
    # bucket instead of crashing when a shifted target contains new values.
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat),
    ])


def make_model(name, X, seed=0):
    clf = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "GBT": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             n_jobs=-1, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             max_iter=200, random_state=seed),
    }[name]
    return Pipeline([("prep", make_preprocessor(X)), ("clf", clf)])
