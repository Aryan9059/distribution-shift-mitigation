"""Our final model: Detect-Diagnose-Adapt (D2A).

Failing Loudly stops at "a shift was detected". D2A wraps any trained classifier and, using only
UNLABELED target rows, also decides what kind of shift it is and applies the matching fix:
  1. Detect:    is there a shift?                    BBSDs + KS (the paper's best detector)
  2. Diagnose:  does the target contain NEW rows, unlike anything a re-weighting of the source
                would give?                                  -> off-support (drifted / corrupted features)
                else, does re-weighting by class explain it? -> label shift
                else                                         -> covariate shift
  3. Adapt:     off-support -> CORAL feature alignment
                label shift -> BBSE prior correction
                covariate   -> importance-weighted retraining (the safe default)
No detected shift -> the source model is kept unchanged.
ATC (Garg et al., 2022) estimates the target accuracy without labels and raises the "harmful"
alarm. Using that alarm as a gate (fix only if harmful) was our first design; it is kept as an
ablation (action_gated) because ATC misses many harmful shifts.
"""
import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from src.detect import ALPHA, domain_test, ks_bonferroni
from src.models import make_model

TAU = 0.02          # an estimated accuracy drop above 2 points counts as harmful
EXCESS_MAX = 0.05   # off-support if > 5 points more target rows are 'far' than re-weighting explains
ACTIONS = ("none", "bbse", "iw", "coral")


# ---------------------------------------------------------------- building blocks
def atc_threshold(probs_val, y_val, classes):
    """Average Thresholded Confidence: pick t so that, on source-validation, the share of rows
    with confidence > t equals the accuracy. On new data, that share estimates accuracy."""
    acc = (classes[probs_val.argmax(1)] == y_val).mean()
    return np.quantile(probs_val.max(1), 1 - acc)


def atc_estimate(probs, t):
    return float((probs.max(1) > t).mean())


def bbse_weights(pred_val, y_val, pred_target, n_classes):
    """Black Box Shift Estimation (Lipton et al., 2018). Under label shift,
    P_t(yhat = i) = sum_j P_s(yhat = i, y = j) * w_j with w_j = q(y=j) / p(y=j),
    so w solves a K x K linear system C w = mu."""
    C = np.zeros((n_classes, n_classes))
    np.add.at(C, (pred_val, y_val), 1)
    C /= len(y_val)
    mu = np.bincount(pred_target, minlength=n_classes) / len(pred_target)
    w = np.linalg.lstsq(C, mu, rcond=None)[0]
    return np.clip(w, 0, None)


class DensityRatio:
    """Covariate-shift weights w(x) = q(x)/p(x) = d(x) / (1 - d(x)) for a balanced logistic-regression
    domain classifier d, normalised to mean 1. d is clipped to [0.05, 0.95] so a few rows cannot dominate."""

    def __init__(self, F_ref, F_target, seed=0):
        m = min(len(F_ref), len(F_target))
        rng = np.random.default_rng(seed)
        X = np.vstack([F_ref[rng.choice(len(F_ref), m, replace=False)],
                       F_target[rng.choice(len(F_target), m, replace=False)]])
        self.clf = LogisticRegression(max_iter=1000).fit(X, np.r_[np.zeros(m), np.ones(m)])

    def weights(self, F):
        d = np.clip(self.clf.predict_proba(F)[:, 1], 0.05, 0.95)
        w = d / (1 - d)
        return w / w.mean()


class Coral:
    """CORAL (Sun et al., 2016) on the numeric columns: re-colour the target so its mean and
    covariance match the source, i.e. undo the shift in feature space before predicting."""

    def __init__(self, X_source, X_target, eps=0.1):
        self.cols = list(X_source.select_dtypes(include="number").columns)
        S = X_source[self.cols].to_numpy(float)
        self.mu_s, sd = S.mean(0), S.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)  # constant columns stay as they are
        Zs = (S - self.mu_s) / self.sd
        Zt = (X_target[self.cols].to_numpy(float) - self.mu_s) / self.sd
        self.mu_t = Zt.mean(0)
        I = np.eye(len(self.cols))
        self.A = _mat_pow(np.cov(Zt, rowvar=False) + eps * I, -0.5) @ _mat_pow(np.cov(Zs, rowvar=False) + eps * I, 0.5)

    def transform(self, X):
        X = X.copy()
        Z = (X[self.cols].to_numpy(float) - self.mu_s) / self.sd
        X[self.cols] = ((Z - self.mu_t) @ self.A) * self.sd + self.mu_s
        return X


def _mat_pow(M, p):
    vals, vecs = np.linalg.eigh(M)
    return (vecs * np.maximum(vals, 1e-8) ** p) @ vecs.T


def _dense(Z):
    return Z.toarray() if issparse(Z) else np.asarray(Z, dtype=float)


# ---------------------------------------------------------------- the final model
class D2A:
    """Detect-Diagnose-Adapt wrapper around one model type ("LR", "RF", "GBT", "MLP").

    fit(source)            trains the source model and stores what it needs from validation data
    adapt(X_target)        looks at UNLABELED target rows and decides on an action
    predict(X)             predicts with the chosen fix
    fixed(action)          a predictor for any action (used to compare against "always do X")
    """

    def __init__(self, name, seed=0, alpha=ALPHA, tau=TAU, excess_max=EXCESS_MAX):
        self.name, self.seed, self.alpha, self.tau, self.excess_max = name, seed, alpha, tau, excess_max

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.model = make_model(self.name, X_train, self.seed).fit(X_train, y_train)
        self.classes = self.model.classes_
        self.probs_val = self.model.predict_proba(X_val)
        self.acc_val = float((self.classes[self.probs_val.argmax(1)] == y_val).mean())
        self.t_atc = atc_threshold(self.probs_val, y_val, self.classes)
        self.F_val = self._features(X_val)
        # A row is "far" if its nearest source row is farther away than for 99% of source rows
        # (for source rows: the 2nd neighbour, because the nearest one is the row itself).
        self.nn = NearestNeighbors(n_neighbors=2).fit(self.F_val)
        d_val = self.nn.kneighbors(self.F_val)[0][:, 1]
        self.d99 = np.quantile(d_val, 0.99)
        self.far_val = (d_val > self.d99).astype(float)
        return self

    def _features(self, X):
        return _dense(self.model.named_steps["prep"].transform(X))

    def _idx(self, labels):
        return np.searchsorted(self.classes, labels)

    def adapt(self, X_target):
        self.X_target, self._cache = X_target, {}
        self.rng = np.random.default_rng(self.seed)
        probs_t = self.model.predict_proba(X_target)
        self.F_t = self._features(X_target)
        info = dict(p_detect=ks_bonferroni(self.probs_val, probs_t),
                    est_acc=atc_estimate(probs_t, self.t_atc), p_label_check=np.nan, far_target=np.nan,
                    far_reweighted=np.nan)
        info["detected"] = bool(info["p_detect"] < self.alpha)
        info["harmful"] = bool(info["detected"] and self.acc_val - info["est_acc"] > self.tau)
        info["diagnosis"], info["action"] = "none", "none"
        if info["detected"]:
            # Off-support check: share of far target rows minus the share a re-weighted source would
            # have. Re-weighting (covariate or label shift) only moves mass between existing rows, so a
            # large excess means the target holds new kinds of rows, which only feature alignment fixes.
            w = self._ratio().weights(self.F_val)
            info["far_target"] = float((self.nn.kneighbors(self.F_t, n_neighbors=1)[0][:, 0] > self.d99).mean())
            info["far_reweighted"] = float((w * self.far_val).sum() / w.sum())
            if info["far_target"] - info["far_reweighted"] > self.excess_max:
                info["diagnosis"], info["action"] = "off-support", "coral"
            else:
                info["p_label_check"] = self._explains(self._bbse()[self._idx(self.y_val)])
                if info["p_label_check"] >= self.alpha:
                    info["diagnosis"], info["action"] = "label", "bbse"
                else:
                    info["diagnosis"], info["action"] = "covariate", "iw"
        info["action_gated"] = info["action"] if info["harmful"] else "none"  # ablation: first design
        self.info, self.action = info, info["action"]
        return info

    def _explains(self, weights):
        """Resample source-validation rows with the given per-row weights and test whether a domain
        classifier can still tell them from the target. A high p-value = the re-weighting explains
        the shift. Each half is resampled separately so that a duplicated row never appears in both
        the training and the test half of the classifier (it would be easy to recognise)."""
        rng = self.rng
        halves = np.array_split(rng.permutation(len(weights)), 2)
        t = np.array_split(rng.permutation(len(self.F_t)), 2)
        a = [rng.choice(h, size=len(tt), p=weights[h] / weights[h].sum()) for h, tt in zip(halves, t)]
        return domain_test(self.F_val[a[0]], self.F_t[t[0]], self.F_val[a[1]], self.F_t[t[1]])[0]

    # ---- the three fixes (computed lazily, cached per target sample)
    def _ratio(self):
        if "ratio" not in self._cache:
            self._cache["ratio"] = DensityRatio(self.F_val, self.F_t, self.seed)
        return self._cache["ratio"]

    def _bbse(self):
        if "w" not in self._cache:
            self._cache["w"] = bbse_weights(self._idx(self.model.predict(self.X_val)), self._idx(self.y_val),
                                            self._idx(self.model.predict(self.X_target)), len(self.classes))
        return self._cache["w"]

    def fixed(self, action):
        """Return a predict(X) function for the given action."""
        if action not in self._cache:
            if action == "none":
                fn = self.model.predict
            elif action == "bbse":
                w = self._bbse()
                fn = lambda X: self.classes[(self.model.predict_proba(X) * w).argmax(1)]
            elif action == "iw":
                weights = self._ratio().weights(self._features(self.X_train))
                m = make_model(self.name, self.X_train, self.seed)
                m.fit(self.X_train, self.y_train, clf__sample_weight=weights)
                fn = m.predict
            elif action == "coral":
                coral = Coral(self.X_train, self.X_target)
                fn = lambda X: self.model.predict(coral.transform(X))
            self._cache[action] = fn
        return self._cache[action]

    def predict(self, X):
        return self.fixed(self.action)(X)
