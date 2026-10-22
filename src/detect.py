"""Base paper: "Failing Loudly" (Rabanser et al., NeurIPS 2019) on tabular data.

Shift detection = dimensionality reduction (DR) + two-sample test.
  source reference sample  --DR-->  low-dim features  --test-->  p-value < alpha  =>  "shift!"
Representations: NoRed (preprocessed features), PCA, SRP (sparse random projection),
BBSDs (softmax of the trained model), BBSDh (its hard predictions), Classif (domain classifier).
Tests: KS per dimension + Bonferroni, MMD (multivariate, permutation), chi-squared (BBSDh),
binomial (domain classifier accuracy vs 0.5).
"""
import numpy as np
from scipy import stats
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.random_projection import SparseRandomProjection

ALPHA = 0.05
MMD_MAX_N = 1000  # kernel tests scale badly; like the paper we stop at 1000 target samples


def dense(Z):
    return Z.toarray() if issparse(Z) else np.asarray(Z, dtype=float)


# ---------------------------------------------------------------- two-sample tests
def ks_bonferroni(A, B):
    """One KS test per column; reject if the smallest p-value < alpha / K. Returns the
    Bonferroni-adjusted p-value min(1, K * min p) so it can be compared with alpha directly."""
    A, B = np.atleast_2d(A.T).T, np.atleast_2d(B.T).T
    # asymptotic p-values: scipy's exact mode (used up to 10,000 rows) is very slow and gives the same answer here
    ps = [stats.ks_2samp(A[:, j], B[:, j], method="asymp").pvalue for j in range(A.shape[1])]
    return min(1.0, A.shape[1] * float(np.min(ps)))


def mmd_perm(A, B, rng, n_perm=100):
    """MMD with an RBF kernel (bandwidth = median pairwise distance), p-value by permutation.
    The source sample is subsampled to the target size so both sides are balanced."""
    A = A[rng.choice(len(A), size=min(len(A), len(B)), replace=False)]
    Z = np.vstack([A, B])
    d2 = np.maximum((Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Z @ Z.T, 0)
    sigma2 = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    K = np.exp(-d2 / sigma2)
    n, m = len(A), len(B)
    sign = np.r_[np.full(n, 1 / n), np.full(m, -1 / m)]
    # MMD^2 (biased form) = a^T K a with a = +1/n on source rows, -1/m on target rows.
    perms = np.array([rng.permutation(sign) for _ in range(n_perm)])
    stat = sign @ K @ sign
    null = np.einsum("ij,ij->i", perms @ K, perms)
    return (1 + np.sum(null >= stat)) / (1 + n_perm)


def chi2_labels(a, b, n_classes):
    """Chi-squared test of homogeneity on predicted-class counts (BBSDh)."""
    table = np.array([np.bincount(a, minlength=n_classes), np.bincount(b, minlength=n_classes)])
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return stats.chi2_contingency(table)[1]


def domain_classifier(seed=0):
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=seed)


def c2st(A, B, rng, return_scores=False):
    """Classifier two-sample test (the paper's "Classif" detector).

    Train a classifier to tell source (0) from target (1) on one half of each sample, then check
    with a binomial test whether its accuracy on the other half is better than a coin flip.
    With return_scores, also give the held-out target rows (indices into B) and their
    P(target), which the paper uses to find the "most anomalous" samples.
    """
    m = min(len(A), len(B))
    ia, ib = rng.choice(len(A), m, replace=False), rng.choice(len(B), m, replace=False)
    h = m // 2
    p, scores = domain_test(A[ia[:h]], B[ib[:h]], A[ia[h:]], B[ib[h:]])
    return (p, ib[h:], scores) if return_scores else p


def domain_test(A_tr, B_tr, A_te, B_te):
    """Fit source-vs-target on the training halves, binomial test on the held-out halves.
    Returns the p-value and P(target) of the held-out target rows."""
    if len(A_tr) < 2:
        return 1.0, np.full(len(B_te), 0.5)
    clf = domain_classifier().fit(np.vstack([A_tr, B_tr]), np.r_[np.zeros(len(A_tr)), np.ones(len(B_tr))])
    prob = clf.predict_proba(np.vstack([A_te, B_te]))[:, 1]
    yte = np.r_[np.zeros(len(A_te)), np.ones(len(B_te))]
    correct = int(((prob > 0.5) == yte).sum())
    return stats.binomtest(correct, len(yte), 0.5, alternative="greater").pvalue, prob[len(A_te):]


# ---------------------------------------------------------------- the detector
class FailingLoudly:
    """All (representation, test) combinations of the paper, fitted on source data.

    model: a fitted sklearn Pipeline("prep", "clf") trained on source-train. Its preprocessor
    gives the NoRed features and its predict_proba gives BBSDs/BBSDh.
    """

    def __init__(self, model, X_train, X_ref, seed=0):
        self.model, self.seed = model, seed
        self.n_classes = len(model.classes_)
        F_train = self.features(X_train)
        k = min(32, F_train.shape[1])
        self.pca = PCA(n_components=k, random_state=seed).fit(F_train)
        self.srp = SparseRandomProjection(n_components=k, random_state=seed).fit(F_train)
        self.ref = self.represent(X_ref)

    def features(self, X):
        return dense(self.model.named_steps["prep"].transform(X))

    def represent(self, X):
        F = self.features(X)
        return {"NoRed": F, "PCA": self.pca.transform(F), "SRP": dense(self.srp.transform(F)),
                "BBSDs": self.model.predict_proba(X), "BBSDh": self.model.predict(X)}

    def run(self, X_target, rng):
        """p-value for every (representation, test) pair on one target sample."""
        tgt = self.represent(X_target)
        out = {}
        for rep in ("NoRed", "PCA", "SRP", "BBSDs"):
            out[(rep, "KS")] = ks_bonferroni(self.ref[rep], tgt[rep])
            if len(X_target) <= MMD_MAX_N:
                out[(rep, "MMD")] = mmd_perm(self.ref[rep], tgt[rep], rng)
        hard = lambda a: np.searchsorted(self.model.classes_, a)
        out[("BBSDh", "Chi2")] = chi2_labels(hard(self.ref["BBSDh"]), hard(tgt["BBSDh"]), self.n_classes)
        out[("Classif", "Binomial")] = c2st(self.ref["NoRed"], tgt["NoRed"], rng)
        return out

    def malignancy_labeled(self, X_target, y_target, rng, k=100):
        """Paper Sec. 3.4: label the k samples the domain classifier finds most target-like and
        measure the model's accuracy on them. Returns that accuracy estimate."""
        _, idx, scores = c2st(self.ref["NoRed"], self.features(X_target), rng, return_scores=True)
        top = idx[np.argsort(-scores)[:k]]
        return float((self.model.predict(X_target.iloc[top]) == y_target[top]).mean())
