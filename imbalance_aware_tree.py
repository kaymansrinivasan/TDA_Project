"""
imbalance_aware_tree.py
=================================================================
PROPOSED IMPROVEMENT (Part 2e):
Imbalance-Aware Decision Tree with Node-Adaptive Class Weighting
(IA-CART)

Standard ID3 / C4.5 / CART optimise OVERALL node purity. On an
imbalanced dataset (e.g. UCI Adult: ~76% <=50K, ~24% >50K) this
biases the tree toward the majority class and produces poor
minority-class recall.

This module modifies the CART splitting criterion so that the class
weight is applied DIRECTLY inside the Gini computation and is
RE-COMPUTED LOCALLY at every node from that node's own class
distribution. The strength of the correction therefore adapts to the
local imbalance:

    weighted count   m_c = n_c * (N_node / n_c)^alpha
                         = N_node^alpha * n_c^(1 - alpha)

    alpha = 0  -> m_c = n_c          (standard CART, no correction)
    alpha = 1  -> m_c = N_node / ... (fully balanced at the node)
    0<alpha<1  -> partial correction

Because (N_node / n_c) -> 1 when a node is already locally balanced,
the differential reweighting automatically fades out in balanced or
pure regions and only acts where the local imbalance is real. This
is the distinguishing feature compared with a single FIXED GLOBAL
class weight, which over-corrects already-balanced sub-regions.

Setting alpha = 0 reproduces ordinary CART exactly, so the two can be
compared on identical code.
=================================================================
"""

from __future__ import annotations
import numpy as np


class _Node:
    __slots__ = ("is_leaf", "prediction", "feature", "threshold",
                 "left", "right", "majority")

    def __init__(self):
        self.is_leaf = False
        self.prediction = None
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.majority = None


def weighted_counts(y, n_classes, alpha):
    """Node-adaptive weighted class counts m_c = N^alpha * n_c^(1-alpha)."""
    c = np.bincount(y, minlength=n_classes).astype(np.float64)
    N = c.sum()
    if N == 0 or alpha == 0.0:
        return c
    m = np.zeros_like(c)
    nz = c > 0
    m[nz] = (N ** alpha) * (c[nz] ** (1.0 - alpha))
    return m


def weighted_gini(y, n_classes, alpha):
    m = weighted_counts(y, n_classes, alpha)
    tot = m.sum()
    if tot == 0:
        return 0.0
    p = m / tot
    return float(1.0 - np.sum(p * p))


class WeightedCARTClassifier:
    """CART with node-adaptive class weighting.

    Parameters
    ----------
    imbalance_alpha : float in [0, 1]
        0   -> standard CART (no imbalance correction)
        >0  -> node-adaptive imbalance-aware splitting
    """

    def __init__(self, imbalance_alpha=0.0, max_depth=15,
                 min_samples_split=10, max_thresholds=32):
        self.alpha = float(imbalance_alpha)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None
        self.n_classes = None

    # ---- training ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(y.max()) + 1 if len(y) else 1
        self.root = self._build(X, y, depth=0)
        return self

    def _leaf(self, y):
        node = _Node()
        node.is_leaf = True
        # leaf predicts the WEIGHTED majority, so a boosted minority can win
        node.prediction = int(np.argmax(weighted_counts(y, self.n_classes, self.alpha)))
        return node

    def _candidate_thresholds(self, col):
        uniq = np.unique(col)
        if len(uniq) <= 1:
            return np.empty(0)
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if self.max_thresholds is not None and len(mids) > self.max_thresholds:
            idx = np.linspace(0, len(mids) - 1, self.max_thresholds).astype(int)
            mids = mids[idx]
        return mids

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split or depth >= self.max_depth
                or len(np.unique(y)) == 1):
            return self._leaf(y)

        base = weighted_gini(y, self.n_classes, self.alpha)
        total = len(y)
        best = None  # (gain, feature, threshold)

        for f in range(X.shape[1]):
            col = X[:, f]
            order = np.argsort(col)
            col_s = col[order]
            y_s = y[order]
            for thr in self._candidate_thresholds(col):
                left = col_s <= thr
                nl = int(left.sum())
                nr = total - nl
                if nl == 0 or nr == 0:
                    continue
                g = base - ((nl / total) * weighted_gini(y_s[left], self.n_classes, self.alpha)
                            + (nr / total) * weighted_gini(y_s[~left], self.n_classes, self.alpha))
                if best is None or g > best[0]:
                    best = (g, f, thr)

        if best is None or best[0] <= 0:
            return self._leaf(y)

        _, f, thr = best
        node = _Node()
        node.feature = f
        node.threshold = thr
        node.majority = int(np.argmax(weighted_counts(y, self.n_classes, self.alpha)))
        left_mask = X[:, f] <= thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # ---- prediction ----
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(r, self.root) for r in X])

    def _predict_row(self, row, node):
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction