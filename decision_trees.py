"""
decision_trees.py
=================================================================
From-scratch implementations of the three decision-tree induction
algorithms compared in this project:

    * ID3   (Quinlan, 1986)  - Information Gain (entropy), multi-way
                               splits on CATEGORICAL attributes only.
    * C4.5  (Quinlan, 1993)  - Gain Ratio; multi-way splits on
                               categorical attributes, BINARY threshold
                               splits on continuous attributes.
    * CART  (Breiman et al., 1984) - Gini impurity, strictly BINARY
                               splits on every attribute.

All three share ONE tree-building framework (this file) so that the
execution-time comparison in the experiment reflects genuine
*algorithmic* differences (split criterion + split type), NOT
differences in programming language or library optimisation.
This is required for a fair theoretical-vs-experimental comparison.

-----------------------------------------------------------------
DESIGN NOTE / CITATION
The structure below (recursive top-down induction, entropy / gain
ratio / gini split criteria) follows the standard textbook
formulations of the three algorithms:
    J. R. Quinlan, "Induction of decision trees," Machine Learning,
        vol. 1, no. 1, pp. 81-106, 1986.                      [ID3]
    J. R. Quinlan, C4.5: Programs for Machine Learning.
        Morgan Kaufmann, 1993.                                [C4.5]
    L. Breiman, J. Friedman, R. Olshen, C. Stone, Classification
        and Regression Trees. Wadsworth, 1984.                [CART]
The code itself is written by the group; replace this note with
your own citation policy as required by the assignment.
-----------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Tree node
# ----------------------------------------------------------------------
class _Node:
    """A single node. Either a leaf (prediction set) or an internal split."""
    __slots__ = ("is_leaf", "prediction", "feature", "kind",
                 "threshold", "children", "majority")

    def __init__(self):
        self.is_leaf = False
        self.prediction = None      # class label if leaf
        self.feature = None         # index of the feature this node splits on
        self.kind = None            # "categorical" or "numeric"
        self.threshold = None       # float, for numeric binary splits
        self.children = {}          # dict: value/"left"/"right" -> _Node
        self.majority = None        # fallback class for unseen branches


# ----------------------------------------------------------------------
# Impurity helpers (vectorised with numpy)
# ----------------------------------------------------------------------
def _counts(y, n_classes):
    return np.bincount(y, minlength=n_classes)


def _entropy(y, n_classes):
    c = _counts(y, n_classes)
    total = c.sum()
    if total == 0:
        return 0.0
    p = c[c > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _gini(y, n_classes):
    c = _counts(y, n_classes)
    total = c.sum()
    if total == 0:
        return 0.0
    p = c / total
    return float(1.0 - np.sum(p * p))


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class _BaseTree:
    """
    Shared recursive induction engine.

    Parameters
    ----------
    max_depth : int       hard depth cap (pre-pruning, controls runtime)
    min_samples_split : int   a node with fewer samples becomes a leaf
    max_thresholds : int  for numeric features, evaluate at most this many
                          candidate thresholds (quantile-spaced). This is a
                          standard speed optimisation used by real
                          implementations; set to None to test every
                          midpoint.
    """

    def __init__(self, max_depth=15, min_samples_split=2, max_thresholds=32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None
        self.n_classes = None
        # feature_types[i] == "categorical" or "numeric"
        self.feature_types = None

    # -- to be overridden by each algorithm -----------------------------
    def _score_categorical(self, X_col, y):
        """Return (score, None) for a multi-way categorical split, or None."""
        raise NotImplementedError

    def _score_numeric(self, X_col, y):
        """Return (score, threshold) for the best binary numeric split, or None."""
        raise NotImplementedError

    def _allow_categorical(self):
        """CART forbids native multi-way categorical splits."""
        return True

    # -- training -------------------------------------------------------
    def fit(self, X, y, feature_types):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(y.max()) + 1 if len(y) else 1
        self.feature_types = list(feature_types)
        available = list(range(X.shape[1]))
        self.root = self._build(X, y, available, depth=0)
        return self

    def _make_leaf(self, y):
        node = _Node()
        node.is_leaf = True
        node.prediction = int(np.bincount(y, minlength=self.n_classes).argmax())
        return node

    def _build(self, X, y, available, depth):
        # stopping conditions ------------------------------------------
        if (len(y) < self.min_samples_split
                or depth >= self.max_depth
                or len(np.unique(y)) == 1
                or len(available) == 0):
            return self._make_leaf(y)

        best = None  # (score, feature, kind, threshold)
        for f in available:
            kind = self.feature_types[f]
            col = X[:, f]
            if kind == "categorical" and self._allow_categorical():
                res = self._score_categorical(col, y)
                if res is not None:
                    score, _ = res
                    if best is None or score > best[0]:
                        best = (score, f, "categorical", None)
            else:
                # numeric, OR categorical handled as numeric (CART)
                col_num = col.astype(np.float64)
                res = self._score_numeric(col_num, y)
                if res is not None:
                    score, thr = res
                    if best is None or score > best[0]:
                        best = (score, f, "numeric", thr)

        # no useful split found
        if best is None or best[0] <= 0:
            return self._make_leaf(y)

        score, f, kind, thr = best
        node = _Node()
        node.feature = f
        node.kind = kind
        node.majority = int(np.bincount(y, minlength=self.n_classes).argmax())

        if kind == "categorical":
            # multi-way split; that feature is consumed on this path (ID3/C4.5)
            col = X[:, f]
            remaining = [a for a in available if a != f]
            for v in np.unique(col):
                mask = col == v
                node.children[v] = self._build(X[mask], y[mask], remaining, depth + 1)
        else:
            # binary threshold split; numeric features may be reused below
            node.threshold = thr
            col_num = X[:, f].astype(np.float64)
            left_mask = col_num <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                return self._make_leaf(y)
            node.children["left"] = self._build(
                X[left_mask], y[left_mask], available, depth + 1)
            node.children["right"] = self._build(
                X[right_mask], y[right_mask], available, depth + 1)
        return node

    # -- numeric threshold candidate generation (shared) ----------------
    def _candidate_thresholds(self, col_num):
        uniq = np.unique(col_num)
        if len(uniq) <= 1:
            return np.empty(0)
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if self.max_thresholds is not None and len(mids) > self.max_thresholds:
            idx = np.linspace(0, len(mids) - 1, self.max_thresholds).astype(int)
            mids = mids[idx]
        return mids

    # -- prediction -----------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=object)
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row, node):
        while not node.is_leaf:
            if node.kind == "categorical":
                v = row[node.feature]
                nxt = node.children.get(v)
                if nxt is None:               # unseen category at this node
                    return node.majority
                node = nxt
            else:
                val = float(row[node.feature])
                node = node.children["left"] if val <= node.threshold \
                    else node.children["right"]
        return node.prediction


# ----------------------------------------------------------------------
# ID3 - Information Gain, categorical multi-way only
# ----------------------------------------------------------------------
class ID3Classifier(_BaseTree):
    """Quinlan (1986). Requires all features to be categorical
    (continuous attributes must be discretised before fitting)."""

    def _score_categorical(self, col, y):
        base = _entropy(y, self.n_classes)
        total = len(y)
        weighted = 0.0
        for v in np.unique(col):
            mask = col == v
            weighted += (mask.sum() / total) * _entropy(y[mask], self.n_classes)
        return (base - weighted, None)

    def _score_numeric(self, col_num, y):
        # ID3 has no native continuous handling. If a numeric column slips
        # through, treat each distinct value as a category (rarely used,
        # because the ID3 pipeline discretises beforehand).
        base = _entropy(y, self.n_classes)
        total = len(y)
        weighted = 0.0
        for v in np.unique(col_num):
            mask = col_num == v
            weighted += (mask.sum() / total) * _entropy(y[mask], self.n_classes)
        return (base - weighted, None)


# ----------------------------------------------------------------------
# C4.5 - Gain Ratio, categorical multi-way + numeric binary threshold
# ----------------------------------------------------------------------
class C45Classifier(_BaseTree):
    """Quinlan (1993). Gain ratio = information gain / split information."""

    @staticmethod
    def _split_info(group_sizes, total):
        p = group_sizes[group_sizes > 0] / total
        return float(-np.sum(p * np.log2(p)))

    def _score_categorical(self, col, y):
        base = _entropy(y, self.n_classes)
        total = len(y)
        weighted = 0.0
        sizes = []
        for v in np.unique(col):
            mask = col == v
            s = mask.sum()
            sizes.append(s)
            weighted += (s / total) * _entropy(y[mask], self.n_classes)
        gain = base - weighted
        si = self._split_info(np.array(sizes), total)
        if si == 0:
            return (0.0, None)
        return (gain / si, None)

    def _score_numeric(self, col_num, y):
        base = _entropy(y, self.n_classes)
        total = len(y)
        best = None
        order = np.argsort(col_num)
        col_sorted = col_num[order]
        y_sorted = y[order]
        for thr in self._candidate_thresholds(col_num):
            left = col_sorted <= thr
            nl = left.sum()
            nr = total - nl
            if nl == 0 or nr == 0:
                continue
            wl = nl / total
            wr = nr / total
            gain = base - (wl * _entropy(y_sorted[left], self.n_classes)
                           + wr * _entropy(y_sorted[~left], self.n_classes))
            si = self._split_info(np.array([nl, nr]), total)
            if si == 0:
                continue
            ratio = gain / si
            if best is None or ratio > best[0]:
                best = (ratio, thr)
        return best


# ----------------------------------------------------------------------
# CART - Gini impurity, strictly binary splits on every feature
# ----------------------------------------------------------------------
class CARTClassifier(_BaseTree):
    """Breiman et al. (1984). All features are evaluated as binary
    threshold splits (categorical features are integer-encoded upstream,
    mirroring how scikit-learn's CART treats them)."""

    def _allow_categorical(self):
        return False     # force every feature through the numeric (binary) path

    def _score_categorical(self, col, y):     # never called
        return None

    def _score_numeric(self, col_num, y):
        base = _gini(y, self.n_classes)
        total = len(y)
        best = None
        order = np.argsort(col_num)
        col_sorted = col_num[order]
        y_sorted = y[order]
        for thr in self._candidate_thresholds(col_num):
            left = col_sorted <= thr
            nl = left.sum()
            nr = total - nl
            if nl == 0 or nr == 0:
                continue
            wl = nl / total
            wr = nr / total
            gini_split = (wl * _gini(y_sorted[left], self.n_classes)
                          + wr * _gini(y_sorted[~left], self.n_classes))
            gain = base - gini_split        # Gini gain (>0 is good)
            if best is None or gain > best[0]:
                best = (gain, thr)
        return best


# Convenience registry used by the experiment script
ALGORITHMS = {
    "ID3": ID3Classifier,
    "C4.5": C45Classifier,
    "CART": CARTClassifier,
}