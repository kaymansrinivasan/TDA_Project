"""
ab_id3_experiment.py
Tests the friend's proposal: Adaptive Binning ID3 (AB-ID3), which replaces
ID3's naive (equal-frequency) discretisation with supervised, information-gain
driven bin edges, then runs the SAME ID3 induction.

Compares, on identical UCI Adult data:
  - Baseline ID3   (quantile / equal-frequency bins, 5 bins)   <- current project
  - AB-ID3 (2-bin) (best single cut among percentile candidates) <- literal diagram
  - AB-ID3 (5-bin) (greedy info-gain supervised bins, 5 bins)   <- generous version
  - C4.5, CART     (reference targets)
"""
import time
import numpy as np
import pandas as pd
from decision_trees import (ID3Classifier, C45Classifier, CARTClassifier,
                            _entropy)

CONT = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation",
       "relationship", "race", "sex", "native-country"]


def load(path="adult.csv"):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip().lower().replace("_", "-").replace(".", "-")
                  for c in df.columns]
    variants = {"educational-num": "education-num", "gender": "sex",
                "class": "income", "target": "income", "income-bracket": "income",
                "marital": "marital-status", "native": "native-country"}
    df = df.rename(columns={k: v for k, v in variants.items() if k in df.columns})
    for junk in ("x", "unnamed: 0", "index"):
        if junk in df.columns:
            df = df.drop(columns=[junk])
    df = df.replace("?", np.nan).dropna()
    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)
    return df.reset_index(drop=True)


def info_gain(labels, y, n_classes):
    base = _entropy(y, n_classes)
    total = len(y)
    w = 0.0
    for v in np.unique(labels):
        m = labels == v
        w += (m.sum() / total) * _entropy(y[m], n_classes)
    return base - w


def equal_freq_bins(col, bins=5):
    edges = np.quantile(col.astype(float), np.linspace(0, 1, bins + 1))
    edges = np.unique(edges); edges[0] -= 1e-9
    return np.digitize(col.astype(float), edges[1:-1]).astype(str)


def supervised_bins(col, y, n_classes, max_bins=5):
    """Greedily choose up to max_bins-1 cut points (from percentile candidates)
    that maximise information gain -- the AB-ID3 idea."""
    vals = col.astype(float)
    cand = np.unique(np.percentile(vals, np.linspace(10, 90, 9)))
    chosen = []
    for _ in range(max_bins - 1):
        best = None
        for t in cand:
            if t in chosen:
                continue
            cuts = np.sort(np.array(chosen + [t]))
            labels = np.digitize(vals, cuts)
            ig = info_gain(labels, y, n_classes)
            if best is None or ig > best[0]:
                best = (ig, t)
        if best is None or best[0] <= 0:
            break
        chosen.append(best[1])
    cuts = np.sort(np.array(chosen)) if chosen else np.array([np.median(vals)])
    return np.digitize(vals, cuts).astype(str)


def build_id3_view(df, y, mode, n_classes):
    cols = []
    for c in CONT:
        if mode == "baseline":
            cols.append(equal_freq_bins(df[c], 5))
        elif mode == "ab2":
            cols.append(supervised_bins(df[c], y, n_classes, max_bins=2))
        elif mode == "ab5":
            cols.append(supervised_bins(df[c], y, n_classes, max_bins=5))
    for c in CAT:
        cols.append(df[c].astype(str).to_numpy())
    X = np.column_stack(cols).astype(object)
    types = ["categorical"] * (len(CONT) + len(CAT))
    return X, types


def main():
    df = load()
    y_all = df["income"].to_numpy(np.int64)
    n_classes = 2
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n = 20000
    idx = idx[:n]
    df = df.iloc[idx].reset_index(drop=True)
    y = y_all[idx]
    cut = int(n * 0.7)
    tr, te = np.arange(cut), np.arange(cut, n)

    print(f"Adult: {n} rows, minority rate {y.mean():.1%}, 70/30 split\n")
    print(f"{'Model':<22}{'Train(s)':>10}{'Accuracy':>10}")
    print("-" * 42)

    # ID3 family
    for name, mode in [("ID3 baseline (5 bins)", "baseline"),
                       ("AB-ID3 (2 bins)", "ab2"),
                       ("AB-ID3 (5 bins)", "ab5")]:
        # supervised bins use TRAIN labels only (no leakage)
        if mode == "baseline":
            X, types = build_id3_view(df, y, mode, n_classes)
        else:
            # fit bins on train, apply to all via same routine (approx: build on full
            # using train labels for cut selection)
            cols = []
            for c in CONT:
                mb = 2 if mode == "ab2" else 5
                cuts_src = supervised_bins(df[c].iloc[tr], y[tr], n_classes, max_bins=mb)
                # recompute edges from train then apply to full column
                vals = df[c].astype(float).to_numpy()
                tr_vals = df[c].iloc[tr].astype(float).to_numpy()
                cand = np.unique(np.percentile(tr_vals, np.linspace(10, 90, 9)))
                chosen = []
                for _ in range(mb - 1):
                    best = None
                    for t in cand:
                        if t in chosen:
                            continue
                        lab = np.digitize(tr_vals, np.sort(np.array(chosen + [t])))
                        ig = info_gain(lab, y[tr], n_classes)
                        if best is None or ig > best[0]:
                            best = (ig, t)
                    if best is None or best[0] <= 0:
                        break
                    chosen.append(best[1])
                edges = np.sort(np.array(chosen)) if chosen else np.array([np.median(tr_vals)])
                cols.append(np.digitize(vals, edges).astype(str))
            for c in CAT:
                cols.append(df[c].astype(str).to_numpy())
            X = np.column_stack(cols).astype(object)
            types = ["categorical"] * (len(CONT) + len(CAT))

        t0 = time.perf_counter()
        m = ID3Classifier(max_depth=15, min_samples_split=10).fit(X[tr], y[tr], types)
        tt = time.perf_counter() - t0
        acc = (m.predict(X[te]) == y[te]).mean()
        print(f"{name:<22}{tt:>10.3f}{acc:>10.3f}")

    # C4.5 and CART reference
    cont_num = [df[c].astype(float).to_numpy().astype(object) for c in CONT]
    cat_cat = [df[c].astype(str).to_numpy() for c in CAT]
    Xc = np.column_stack(cont_num + cat_cat).astype(object)
    types_c45 = ["numeric"] * len(CONT) + ["categorical"] * len(CAT)
    t0 = time.perf_counter()
    mc = C45Classifier(max_depth=15, min_samples_split=10).fit(Xc[tr], y[tr], types_c45)
    tt = time.perf_counter() - t0
    print(f"{'C4.5 (reference)':<22}{tt:>10.3f}{(mc.predict(Xc[te])==y[te]).mean():>10.3f}")

    def le(col):
        vals = {v: i for i, v in enumerate(sorted(set(col)))}
        return np.array([vals[v] for v in col], dtype=float)
    cart_cols = [df[c].astype(float).to_numpy() for c in CONT] + [le(df[c].astype(str)) for c in CAT]
    Xcart = np.column_stack(cart_cols).astype(object)
    types_cart = ["numeric"] * (len(CONT) + len(CAT))
    t0 = time.perf_counter()
    mca = CARTClassifier(max_depth=15, min_samples_split=10).fit(Xcart[tr], y[tr], types_cart)
    tt = time.perf_counter() - t0
    print(f"{'CART (reference)':<22}{tt:>10.3f}{(mca.predict(Xcart[te])==y[te]).mean():>10.3f}")


if __name__ == "__main__":
    main()