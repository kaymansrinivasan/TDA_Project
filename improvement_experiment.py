"""
improvement_experiment.py
Compares standard CART (alpha=0) against the proposed Imbalance-Aware
CART (alpha>0) on the UCI Adult dataset, focusing on minority-class
(>50K) detection.
"""
import numpy as np
import pandas as pd
from imbalance_aware_tree import WeightedCARTClassifier

CONT = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation",
       "relationship", "race", "sex", "native-country"]


def _harmonise(df):
    """Normalise column names so any Adult naming variant works
    (handles dots, underscores, 'educational-num', 'gender', 'class', etc.)."""
    df.columns = [c.strip().lower().replace("_", "-").replace(".", "-")
                  for c in df.columns]
    variants = {"educational-num": "education-num", "gender": "sex",
                "class": "income", "target": "income", "income-bracket": "income",
                "marital": "marital-status", "native": "native-country"}
    df = df.rename(columns={k: v for k, v in variants.items() if k in df.columns})
    for junk in ("x", "unnamed: 0", "index"):
        if junk in df.columns:
            df = df.drop(columns=[junk])
    return df


def load(path="adult.csv"):
    df = pd.read_csv(path, skipinitialspace=True)
    df = _harmonise(df)
    df = df.replace("?", np.nan).dropna()
    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)
    return df.reset_index(drop=True)


def encode(df):
    cols = []
    for c in CONT:
        if c in df.columns:
            cols.append(df[c].astype(float).to_numpy())
    for c in CAT:
        if c in df.columns:
            vals = {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
            cols.append(df[c].astype(str).map(vals).to_numpy(dtype=float))
    X = np.column_stack(cols)
    y = df["income"].to_numpy(np.int64)
    return X, y


def metrics(y_true, y_pred, pos=1):
    tp = int(np.sum((y_pred == pos) & (y_true == pos)))
    fp = int(np.sum((y_pred == pos) & (y_true != pos)))
    fn = int(np.sum((y_pred != pos) & (y_true == pos)))
    tn = int(np.sum((y_pred != pos) & (y_true != pos)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # macro-F1 over both classes
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = (f1 + f10) / 2
    return acc, prec, rec, f1, macro_f1, (tp, fp, fn, tn)


def main():
    df = load()
    X, y = encode(df)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    # use a manageable subsample for the from-scratch tree
    n = min(20000, len(y))
    X, y = X[:n], y[:n]
    cut = int(n * 0.7)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    pos_rate = y.mean()
    print(f"Dataset: {len(df)} usable rows; minority (>50K) rate = {pos_rate:.1%}")
    print(f"Using {n} rows ({cut} train / {n-cut} test)\n")
    print(f"{'Model':<22}{'Acc':>7}{'Min-Prec':>10}{'Min-Rec':>9}{'Min-F1':>8}{'MacroF1':>9}")
    print("-" * 65)

    configs = [("Standard CART (a=0)", 0.0),
               ("IA-CART (a=0.5)", 0.5),
               ("IA-CART (a=0.8)", 0.8),
               ("IA-CART (a=1.0)", 1.0)]
    results = {}
    for name, a in configs:
        m = WeightedCARTClassifier(imbalance_alpha=a, max_depth=12,
                                   min_samples_split=20).fit(Xtr, ytr)
        pred = m.predict(Xte)
        acc, prec, rec, f1, macro, cm = metrics(yte, pred)
        results[name] = (acc, prec, rec, f1, macro, cm)
        print(f"{name:<22}{acc:>7.3f}{prec:>10.3f}{rec:>9.3f}{f1:>8.3f}{macro:>9.3f}")

    print("\nConfusion (minority=>50K)  [tp, fp, fn, tn]")
    for name in results:
        print(f"  {name:<22}{results[name][5]}")


if __name__ == "__main__":
    main()