"""
experiment.py
=================================================================
Part 2 experiment driver for:
"Performance Analysis of Decision Tree Variants in Classification"
(ID3 vs C4.5 vs CART on the UCI Adult / Census Income dataset).

WHAT THIS SCRIPT DOES
  1. Loads the UCI Adult dataset.
  2. Builds three preprocessed "views" of the data, one per algorithm:
        ID3  : all features categorical  (continuous -> quantile bins)
        C4.5 : continuous numeric + categorical kept as categories
        CART : all features numeric      (categoricals integer-encoded)
  3. Subsamples the data at increasing input sizes.
  4. Trains each algorithm at each size, recording execution time
     (averaged over several repeats) and test accuracy.
  5. Saves results to results.csv and plots execution time vs input
     size to execution_time_vs_size.png  (the graph required by 2c).

HOW TO GET THE DATASET (pick ONE)
  Option A (recommended) -- automatic download via the official UCI package:
        pip install ucimlrepo
     then just run:  python experiment.py
  Option B -- scikit-learn / OpenML:
        the script falls back to fetch_openml('adult') automatically.
  Option C -- offline:
        download "adult.csv" from
        https://archive.ics.uci.edu/dataset/2/adult
        place it next to this script, and run:
        python experiment.py --csv adult.csv
=================================================================
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decision_trees import ID3Classifier, C45Classifier, CARTClassifier

# Adult columns. The target is "income" (>50K vs <=50K).
CONTINUOUS = ["age", "fnlwgt", "education-num", "capital-gain",
              "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
TARGET = "income"


# ----------------------------------------------------------------------
# 1. Data loading (three fallbacks)
# ----------------------------------------------------------------------
def load_adult(csv_path=None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        return _normalise_columns(df)

    # Option A: official UCI repo package
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        df = adult.data.features.copy()
        df[TARGET] = adult.data.targets.values.ravel()
        print("Loaded Adult via ucimlrepo.")
        return _normalise_columns(df)
    except Exception as e:
        print(f"ucimlrepo unavailable ({e}); trying OpenML...")

    # Option B: OpenML through scikit-learn
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"class": TARGET})
    print("Loaded Adult via fetch_openml.")
    return _normalise_columns(df)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower().replace("_", "-")
                            for c in df.columns})
    # harmonise a couple of common name variants
    df = df.rename(columns={"educationnum": "education-num",
                            "capitalgain": "capital-gain",
                            "capitalloss": "capital-loss",
                            "hoursperweek": "hours-per-week",
                            "marital-status": "marital-status",
                            "native-country": "native-country"})
    df = df.replace("?", np.nan).dropna()
    # binarise the target robustly (handles ">50K", ">50K.", etc.)
    df[TARGET] = (df[TARGET].astype(str).str.contains(">50K")).astype(int)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Build the three algorithm-specific views
# ----------------------------------------------------------------------
def quantile_bin(col: pd.Series, bins: int = 5) -> np.ndarray:
    edges = np.quantile(col.astype(float), np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    edges[0] -= 1e-9
    return np.digitize(col.astype(float), edges[1:-1]).astype(str)


def label_encode(col: pd.Series) -> np.ndarray:
    vals = {v: i for i, v in enumerate(sorted(col.unique()))}
    return col.map(vals).to_numpy(dtype=float)


def build_views(df: pd.DataFrame):
    cont = [c for c in CONTINUOUS if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    y = df[TARGET].to_numpy(np.int64)

    # ID3: every column categorical
    id3_cols = [quantile_bin(df[c]) for c in cont] + \
               [df[c].astype(str).to_numpy() for c in cat]
    X_id3 = np.column_stack(id3_cols).astype(object)
    types_id3 = ["categorical"] * (len(cont) + len(cat))

    # C4.5: continuous numeric, categorical as categories
    c45_cols = [df[c].astype(float).to_numpy().astype(object) for c in cont] + \
               [df[c].astype(str).to_numpy() for c in cat]
    X_c45 = np.column_stack(c45_cols).astype(object)
    types_c45 = ["numeric"] * len(cont) + ["categorical"] * len(cat)

    # CART: everything numeric (categoricals integer-encoded)
    cart_cols = [df[c].astype(float).to_numpy() for c in cont] + \
                [label_encode(df[c].astype(str)) for c in cat]
    X_cart = np.column_stack(cart_cols).astype(object)
    types_cart = ["numeric"] * (len(cont) + len(cat))

    return {
        "ID3":  (ID3Classifier,  X_id3,  types_id3),
        "C4.5": (C45Classifier,  X_c45,  types_c45),
        "CART": (CARTClassifier, X_cart, types_cart),
    }, y


# ----------------------------------------------------------------------
# 3-4. Run the timing experiment
# ----------------------------------------------------------------------
def run(df, sizes, repeats, max_depth, min_samples_split, seed=42):
    views, y_full = build_views(df)
    rng = np.random.default_rng(seed)
    n_total = len(y_full)
    rows = []

    for n in sizes:
        if n > n_total:
            print(f"  (skipping size {n}: dataset only has {n_total} rows)")
            continue
        idx = rng.choice(n_total, size=n, replace=False)
        # fixed 70/30 train/test split of the subsample
        cut = int(n * 0.7)
        tr, te = idx[:cut], idx[cut:]

        for name, (cls, X, types) in views.items():
            times = []
            acc = np.nan
            for r in range(repeats):
                model = cls(max_depth=max_depth,
                            min_samples_split=min_samples_split)
                t0 = time.perf_counter()
                model.fit(X[tr], y_full[tr], types)
                times.append(time.perf_counter() - t0)
                if r == 0:
                    acc = float((model.predict(X[te]) == y_full[te]).mean())
            best = min(times)          # min = least noise from the OS
            mean = float(np.mean(times))
            rows.append({"algorithm": name, "size": n,
                         "time_best_s": best, "time_mean_s": mean,
                         "accuracy": acc})
            print(f"  n={n:6d}  {name:5s}  time={best:7.4f}s  acc={acc:.3f}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5. Plot
# ----------------------------------------------------------------------
def plot(results: pd.DataFrame, outfile="execution_time_vs_size.png"):
    plt.figure(figsize=(8, 5.5))
    markers = {"ID3": "o", "C4.5": "s", "CART": "^"}
    for name, g in results.groupby("algorithm"):
        g = g.sort_values("size")
        plt.plot(g["size"], g["time_best_s"],
                 marker=markers.get(name, "o"), linewidth=2, label=name)
    plt.xlabel("Input size (number of training samples)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Decision Tree Training Time vs Input Size\n"
              "(ID3 vs C4.5 vs CART, UCI Adult dataset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved graph -> {outfile}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="path to a local adult.csv (offline mode)")
    ap.add_argument("--sizes", default="1000,2000,5000,10000,20000,30000",
                    help="comma-separated training sizes")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--min-samples-split", type=int, default=10)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print("Loading Adult dataset...")
    df = load_adult(args.csv)
    print(f"Dataset ready: {len(df)} rows, {df.shape[1]-1} features.\n")

    print("Running timing experiment...")
    results = run(df, sizes, args.repeats,
                  args.max_depth, args.min_samples_split)
    results.to_csv("results.csv", index=False)
    print("\nSaved raw numbers -> results.csv")
    plot(results)


if __name__ == "__main__":
    main()