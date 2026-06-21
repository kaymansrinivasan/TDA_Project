# Part 2 — Code Package
### Performance Analysis of Decision Tree Variants (ID3 vs C4.5 vs CART)
TDA6323 Algorithm Design and Analysis · Lab A1C · Group 5

---

## Files

| File | Purpose |
|------|---------|
| `decision_trees.py` | From-scratch ID3, C4.5 and CART, all in **one shared framework** so the timing comparison is fair. |
| `experiment.py` | Loads the UCI Adult dataset, subsamples it at increasing sizes, times each algorithm, and produces the required graph. |
| `README.md` | This file. |

Outputs produced when you run the experiment:
- `results.csv` — raw timing + accuracy numbers (use these in your tables).
- `execution_time_vs_size.png` — the execution-time-vs-input-size graph (assignment requirement 2c).

---

## Step 1 — Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

## Step 2 — Get the dataset (pick ONE)

**Option A — automatic (recommended).** Just run the script; `ucimlrepo`
downloads the official UCI Adult data for you.

**Option B — scikit-learn fallback.** If `ucimlrepo` fails, the script
automatically falls back to `fetch_openml('adult')`.

**Option C — offline.** Download `adult.csv` from
<https://archive.ics.uci.edu/dataset/2/adult>, put it beside the scripts,
and pass `--csv adult.csv`.

## Step 3 — Run

```bash
python experiment.py
```

or, with options:

```bash
python experiment.py --csv adult.csv \
    --sizes 1000,2000,5000,10000,20000,30000 \
    --repeats 3 --max-depth 15 --min-samples-split 10
```

---

## How the three algorithms are made genuinely different

The same raw data is preprocessed into three **views**, each matching the
real design of the algorithm (this mirrors Table 2 in your Part 1 report):

| View | Continuous attributes | Categorical attributes | Split type |
|------|----------------------|------------------------|-----------|
| **ID3** | quantile-binned into categories (ID3 cannot take continuous input) | kept as categories | multi-way, Information Gain |
| **C4.5** | kept numeric, binary threshold | kept as categories, multi-way | Gain Ratio |
| **CART** | kept numeric | integer-encoded numeric | strictly binary, Gini |

The fact that **ID3 needs discretisation** is one of its documented
weaknesses — note the small accuracy cost it causes in your write-up.

---

## How this maps to the Part 2 report sections

- **(g) Theoretical analysis** — derive Big-O for each `_build` recursion in
  `decision_trees.py`. All three are ≈ `O(n · m · d)` (n = samples,
  m = features, d = depth); CART/C4.5 carry the extra threshold-search
  factor `O(n log n)` per numeric feature. Point to the actual loops.
- **(h) Experimental analysis** — use `results.csv` and the PNG graph.
- **(i) Comparison & findings** — compare the theoretical growth above with
  the measured curve shapes; explain discrepancies (constant factors,
  threshold capping via `max_thresholds`, Python overhead).
- **(k) Appendix** — paste `decision_trees.py` and `experiment.py`.

---

## Citation note

The algorithm logic follows the standard formulations of Quinlan (1986,
1993) and Breiman et al. (1984), already in your reference list. The code is
written by the group. If you adapt any external snippet, cite it inline to
stay within the assignment's plagiarism rules.
