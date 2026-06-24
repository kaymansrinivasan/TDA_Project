# TDA6323 Algorithm Design and Analysis — Part 2 Submission

**Project title:** Performance Analysis of Decision Tree Variants in Classification Problems
**Algorithms compared:** ID3, C4.5, and CART
**Dataset:** UCI Adult / Census Income
**Language:** Python 3
**Lab section:** A1C  **Group:** 5

| Member | Student ID |
|--------|-----------|
| Yogeswary A/P Saravanan | 241UT240ZU |
| Kayman Srinivasan Arumugam | 1231301161 |
| Harith Irfan Bin Sufi Adrian | 1221308953 |
| Teo Jin Jie | 1201200850 |

---

## 1. Overview

This project implements and compares three classic decision tree induction
algorithms — ID3, C4.5 and CART — entirely from scratch in a single,
consistent Python framework. Using one shared implementation (rather than
mixing optimised libraries) ensures that the measured execution times reflect
genuine algorithmic differences rather than differences in language or library
optimisation, which is essential for a fair theoretical-versus-experimental
comparison.

The work covers the asymptotic (Big-O) analysis of each algorithm, an
experimental study of training time against input size on the Adult dataset,
a comparison of the theoretical and experimental results, and two proposed
algorithm improvements (AB-ID3 and IA-CART), both implemented and benchmarked.

---

## 2. Quick start

**Step 1 — Install dependencies**

```
pip install numpy pandas matplotlib scikit-learn
```

**Step 2 — Dataset**

The dataset `adult.csv` is included in this folder, so no download is needed.
(The scripts also tolerate alternative Adult column-naming conventions
automatically.)

**Step 3 — Run the experiments**

```
python experiment.py              # main timing experiment (time vs input size)
python improvement_experiment.py  # IA-CART improvement (imbalance handling)
python ab_id3_experiment.py       # AB-ID3 improvement (adaptive binning)
```

Each script prints its results to the console and, where relevant, saves a
figure and a CSV of the raw numbers.

---

## 3. File inventory

**Core implementation**

| File | Description |
|------|-------------|
| `decision_trees.py` | From-scratch ID3, C4.5 and CART in one shared framework |
| `experiment.py` | Loads Adult, subsamples at increasing sizes, times each algorithm, plots time vs input size |
| `adult.csv` | UCI Adult / Census Income dataset (ready to use) |

**Proposed improvement 1 — AB-ID3 (Adaptive Binning ID3)**

| File | Description |
|------|-------------|
| `ab_id3_experiment.py` | Supervised information-gain binning for ID3; compares against baseline ID3, C4.5, CART |
| `ab_id3_figure.py` | Generates the accuracy/speed comparison figure |

**Proposed improvement 2 — IA-CART (Imbalance-Aware CART)**

| File | Description |
|------|-------------|
| `imbalance_aware_tree.py` | CART with node-adaptive class weighting |
| `improvement_experiment.py` | Compares standard CART against IA-CART on minority-class metrics |

**Generated outputs (sample runs included for reference)**

| File | Description |
|------|-------------|
| `execution_time_vs_size.png` | Training time vs input size graph (ID3, C4.5, CART) |
| `results.csv` | Raw timing and accuracy numbers from the main experiment |
| `ab_id3_comparison.png` | Accuracy and speed of AB-ID3 vs the baselines |
| `ia_cart_alpha_sweep.png` | Effect of the correction strength on IA-CART |

---

## 4. Mapping to the report requirements

| Report section | Where it is produced |
|----------------|---------------------|
| Theoretical (asymptotic) analysis | Derived from the `_build` recursion in `decision_trees.py` |
| Experimental analysis (time vs input size) | `experiment.py` → `results.csv`, `execution_time_vs_size.png` |
| Comparison of theoretical vs experimental results | Based on the above outputs |
| Proposal 1: AB-ID3 | `ab_id3_experiment.py`, `ab_id3_figure.py` |
| Proposal 2: IA-CART | `imbalance_aware_tree.py`, `improvement_experiment.py` |

---

## 5. Summary of key results (on UCI Adult)

- **Execution time:** ID3 is fastest at every input size, followed by C4.5 and
  CART, which track closely together. This matches the theoretical analysis:
  ID3 is O(Nmd), while C4.5 and CART both simplify to O(dmN log N) due to the
  sorting and threshold search they perform on continuous attributes.
- **Accuracy:** ID3 is the least accurate (its discretisation of continuous
  attributes loses information), C4.5 the most accurate, with CART close behind.
- **AB-ID3 (Proposal 1):** raises ID3's accuracy from about 0.802 to 0.810 while
  preserving its speed advantage, narrowing the gap to C4.5.
- **IA-CART (Proposal 2):** improves minority-class recall and F1 on the
  imbalanced data at no cost to overall accuracy.

(Exact numbers vary slightly between machines, but the rankings and trends are
stable and reproducible.)

---

## 6. Notes on implementation and reproducibility

- All three algorithms share one framework for a fair runtime comparison.
- Random seeds are fixed in the experiment scripts for reproducibility; absolute
  timings depend on the host machine, but relative rankings are consistent.
- For numeric features, the implementation evaluates a bounded number of
  candidate thresholds per feature — a standard optimisation that keeps runtime
  practical without affecting the comparison.

## 7. Citations and attribution

The algorithm logic follows the standard formulations of Quinlan (1986, 1993)
for ID3 and C4.5 and Breiman et al. (1984) for CART. The supervised binning idea
in AB-ID3 draws on entropy-based discretisation (Fayyad & Irani, 1993). All code
in this submission was written by the group within a single Python framework.
Full references are provided in the report.
