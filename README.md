# ktspy

k top scoring pairs (k-TSP): rank-based classifiers that work on **single samples**.

A *top scoring pair* is a pair of features (genes) whose order **within a sample** differs between two
classes: `gene_a > gene_b` in one class, `<` in the other. A k-TSP classifier is a majority vote of k such
pairs. Because only the order of values inside a sample matters:

- the rule is **intuitive**: it can be read pair by pair ("MLH1 is lower than INTS12 in MSI tumours");
- it is **robust for single samples**: no normalisation across samples, the result does not depend on
  which other samples are analysed with it; any monotone per-sample transformation (library size, log,
  scale, offset) leaves it unchanged.

Not unchanged: a per-*gene* transformation that differs between cohorts (e.g. standardising each gene
within a cohort), and gene-specific platform effects, which can reverse the order of two genes. Train on
unstandardised values and validate on another platform.

The binary core (`KTSP`, `KTSPClassifier`) is a pure-NumPy port of Bioconductor
[`switchBox`](https://bioconductor.org/packages/switchBox/) (selected pairs, scores and tie votes are
identical to R, see `tests/test_core.py`). `MultiKTSPClassifier` extends it to more than two classes.
Extracted from `cbrmb.ml.ktsp`, which now re-exports it.

## Install

```bash
pip install -e '.[test]'
pip install -e '.[r,test]'     # also runs the equivalence tests against R switchBox
```

## Usage

```python
from ktspy import MultiKTSPClassifier

clf = MultiKTSPClassifier(k=5, prefilter=300).fit(X_train, y_train)    # X: samples x genes (DataFrame), log expression
clf.predict(X_new)                      # works for one sample at a time
clf.predict_proba(X_new)                # vote shares (not calibrated probabilities)

clf.get_pairs()                         # the rules: class_a, class_b, gene a, gene b, comparison, score
clf.explain(X_new.iloc[[0]])            # every pair, the two values compared, and its vote for that sample
```

- One rule per pair of classes (one-vs-one); `k` pairs each; soft voting (`voting="soft"`, default) adds
  up the share of pairs voting for a class, `"hard"` counts won class pairs.
- `prefilter=300` keeps the 300 most class-discriminating genes per class pair (switchBox Wilcoxon filter)
  before pairing; pair scoring is O(genes²). Inside cross-validation it is part of the model, so there is
  no leakage.
- Binary problems: `KTSPClassifier` (same arguments; `KTSP` as a transformer producing the k pair
  indicators as features for any other classifier). `k` may be a sequence (switchBox `krange`).
- All estimators follow the scikit-learn API (pipelines, `clone`, cross-validation).

## Several feature types (expression, copy number, mutations, ...)

Only features on the same scale can be ordered within a sample, so pairs are formed **inside** a group
of features. Give one label per column:

```python
groups = ["expression"] * n_expr + ["cna"] * n_cna + ["mutation"] * n_mut
clf = MultiKTSPClassifier(k=5, handle_ties=True, feature_groups=groups,
                          prefilter={"expression": 300, "cna": 300}).fit(X, y)    # X: samples x all features
clf.get_pairs()                 # has a 'group' column
```

The groups compete for the k places by their TSP score (a difference of order frequencies, on the same
scale for every group). `handle_ties=True` is recommended for binary features. On the TCGA gastric
subtypes this did not help: expression pairs always had the higher scores, copy-number and mutation pairs
alone reach about 0.6 accuracy (expression 0.85), and a late fusion of the vote shares did not
improve on expression either. A group that is globally weaker never enters the top k; if every group is
meant to contribute, train one model per group and combine the vote shares (`predict_proba`).

## Notes

- `a > b` being true votes for the *second* class of a pair (classes sorted); see `get_pairs()`.
- Ties (equal values, e.g. zeros in count data): `handle_ties=True` is recommended for zero-inflated data.
- `predict_proba` is the share of votes; it is not calibrated and says little about confidence in
  absolute terms. Check the distribution of the maximum share in your own data.
