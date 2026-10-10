# Changelog

## Unreleased

- `feature_groups` (`KTSP`, `KTSPClassifier`, `MultiKTSPClassifier`): one label per feature; pairs are formed
  only within a group (same scale), the groups compete by TSP score; `prefilter` per group (int or dict);
  `pair_groups_`, a `group` column in `get_pairs()` / `rules_`.

## 0.1.0

- Binary k-TSP (`KTSP`, `KTSPClassifier`, `tsp_scores`, `wilcoxon_filter`), moved unchanged from
  `cbrmb.ml.ktsp`; identical to Bioconductor `switchBox` in the tests.
- `MultiKTSPClassifier`: more than two classes (one-vs-one, soft or hard voting), `get_pairs` for the
  rules, `explain` for the vote of every gene pair of a sample.
- Tests: single-sample prediction independent of the other samples; invariance to per-sample scale,
  offset and monotone transformations.
