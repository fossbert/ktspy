"""k-TSP for more than two classes: one classifier per pair of classes, soft voting.

For ``C`` classes, ``C (C - 1) / 2`` binary k-TSP classifiers are trained, each on the samples of its
two classes only (one-vs-one). A sample is classified by looking only at its own values: every
binary classifier compares the k gene pairs of its rule within the sample, and the share of pairs
that vote for each class is added up. No normalisation across samples is involved, so a single
sample can be classified, and the decision can be read off pair by pair (:meth:`explain`).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .core import KTSPClassifier

__all__ = ["MultiKTSPClassifier"]


class MultiKTSPClassifier(ClassifierMixin, BaseEstimator):
    """One-vs-one k-TSP classifier with soft voting.

    Parameters
    ----------
    k : int or sequence of int
        Pairs per class pair (a sequence is the switchBox ``krange``, see :class:`ktspy.KTSP`).
    prefilter : int, optional
        Keep this many features per class pair before pairing (switchBox Wilcoxon filter).
        Pair scoring is O(n_features**2): use it for genome-wide data.
    updown, handle_ties, disjoint :
        As in :class:`ktspy.KTSP`.
    voting : {"soft", "hard"}
        ``"soft"`` (default): the score of a class is the sum, over the class pairs it belongs to, of
        the share of that classifier's pairs voting for it. ``"hard"``: number of won class-pair
        decisions (more than half of the pairs), ties broken by the soft score.

    Attributes
    ----------
    classes_ : the classes, sorted
    estimators_ : dict {(class_a, class_b): fitted :class:`ktspy.KTSPClassifier`}; a pair
        ``x_a > x_b`` being true votes for ``class_b``
    rules_ : DataFrame with one row per gene pair (``class_a, class_b, a, b, comparison, score``)
    """

    def __init__(self, k: Union[int, Sequence[int]] = 5, *, prefilter=None, updown: bool = True,
                 handle_ties: bool = False, disjoint: bool = True, voting: str = "soft"):
        self.k = k
        self.prefilter = prefilter
        self.updown = updown
        self.handle_ties = handle_ties
        self.disjoint = disjoint
        self.voting = voting

    def fit(self, X, y):
        if self.voting not in ("soft", "hard"):
            raise ValueError("voting must be 'soft' or 'hard'")
        X = validate_data(self, X, dtype=float)
        check_classification_targets(y)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        self._names = np.asarray(names, dtype=object)

        self.estimators_ = {}
        rules = []
        for a, b in combinations(self.classes_, 2):
            m = (y == a) | (y == b)
            est = KTSPClassifier(k=self.k, prefilter=self.prefilter, updown=self.updown,
                                 handle_ties=self.handle_ties, disjoint=self.disjoint).fit(X[m], y[m])
            self.estimators_[(a, b)] = est
            pairs = est.get_pairs(self._names)
            pairs.insert(0, "class_b", b)
            pairs.insert(0, "class_a", a)
            rules.append(pairs.drop(columns="tie_vote"))
        self.rules_ = pd.concat(rules, ignore_index=True)
        return self

    # ------------------------------------------------------------------ scores
    def _scores(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, dtype=float, reset=False)
        idx = {c: i for i, c in enumerate(self.classes_)}
        soft = np.zeros((X.shape[0], len(self.classes_)))
        wins = np.zeros_like(soft)
        for (a, b), est in self.estimators_.items():
            frac_b = est.predict_proba(X)[:, 1]
            soft[:, idx[a]] += 1 - frac_b
            soft[:, idx[b]] += frac_b
            wins[:, idx[b]] += frac_b > 0.5
            wins[:, idx[a]] += frac_b <= 0.5
        return soft, wins

    def decision_function(self, X):
        """Soft vote score per class (``n_samples x n_classes``)."""
        return self._scores(X)[0]

    def predict_proba(self, X):
        """Vote shares (soft scores normalised to sum 1). A share of votes, not a calibrated probability."""
        soft = self._scores(X)[0]
        return soft / soft.sum(axis=1, keepdims=True)

    def predict(self, X):
        soft, wins = self._scores(X)
        score = wins + soft / (soft.max() + 1.0) if self.voting == "hard" else soft
        return self.classes_[score.argmax(axis=1)]

    # ------------------------------------------------------------------ interpretation
    def explain(self, X) -> pd.DataFrame:
        """Every gene-pair vote for every sample (long format).

        Columns: ``sample`` (row index of ``X``), ``class_a``, ``class_b``, ``a``, ``b``,
        ``value_a``, ``value_b`` (the expression values compared), ``comparison``, ``vote`` (the
        class the pair votes for). Use it to see why a single sample received its class.
        """
        check_is_fitted(self, "estimators_")
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        Xv = validate_data(self, X, dtype=float, reset=False)
        rows = []
        for (ca, cb), est in self.estimators_.items():
            votes = est._votes(Xv)
            pairs = est.get_pairs(self._names)
            for j, p in pairs.iterrows():
                a_idx, b_idx = est.pairs_[j]
                va, vb = Xv[:, a_idx], Xv[:, b_idx]
                rows.append(pd.DataFrame({
                    "sample": index, "class_a": ca, "class_b": cb, "a": p["a"], "b": p["b"],
                    "value_a": va, "value_b": vb, "comparison": p["comparison"],
                    "vote": np.where(votes[:, j], cb, ca),
                }))
        return pd.concat(rows, ignore_index=True)

    def get_pairs(self) -> pd.DataFrame:
        """The rules: gene pairs per class pair (alias of ``rules_``)."""
        check_is_fitted(self, "rules_")
        return self.rules_.copy()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
