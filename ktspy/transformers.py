"""k-TSP as a feature for other models: the vote share of the pairs, optionally out-of-fold.

``KTSPScore`` condenses a block of features (e.g. expression, activity or copy number) into one score per
sample (two classes) or one score per class (more than two): the share of the k-TSP pairs that vote for the
class. The score is computed from the values of one sample only, so it keeps the single-sample property, and
it can be combined with other blocks (mutations, clinical covariates) in a penalised model::

    ColumnTransformer([("expr", Pipeline([("ktsp", KTSPScore(k=5, prefilter=300)), ("sc", StandardScaler())]), expr_cols),
                       ("clin", clin_pipeline, clin_cols)])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .core import KTSPClassifier
from .multiclass import MultiKTSPClassifier

__all__ = ["KTSPScore"]


class KTSPScore(TransformerMixin, BaseEstimator):
    """Vote share of a k-TSP classifier as feature(s) for a downstream model.

    Parameters
    ----------
    k, prefilter, updown, handle_ties, disjoint, feature_groups :
        As in :class:`ktspy.KTSPClassifier` / :class:`ktspy.MultiKTSPClassifier`.
    cross_fit : bool
        True (default): ``fit_transform`` returns **out-of-fold** scores for the training samples (each score from
        a k-TSP model that did not see the sample), while ``transform`` uses the model fitted on all training
        samples. Without it the downstream model sees optimistic in-sample scores and tends to over-weight this
        block against the others (stacking bias).
    n_splits : int
        Folds for the cross-fitting (reduced if a class has fewer samples).
    random_state : int
        Seed of the cross-fitting folds.

    Attributes
    ----------
    classes_ : the classes, sorted
    estimator_ : the k-TSP model fitted on all training samples (``get_pairs()`` gives the rules)
    oof_scores_ : out-of-fold scores of the training samples (with ``cross_fit``)

    Notes
    -----
    Output: one column (share of the pairs voting for ``classes_[1]``) for two classes; for more than two
    classes one column per class except the first (the shares sum to 1). Use inside a ``Pipeline`` or
    ``ColumnTransformer``; it needs ``y`` in ``fit``.
    """

    def __init__(self, k=5, *, prefilter=None, updown=True, handle_ties=False, disjoint=True,
                 feature_groups=None, cross_fit=True, n_splits=5, random_state=0):
        self.k = k
        self.prefilter = prefilter
        self.updown = updown
        self.handle_ties = handle_ties
        self.disjoint = disjoint
        self.feature_groups = feature_groups
        self.cross_fit = cross_fit
        self.n_splits = n_splits
        self.random_state = random_state

    def _base(self, n_classes):
        kw = dict(k=self.k, prefilter=self.prefilter, updown=self.updown, handle_ties=self.handle_ties,
                  disjoint=self.disjoint, feature_groups=self.feature_groups)
        return KTSPClassifier(**kw) if n_classes == 2 else MultiKTSPClassifier(**kw)

    def _frame(self, X):
        """Keep the feature names so that the rules (get_pairs) show them, also inside a ColumnTransformer."""
        names = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=names) if names is not None else X

    def _scores(self, est, X):
        p = est.predict_proba(X)
        return p[:, [1]] if p.shape[1] == 2 else p[:, 1:]

    def fit(self, X, y):
        X = validate_data(self, X, dtype=float)
        check_classification_targets(y)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        base = self._base(len(self.classes_))
        Xf = self._frame(X)
        self.estimator_ = clone(base).fit(Xf, y)
        self.oof_scores_ = None
        if self.cross_fit:
            n_splits = int(min(self.n_splits, np.bincount(np.searchsorted(self.classes_, y)).min()))
            if n_splits < 2:
                raise ValueError("cross_fit needs at least two samples in every class")
            out = np.zeros((len(y), len(self.classes_) - 1))
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
            for tr, te in cv.split(X, y):
                if len(np.unique(y[tr])) != len(self.classes_):
                    raise ValueError("a training fold lost a class; lower n_splits")
                est = clone(base).fit(Xf.iloc[tr] if hasattr(Xf, "iloc") else Xf[tr], y[tr])
                out[te] = self._scores(est, Xf.iloc[te] if hasattr(Xf, "iloc") else Xf[te])
            self.oof_scores_ = out
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        if self.cross_fit:
            return self.oof_scores_
        return self._scores(self.estimator_, self._frame(validate_data(self, X, dtype=float, reset=False)))

    def transform(self, X):
        check_is_fitted(self, "estimator_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self._scores(self.estimator_, self._frame(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "estimator_")
        if len(self.classes_) == 2:
            return np.array(["ktsp_score"], dtype=object)
        return np.array([f"ktsp_score[{c}]" for c in self.classes_[1:]], dtype=object)

    def get_pairs(self, input_features=None):
        """The k-TSP rules of the model fitted on all training samples."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.get_pairs(input_features) if isinstance(self.estimator_, KTSPClassifier) else self.estimator_.get_pairs()  # names come from the fitted estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        return tags
