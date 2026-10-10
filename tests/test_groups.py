"""feature_groups: pairs only within a group (same scale), groups compete by their TSP score."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ktspy import KTSP, KTSPClassifier, MultiKTSPClassifier


@pytest.fixture
def two_modalities(rng):
    """expression (continuous) + mutation (binary). Expression pair e0 > e1 flips between classes
    (strong), mutation pair m0 > m1 flips weakly. All other features are noise."""
    n = 80
    y = np.array(["case"] * (n // 2) + ["ctrl"] * (n // 2))
    E = rng.normal(size=(n, 10))
    E[y == "ctrl", 0] += 3
    E[y == "case", 1] += 3
    M = (rng.random((n, 6)) < 0.3).astype(float)
    M[y == "ctrl", 0] = (rng.random((n // 2,)) < 0.8)
    M[y == "case", 0] = (rng.random((n // 2,)) < 0.1)
    M[:, 1] = (rng.random(n) < 0.1)
    names = [f"e{i}" for i in range(10)] + [f"m{i}" for i in range(6)]
    groups = np.array(["expression"] * 10 + ["mutation"] * 6)
    return pd.DataFrame(np.column_stack([E, M]), columns=names), y, groups


def test_pairs_stay_inside_their_group_and_groups_compete(two_modalities):
    X, y, groups = two_modalities
    est = KTSP(k=3, feature_groups=groups, handle_ties=True).fit(X, y)
    pairs = est.get_pairs()
    assert "group" in pairs.columns and len(pairs) == 3
    for _, r in pairs.iterrows():
        assert r["a"][0] == r["b"][0]                           # e-e or m-m, never e-m
        assert {r["a"][0]} == {"e"} if r["group"] == "expression" else {r["a"][0]} == {"m"}
    assert (pairs["group"] == "expression").any() and (pairs["group"] == "mutation").any()
    assert np.all(np.diff(est.scores_) <= 1e-12)                # best first across groups
    assert (est.pairs_[:, 0] < 10).tolist().count(True) == (pairs["group"] == "expression").sum()


def test_without_groups_cross_scale_pairs_are_possible(two_modalities):
    X, y, groups = two_modalities
    plain = KTSP(k=6, handle_ties=True).fit(X, y).get_pairs()
    assert any(r["a"][0] != r["b"][0] for _, r in plain.iterrows())      # e-m pairs exist without groups
    grouped = KTSP(k=6, feature_groups=groups, handle_ties=True).fit(X, y).get_pairs()
    assert all(r["a"][0] == r["b"][0] for _, r in grouped.iterrows())


def test_one_group_equals_no_groups(two_modalities):
    X, y, _ = two_modalities
    a = KTSP(k=4).fit(X, y)
    b = KTSP(k=4, feature_groups=["all"] * X.shape[1]).fit(X, y)
    np.testing.assert_array_equal(a.pairs_, b.pairs_)
    np.testing.assert_array_equal(a.scores_, b.scores_)
    assert a.pair_groups_ is None and list(b.pair_groups_) == ["all"] * len(b.pairs_)


def test_prefilter_per_group_and_dict(two_modalities):
    X, y, groups = two_modalities
    est = KTSP(k=2, prefilter={"expression": 6}, feature_groups=groups, handle_ties=True).fit(X, y)
    assert len(est.pairs_) == 2
    est2 = KTSP(k=2, prefilter=6, feature_groups=groups, handle_ties=True).fit(X, y)
    assert len(est2.pairs_) == 2


def test_small_groups_are_skipped_and_errors(two_modalities):
    X, y, groups = two_modalities
    g = groups.copy(); g[-2:] = "tiny"; g[:-2] = np.where(g[:-2] == "mutation", "expression", g[:-2])
    g[10:14] = "mutation"
    est = KTSP(k=2, feature_groups=g, handle_ties=True).fit(X, y)       # 'tiny' (2 features) is ignored
    assert "tiny" not in set(est.pair_groups_)
    with pytest.raises(ValueError, match="one label per feature"):
        KTSP(feature_groups=["a"] * 3).fit(X, y)
    with pytest.raises(ValueError, match="at least 4"):                # no group has 4 features
        KTSP(feature_groups=np.repeat(list("abcd"), 3)).fit(X.iloc[:, :12], y)


def test_classifier_clone_and_multiclass_rules(two_modalities, rng):
    X, y, groups = two_modalities
    clf = KTSPClassifier(k=3, feature_groups=groups, handle_ties=True).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.85
    assert list(clone(clf).get_params()["feature_groups"]) == list(groups)
    y3 = np.concatenate([y, ["other"] * 40])
    X3 = pd.concat([X, pd.DataFrame(rng.normal(size=(40, X.shape[1])), columns=X.columns)], ignore_index=True)
    multi = MultiKTSPClassifier(k=2, feature_groups=groups, handle_ties=True).fit(X3, y3)
    assert "group" in multi.rules_.columns and set(multi.rules_["group"]) <= {"expression", "mutation"}
    assert len(multi.predict(X3.iloc[:5])) == 5
