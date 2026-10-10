"""ktspy.MultiKTSPClassifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ktspy import KTSPClassifier, MultiKTSPClassifier


@pytest.fixture
def data(rng):
    """Three classes; gene pairs (g0,g1), (g2,g3), (g4,g5) flip between class pairs, the rest is noise."""
    n_per, m = 40, 30
    y = np.repeat(["A", "B", "C"], n_per)
    X = rng.normal(size=(3 * n_per, m))
    X[y == "A", 0] += 3                       # A: g0 > g1
    X[y != "A", 1] += 3
    X[y == "B", 2] += 3                       # B: g2 > g3
    X[y != "B", 3] += 3
    X[y == "C", 4] += 3                       # C: g4 > g5
    X[y != "C", 5] += 3
    return pd.DataFrame(X, columns=[f"g{i}" for i in range(m)]), y


def test_fit_predict_and_rules(data):
    X, y = data
    clf = MultiKTSPClassifier(k=3).fit(X, y)
    assert list(clf.classes_) == ["A", "B", "C"]
    assert set(clf.estimators_) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert (clf.predict(X) == y).mean() > 0.9
    rules = clf.get_pairs()
    assert list(rules.columns) == ["class_a", "class_b", "a", "b", "comparison", "score"]
    assert len(rules) == 9 and set(rules["class_a"] + rules["class_b"]) == {"AB", "AC", "BC"}
    top_ab = rules[(rules.class_a == "A") & (rules.class_b == "B")].iloc[0]
    assert {top_ab["a"], top_ab["b"]} & {"g0", "g1", "g2", "g3"}


def test_probabilities_and_hard_voting(data):
    X, y = data
    clf = MultiKTSPClassifier(k=3).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert (clf.classes_[p.argmax(axis=1)] == clf.predict(X)).all()
    hard = clone(clf).set_params(voting="hard").fit(X, y)
    assert (hard.predict(X) == y).mean() > 0.9
    assert clf.decision_function(X).shape == (len(X), 3)
    with pytest.raises(ValueError):
        MultiKTSPClassifier(voting="x").fit(X, y)


def test_two_classes_equals_binary_ktsp(data):
    X, y = data
    m = y != "C"
    multi = MultiKTSPClassifier(k=3).fit(X[m], y[m])
    binary = KTSPClassifier(k=3).fit(X[m], y[m])
    assert (multi.predict(X[m]) == binary.predict(X[m])).all()
    np.testing.assert_allclose(multi.predict_proba(X[m]), binary.predict_proba(X[m]))


def test_single_sample_prediction_is_independent_of_the_other_samples(data):
    X, y = data
    clf = MultiKTSPClassifier(k=3).fit(X, y)
    batch = clf.predict(X)
    single = np.array([clf.predict(X.iloc[[i]])[0] for i in range(len(X))])
    assert (batch == single).all()


def test_invariant_to_monotone_per_sample_transformations(data):
    X, y = data
    clf = MultiKTSPClassifier(k=3).fit(X, y)
    base = clf.predict(X)
    rng = np.random.default_rng(5)
    scale = rng.uniform(0.2, 8, size=(len(X), 1))
    offset = rng.normal(0, 20, size=(len(X), 1))
    assert (clf.predict(X * scale + offset) == base).all()          # per-sample scale and shift
    assert (clf.predict(np.exp(X / 5.0)) == base).all()              # monotone
    # a per-GENE shift (what cohort-wise standardisation does) is not harmless
    gene_shift = rng.normal(0, 3, size=(1, X.shape[1]))
    assert (clf.predict(X + gene_shift) != base).any()


def test_explain_lists_every_vote(data):
    X, y = data
    clf = MultiKTSPClassifier(k=2).fit(X, y)
    ex = clf.explain(X.iloc[:4])
    assert len(ex) == 4 * len(clf.rules_)
    assert set(ex.columns) >= {"sample", "class_a", "class_b", "a", "b", "value_a", "value_b", "comparison", "vote"}
    first = ex[ex["sample"] == X.index[0]].iloc[0]
    assert first["value_a"] == X.iloc[0][first["a"]] and first["value_b"] == X.iloc[0][first["b"]]
    # the vote is the class whose side of the comparison is true
    gt = first["value_a"] > first["value_b"]
    assert first["vote"] == (first["class_b"] if gt else first["class_a"])
    # winners reproduce the soft scores
    soft = clf.decision_function(X.iloc[:4])
    for i in range(4):
        votes = ex[ex["sample"] == X.index[i]]["vote"].value_counts()
        k = 2
        for j, c in enumerate(clf.classes_):
            assert soft[i, j] == pytest.approx(votes.get(c, 0) / k)


def test_numpy_input_and_clone(data):
    X, y = data
    clf = MultiKTSPClassifier(k=[1, 3, 5]).fit(X.to_numpy(), y)
    assert clf.rules_["a"].str.startswith("x").all()
    assert clone(clf).get_params()["k"] == [1, 3, 5]
    with pytest.raises(ValueError):
        MultiKTSPClassifier().fit(X, np.array(["A"] * len(X)))
