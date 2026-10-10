"""KTSPScore: k-TSP vote share as a feature, out-of-fold for the training data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ktspy import KTSPClassifier, KTSPScore


@pytest.fixture
def binary(rng):
    n, m = 90, 40
    y = np.array(["a"] * 45 + ["b"] * 45)
    X = rng.normal(size=(n, m))
    X[y == "b", 0] += 1.0                     # weak pair signal ...
    X[y == "a", 1] += 1.0
    return pd.DataFrame(X, columns=[f"g{i}" for i in range(m)]), y


@pytest.fixture
def three(rng):
    y = np.repeat(["A", "B", "C"], 30)
    X = rng.normal(size=(90, 20))
    X[y == "A", 0] += 3; X[y != "A", 1] += 3
    X[y == "B", 2] += 3; X[y != "B", 3] += 3
    return pd.DataFrame(X, columns=[f"g{i}" for i in range(20)]), y


def test_binary_shapes_names_and_values(binary):
    X, y = binary
    t = KTSPScore(k=3)
    Z = t.fit_transform(X, y)
    assert Z.shape == (len(X), 1) and Z.min() >= 0 and Z.max() <= 1
    assert list(t.get_feature_names_out()) == ["ktsp_score"]
    assert t.transform(X).shape == (len(X), 1)
    clf = KTSPClassifier(k=3).fit(X, y)
    np.testing.assert_allclose(t.transform(X)[:, 0], clf.predict_proba(X)[:, 1])     # transform = the final model


def test_out_of_fold_scores_are_less_optimistic_than_in_sample(binary):
    X, y = binary
    yb = (y == "b").astype(int)
    oof = KTSPScore(k=5, cross_fit=True).fit_transform(X, y)[:, 0]
    ins = KTSPScore(k=5, cross_fit=False).fit_transform(X, y)[:, 0]
    assert roc_auc_score(yb, ins) >= roc_auc_score(yb, oof) - 1e-12
    assert roc_auc_score(yb, ins) - roc_auc_score(yb, oof) > 0.02                    # selection bias is visible
    t = KTSPScore(k=5, cross_fit=True).fit(X, y)
    assert t.oof_scores_.shape == (len(X), 1) and not np.allclose(t.oof_scores_, t.transform(X))


def test_multiclass_one_column_per_class_but_the_first(three):
    X, y = three
    t = KTSPScore(k=2)
    Z = t.fit_transform(X, y)
    assert Z.shape == (90, 2) and list(t.get_feature_names_out()) == ["ktsp_score[B]", "ktsp_score[C]"]
    assert (t.transform(X).argmax(axis=1) == pd.Series(y).map({"A": -1, "B": 0, "C": 1}).to_numpy()).mean() > 0.5
    assert len(t.get_pairs()) > 0


def test_single_sample_transform_is_independent_of_other_samples(binary):
    X, y = binary
    t = KTSPScore(k=3).fit(X, y)
    batch = t.transform(X)
    single = np.vstack([t.transform(X.iloc[[i]]) for i in range(10)])
    np.testing.assert_allclose(batch[:10], single)
    scale = np.random.default_rng(1).uniform(0.5, 3, size=(len(X), 1))
    np.testing.assert_allclose(t.transform(X * scale + 5), batch)                    # per-sample scale and offset


def test_in_pipeline_with_other_blocks_and_clone(binary, rng):
    X, y = binary
    D = X.copy(); D["age"] = rng.normal(60, 10, len(D)); D["male"] = (rng.random(len(D)) < 0.5).astype(float)
    ct = ColumnTransformer([("expr", Pipeline([("k", KTSPScore(k=3)), ("sc", StandardScaler())]), list(X.columns)),
                            ("clin", StandardScaler(), ["age", "male"])])
    pipe = Pipeline([("ct", ct), ("lr", LogisticRegression())]).fit(D, y)
    assert pipe.predict(D).shape == (len(D),)
    assert list(pipe[:-1].get_feature_names_out())[0].endswith("ktsp_score")
    assert clone(pipe).get_params()["ct__expr__k__k"] == 3
    with pytest.raises(ValueError, match="at least two"):
        KTSPScore().fit(X, np.array(["a"] * len(X)))


def test_rules_carry_the_feature_names_also_inside_a_column_transformer(binary):
    X, y = binary
    t = KTSPScore(k=3).fit(X, y)
    assert set(t.get_pairs()["a"]) | set(t.get_pairs()["b"]) <= set(X.columns)
    ct = ColumnTransformer([("expr", Pipeline([("k", KTSPScore(k=3))]), list(X.columns))]).fit(X, y)
    pairs = ct.named_transformers_["expr"][0].get_pairs()
    assert pairs["a"].str.startswith("g").all()
    multi = KTSPScore(k=2).fit(X.assign(extra=1.0).iloc[:, :20], np.where(np.arange(len(X)) % 3 == 0, "c", y))
    assert multi.get_pairs()["a"].str.startswith("g").all()
