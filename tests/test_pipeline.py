"""Automated checks for leakage, protocols and metrics.  Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitsdp import config as C  # noqa: E402
from jitsdp.metrics import effort_aware, evaluate  # noqa: E402
from jitsdp.models import JITPreprocessor, make_model  # noqa: E402
from jitsdp.protocols import (LeakageError, Split, check_split, p1_random, p2_chrono,  # noqa: E402
                              p3_latency, random_counterpart, rolling_windows)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 3000
    t0 = pd.Timestamp("2015-01-01").value // 10**9
    df = pd.DataFrame({f: rng.integers(0, 200, n) for f in C.FEATURES})
    df["ent"] = rng.random(n)
    df["fix"] = rng.integers(0, 2, n)
    df[C.TIME] = np.sort(t0 + rng.integers(0, 4 * 365 * 86400, n))
    df["date"] = pd.to_datetime(df[C.TIME], unit="s")
    df[C.ID] = [f"c{i}" for i in range(n)]
    df[C.PROJECT] = "toy"
    df[C.LABEL] = (rng.random(n) < 0.15 + 0.002 * np.log1p(df["la"])).astype(int)
    return df


def test_p2_is_chronological(toy):
    s = p2_chrono(toy)
    check_split(s)
    assert s.train[C.TIME].max() <= s.test[C.TIME].min()
    assert abs(len(s.test) / len(toy) - C.TEST_FRAC) < 0.01


@pytest.mark.parametrize("gap", C.GAPS)
def test_p3_same_test_and_gap(toy, gap):
    s2, s3 = p2_chrono(toy), p3_latency(toy, gap)
    check_split(s3)
    assert list(s2.test[C.ID]) == list(s3.test[C.ID])          # same test commits
    assert s3.test[C.TIME].min() - s3.train[C.TIME].max() >= gap * 86400
    assert len(s3.train) + s3.info["withheld_unlabeled"] == len(s2.train)


def test_p1_disjoint_and_stratified(toy):
    s = p1_random(toy, 0)
    check_split(s)
    assert abs(s.train[C.LABEL].mean() - s.test[C.LABEL].mean()) < 0.02


def test_leakage_detected(toy):
    s = p2_chrono(toy)
    bad = Split("P2", s.test, s.train, True)                   # reversed in time
    with pytest.raises(LeakageError):
        check_split(bad)
    with pytest.raises(LeakageError):
        check_split(s, features=C.FEATURES + ["year"])


def test_windows_respect_gap(toy):
    for _, _, s in rolling_windows(toy, 90):
        check_split(s)
        rnd = random_counterpart(s, 0)
        assert len(rnd.test) == len(s.test)


def test_preprocessor_fits_on_train_only(toy):
    s = p2_chrono(toy)
    p = JITPreprocessor(scale=True).fit(s.train[C.FEATURES])
    z_tr = p.transform(s.train[C.FEATURES])
    assert np.allclose(z_tr.mean().to_numpy(), 0, atol=1e-8)   # train-fitted stats
    shifted = s.test[C.FEATURES] + 1000
    z_te = p.transform(shifted)
    assert (z_te["la"].mean()) > 1                              # not re-centred on test


def test_undersampling_never_touches_test(toy):
    s = p2_chrono(toy)
    m = make_model("LR", "under", {"C": 1.0}, s.train[C.LABEL]).fit(
        s.train[C.FEATURES], s.train[C.LABEL])
    assert len(m.predict_proba(s.test[C.FEATURES])) == len(s.test)


def test_effort_metrics_bounds():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    eff = np.array([10, 50, 20, 5, 100, 30, 40, 15], float)
    perfect = y / eff
    r, popt = effort_aware(y, perfect, eff)
    assert popt == pytest.approx(1.0)
    r_w, popt_w = effort_aware(y, -perfect + (1 - y) * 0, eff)
    assert 0 <= popt_w <= 1 and r >= r_w


def test_constant_score_is_not_size_baseline():
    """Ties must be broken randomly, not by effort."""
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.3).astype(int)
    eff = rng.integers(1, 1000, 2000).astype(float)
    r_const, _ = effort_aware(y, np.ones(2000), eff)
    r_lt, _ = effort_aware(y, 1 / eff, eff)
    assert r_const < r_lt


def test_evaluate_keys():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = evaluate(y, np.array([.1, .9, .2, .7, .4, .3]), 0.5, np.ones(6))
    for k in ["precision", "recall", "f1", "mcc", "pr_auc", "roc_auc", "recall20", "popt",
              "pr_auc_lift", "brier"]:
        assert k in m
