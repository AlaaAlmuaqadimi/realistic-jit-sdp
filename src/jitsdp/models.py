"""Leakage-free preprocessing, the three classifiers, the two baselines,
time-ordered hyper-parameter tuning and validation-based thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from xgboost import XGBClassifier

from . import config as C
from .data import effort
from .metrics import best_f1_threshold, recall_threshold


class JITPreprocessor(BaseEstimator, TransformerMixin):
    """log1p on skewed counts -> optional Spearman collinearity filter ->
    optional standardization. Everything learned is fitted on training data
    only and then applied unchanged to validation/test data."""

    def __init__(self, features=tuple(C.FEATURES), corr_filter=False, scale=False,
                 threshold=C.CORR_THRESHOLD):
        self.features = features
        self.corr_filter = corr_filter
        self.scale = scale
        self.threshold = threshold

    def _log(self, X):
        X = pd.DataFrame(X, columns=list(self.features)).astype(float).copy()
        for f in C.LOG_FEATURES:
            if f in X:
                X[f] = np.log1p(X[f].clip(lower=0))
        return X

    def fit(self, X, y=None):
        X = self._log(X)
        keep = list(X.columns)
        if self.corr_filter:
            corr = X.corr(method="spearman").abs()
            keep = []
            for f in X.columns:          # greedy: keep earlier (core) metric
                if all(not (corr.loc[f, k] > self.threshold) for k in keep):
                    keep.append(f)
        self.keep_ = keep
        self.dropped_ = [f for f in X.columns if f not in keep]
        self.mean_ = X[keep].mean()
        self.std_ = X[keep].std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X):
        X = self._log(X)[self.keep_]
        if self.scale:
            X = (X - self.mean_) / self.std_
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


# ------------------------------------------------------------------- grids
GRIDS = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "RF": [{"max_depth": d, "min_samples_leaf": l} for d in (None, 12) for l in (1, 5)],
    "XGB": [{"max_depth": d, "learning_rate": lr} for d in (3, 6) for lr in (0.03, 0.1)],
}
DEFAULTS = {"LR": {"C": 1.0}, "RF": {"max_depth": 12, "min_samples_leaf": 5},
            "XGB": {"max_depth": 4, "learning_rate": 0.05}}


def make_model(name: str, strategy: str, params: dict, y_train, seed=C.SEED,
               n_jobs=-1):
    weight = strategy == "weight"
    if name == "LR":
        clf = LogisticRegression(C=params["C"], max_iter=3000,
                                 class_weight="balanced" if weight else None)
        prep = JITPreprocessor(corr_filter=True, scale=True)
    elif name == "RF":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], max_features="sqrt",
            class_weight="balanced" if weight else None, n_jobs=n_jobs,
            random_state=seed)
        prep = JITPreprocessor()
    elif name == "XGB":
        y = np.asarray(y_train)
        spw = (len(y) - y.sum()) / max(y.sum(), 1) if weight else 1.0
        clf = XGBClassifier(
            n_estimators=300, max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=spw,
            tree_method="hist", eval_metric="logloss", n_jobs=n_jobs,
            random_state=seed, verbosity=0)
        prep = JITPreprocessor()
    else:
        raise ValueError(name)
    steps = [("prep", prep)]
    if strategy == "under":   # resampling happens inside fit() only
        steps.append(("under", RandomUnderSampler(random_state=seed)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def _folds(X, y, time_ordered, seed):
    if time_ordered:          # every fold validates on later data
        return TimeSeriesSplit(n_splits=C.TUNING_FOLDS).split(X)
    return StratifiedKFold(C.TUNING_FOLDS, shuffle=True, random_state=seed).split(X, y)


def tune(name, strategy, X, y, time_ordered, seed=C.SEED, n_jobs=-1) -> dict:
    """Pick hyper-parameters by mean PR-AUC over training-only folds."""
    best, best_score = DEFAULTS[name], -np.inf
    folds = list(_folds(X, y, time_ordered, seed))
    for params in GRIDS[name]:
        scores = []
        for tr, va in folds:
            if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
                continue
            m = make_model(name, strategy, params, y[tr], seed, n_jobs).fit(X.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        if scores and np.mean(scores) > best_score:
            best, best_score = params, float(np.mean(scores))
    return best


def _val_split(n, y, time_ordered, seed):
    idx = np.arange(n)
    if time_ordered:
        k = int(n * (1 - C.VAL_FRAC))
        return idx[:k], idx[k:]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * (1 - C.VAL_FRAC))
    return np.sort(perm[:k]), np.sort(perm[k:])


def fit_with_threshold(name, strategy, params, X, y, time_ordered, seed=C.SEED, n_jobs=-1):
    """Fit on the first 80% of training, choose thresholds on the last 20%
    (validation), then refit on all training data."""
    tr, va = _val_split(len(y), y, time_ordered, seed)
    m = make_model(name, strategy, params, y[tr], seed, n_jobs).fit(X.iloc[tr], y[tr])
    pv = m.predict_proba(X.iloc[va])[:, 1]
    thr = best_f1_threshold(y[va], pv) if y[va].min() != y[va].max() else 0.5
    thr_low = recall_threshold(y[va], pv, 0.90) if y[va].sum() else thr
    final = make_model(name, strategy, params, y, seed, n_jobs).fit(X, y)
    return final, {"threshold": thr, "low_threshold": min(thr_low, thr)}


# --------------------------------------------------------------- baselines
def baseline_scores(name, train: pd.DataFrame, test: pd.DataFrame):
    """Returns (test scores, threshold).
    Prevalence: constant base rate (chance level for PR-AUC; predicts the
    majority class). LT: rank by inverse size 1/(la+ld) (Yang et al., 2016);
    flags the same share of commits as the training prevalence."""
    prev = float(train[C.LABEL].mean())
    if name == "Prevalence":
        s = np.full(len(test), prev)
        return s, (1.0 if prev < 0.5 else 0.0)
    if name == "LT":
        s_tr = 1.0 / effort(train)
        s = 1.0 / effort(test)
        thr = float(np.quantile(s_tr, 1 - prev))
        return s, thr
    raise ValueError(name)
