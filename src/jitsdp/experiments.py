"""Experiment runners: main protocol comparison (per project + pooled),
imbalance strategies, rolling windows with SHAP, and the prototype model."""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

from . import config as C
from .data import effort
from .metrics import evaluate
from .models import DEFAULTS, baseline_scores, fit_with_threshold, tune
from .protocols import (check_split, p1_random, p2_chrono, p3_latency,
                        random_counterpart, rolling_windows)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _xy(df):
    return df[C.FEATURES].reset_index(drop=True), df[C.LABEL].to_numpy()


def eval_split(s, project, combos, *, tuned=True, n_jobs=-1, keep=False,
               baselines=True):
    check_split(s)
    Xtr, ytr = _xy(s.train)
    Xte, yte = _xy(s.test)
    eff = effort(s.test)
    seed = C.SEED if s.seed is None else s.seed
    base = dict(project=project, protocol=s.protocol, gap=s.gap, seed=s.seed,
                n_train=len(ytr), n_test=len(yte), train_prev=float(ytr.mean()),
                withheld=s.info.get("withheld_unlabeled", 0))
    rows, fitted = [], {}
    for name, strat in combos:
        t0 = time.time()
        params = tune(name, strat, Xtr, ytr, s.time_ordered, seed, n_jobs) if tuned else DEFAULTS[name]
        model, th = fit_with_threshold(name, strat, params, Xtr, ytr, s.time_ordered, seed, n_jobs)
        prob = model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, prob, th["threshold"], eff, C.EFFORT_BUDGET, seed)
        rows.append({**base, "model": name, "strategy": strat,
                     "params": json.dumps(params), "fit_seconds": round(time.time() - t0, 2), **m})
        if keep:
            fitted[(name, strat)] = dict(model=model, thresholds=th, params=params,
                                        X_test=Xte, y_test=yte, prob=prob, effort=eff)
    if baselines:
        for b in C.BASELINES:
            sc, thr = baseline_scores(b, s.train, s.test)
            m = evaluate(yte, sc, thr, eff, C.EFFORT_BUDGET, seed)
            m["brier"] = np.nan
            rows.append({**base, "model": b, "strategy": "-", "params": "", **m})
    return rows, fitted


def protocol_suite(project, g, n_jobs=-1, p1_seeds=C.P1_SEEDS, keep_p3=False):
    weight = [(m, C.DEFAULT_STRATEGY) for m in C.MODELS]
    rows = []
    for seed in p1_seeds:
        rows += eval_split(p1_random(g, seed), project, weight, n_jobs=n_jobs)[0]
    rows += eval_split(p2_chrono(g), project, weight, n_jobs=n_jobs)[0]
    fitted = {}
    for G in C.GAPS:
        combos = weight
        if G == C.DEFAULT_GAP:   # imbalance strategies compared under P3
            combos = [(m, s) for m in C.MODELS for s in C.STRATEGIES]
        r, f = eval_split(p3_latency(g, G), project, combos, n_jobs=n_jobs,
                          keep=keep_p3 and G == C.DEFAULT_GAP)
        rows += r
        fitted.update(f)
    return rows, fitted


def run_main(df, projects, n_parallel=6, log=print):
    log(f"[main] {len(projects)} projects in parallel ({n_parallel} workers)")
    jobs = (delayed(protocol_suite)(p, df[df[C.PROJECT] == p], 2) for p in projects)
    out = Parallel(n_jobs=n_parallel, verbose=5)(jobs)
    rows = [r for rs, _ in out for r in rs]
    return pd.DataFrame(rows)


def run_pooled(df, projects, log=print):
    """Pooled analysis over all retained projects (global time order)."""
    log("[pooled] running protocol suite on pooled data")
    g = df[df[C.PROJECT].isin(projects)]
    rows, fitted = protocol_suite("POOLED", g, n_jobs=-1, p1_seeds=[0], keep_p3=True)
    return pd.DataFrame(rows), fitted


# --------------------------------------------------------- rolling windows
def _window_project(project, g):
    rows, shap_rows = [], []
    combos = [(m, C.DEFAULT_STRATEGY) for m in C.MODELS]
    for wid, t, s in rolling_windows(g, C.DEFAULT_GAP):
        r, fitted = eval_split(s, project, combos, tuned=False, n_jobs=1, keep=True)
        for x in r:
            x.update(window=wid, test_start=str(t.date()), setting="realistic")
        rows += r
        rnd = random_counterpart(s, seed=wid)
        r2, _ = eval_split(rnd, project, combos, tuned=False, n_jobs=1, baselines=False)
        for x in r2:
            x.update(window=wid, test_start=str(t.date()), setting="random")
        rows += r2
        # SHAP of the XGBoost model in this window (RQ3)
        f = fitted[("XGB", C.DEFAULT_STRATEGY)]
        pipe = f["model"]
        Xs = f["X_test"]
        if len(Xs) > C.SHAP_SAMPLE:
            Xs = Xs.sample(C.SHAP_SAMPLE, random_state=wid)
        Z = pipe.named_steps["prep"].transform(Xs)
        sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Z)
        imp = np.abs(sv).mean(axis=0)
        for feat, v in zip(Z.columns, imp):
            shap_rows.append(dict(project=project, window=wid, test_start=str(t.date()),
                                  feature=feat, mean_abs_shap=float(v)))
    return rows, shap_rows


def run_windows(df, projects, n_parallel=10, log=print):
    log(f"[windows] rolling windows for {len(projects)} projects")
    out = Parallel(n_jobs=n_parallel, verbose=5)(
        delayed(_window_project)(p, df[df[C.PROJECT] == p]) for p in projects)
    rows = [r for rs, _ in out for r in rs]
    shap_rows = [r for _, ss in out for r in ss]
    return pd.DataFrame(rows), pd.DataFrame(shap_rows)


# ------------------------------------------------- explanations + prototype
def pooled_explanations(fitted, n=2000):
    """Global SHAP for the pooled P3 XGBoost model + LR coefficients."""
    f = fitted[("XGB", C.DEFAULT_STRATEGY)]
    Xs = f["X_test"].sample(min(n, len(f["X_test"])), random_state=C.SEED)
    Z = f["model"].named_steps["prep"].transform(Xs)
    sv = shap.TreeExplainer(f["model"].named_steps["clf"]).shap_values(Z)
    lr = fitted[("LR", C.DEFAULT_STRATEGY)]["model"]
    prep, clf = lr.named_steps["prep"], lr.named_steps["clf"]
    coefs = pd.DataFrame({"feature": prep.keep_, "coef": clf.coef_[0]})
    coefs["odds_ratio_per_sd"] = np.exp(coefs["coef"])
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    return dict(shap_values=sv, shap_X=Z, lr_coefs=coefs, lr_dropped=prep.dropped_)


def save_prototype(fitted, pooled_rows: pd.DataFrame, sha: str):
    """Persist the recommended model for the command-line prototype."""
    p3 = pooled_rows[(pooled_rows.protocol == f"P3-G{C.DEFAULT_GAP}")
                     & (pooled_rows.strategy == C.DEFAULT_STRATEGY)
                     & pooled_rows.model.isin(C.MODELS)]
    best = p3.sort_values("pr_auc", ascending=False).iloc[0]
    f = fitted[(best.model, C.DEFAULT_STRATEGY)]
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = dict(
        pipeline=f["model"], features=C.FEATURES,
        thresholds={"high": f["thresholds"]["threshold"],
                    "medium": f["thresholds"]["low_threshold"]},
        meta=dict(model=best.model, strategy=C.DEFAULT_STRATEGY, params=f["params"],
                  protocol=f"P3-G{C.DEFAULT_GAP} (pooled)", dataset_sha256=sha,
                  trained_on=int(best.n_train), test_pr_auc=round(float(best.pr_auc), 4),
                  test_recall20=round(float(best.recall20), 4),
                  version="1.0", created=pd.Timestamp.now().isoformat(timespec="seconds")))
    joblib.dump(bundle, C.MODELS_DIR / "jit_model.joblib")
    held = f["X_test"].copy()
    held["buggy"] = f["y_test"]
    held.sample(200, random_state=C.SEED).to_csv(C.MODELS_DIR / "heldout_examples.csv", index=False)
    return bundle["meta"]
