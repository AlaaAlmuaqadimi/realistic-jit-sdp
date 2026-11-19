"""Threshold, ranking and effort-aware metrics (Sec 8.7)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y, prob) -> float:
    """Threshold maximizing F1 -- used on *validation* data only."""
    p, r, t = precision_recall_curve(y, prob)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    if len(t) == 0:
        return 0.5
    return float(t[int(np.argmax(f1))])


def recall_threshold(y, prob, target=0.90) -> float:
    """Largest threshold that still reaches `target` recall on validation."""
    p, r, t = precision_recall_curve(y, prob)
    ok = np.where(r[:-1] >= target)[0]
    return float(t[ok.max()]) if len(ok) else float(np.min(prob))


def _effort_order(score, effort, rng):
    # sort by score desc; ties broken at random (never by effort, which would
    # silently turn a constant-score model into the size baseline)
    return np.lexsort((rng.random(len(score)), -np.asarray(score, float)))


def _curve(y, effort, order):
    e = np.concatenate([[0.0], np.cumsum(effort[order]) / effort.sum()])
    d = np.concatenate([[0.0], np.cumsum(y[order]) / max(y.sum(), 1)])
    return e, d


def effort_aware(y, score, effort, budget=0.20, seed=0) -> tuple[float, float]:
    """Recall@budget effort and Popt (Kamei et al., 2013)."""
    y = np.asarray(y, int)
    effort = np.asarray(effort, float)
    if y.sum() == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    order = _effort_order(score, effort, rng)
    e, d = _curve(y, effort, order)
    recall_at = float(np.interp(budget, e, d))
    density = y / effort
    opt = np.lexsort((effort, -density))            # best possible ordering
    worst = np.lexsort((-effort, density))          # worst possible ordering
    area = lambda o: np.trapezoid(*_curve(y, effort, o)[::-1])
    a_m, a_opt, a_worst = np.trapezoid(d, e), area(opt), area(worst)
    popt = 1 - (a_opt - a_m) / max(a_opt - a_worst, 1e-12)
    return recall_at, float(popt)


def evaluate(y, prob, threshold, effort, budget=0.20, seed=0) -> dict:
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prev = float(y.mean())
    one_class = len(np.unique(y)) < 2
    pr_auc = np.nan if one_class else float(average_precision_score(y, prob))
    out = dict(
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)) if not one_class else np.nan,
        accuracy=float(accuracy_score(y, pred)),
        pr_auc=pr_auc,
        roc_auc=np.nan if one_class else float(roc_auc_score(y, prob)),
        pr_auc_lift=pr_auc / prev if prev > 0 else np.nan,   # improvement
        brier=float(brier_score_loss(y, np.clip(prob, 0, 1))),
        prevalence=prev, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        threshold=float(threshold),
    )
    out["recall20"], out["popt"] = effort_aware(y, prob, effort, budget, seed)
    # effort-aware *ranking* of a supervised model: risk per changed line
    out["recall20_density"], out["popt_density"] = effort_aware(
        y, prob / effort, effort, budget, seed)
    return out
