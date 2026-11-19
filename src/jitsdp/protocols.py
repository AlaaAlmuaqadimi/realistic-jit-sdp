"""Evaluation protocols P1 (random), P2 (chronological), P3 (chronological +
label-latency gap) and rolling windows, plus automated leakage checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .data import chrono_split_index


@dataclass
class Split:
    protocol: str
    train: pd.DataFrame
    test: pd.DataFrame
    time_ordered: bool          # tune/validate with time-ordered folds?
    gap: int = 0
    seed: int | None = None
    info: dict = field(default_factory=dict)


def p1_random(g: pd.DataFrame, seed: int) -> Split:
    tr, te = train_test_split(g, test_size=C.TEST_FRAC, stratify=g[C.LABEL],
                              random_state=seed)
    return Split("P1", tr.sort_values(C.TIME, kind="mergesort"),
                 te.sort_values(C.TIME, kind="mergesort"), False, seed=seed)


def p2_chrono(g: pd.DataFrame) -> Split:
    g = g.sort_values(C.TIME, kind="mergesort")
    k = chrono_split_index(len(g))
    return Split("P2", g.iloc[:k], g.iloc[k:], True)


def p3_latency(g: pd.DataFrame, gap_days: int) -> Split:
    """Same test commits as P2; commits authored in the G days before the
    cut-off are treated as *unlabeled* and removed from training (their label
    would not yet be observable), never relabeled as clean."""
    s = p2_chrono(g)
    cutoff = s.test[C.TIME].min()
    keep = s.train[C.TIME] < cutoff - gap_days * 86400
    return Split(f"P3-G{gap_days}", s.train[keep], s.test, True, gap=gap_days,
                 info={"withheld_unlabeled": int((~keep).sum())})


def rolling_windows(g: pd.DataFrame, gap_days: int = C.DEFAULT_GAP):
    """Train on 12 months, (skip G days of unlabeled commits), test on the
    next 3 months, step 3 months. Yields (window_id, test_start, Split)."""
    g = g.sort_values(C.TIME, kind="mergesort")
    dates = g["date"]
    start = dates.min().normalize() + pd.DateOffset(months=C.WIN_TRAIN_MONTHS) \
        + pd.Timedelta(days=gap_days)
    end = dates.max()
    wid = 0
    t = start
    while t + pd.DateOffset(months=C.WIN_TEST_MONTHS) <= end + pd.Timedelta(days=1):
        test_end = t + pd.DateOffset(months=C.WIN_TEST_MONTHS)
        train_end = t - pd.Timedelta(days=gap_days)
        train_start = train_end - pd.DateOffset(months=C.WIN_TRAIN_MONTHS)
        tr = g[(dates >= train_start) & (dates < train_end)]
        te = g[(dates >= t) & (dates < test_end)]
        ok = (tr[C.LABEL].sum() >= C.WIN_MIN_POS_TRAIN
              and (1 - tr[C.LABEL]).sum() >= C.WIN_MIN_POS_TRAIN
              and te[C.LABEL].sum() >= C.WIN_MIN_POS_TEST
              and (1 - te[C.LABEL]).sum() >= C.WIN_MIN_NEG_TEST)
        if ok:
            yield wid, t, Split(f"W-G{gap_days}", tr, te, True, gap=gap_days)
            wid += 1
        t = t + pd.DateOffset(months=C.WIN_STEP_MONTHS)


def random_counterpart(s: Split, seed: int) -> Split:
    """Shuffle the commits of a window and re-split with the same test size:
    the paired 'retrospective' version of a realistic window (for RQ1)."""
    pool = pd.concat([s.train, s.test])
    tr, te = train_test_split(pool, test_size=len(s.test), stratify=pool[C.LABEL],
                              random_state=seed)
    return Split("W-random", tr, te, False, seed=seed)


# ------------------------------------------------------------ leakage checks
class LeakageError(AssertionError):
    pass


def check_split(s: Split, features=C.FEATURES) -> None:
    """Automated leakage checks required by the success criteria (Sec 9)."""
    bad = set(features) & set(C.FORBIDDEN)
    if bad:
        raise LeakageError(f"forbidden feature(s) used: {bad}")
    if set(s.train[C.ID]) & set(s.test[C.ID]):
        raise LeakageError("train and test share commits")
    if s.time_ordered:
        # ties within the same second are allowed (commits pushed together)
        if s.train[C.TIME].max() > s.test[C.TIME].min():
            raise LeakageError(f"{s.protocol}: training dates do not precede test dates")
        if s.gap:
            margin = s.test[C.TIME].min() - s.train[C.TIME].max()
            if margin < s.gap * 86400:
                raise LeakageError(f"{s.protocol}: latency gap violated")
