"""Statistical tests: Wilcoxon, Cliff's delta, Friedman + Nemenyi (Demsar 2006)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

# Nemenyi critical values q_0.05 (Demsar, 2006, Table 5), k = 2..10
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164}


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def cliffs_magnitude(d: float) -> str:
    d = abs(d)
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


def wilcoxon(a, b, alternative="two-sided") -> dict:
    """Paired Wilcoxon signed-rank test of a vs b + Cliff's delta."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a, b):
        p = np.nan
    else:
        p = float(stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)
    d = cliffs_delta(a, b)
    return dict(n=int(len(a)), median_a=float(np.median(a)), median_b=float(np.median(b)),
                median_diff=float(np.median(a - b)), p=p, cliffs_delta=d,
                magnitude=cliffs_magnitude(d))


def friedman_nemenyi(table: pd.DataFrame) -> dict:
    """table: rows = blocks (projects), columns = methods; higher = better."""
    t = table.dropna()
    k, n = t.shape[1], t.shape[0]
    stat, p = stats.friedmanchisquare(*[t[c].to_numpy() for c in t.columns])
    ranks = t.rank(axis=1, ascending=False).mean().sort_values()
    cd = _Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n))
    nem = sp.posthoc_nemenyi_friedman(t.to_numpy())
    nem.index = nem.columns = t.columns
    return dict(chi2=float(stat), p=float(p), n=n, k=k, avg_ranks=ranks, cd=float(cd),
                nemenyi=nem)


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
