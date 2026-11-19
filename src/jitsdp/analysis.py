"""Turns raw result tables into hypothesis tests (H1-H3), figures and a
Markdown report. Every number in report.md is produced here."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from . import config as C  # noqa: E402
from .metrics import _curve  # noqa: E402
from .stats import fmt_p, friedman_nemenyi, wilcoxon  # noqa: E402

P3 = f"P3-G{C.DEFAULT_GAP}"
PROTO_ORDER = ["P1", "P2"] + [f"P3-G{g}" for g in C.GAPS]
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52",
           "LT": "#8172B2", "Prevalence": "#937860"}


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES / name, dpi=150)
    plt.close(fig)
    return (C.FIGURES / name).relative_to(C.ROOT).as_posix()


def per_project_protocol(main: pd.DataFrame) -> pd.DataFrame:
    """One value per (project, protocol, model) under the default strategy;
    P1 is averaged over its random seeds."""
    d = main[main.strategy.isin([C.DEFAULT_STRATEGY, "-"])]
    num = ["pr_auc", "pr_auc_lift", "roc_auc", "f1", "mcc", "precision", "recall",
           "recall20", "popt", "recall20_density", "popt_density", "brier",
           "prevalence", "train_prev", "n_train", "withheld"]
    return d.groupby(["project", "protocol", "model"])[num].mean().reset_index()


# ======================================================================= RQ1
def rq1(pp: pd.DataFrame, windows: pd.DataFrame, out: list):
    ml = pp[pp.model.isin(C.MODELS)]
    piv = lambda metric: ml.pivot_table(index=["project", "model"], columns="protocol",
                                        values=metric)
    out.append("## RQ1 - How much does performance fall under realistic evaluation?\n")
    summary = (ml.groupby(["protocol", "model"])[["pr_auc", "pr_auc_lift", "roc_auc", "f1",
                                                  "mcc", "recall20"]]
               .median().reindex(PROTO_ORDER, level=0))
    out.append("Median across the retained projects (default strategy = class weighting):\n")
    out.append(_md(summary) + "\n")

    tests = []
    for metric in ["pr_auc", "f1", "pr_auc_lift", "mcc"]:
        t = piv(metric)
        for a, b, label in [("P1", P3, "P1 vs P3 (H1)"), ("P1", "P2", "P1 vs P2 (splitting)"),
                            ("P2", P3, "P2 vs P3 (latency gap)")]:
            r = wilcoxon(t[a], t[b], alternative="greater")
            tests.append(dict(comparison=label, metric=metric, **r))
    tests = pd.DataFrame(tests)
    out.append("Paired one-sided Wilcoxon signed-rank tests over (project, model) pairs "
               "(alternative: the first protocol scores higher):\n")
    show = tests.assign(p=tests.p.map(fmt_p))
    out.append(_md(show.set_index(["comparison", "metric"])) + "\n")

    # windows: realistic vs shuffled counterpart, paired per window x model
    w = windows[windows.model.isin(C.MODELS)]
    wp = w.pivot_table(index=["project", "window", "model"], columns="setting",
                       values=["pr_auc", "f1", "pr_auc_lift"])
    wt = []
    for metric in ["pr_auc", "f1", "pr_auc_lift"]:
        r = wilcoxon(wp[(metric, "random")], wp[(metric, "realistic")], alternative="greater")
        wt.append(dict(metric=metric, **r))
    wt = pd.DataFrame(wt)
    out.append("Rolling windows: each realistic window (train 12 m -> gap 90 d -> test 3 m) "
               "is paired with a shuffled split of the *same commits* (random vs realistic):\n")
    out.append(_md(wt.assign(p=wt.p.map(fmt_p)).set_index("metric")) + "\n")

    h1 = tests[(tests.comparison == "P1 vs P3 (H1)") & tests.metric.isin(["pr_auc", "f1"])]
    ok = bool((h1.p < C.ALPHA).all())
    verdict = "SUPPORTED" if ok else "NOT SUPPORTED"
    drop_pr = float(h1[h1.metric == "pr_auc"].median_diff.iloc[0])
    drop_f1 = float(h1[h1.metric == "f1"].median_diff.iloc[0])
    out.append(f"**H1 verdict: {verdict}.** Median paired drop P1 -> {P3}: "
               f"PR-AUC {drop_pr:+.3f}, F1 {drop_f1:+.3f}.\n")

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], "pr_auc", "PR-AUC"), (axes[1], "f1", "F1")]:
        sns.boxplot(data=ml, x="protocol", y=metric, hue="model", order=PROTO_ORDER,
                    palette=PALETTE, ax=ax, fliersize=2)
        ax.set_title(f"{title} by evaluation protocol ({ml.project.nunique()} projects)")
        ax.set_xlabel("")
    f1 = _save(fig, "rq1_protocols.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    gs = ml[ml.protocol.isin(["P2"] + [f"P3-G{g}" for g in C.GAPS])]
    gs = gs.assign(G=gs.protocol.map(lambda p: 0 if p == "P2" else int(p.split("G")[1])))
    sns.lineplot(data=gs, x="G", y="pr_auc", hue="model", estimator="median",
                 errorbar=None, marker="o", palette=PALETTE, ax=ax)
    ax.set_xlabel("latency gap G (days); 0 = P2")
    ax.set_ylabel("median PR-AUC")
    ax.set_title("Sensitivity to the label-latency gap")
    f2 = _save(fig, "rq1_gap_sensitivity.png")
    out.append(f"![protocols]({f1})\n\n![gap]({f2})\n")

    gap_tab = (ml[ml.protocol.str.startswith("P3") | (ml.protocol == "P2")]
               .groupby(["protocol", "model"])[["pr_auc", "f1", "recall20", "n_train", "withheld"]]
               .median().unstack("model").reindex(PROTO_ORDER[1:]))
    out.append("Sensitivity analysis over G (medians):\n")
    out.append(_md(gap_tab[["pr_auc", "f1"]]) + "\n")
    return dict(h1=verdict, tests=tests, window_tests=wt)


# ======================================================================= RQ2
def rq2(main: pd.DataFrame, pp: pd.DataFrame, out: list):
    out.append("## RQ2 - Imbalance strategies, models and baselines under P3\n")
    d = main[(main.protocol == P3) & main.model.isin(C.MODELS)]
    tab = d.groupby(["strategy", "model"])[["precision", "recall", "f1", "mcc", "pr_auc",
                                            "recall20"]].median()
    out.append(f"Median over projects under {P3}:\n")
    out.append(_md(tab) + "\n")

    tests = []
    for metric, alt in [("recall", "greater"), ("f1", "greater"), ("pr_auc", "two-sided")]:
        p = d.pivot_table(index=["project", "model"], columns="strategy", values=metric)
        for other in ["weight", "under"]:
            r = wilcoxon(p[other], p["none"], alternative=alt)
            tests.append(dict(comparison=f"{other} vs none", metric=metric, alternative=alt, **r))
    tests = pd.DataFrame(tests)
    out.append("Wilcoxon tests (pairs = project x model):\n")
    out.append(_md(tests.assign(p=tests.p.map(fmt_p)).set_index(["comparison", "metric"])) + "\n")

    w = tests[tests.comparison == "weight vs none"].set_index("metric")
    h2a = (w.loc["recall", "p"] < C.ALPHA and w.loc["f1", "p"] < C.ALPHA
           and not (w.loc["pr_auc", "p"] < C.ALPHA and w.loc["pr_auc", "median_diff"] < 0))

    # Friedman + Nemenyi over rankers (PR-AUC) -------------------------------
    p3 = pp[pp.protocol == P3]
    rank_pr = p3.pivot_table(index="project", columns="model", values="pr_auc")
    fr_pr = friedman_nemenyi(rank_pr)
    # effort-aware: raw ranking, density ranking (improvement) and baselines
    eff = p3.pivot_table(index="project", columns="model", values="recall20")
    dens = p3[p3.model.isin(C.MODELS)].pivot_table(index="project", columns="model",
                                                   values="recall20_density")
    dens.columns = [f"{c}-density" for c in dens.columns]
    eff_all = pd.concat([eff, dens], axis=1)
    fr_eff = friedman_nemenyi(eff_all)

    out.append(f"Friedman test on PR-AUC across {fr_pr['n']} projects: chi2 = {fr_pr['chi2']:.2f}, "
               f"p = {fmt_p(fr_pr['p'])}; Nemenyi CD = {fr_pr['cd']:.2f}. Average ranks (1 = best):\n")
    out.append(_md(fr_pr["avg_ranks"].to_frame("avg_rank")) + "\n")
    out.append(f"Friedman test on Recall@20% effort: chi2 = {fr_eff['chi2']:.2f}, "
               f"p = {fmt_p(fr_eff['p'])}; CD = {fr_eff['cd']:.2f}. Average ranks:\n")
    out.append(_md(fr_eff["avg_ranks"].to_frame("avg_rank")) + "\n")

    vs_lt = []
    for col in eff_all.columns:
        if col in C.BASELINES:
            continue
        r = wilcoxon(eff_all[col], eff_all["LT"], alternative="greater")
        vs_lt.append(dict(ranker=col, **r))
    vs_lt = pd.DataFrame(vs_lt)
    out.append("Recall@20% effort of each supervised ranker vs the unsupervised LT baseline "
               "(one-sided Wilcoxon over projects):\n")
    out.append(_md(vs_lt.assign(p=vs_lt.p.map(fmt_p)).set_index("ranker")) + "\n")
    raw = vs_lt[vs_lt.ranker.isin(C.MODELS)]
    h2b = bool((raw.p < C.ALPHA).any())
    h2b_density = bool((vs_lt[~vs_lt.ranker.isin(C.MODELS)].p < C.ALPHA).any())
    verdict = "SUPPORTED" if (h2a and h2b) else (
        "PARTIALLY SUPPORTED" if (h2a or h2b) else "NOT SUPPORTED")
    out.append(f"**H2 verdict: {verdict}.** (a) class weighting improves recall and F1 without "
               f"lowering PR-AUC: {'yes' if h2a else 'no'}; (b) an ML model (risk ranking) beats "
               f"LT on Recall@20% effort: {'yes' if h2b else 'no'}. Improvement - ranking by "
               f"*risk per changed line* (density) beats LT: {'yes' if h2b_density else 'no'}.\n")

    # model-selection rule (Sec 8.7)
    ml = p3[p3.model.isin(C.MODELS)].groupby("model")[["pr_auc", "recall", "f1", "recall20",
                                                       "recall20_density", "popt_density"]].median()
    best_pr, best_eff = ml.pr_auc.idxmax(), ml.recall20_density.idxmax()
    out.append("Model-selection rule (median over projects):\n")
    out.append(_md(ml) + "\n")
    tradeoff = ("the same model" if best_pr == best_eff else
                f"different models -> trade-off: {best_pr} ranks risk best, {best_eff} "
                "finds the most defects per review effort")
    out.append(f"Best PR-AUC: **{best_pr}**; best effort-aware recall (density ranking): "
               f"**{best_eff}** ({tradeoff}).\n")

    # figures
    fig, ax = plt.subplots(figsize=(9, 4))
    m = d.melt(id_vars=["strategy", "model"], value_vars=["recall", "f1", "pr_auc"])
    sns.barplot(data=m, x="variable", y="value", hue="strategy", estimator=np.median,
                errorbar=None, ax=ax, hue_order=C.STRATEGIES)
    ax.set_title(f"Imbalance strategies under {P3} (median over projects x models)")
    ax.set_xlabel("")
    f1 = _save(fig, "rq2_imbalance.png")
    f2 = _rank_plot(fr_pr, "PR-AUC", "rq2_ranks_prauc.png")
    f3 = _rank_plot(fr_eff, "Recall@20% effort", "rq2_ranks_effort.png")
    out.append(f"![imbalance]({f1})\n\n![ranks]({f2})\n\n![ranks]({f3})\n")
    return dict(h2=verdict, tests=tests, vs_lt=vs_lt, friedman_pr=fr_pr, friedman_eff=fr_eff,
                best_pr=best_pr, best_eff=best_eff)


def _rank_plot(fr, metric, name):
    r = fr["avg_ranks"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(r) + 1.2))
    colors = [PALETTE.get(i.split("-")[0], "#777") for i in r.index]
    ax.barh(r.index[::-1], r.values[::-1], color=colors[::-1])
    best = r.min()
    ax.axvline(best + fr["cd"], ls="--", color="k", lw=1)
    ax.text(best + fr["cd"], len(r) - 0.4, f" best + CD ({fr['cd']:.2f})", fontsize=8)
    ax.set_xlabel("average rank (lower = better)")
    ax.set_title(f"Friedman/Nemenyi average ranks - {metric}")
    return _save(fig, name)


# ======================================================================= RQ3
def rq3(shapw: pd.DataFrame, expl: dict, out: list):
    out.append("## RQ3 - Which features matter, and are they stable over time?\n")
    rows = []
    for proj, g in shapw.groupby("project"):
        piv = g.pivot_table(index="window", columns="feature", values="mean_abs_shap").sort_index()
        for i in range(1, len(piv)):
            a, b = piv.iloc[i - 1], piv.iloc[i]
            top_a, top_b = set(a.nlargest(5).index), set(b.nlargest(5).index)
            union = sorted(top_a | top_b)
            rho_all = spearmanr(a, b).statistic
            rho_top = spearmanr(a[union], b[union]).statistic
            rows.append(dict(project=proj, window=int(piv.index[i]), rho_all=rho_all,
                             rho_top5=rho_top, jaccard_top5=len(top_a & top_b) / len(top_a | top_b)))
    st = pd.DataFrame(rows)
    by_proj = st.groupby("project")[["rho_top5", "rho_all", "jaccard_top5"]].median()
    by_proj["pairs"] = st.groupby("project").size()
    med_top = float(st.rho_top5.median())
    share = float((st.rho_top5 >= 0.7).mean())
    out.append("Consecutive-window stability of XGBoost mean |SHAP| rankings:\n")
    out.append(_md(by_proj) + "\n")
    verdict = "SUPPORTED" if med_top >= 0.7 else "NOT SUPPORTED"
    out.append(f"Overall median Spearman (top-5 union) = {med_top:.2f}; "
               f"{share:.0%} of consecutive pairs reach 0.7; median full-ranking Spearman = "
               f"{st.rho_all.median():.2f}; median top-5 Jaccard = {st.jaccard_top5.median():.2f}.\n")
    out.append(f"**H3 verdict: {verdict}.**\n")

    shapw = shapw.copy()
    shapw["rank"] = shapw.groupby(["project", "window"]).mean_abs_shap.rank(ascending=False)
    freq = (shapw[shapw["rank"] <= 5].groupby("feature").size()
            / shapw.groupby(["project", "window"]).ngroups).sort_values(ascending=False)
    out.append("How often each feature is in the top-5 across all windows:\n")
    out.append(_md(freq.to_frame("share_of_windows_in_top5")) + "\n")

    # global pooled SHAP + LR coefficients
    sv, Z = expl["shap_values"], expl["shap_X"]
    imp = pd.Series(np.abs(sv).mean(0), index=Z.columns).sort_values(ascending=False)
    out.append(f"Pooled {P3} XGBoost - global mean |SHAP|:\n")
    out.append(_md(imp.to_frame("mean_abs_shap")) + "\n")
    out.append(f"Pooled {P3} Logistic Regression coefficients (standardized, after log1p; "
               f"collinearity filter dropped: {', '.join(expl['lr_dropped']) or 'none'}):\n")
    out.append(_md(expl["lr_coefs"].set_index("feature")) + "\n")

    plt.figure()
    shap.summary_plot(sv, Z, show=False, max_display=13)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary - pooled XGBoost ({P3})", y=1.0)
    f1 = _save(fig, "rq3_shap_beeswarm.png")

    big = shapw.groupby("project").window.nunique().idxmax()
    hm = (shapw[shapw.project == big].pivot_table(index="feature", columns="test_start",
                                                  values="rank"))
    hm = hm.loc[hm.mean(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(hm, cmap="viridis_r", ax=ax, cbar_kws={"label": "rank (1 = most important)"})
    ax.set_title(f"Feature-importance rank over time - {big}")
    ax.set_xlabel("test window start")
    f2 = _save(fig, "rq3_rank_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.histplot(st.rho_top5.dropna(), bins=20, ax=ax)
    ax.axvline(0.7, color="r", ls="--")
    ax.set_xlabel("Spearman rho between consecutive windows (top-5 union)")
    ax.set_title("Stability of feature rankings (all projects)")
    f3 = _save(fig, "rq3_stability_hist.png")
    out.append(f"![beeswarm]({f1})\n\n![heatmap]({f2})\n\n![hist]({f3})\n")
    return dict(h3=verdict, stability=st, top5_freq=freq)


# ============================================================ extra analyses
def data_section(val: dict, excl: pd.DataFrame, out: list):
    s = val["summary"]
    out.append("## Dataset readiness gate and validation\n")
    out.append(f"- Commits: **{s['n_commits']:,}** from **{s['n_projects']}** projects; "
               f"{s['date_range'][0][:10]} -> {s['date_range'][1][:10]}; prevalence "
               f"{s['prevalence']:.3f}.\n- SHA-256: `{s['sha256']}`\n"
               f"- Missing values: {s['missing_values']}; duplicate commit ids: "
               f"{s['duplicate_commit_ids']}; duplicate feature rows: {s['duplicate_feature_rows']}; "
               f"negative values: {s['negative_values'] or 'none'}.\n"
               f"- Fix-commit timestamps available: **no** -> P3 uses the gap approximation.\n"
               f"- Readiness gate: {json.dumps(s['readiness_gate'])} -> "
               f"**{'PASSED' if s['readiness_passed'] else 'FAILED'}**\n")
    out.append("Pre-registered exclusion rule: a project is kept only if the chronological "
               f"70/30 split has at least {C.MIN_CLASS} commits of each class in both parts.\n")
    out.append(_md(excl.set_index("project"), floatfmt=".0f") + "\n")
    if C.DATASET == "apachejit":
        out.append("> **Finding (data audit):** `apache/hadoop` has *zero* defect-inducing commits "
                   "although its sub-projects `hadoop-hdfs` and `hadoop-mapreduce` have 76% and 63%; "
                   "this looks like a labeling artefact of the repository split, so it is excluded "
                   "rather than treated as a defect-free project.\n")

    py = val["per_year"]
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    ax1.bar(py.year, py.commits, color="#ccc")
    ax1.set_ylabel("commits")
    ax2 = ax1.twinx()
    ax2.plot(py.year, py.prevalence, color="#C44E52", marker="o")
    ax2.set_ylabel("share defect-inducing", color="#C44E52")
    ax2.grid(False)
    ax1.set_title("Label maturity: defect prevalence by commit year")
    f = _save(fig, "data_prevalence_by_year.png")
    out.append(f"![prevalence]({f})\n")


def label_maturity(pp: pd.DataFrame, out: list):
    d = pp[(pp.protocol == "P2") & (pp.model == "LR")][["project", "train_prev", "prevalence"]]
    d = d.rename(columns={"prevalence": "test_prev"}).set_index("project")
    d["ratio"] = d.test_prev / d.train_prev
    out.append("### Improvement: label-maturity (right-censoring) check\n")
    out.append("Recent commits have had less time for their bugs to be found, so the newest "
               "30% (test) has a lower observed prevalence than the oldest 70% in most projects. "
               "This is why we also report **PR-AUC lift** (PR-AUC / test prevalence), which "
               "compares protocols fairly when prevalence differs:\n")
    out.append(_md(d) + "\n")


def effort_curves(fitted: dict, out: list):
    f = fitted[("XGB", C.DEFAULT_STRATEGY)]
    X, y, prob = f["X_test"], f["y_test"], f["prob"]
    eff = f["effort"] if "effort" in f else \
        np.maximum(X["la"].to_numpy() + X["ld"].to_numpy(), 1).astype(float)
    rng = np.random.default_rng(0)
    rankers = {"XGB (risk)": prob, "XGB (risk / LOC)": prob / eff, "LT (1 / LOC)": 1 / eff,
               "Random": rng.random(len(y))}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in rankers.items():
        order = np.lexsort((rng.random(len(s)), -s))
        e, d = _curve(y, eff, order)
        ax.plot(e, d, label=name)
    ax.axvline(0.2, color="k", ls=":", lw=1)
    ax.set_xlabel("share of review effort (LOC inspected)")
    ax.set_ylabel("share of defect-inducing commits found")
    ax.set_title(f"Effort-aware curves - pooled {P3} test set")
    ax.legend()
    out.append(f"![effort]({_save(fig, 'rq2_effort_curves.png')})\n")


def bootstrap_ci(fitted: dict, n=1000) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score
    rows = []
    rng = np.random.default_rng(C.SEED)
    for (name, strat), f in fitted.items():
        if strat != C.DEFAULT_STRATEGY:
            continue
        y, p = f["y_test"], f["prob"]
        vals = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(average_precision_score(y[i], p[i]))
        rows.append(dict(model=name, pr_auc=average_precision_score(y, p),
                         ci_low=np.percentile(vals, 2.5), ci_high=np.percentile(vals, 97.5)))
    return pd.DataFrame(rows).set_index("model")
