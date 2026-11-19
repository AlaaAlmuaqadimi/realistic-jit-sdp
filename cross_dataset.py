"""Cross-dataset analysis (ApacheJIT + Kamei et al. 2013).

Reads the per-dataset raw results written by `run_all.py --dataset ...` and
re-tests H1-H3 on all 19 projects together and per dataset. Writes
results_combined/summary.md, CSV tables and paper figures.

    python cross_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from jitsdp import analysis as A  # noqa: E402
from jitsdp import config as C  # noqa: E402
from jitsdp.stats import fmt_p, friedman_nemenyi, wilcoxon  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "results_combined"
FIG = ROOT / "paper" / "figures"
SRC = {"ApacheJIT": ROOT / "results", "Kamei": ROOT / "results_kamei"}
P3 = A.P3
MODELS = C.MODELS
lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def load():
    pp, main, win, shp = [], [], [], []
    for name, d in SRC.items():
        a = pd.read_csv(d / "per_project_protocol.csv").assign(dataset=name)
        b = pd.read_csv(d / "raw" / "main_results.csv").assign(dataset=name)
        c = pd.read_csv(d / "raw" / "window_results.csv").assign(dataset=name)
        s = pd.read_csv(d / "raw" / "window_shap.csv").assign(dataset=name)
        pp.append(a); main.append(b); win.append(c); shp.append(s)
    return (pd.concat(pp, ignore_index=True), pd.concat(main, ignore_index=True),
            pd.concat(win, ignore_index=True), pd.concat(shp, ignore_index=True))


def table(df, fmt=".3f"):
    say(df.to_markdown(floatfmt=fmt))
    say()


def rq1(pp, win):
    say("## RQ1")
    ml = pp[pp.model.isin(MODELS)]
    med = ml.groupby(["dataset", "protocol", "model"])[["pr_auc", "pr_auc_lift", "f1", "mcc",
                                                         "recall20"]].median()
    table(med.reindex(A.PROTO_ORDER, level=1))
    rows = []
    for scope, d in [("All", ml)] + [(k, g) for k, g in ml.groupby("dataset")]:
        for metric in ["pr_auc", "f1", "mcc", "pr_auc_lift"]:
            t = d.pivot_table(index=["dataset", "project", "model"], columns="protocol", values=metric)
            for a, b in [("P1", P3), ("P1", "P2"), ("P2", P3)]:
                r = wilcoxon(t[a], t[b], "greater")
                rows.append(dict(scope=scope, comparison=f"{a} vs {b}", metric=metric, **r))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rq1_tests.csv", index=False)
    table(t.assign(p=t.p.map(fmt_p)).set_index(["scope", "comparison", "metric"]))

    say("Per-model P1 vs P3 (all 19 projects), Holm-corrected:")
    pm = []
    for m in MODELS:
        d = ml[ml.model == m]
        for metric in ["pr_auc", "f1"]:
            p = d.pivot_table(index=["dataset", "project"], columns="protocol", values=metric)
            pm.append(dict(model=m, metric=metric, **wilcoxon(p["P1"], p[P3], "greater")))
    pm = pd.DataFrame(pm)
    pm["p_holm"] = np.nan
    for metric in ["pr_auc", "f1"]:
        i = pm.metric == metric
        pm.loc[i, "p_holm"] = holm(pm.loc[i, "p"])
    table(pm.set_index(["model", "metric"]), ".4f")

    say("Prevalence shift vs drop (Spearman over projects):")
    cors = []
    for scope, d in [("All", ml)] + [(k, g) for k, g in ml.groupby("dataset")]:
        for m in MODELS:
            x = d[d.model == m]
            pr = x.pivot_table(index=["dataset", "project"], columns="protocol", values="pr_auc")
            pv = x.pivot_table(index=["dataset", "project"], columns="protocol", values="prevalence")
            r = spearmanr(pv[P3] / pv["P1"], pr["P1"] - pr[P3])
            cors.append(dict(scope=scope, model=m, n=len(pr), rho=r.statistic, p=r.pvalue))
    cors = pd.DataFrame(cors)
    table(cors.set_index(["scope", "model"]), ".4f")

    w = win[win.model.isin(MODELS)]
    say("Paired windows (random vs realistic on identical commits):")
    wr = []
    for scope, d in [("All", w)] + [(k, g) for k, g in w.groupby("dataset")]:
        wp = d.pivot_table(index=["dataset", "project", "window", "model"], columns="setting",
                           values=["pr_auc", "f1", "pr_auc_lift"])
        for metric in ["pr_auc", "f1", "pr_auc_lift"]:
            r = wilcoxon(wp[(metric, "random")], wp[(metric, "realistic")], "greater")
            wr.append(dict(scope=scope, metric=metric, **r))
    wr = pd.DataFrame(wr)
    table(wr.assign(p=wr.p.map(fmt_p)).set_index(["scope", "metric"]))
    say(f"windows per dataset: {w[w.setting == 'realistic'].groupby('dataset').apply(lambda g: g.groupby('project').window.nunique().sum()).to_dict()}")
    say()

    # figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (name, d) in zip(axes, ml.groupby("dataset")):
        sns.boxplot(data=d, x="protocol", y="pr_auc", hue="model", order=A.PROTO_ORDER,
                    palette=A.PALETTE, ax=ax, fliersize=2)
        ax.set_title(f"{name} ({d.project.nunique()} projects)")
        ax.set_xlabel("")
        ax.set_ylabel("PR-AUC")
    fig.tight_layout()
    fig.savefig(FIG / "x_rq1_protocols.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    mk = {"ApacheJIT": "o", "Kamei": "s"}
    for (ds, m), x in ml.groupby(["dataset", "model"]):
        pr = x.pivot_table(index="project", columns="protocol", values="pr_auc")
        pv = x.pivot_table(index="project", columns="protocol", values="prevalence")
        ax.scatter(pv[P3] / pv["P1"], pr["P1"] - pr[P3], marker=mk[ds], c=A.PALETTE[m],
                   s=38, alpha=.85, edgecolors="k" if ds == "Kamei" else "none", linewidths=.6)
    allr = cors[(cors.scope == "All")]
    for m in MODELS:
        ax.scatter([], [], c=A.PALETTE[m], label=f"{m} (ρ = {allr[allr.model == m].rho.iloc[0]:.2f})")
    for ds in mk:
        ax.scatter([], [], marker=mk[ds], c="grey", label=ds)
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(1, color="k", lw=.8, ls=":")
    ax.set_xlabel("test prevalence under P3-G90 / test prevalence under P1")
    ax.set_ylabel("PR-AUC drop (P1 − P3-G90)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "x_rq1_drop_vs_prevalence.png", dpi=200)
    plt.close(fig)
    return t, cors, wr


def rq2(main, pp):
    say("## RQ2")
    d = main[(main.protocol == P3) & main.model.isin(MODELS)]
    table(d.groupby(["dataset", "strategy", "model"])[["precision", "recall", "f1", "mcc",
                                                        "pr_auc"]].median())
    rows = []
    for scope, x in [("All", d)] + [(k, g) for k, g in d.groupby("dataset")]:
        for metric, alt in [("recall", "greater"), ("f1", "greater"), ("pr_auc", "two-sided"),
                            ("mcc", "two-sided")]:
            p = x.pivot_table(index=["dataset", "project", "model"], columns="strategy", values=metric)
            for o in ["weight", "under"]:
                rows.append(dict(scope=scope, comparison=f"{o} vs none", metric=metric,
                                 alternative=alt, **wilcoxon(p[o], p["none"], alt)))
    t = pd.DataFrame(rows)
    table(t.assign(p=t.p.map(fmt_p)).set_index(["scope", "comparison", "metric"]))

    p3 = pp[pp.protocol == P3].copy()
    p3["block"] = p3.dataset + "/" + p3.project
    for scope, x in [("All", p3)] + [(k, g) for k, g in p3.groupby("dataset")]:
        fr = friedman_nemenyi(x.pivot_table(index="block", columns="model", values="pr_auc"))
        say(f"Friedman PR-AUC [{scope}] chi2={fr['chi2']:.2f} p={fmt_p(fr['p'])} n={fr['n']} "
            f"CD={fr['cd']:.2f} ranks={fr['avg_ranks'].round(2).to_dict()}")
    eff = p3.pivot_table(index="block", columns="model", values="recall20")
    dens = p3[p3.model.isin(MODELS)].pivot_table(index="block", columns="model",
                                                 values="recall20_density")
    dens.columns = [f"{c}-density" for c in dens.columns]
    alle = pd.concat([eff, dens], axis=1)
    fr = friedman_nemenyi(alle)
    say(f"Friedman Recall@20 [All] chi2={fr['chi2']:.2f} p={fmt_p(fr['p'])} CD={fr['cd']:.2f} "
        f"ranks={fr['avg_ranks'].round(2).to_dict()}")
    vs = []
    for scope, x in [("All", alle)] + [(k, alle[alle.index.str.startswith(k)]) for k in SRC]:
        for col in x.columns:
            if col in C.BASELINES:
                continue
            vs.append(dict(scope=scope, ranker=col, **wilcoxon(x[col], x["LT"], "greater")))
    vs = pd.DataFrame(vs)
    table(vs.assign(p=vs.p.map(fmt_p)).set_index(["scope", "ranker"]))
    say("Median Recall@20 by dataset:")
    table(p3.groupby(["dataset", "model"])[["recall20", "recall20_density", "pr_auc"]].median())

    # figure: average ranks for both criteria over 19 projects
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tab, title) in zip(axes, [(p3.pivot_table(index="block", columns="model",
                                                       values="pr_auc"), "PR-AUC"),
                                       (alle, "Recall@20% effort")]):
        f = friedman_nemenyi(tab)
        r = f["avg_ranks"]
        ax.barh(r.index[::-1], r.values[::-1],
                color=[A.PALETTE.get(i.split("-")[0], "#777") for i in r.index[::-1]])
        ax.axvline(r.min() + f["cd"], ls="--", color="k", lw=1)
        ax.set_title(f"{title}: average rank over {f['n']} projects (CD = {f['cd']:.2f})")
        ax.set_xlabel("average rank (lower is better)")
    fig.tight_layout()
    fig.savefig(FIG / "x_rq2_ranks.png", dpi=170)
    plt.close(fig)
    return t, vs


def rq3(shp):
    say("## RQ3")
    rows = []
    for (ds, proj), g in shp.groupby(["dataset", "project"]):
        piv = g.pivot_table(index="window", columns="feature", values="mean_abs_shap").sort_index()
        for i in range(1, len(piv)):
            a, b = piv.iloc[i - 1], piv.iloc[i]
            ta, tb = set(a.nlargest(5).index), set(b.nlargest(5).index)
            u = sorted(ta | tb)
            rows.append(dict(dataset=ds, project=proj, rho_top5=spearmanr(a[u], b[u]).statistic,
                             rho_all=spearmanr(a, b).statistic, jacc=len(ta & tb) / len(ta | tb)))
    st = pd.DataFrame(rows)
    table(st.groupby("dataset")[["rho_top5", "rho_all", "jacc"]].median())
    say(f"All: median rho_top5={st.rho_top5.median():.3f}, share>=0.7={(st.rho_top5 >= .7).mean():.3f}, "
        f"rho_all={st.rho_all.median():.3f}, pairs={len(st)}")
    s = shp.copy()
    s["rank"] = s.groupby(["dataset", "project", "window"]).mean_abs_shap.rank(ascending=False)
    freq = (s[s["rank"] <= 5].groupby(["dataset", "feature"]).size()
            / s.groupby(["dataset"]).apply(lambda g: g.groupby(["project", "window"]).ngroups))
    table(freq.unstack(0).sort_values("Kamei", ascending=False))
    return st


def main():
    OUT.mkdir(exist_ok=True)
    pp, mainr, win, shp = load()
    say("# Cross-dataset summary (ApacheJIT + Kamei)\n")
    say(f"projects: {pp.groupby('dataset').project.nunique().to_dict()}")
    rq1(pp, win)
    rq2(mainr, pp)
    rq3(shp)
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
