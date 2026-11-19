"""Extra figure for the paper: PR-AUC drop vs. test-prevalence shift."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
pp = pd.read_csv("results/per_project_protocol.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
col = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52"}; mk = {"LR": "o", "RF": "s", "XGB": "^"}
for m in col:
    d = pp[pp.model == m].pivot(index="project", columns="protocol", values="pr_auc")
    p1 = pp[(pp.model == m) & (pp.protocol == "P1")].set_index("project").prevalence
    p3 = pp[(pp.model == m) & (pp.protocol == "P3-G90")].set_index("project").prevalence
    x, y = p3 / p1, d["P1"] - d["P3-G90"]
    r = spearmanr(x, y).statistic
    ax.scatter(x, y, c=col[m], marker=mk[m], label=f"{m} (Spearman ρ = {r:.2f})", s=40, alpha=.85)
ax.axhline(0, color="k", lw=.8); ax.axvline(1, color="k", lw=.8, ls=":")
ax.set_xlabel("test prevalence under P3-G90 / test prevalence under P1")
ax.set_ylabel("PR-AUC drop (P1 − P3-G90)")
ax.legend(frameon=False); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig("paper/figures/rq1_drop_vs_prevalence.png", dpi=200)
