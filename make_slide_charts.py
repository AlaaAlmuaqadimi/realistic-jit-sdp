"""Large-font charts for the presentation slides (results/slides/)."""
import sys; sys.path.insert(0, "src")
import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from jitsdp.metrics import _curve
OUT = "results/slides/"
plt.rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 21,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
COL = {"LR": "#9E9E9E", "RF": "#4A4A4A", "XGB": "#E8B400"}

pp = pd.read_csv("results/per_project_protocol.csv")
order = ["P1", "P2", "P3-G30", "P3-G90", "P3-G180"]
labels = ["P1\nRandom", "P2\nChrono", "P3\nG=30", "P3\nG=90", "P3\nG=180"]
med = pp[pp.model.isin(COL)].groupby(["protocol", "model"]).pr_auc.median().unstack().loc[order]
fig, ax = plt.subplots(figsize=(11, 7)); x = np.arange(len(order)); w = 0.26
for i, m in enumerate(COL):
    b = ax.bar(x + (i - 1) * w, med[m], w, label=m, color=COL[m])
    ax.bar_label(b, fmt="%.2f", fontsize=14, padding=3)
ax.set_xticks(x, labels); ax.set_ylim(0.5, 0.9); ax.set_ylabel("Median PR-AUC (13 projects)")
ax.set_title("Performance drops under realistic evaluation"); ax.legend(ncol=3, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "slide_rq1.png", dpi=170); plt.close(fig)

art = joblib.load("results/raw/pooled_artifacts.joblib")
f = art["curves"][("XGB", "weight")]; X, y, p = f["X_test"], f["y_test"], f["prob"]
eff = np.maximum(X["la"].to_numpy() + X["ld"].to_numpy(), 1).astype(float); rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(11, 7))
for name, s, c, lw in [("XGB: risk / LOC (ours)", p / eff, "#E8B400", 5), ("LT baseline: 1 / LOC", 1 / eff, "#4A4A4A", 3.5),
                       ("XGB: risk only", p, "#9E9E9E", 3.5), ("Random", rng.random(len(y)), "#CCCCCC", 2.5)]:
    e, d = _curve(y, eff, np.lexsort((rng.random(len(s)), -s)))
    ax.plot(e, d, label=f"{name}  ({np.interp(0.2, e, d):.0%})", color=c, lw=lw)
ax.axvline(0.2, color="k", ls=":", lw=1.5); ax.text(0.21, 0.04, "20% effort", fontsize=16)
ax.set_xlabel("Share of code reviewed (lines)"); ax.set_ylabel("Share of buggy commits found")
ax.set_title("Effort-aware ranking: defects found per review effort"); ax.legend(frameon=False, fontsize=16, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "slide_rq2.png", dpi=170); plt.close(fig)

sv, Z = art["expl"]["shap_values"], art["expl"]["shap_X"]
imp = pd.Series(np.abs(sv).mean(0), index=Z.columns).sort_values().tail(8)
names = {"la": "Lines added (la)", "fix": "Is a bug fix (fix)", "ld": "Lines deleted (ld)", "nuc": "Unique changes (nuc)",
         "arexp": "Recent dev. experience", "nf": "Files changed (nf)", "age": "File age (age)", "ndev": "Developers (ndev)"}
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh([names.get(i, i) for i in imp.index], imp.values, color=["#E8B400" if v == imp.max() else "#4A4A4A" for v in imp.values])
ax.set_xlabel("Mean |SHAP value| (pooled XGBoost)"); ax.set_title("Most influential features")
fig.tight_layout(); fig.savefig(OUT + "slide_rq3.png", dpi=170); plt.close(fig)
print("ok")
