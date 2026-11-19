"""Single command that reproduces every table and figure from the raw data.

    python run_all.py                 # full pipeline (~15-30 min on 12 cores)
    python run_all.py --analysis-only # rebuild report from saved raw results
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from jitsdp import analysis as A  # noqa: E402
from jitsdp import config as C  # noqa: E402
from jitsdp import data, experiments  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def versions():
    import imblearn, numpy, scipy, shap, sklearn, xgboost  # noqa: E401
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "imbalanced-learn": imblearn.__version__,
            "shap": shap.__version__, "scipy": scipy.__version__}


def run_experiments(df, projects, val):
    raw = C.RESULTS / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    main = experiments.run_main(df, projects, log=log)
    main.to_csv(raw / "main_results.csv", index=False)
    log(f"main done in {time.time() - t0:.0f}s ({len(main)} rows)")

    pooled, fitted = experiments.run_pooled(df, projects, log=log)
    pooled.to_csv(raw / "pooled_results.csv", index=False)
    log(f"pooled done in {time.time() - t0:.0f}s")
    expl = experiments.pooled_explanations(fitted)
    meta = experiments.save_prototype(fitted, pooled, val["summary"]["sha256"])
    boot = A.bootstrap_ci(fitted)
    small = {k: {kk: v[kk] for kk in ("X_test", "y_test", "prob", "effort")} for k, v in fitted.items()
             if k[0] == "XGB" and k[1] == C.DEFAULT_STRATEGY}
    joblib.dump(dict(expl=expl, curves=small, boot=boot, proto_meta=meta), raw / "pooled_artifacts.joblib")

    windows, shapw = experiments.run_windows(df, projects, log=log)
    windows.to_csv(raw / "window_results.csv", index=False)
    shapw.to_csv(raw / "window_shap.csv", index=False)
    log(f"windows done in {time.time() - t0:.0f}s ({windows.window.nunique()} max windows/project)")


def build_report(df, val, excl):
    raw = C.RESULTS / "raw"
    main = pd.read_csv(raw / "main_results.csv")
    pooled = pd.read_csv(raw / "pooled_results.csv")
    windows = pd.read_csv(raw / "window_results.csv")
    shapw = pd.read_csv(raw / "window_shap.csv")
    art = joblib.load(raw / "pooled_artifacts.joblib")

    pp = A.per_project_protocol(main)
    pp.to_csv(C.RESULTS / "per_project_protocol.csv", index=False)
    sec_data, sec1, sec2, sec3, extra = [], [], [], [], []
    A.data_section(val, excl, sec_data)
    A.label_maturity(pp, sec_data)
    r1 = A.rq1(pp, windows, sec1)
    r2 = A.rq2(main, pp, sec2)
    A.effort_curves(art["curves"], sec2)
    r3 = A.rq3(shapw, art["expl"], sec3)

    # pooled analysis
    pl = pooled[pooled.strategy.isin([C.DEFAULT_STRATEGY, "-"])]
    ptab = pl.pivot_table(index="model", columns="protocol", values="pr_auc")[A.PROTO_ORDER]
    extra.append("## Pooled analysis (all retained projects, global time order)\n")
    extra.append("PR-AUC per protocol:\n")
    extra.append(A._md(ptab) + "\n")
    p3 = pl[pl.protocol == A.P3].set_index("model")[
        ["precision", "recall", "f1", "mcc", "pr_auc", "pr_auc_lift", "roc_auc", "recall20",
         "recall20_density", "popt_density", "brier", "accuracy"]]
    extra.append(f"All metrics, pooled {A.P3}:\n")
    extra.append(A._md(p3) + "\n")
    extra.append("95% bootstrap confidence intervals of PR-AUC (pooled P3 test set, 1000 resamples):\n")
    extra.append(A._md(art["boot"]) + "\n")
    cm = pl[(pl.protocol == A.P3) & pl.model.isin(C.MODELS)].set_index("model")[["tp", "fp", "fn", "tn"]]
    extra.append("Confusion matrices at the validation-selected threshold:\n")
    extra.append(A._md(cm, floatfmt=".0f") + "\n")

    meta = art["proto_meta"]
    nwin = windows[windows.setting == "realistic"].groupby("project").window.nunique()
    out = [
        "# Realistic Evaluation of ML-Based Just-in-Time Defect Prediction - Results\n",
        "*SD566 Artificial Intelligence - Farida Al-Ghariani, Alaa Al-Muqadami. "
        "Generated automatically by `python run_all.py`.*\n",
        "## Executive summary\n",
        f"| Hypothesis | Verdict |\n|---|---|\n| H1 - realistic evaluation lowers PR-AUC and F1 | "
        f"**{r1['h1']}** |\n| H2 - class weighting helps; ML beats the unsupervised baseline "
        f"on Recall@20% | **{r2['h2']}** |\n| H3 - top-5 SHAP features stable across windows "
        f"(rho >= 0.7) | **{r3['h3']}** |\n",
        f"- {len(excl[excl.retained])} projects retained, {int(nwin.sum())} rolling windows, "
        f"{len(main) + len(pooled) + len(windows):,} evaluated model/baseline runs.\n"
        f"- Best ranking model (PR-AUC, P3): **{r2['best_pr']}**; best effort-aware model: "
        f"**{r2['best_eff']}**. Prototype model: **{meta['model']}** "
        f"(pooled test PR-AUC {meta['test_pr_auc']}).\n",
        *sec_data, *sec1, *sec2, *sec3, *extra,
        "## Prototype\n",
        "`python predict.py --example` scores held-out commits; "
        "`python predict.py --la 120 --ld 30 --nf 4 ...` scores a single record. "
        f"Model metadata: `{json.dumps(meta)}`\n",
        "## Automated leakage checks\n",
        "Every split passes `check_split` before training (forbidden fields absent, no shared "
        "commits, training dates precede test dates, gap respected); preprocessing and "
        "resampling are fitted inside the training pipeline only; test data are never "
        "resampled; thresholds come from validation data. See `tests/test_pipeline.py`.\n",
        "## Improvements beyond the proposal\n",
        "1. **Data audit** found and excluded the label-less `apache/hadoop` project.\n"
        "2. **Label-maturity check** + **PR-AUC lift** to compare protocols whose test "
        "prevalence differs.\n"
        "3. **Paired random-vs-realistic windows**: the same commits evaluated both ways, a "
        "much more powerful test of H1 than comparing different test sets.\n"
        "4. **Effort-aware supervised ranking** (risk per changed line) in addition to the "
        "raw risk ranking.\n"
        "5. **Calibration (Brier score)** and **bootstrap confidence intervals**.\n"
        "6. **Collinearity filter** (Spearman |rho| > 0.8) for interpretable LR coefficients.\n"
        "7. **Three-level risk prototype** with thresholds from validation data (High = "
        "F1-optimal threshold, Low = below the 90%-recall threshold).\n",
        "## Limitations\n",
        "- SZZ labels are noisy; no fix timestamps, so latency is approximated by a gap.\n"
        "- Rolling windows use fixed default hyper-parameters (tuning is done for P1/P2/P3).\n"
        "- Effort = lines added + deleted is a simplification of review effort.\n"
        "- Apache Java projects may not generalize to other ecosystems.\n",
        "## Environment\n",
        f"`{json.dumps(versions())}`\n",
    ]
    name = "report.md" if C.DATASET == "apachejit" else f"report_{C.DATASET}.md"
    (C.ROOT / name).write_text("\n".join(out), encoding="utf-8")
    log(f"report written to {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--dataset", choices=["apachejit", "kamei"], default="apachejit")
    args = ap.parse_args()
    C.use_dataset(args.dataset)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df = data.load()
    val = data.validate(df)
    if not val["summary"]["readiness_passed"]:
        sys.exit("dataset-readiness gate failed: " + json.dumps(val["summary"]["readiness_gate"]))
    excl = data.exclusion_table(df)
    excl.to_csv(C.RESULTS / "exclusions.csv", index=False)
    val["per_project"].to_csv(C.RESULTS / "data_per_project.csv", index=False)
    val["per_year"].to_csv(C.RESULTS / "data_per_year.csv", index=False)
    projects = excl[excl.retained].project.tolist()
    log(f"retained projects: {projects}")
    if not args.analysis_only:
        run_experiments(df, projects, val)
    build_report(df, val, excl)


if __name__ == "__main__":
    main()
