# Realistic Evaluation of ML-Based Just-in-Time Defect Prediction

Implementation of the SD566 project proposal (Farida Al-Ghariani, Alaa Al-Muqadami).
It measures how much of the reported performance of standard classifiers survives
when JIT defect prediction is evaluated realistically: chronological splits, label
latency, class imbalance and effort-aware metrics, on **ApacheJIT** (106,674 commits).

## Quick start

```powershell
pip install -r requirements.txt
python run_all.py                  # all experiments + report.md (~20 min, 12 cores)
python run_all.py --analysis-only  # rebuild tables/figures from saved results
python -m pytest -q                # leakage / protocol / metric tests
python predict.py --example        # prototype: score held-out commits
```

The dataset is in `data/apachejit_total.csv` (source:
<https://github.com/hosseinkshvrz/apachejit>, SHA-256 recorded in `report.md`).

## What is implemented

| Proposal item | Where |
|---|---|
| Dataset-readiness gate, validation, exclusion rule | `src/jitsdp/data.py` |
| Leakage-free preprocessing (log1p, collinearity filter, scaling fitted on train only) | `src/jitsdp/models.py` (`JITPreprocessor`) |
| P1 random 70/30, P2 chronological 70/30, P3 chronological + latency gap G = 30/90/180 | `src/jitsdp/protocols.py` |
| Rolling windows (train 12 m, gap 90 d, test 3 m, step 3 m) | `src/jitsdp/protocols.py` |
| Automated leakage checks | `protocols.check_split`, `tests/` |
| LR, RF, XGBoost + Prevalence and LT (inverse size) baselines | `src/jitsdp/models.py` |
| Imbalance: none / class weighting / random under-sampling (in-fold only) | `src/jitsdp/models.py` |
| Time-ordered hyper-parameter tuning; thresholds chosen on validation data | `src/jitsdp/models.py` |
| Precision, recall, F1, MCC, PR-AUC, ROC-AUC, Recall@20% effort, Popt | `src/jitsdp/metrics.py` |
| Wilcoxon, Cliff's delta, Friedman + Nemenyi | `src/jitsdp/stats.py` |
| SHAP (global + per window), LR coefficients, stability (H3) | `src/jitsdp/experiments.py`, `analysis.py` |
| CLI prototype with Low/Medium/High risk | `predict.py` |

## Improvements beyond the proposal

1. Data audit: `apache/hadoop` has 0 defect-inducing commits (labeling artefact) and is excluded.
2. Label-maturity check and **PR-AUC lift** (PR-AUC / prevalence) for fair cross-protocol comparison.
3. **Paired random-vs-realistic windows** on the *same* commits: a stronger test of H1.
4. Effort-aware supervised ranking (risk per changed line) next to the raw risk ranking.
5. Calibration (Brier score) and bootstrap confidence intervals.
6. Three-level prototype thresholds derived from validation data.

## Layout

```
data/            raw dataset
src/jitsdp/      pipeline package
results/         CSV tables, raw results, figures/
models/          prototype model + held-out examples
report.md        generated results report (H1-H3 verdicts)
```
