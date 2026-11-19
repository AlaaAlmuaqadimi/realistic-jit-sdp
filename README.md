# Does Just-in-Time Defect Prediction Survive Realistic Evaluation?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24371575.svg)](https://doi.org/10.5281/zenodo.24371575)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Replication package for the paper
**"Does Just-in-Time Defect Prediction Survive Realistic Evaluation? Time Order, Label Latency, and Review Effort on 321,288 Commits from Two Datasets"**
by Farida Hisham Ghariani and Alla Khaled Almuqadimi
(College of Computer Technology (CCTT), Tripoli, Libya).

The study measures how much of the reported performance of standard JIT defect
predictors survives realistic evaluation. It covers chronological splits, label
latency, class imbalance, and effort-aware metrics on **19 projects** from two
independent datasets:

| Dataset | Projects | Commits | Languages |
|---|---|---|---|
| [ApacheJIT](https://github.com/hosseinkshvrz/apachejit) (Keshavarz & Nagappan, MSR 2022) | 13 retained of 15 | 93,871 | Java |
| Kamei et al. (TSE 2013), via [RevisitUnsupervised](https://github.com/WeiFoo/RevisitUnsupervised) | 6 | 227,417 | C, C++, Perl, Java |

## Key findings

| Hypothesis | ApacheJIT | Kamei | All 19 projects |
|---|---|---|---|
| H1: realistic evaluation lowers PR-AUC and F1 | supported (−0.070) | supported (−0.063) | **supported** |
| H2: class weighting helps; ML beats the unsupervised baseline on Recall@20% effort | not supported | not supported | **not supported**, but risk-per-line ranking beats the baseline in both datasets |
| H3: top-5 SHAP features stable over time (ρ ≥ 0.7) | 0.60 | 0.71 | **0.64, dataset-dependent** |

## Reproduce

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
python download_data.py                     # fetch both datasets into data/ and print SHA-256
python -m pytest -q                         # 12 leakage / protocol / metric tests
python run_all.py --dataset apachejit       # ~19 min on 12 cores -> results/, report.md
python run_all.py --dataset kamei           # ~35 min on 12 cores -> results_kamei/, report_kamei.md
python cross_dataset.py                     # H1-H3 over all 19 projects -> results_combined/, paper/figures/
python predict.py --example                 # prototype (needs a run_all.py run first)
```

Add `--analysis-only` to `run_all.py` to rebuild tables and figures from saved raw results.
Expected dataset hashes:

- ApacheJIT `5097cbbbab8c4611a709b3cf95ac5c9fe5f62e619bab081911a2a247ddbed4e0`
- Kamei `db1ccef8fa1f4be3ed695b6b49a526b04cab9a1758fa5f1c722309027c5caa09`

## Method in brief

| Component | Where |
|---|---|
| Dataset readiness gate, validation, pre-registered exclusion rule | `src/jitsdp/data.py` |
| Leakage-free preprocessing (log1p, collinearity filter, scaling fitted on train only) | `src/jitsdp/models.py` |
| P1 random 70/30 · P2 chronological 70/30 · P3 chronological + latency gap G = 30/90/180 days | `src/jitsdp/protocols.py` |
| 495 rolling windows (train 12 m → gap 90 d → test 3 m), each paired with a shuffled split of the same commits | `src/jitsdp/protocols.py` |
| LR, RF, XGBoost + prevalence and inverse-churn baselines; none / class weighting / under-sampling | `src/jitsdp/models.py` |
| Time-ordered tuning; thresholds selected on validation data only | `src/jitsdp/models.py` |
| PR-AUC, PR-AUC lift, ROC-AUC, F1, MCC, Brier, Recall@20% effort, Popt (risk and risk-per-line ranking) | `src/jitsdp/metrics.py` |
| Wilcoxon + Holm, Cliff's δ, Friedman + Nemenyi, bootstrap CIs | `src/jitsdp/stats.py` |
| SHAP per window and global; ranking stability | `src/jitsdp/experiments.py`, `analysis.py` |
| Automated leakage checks | `protocols.check_split`, `tests/` |

## Layout

```
src/jitsdp/          pipeline package
tests/               unit tests
run_all.py           per-dataset experiments + report
cross_dataset.py     combined analysis over both datasets
download_data.py     fetches the public datasets
predict.py           command-line prototype (Low / Medium / High risk)
results*/            result tables and figures per dataset (raw outputs are regenerated)
report*.md           generated per-dataset reports
paper/               LaTeX source, bibliography, and figures of the paper (see paper/README.md)
CITATION.cff         citation metadata (GitHub "Cite this repository", Zenodo)
```

The datasets are not redistributed here (this includes the prototype's held-out
examples and trained models, which `run_all.py` writes to `models*/`);
`download_data.py` fetches the data from their original sources. Please cite the
original dataset papers when using them.

## Citation

If you use this package, please cite the paper (details will be added after publication)
and the archived software:

> Ghariani, F. H., & Almuqadimi, A. K. (2026). *realistic-jit-sdp: Replication package* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24371576

The DOI above identifies version v1.0.0, which was used in the paper. The concept DOI
[10.5281/zenodo.24371575](https://doi.org/10.5281/zenodo.24371575) always resolves to the latest version.
GitHub's **Cite this repository** button (generated from `CITATION.cff`) gives the same
reference in APA and BibTeX.
