# Paper sources

LaTeX source of the manuscript
*"Does Just-in-Time Defect Prediction Survive Realistic Evaluation? Time Order,
Label Latency, and Review Effort on 321,288 Commits from Two Datasets"*.

## Build

Upload this folder to [Overleaf](https://www.overleaf.com) (compiler: pdfLaTeX),
or build locally with a TeX distribution:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Citations use natbib with the IEEE numeric style `IEEEtranN` (part of TeX Live
and Overleaf). References in `references.bib` were checked against Crossref.

## Where each figure comes from

| File in `figures/` | Produced by |
|---|---|
| `x_rq1_protocols.png`, `x_rq1_drop_vs_prevalence.png`, `x_rq2_ranks.png` | `python cross_dataset.py` |
| `data_prevalence_by_year.png`, `rq2_effort_curves.png`, `rq3_shap_beeswarm.png`, `rq3_stability_hist.png`, `rq3_rank_heatmap.png` | `python run_all.py --dataset apachejit` (copied here by `cross_dataset.py`) |
| `k_data_prevalence_by_year.png`, `k_rq3_shap_beeswarm.png` | `python run_all.py --dataset kamei` (copied here by `cross_dataset.py`) |

All figures are written at 300 dpi.

## Where each number comes from

| Paper section | Source file |
|---|---|
| Per-dataset medians, pooled results, and the appendix tables | `../report.md`, `../report_kamei.md`, `../results*/per_project_protocol.csv` |
| Statistical tests over all 19 projects and per dataset (RQ1-RQ3) | `../results_combined/summary.md`, `../results_combined/rq1_tests.csv` |
