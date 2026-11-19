# Cross-dataset summary (ApacheJIT + Kamei)

projects: {'ApacheJIT': 13, 'Kamei': 6}
## RQ1
|                                 |   pr_auc |   pr_auc_lift |    f1 |   mcc |   recall20 |
|:--------------------------------|---------:|--------------:|------:|------:|-----------:|
| ('ApacheJIT', 'P1', 'LR')       |    0.790 |         1.872 | 0.724 | 0.516 |      0.118 |
| ('ApacheJIT', 'P1', 'RF')       |    0.809 |         1.997 | 0.732 | 0.561 |      0.207 |
| ('ApacheJIT', 'P1', 'XGB')      |    0.834 |         2.001 | 0.740 | 0.578 |      0.229 |
| ('ApacheJIT', 'P2', 'LR')       |    0.700 |         2.533 | 0.627 | 0.393 |      0.120 |
| ('ApacheJIT', 'P2', 'RF')       |    0.657 |         2.445 | 0.626 | 0.399 |      0.158 |
| ('ApacheJIT', 'P2', 'XGB')      |    0.715 |         2.524 | 0.642 | 0.454 |      0.215 |
| ('ApacheJIT', 'P3-G30', 'LR')   |    0.701 |         2.533 | 0.626 | 0.399 |      0.121 |
| ('ApacheJIT', 'P3-G30', 'RF')   |    0.633 |         2.420 | 0.625 | 0.407 |      0.182 |
| ('ApacheJIT', 'P3-G30', 'XGB')  |    0.716 |         2.512 | 0.635 | 0.468 |      0.193 |
| ('ApacheJIT', 'P3-G90', 'LR')   |    0.689 |         2.525 | 0.603 | 0.396 |      0.119 |
| ('ApacheJIT', 'P3-G90', 'RF')   |    0.650 |         2.376 | 0.613 | 0.410 |      0.158 |
| ('ApacheJIT', 'P3-G90', 'XGB')  |    0.715 |         2.510 | 0.626 | 0.443 |      0.202 |
| ('ApacheJIT', 'P3-G180', 'LR')  |    0.688 |         2.520 | 0.591 | 0.372 |      0.121 |
| ('ApacheJIT', 'P3-G180', 'RF')  |    0.610 |         2.304 | 0.606 | 0.387 |      0.163 |
| ('ApacheJIT', 'P3-G180', 'XGB') |    0.682 |         2.477 | 0.626 | 0.428 |      0.178 |
| ('Kamei', 'P1', 'LR')           |    0.454 |         2.171 | 0.470 | 0.285 |      0.282 |
| ('Kamei', 'P1', 'RF')           |    0.541 |         2.667 | 0.515 | 0.363 |      0.361 |
| ('Kamei', 'P1', 'XGB')          |    0.552 |         2.762 | 0.527 | 0.370 |      0.349 |
| ('Kamei', 'P2', 'LR')           |    0.369 |         2.145 | 0.418 | 0.251 |      0.286 |
| ('Kamei', 'P2', 'RF')           |    0.451 |         2.745 | 0.429 | 0.295 |      0.345 |
| ('Kamei', 'P2', 'XGB')          |    0.469 |         2.933 | 0.438 | 0.312 |      0.332 |
| ('Kamei', 'P3-G30', 'LR')       |    0.368 |         2.149 | 0.411 | 0.238 |      0.284 |
| ('Kamei', 'P3-G30', 'RF')       |    0.440 |         2.733 | 0.450 | 0.301 |      0.342 |
| ('Kamei', 'P3-G30', 'XGB')      |    0.458 |         2.897 | 0.471 | 0.325 |      0.325 |
| ('Kamei', 'P3-G90', 'LR')       |    0.363 |         2.151 | 0.412 | 0.240 |      0.297 |
| ('Kamei', 'P3-G90', 'RF')       |    0.432 |         2.689 | 0.447 | 0.292 |      0.350 |
| ('Kamei', 'P3-G90', 'XGB')      |    0.453 |         2.864 | 0.428 | 0.311 |      0.326 |
| ('Kamei', 'P3-G180', 'LR')      |    0.360 |         2.140 | 0.412 | 0.239 |      0.310 |
| ('Kamei', 'P3-G180', 'RF')      |    0.422 |         2.606 | 0.414 | 0.296 |      0.357 |
| ('Kamei', 'P3-G180', 'XGB')     |    0.458 |         2.821 | 0.426 | 0.312 |      0.348 |

|                                              |   n |   median_a |   median_b |   median_diff | p      |   cliffs_delta | magnitude   |
|:---------------------------------------------|----:|-----------:|-----------:|--------------:|:-------|---------------:|:------------|
| ('All', 'P1 vs P3-G90', 'pr_auc')            |  57 |      0.729 |      0.624 |         0.067 | <0.001 |          0.215 | small       |
| ('All', 'P1 vs P2', 'pr_auc')                |  57 |      0.729 |      0.634 |         0.064 | <0.001 |          0.200 | small       |
| ('All', 'P2 vs P3-G90', 'pr_auc')            |  57 |      0.634 |      0.624 |         0.003 | <0.001 |          0.026 | negligible  |
| ('All', 'P1 vs P3-G90', 'f1')                |  57 |      0.650 |      0.577 |         0.071 | <0.001 |          0.237 | small       |
| ('All', 'P1 vs P2', 'f1')                    |  57 |      0.650 |      0.567 |         0.064 | <0.001 |          0.232 | small       |
| ('All', 'P2 vs P3-G90', 'f1')                |  57 |      0.567 |      0.577 |         0.003 | 0.026  |          0.023 | negligible  |
| ('All', 'P1 vs P3-G90', 'mcc')               |  57 |      0.511 |      0.394 |         0.088 | <0.001 |          0.420 | medium      |
| ('All', 'P1 vs P2', 'mcc')                   |  57 |      0.511 |      0.396 |         0.075 | <0.001 |          0.365 | medium      |
| ('All', 'P2 vs P3-G90', 'mcc')               |  57 |      0.396 |      0.394 |         0.004 | 0.014  |          0.055 | negligible  |
| ('All', 'P1 vs P3-G90', 'pr_auc_lift')       |  57 |      2.160 |      2.442 |        -0.110 | 0.993  |         -0.112 | negligible  |
| ('All', 'P1 vs P2', 'pr_auc_lift')           |  57 |      2.160 |      2.498 |        -0.143 | 0.997  |         -0.133 | negligible  |
| ('All', 'P2 vs P3-G90', 'pr_auc_lift')       |  57 |      2.498 |      2.442 |         0.018 | <0.001 |          0.035 | negligible  |
| ('ApacheJIT', 'P1 vs P3-G90', 'pr_auc')      |  39 |      0.809 |      0.681 |         0.070 | <0.001 |          0.256 | small       |
| ('ApacheJIT', 'P1 vs P2', 'pr_auc')          |  39 |      0.809 |      0.700 |         0.071 | <0.001 |          0.239 | small       |
| ('ApacheJIT', 'P2 vs P3-G90', 'pr_auc')      |  39 |      0.700 |      0.681 |         0.004 | <0.001 |          0.034 | negligible  |
| ('ApacheJIT', 'P1 vs P3-G90', 'f1')          |  39 |      0.725 |      0.613 |         0.093 | <0.001 |          0.278 | small       |
| ('ApacheJIT', 'P1 vs P2', 'f1')              |  39 |      0.725 |      0.627 |         0.083 | <0.001 |          0.274 | small       |
| ('ApacheJIT', 'P2 vs P3-G90', 'f1')          |  39 |      0.627 |      0.613 |         0.003 | 0.143  |          0.021 | negligible  |
| ('ApacheJIT', 'P1 vs P3-G90', 'mcc')         |  39 |      0.546 |      0.410 |         0.118 | <0.001 |          0.561 | large       |
| ('ApacheJIT', 'P1 vs P2', 'mcc')             |  39 |      0.546 |      0.424 |         0.114 | <0.001 |          0.499 | large       |
| ('ApacheJIT', 'P2 vs P3-G90', 'mcc')         |  39 |      0.424 |      0.410 |         0.007 | 0.085  |          0.063 | negligible  |
| ('ApacheJIT', 'P1 vs P3-G90', 'pr_auc_lift') |  39 |      1.997 |      2.425 |        -0.132 | 0.983  |         -0.108 | negligible  |
| ('ApacheJIT', 'P1 vs P2', 'pr_auc_lift')     |  39 |      1.997 |      2.472 |        -0.153 | 0.989  |         -0.115 | negligible  |
| ('ApacheJIT', 'P2 vs P3-G90', 'pr_auc_lift') |  39 |      2.472 |      2.425 |         0.016 | <0.001 |          0.037 | negligible  |
| ('Kamei', 'P1 vs P3-G90', 'pr_auc')          |  18 |      0.510 |      0.444 |         0.063 | <0.001 |          0.228 | small       |
| ('Kamei', 'P1 vs P2', 'pr_auc')              |  18 |      0.510 |      0.459 |         0.059 | <0.001 |          0.204 | small       |
| ('Kamei', 'P2 vs P3-G90', 'pr_auc')          |  18 |      0.459 |      0.444 |         0.002 | 0.003  |          0.049 | negligible  |
| ('Kamei', 'P1 vs P3-G90', 'f1')              |  18 |      0.505 |      0.437 |         0.044 | <0.001 |          0.228 | small       |
| ('Kamei', 'P1 vs P2', 'f1')                  |  18 |      0.505 |      0.429 |         0.043 | <0.001 |          0.235 | small       |
| ('Kamei', 'P2 vs P3-G90', 'f1')              |  18 |      0.429 |      0.437 |         0.003 | 0.019  |          0.049 | negligible  |
| ('Kamei', 'P1 vs P3-G90', 'mcc')             |  18 |      0.334 |      0.272 |         0.025 | 0.009  |          0.278 | small       |
| ('Kamei', 'P1 vs P2', 'mcc')                 |  18 |      0.334 |      0.273 |         0.021 | 0.059  |          0.222 | small       |
| ('Kamei', 'P2 vs P3-G90', 'mcc')             |  18 |      0.273 |      0.272 |         0.004 | 0.012  |          0.043 | negligible  |
| ('Kamei', 'P1 vs P3-G90', 'pr_auc_lift')     |  18 |      2.431 |      2.595 |        -0.067 | 0.894  |         -0.080 | negligible  |
| ('Kamei', 'P1 vs P2', 'pr_auc_lift')         |  18 |      2.431 |      2.689 |        -0.104 | 0.963  |         -0.136 | negligible  |
| ('Kamei', 'P2 vs P3-G90', 'pr_auc_lift')     |  18 |      2.689 |      2.595 |         0.023 | 0.005  |          0.056 | negligible  |

Per-model P1 vs P3 (all 19 projects), Holm-corrected:
|                   |   n |   median_a |   median_b |   median_diff |      p |   cliffs_delta | magnitude   |   p_holm |
|:------------------|----:|-----------:|-----------:|--------------:|-------:|---------------:|:------------|---------:|
| ('LR', 'pr_auc')  |  19 |     0.6980 |     0.6302 |        0.0589 | 0.0014 |         0.1745 | small       |   0.0028 |
| ('LR', 'f1')      |  19 |     0.6361 |     0.5169 |        0.0450 | 0.0007 |         0.2410 | small       |   0.0014 |
| ('RF', 'pr_auc')  |  19 |     0.7368 |     0.6107 |        0.0702 | 0.0002 |         0.2687 | small       |   0.0006 |
| ('RF', 'f1')      |  19 |     0.6617 |     0.5774 |        0.0769 | 0.0001 |         0.2465 | small       |   0.0004 |
| ('XGB', 'pr_auc') |  19 |     0.7403 |     0.6405 |        0.0642 | 0.0014 |         0.2410 | small       |   0.0028 |
| ('XGB', 'f1')     |  19 |     0.6616 |     0.5769 |        0.0869 | 0.0010 |         0.2521 | small       |   0.0014 |

Prevalence shift vs drop (Spearman over projects):
|                      |       n |     rho |      p |
|:---------------------|--------:|--------:|-------:|
| ('All', 'LR')        | 19.0000 | -0.7298 | 0.0004 |
| ('All', 'RF')        | 19.0000 | -0.7386 | 0.0003 |
| ('All', 'XGB')       | 19.0000 | -0.7561 | 0.0002 |
| ('ApacheJIT', 'LR')  | 13.0000 | -0.8352 | 0.0004 |
| ('ApacheJIT', 'RF')  | 13.0000 | -0.8736 | 0.0001 |
| ('ApacheJIT', 'XGB') | 13.0000 | -0.8516 | 0.0002 |
| ('Kamei', 'LR')      |  6.0000 | -0.0857 | 0.8717 |
| ('Kamei', 'RF')      |  6.0000 | -0.0857 | 0.8717 |
| ('Kamei', 'XGB')     |  6.0000 | -0.4857 | 0.3287 |

Paired windows (random vs realistic on identical commits):
|                              |    n |   median_a |   median_b |   median_diff | p      |   cliffs_delta | magnitude   |
|:-----------------------------|-----:|-----------:|-----------:|--------------:|:-------|---------------:|:------------|
| ('All', 'pr_auc')            | 1485 |      0.690 |      0.662 |         0.024 | <0.001 |          0.070 | negligible  |
| ('All', 'f1')                | 1485 |      0.614 |      0.582 |         0.025 | <0.001 |          0.088 | negligible  |
| ('All', 'pr_auc_lift')       | 1485 |      2.182 |      2.147 |         0.025 | <0.001 |          0.061 | negligible  |
| ('ApacheJIT', 'pr_auc')      | 1032 |      0.781 |      0.761 |         0.019 | <0.001 |          0.056 | negligible  |
| ('ApacheJIT', 'f1')          | 1032 |      0.682 |      0.667 |         0.022 | <0.001 |          0.066 | negligible  |
| ('ApacheJIT', 'pr_auc_lift') | 1032 |      2.103 |      2.076 |         0.007 | 0.072  |          0.048 | negligible  |
| ('Kamei', 'pr_auc')          |  453 |      0.540 |      0.495 |         0.037 | <0.001 |          0.146 | negligible  |
| ('Kamei', 'f1')              |  453 |      0.513 |      0.465 |         0.033 | <0.001 |          0.180 | small       |
| ('Kamei', 'pr_auc_lift')     |  453 |      2.369 |      2.250 |         0.109 | <0.001 |          0.089 | negligible  |

windows per dataset: {'ApacheJIT': 344, 'Kamei': 151}

## RQ2
|                                |   precision |   recall |    f1 |   mcc |   pr_auc |
|:-------------------------------|------------:|---------:|------:|------:|---------:|
| ('ApacheJIT', 'none', 'LR')    |       0.454 |    0.834 | 0.599 | 0.375 |    0.689 |
| ('ApacheJIT', 'none', 'RF')    |       0.492 |    0.827 | 0.616 | 0.421 |    0.650 |
| ('ApacheJIT', 'none', 'XGB')   |       0.550 |    0.850 | 0.618 | 0.457 |    0.718 |
| ('ApacheJIT', 'under', 'LR')   |       0.477 |    0.833 | 0.602 | 0.364 |    0.687 |
| ('ApacheJIT', 'under', 'RF')   |       0.484 |    0.821 | 0.609 | 0.399 |    0.632 |
| ('ApacheJIT', 'under', 'XGB')  |       0.512 |    0.810 | 0.630 | 0.427 |    0.709 |
| ('ApacheJIT', 'weight', 'LR')  |       0.460 |    0.837 | 0.603 | 0.396 |    0.689 |
| ('ApacheJIT', 'weight', 'RF')  |       0.500 |    0.810 | 0.613 | 0.410 |    0.650 |
| ('ApacheJIT', 'weight', 'XGB') |       0.531 |    0.815 | 0.626 | 0.443 |    0.715 |
| ('Kamei', 'none', 'LR')        |       0.269 |    0.718 | 0.397 | 0.227 |    0.366 |
| ('Kamei', 'none', 'RF')        |       0.288 |    0.691 | 0.422 | 0.272 |    0.438 |
| ('Kamei', 'none', 'XGB')       |       0.306 |    0.682 | 0.435 | 0.300 |    0.444 |
| ('Kamei', 'under', 'LR')       |       0.295 |    0.694 | 0.412 | 0.242 |    0.358 |
| ('Kamei', 'under', 'RF')       |       0.328 |    0.690 | 0.446 | 0.290 |    0.416 |
| ('Kamei', 'under', 'XGB')      |       0.331 |    0.673 | 0.455 | 0.312 |    0.454 |
| ('Kamei', 'weight', 'LR')      |       0.289 |    0.679 | 0.412 | 0.240 |    0.363 |
| ('Kamei', 'weight', 'RF')      |       0.320 |    0.694 | 0.447 | 0.292 |    0.432 |
| ('Kamei', 'weight', 'XGB')     |       0.306 |    0.661 | 0.428 | 0.311 |    0.453 |

|                                           | alternative   |   n |   median_a |   median_b |   median_diff | p      |   cliffs_delta | magnitude   |
|:------------------------------------------|:--------------|----:|-----------:|-----------:|--------------:|:-------|---------------:|:------------|
| ('All', 'weight vs none', 'recall')       | greater       |  57 |      0.776 |      0.798 |        -0.009 | 0.977  |         -0.041 | negligible  |
| ('All', 'under vs none', 'recall')        | greater       |  57 |      0.767 |      0.798 |        -0.003 | 0.889  |         -0.053 | negligible  |
| ('All', 'weight vs none', 'f1')           | greater       |  57 |      0.577 |      0.555 |         0.002 | 0.058  |          0.017 | negligible  |
| ('All', 'under vs none', 'f1')            | greater       |  57 |      0.553 |      0.555 |        -0.000 | 0.451  |          0.003 | negligible  |
| ('All', 'weight vs none', 'pr_auc')       | two-sided     |  57 |      0.624 |      0.637 |        -0.000 | 0.080  |         -0.015 | negligible  |
| ('All', 'under vs none', 'pr_auc')        | two-sided     |  57 |      0.629 |      0.637 |        -0.004 | <0.001 |         -0.029 | negligible  |
| ('All', 'weight vs none', 'mcc')          | two-sided     |  57 |      0.394 |      0.376 |         0.004 | 0.274  |         -0.004 | negligible  |
| ('All', 'under vs none', 'mcc')           | two-sided     |  57 |      0.373 |      0.376 |        -0.003 | 0.971  |         -0.014 | negligible  |
| ('ApacheJIT', 'weight vs none', 'recall') | greater       |  39 |      0.815 |      0.834 |        -0.006 | 0.894  |         -0.030 | negligible  |
| ('ApacheJIT', 'under vs none', 'recall')  | greater       |  39 |      0.821 |      0.834 |         0.000 | 0.449  |         -0.020 | negligible  |
| ('ApacheJIT', 'weight vs none', 'f1')     | greater       |  39 |      0.613 |      0.610 |         0.001 | 0.389  |          0.004 | negligible  |
| ('ApacheJIT', 'under vs none', 'f1')      | greater       |  39 |      0.609 |      0.610 |        -0.003 | 0.955  |         -0.026 | negligible  |
| ('ApacheJIT', 'weight vs none', 'pr_auc') | two-sided     |  39 |      0.681 |      0.678 |        -0.000 | 0.104  |         -0.020 | negligible  |
| ('ApacheJIT', 'under vs none', 'pr_auc')  | two-sided     |  39 |      0.686 |      0.678 |        -0.003 | 0.003  |         -0.036 | negligible  |
| ('ApacheJIT', 'weight vs none', 'mcc')    | two-sided     |  39 |      0.410 |      0.420 |         0.001 | 0.890  |         -0.043 | negligible  |
| ('ApacheJIT', 'under vs none', 'mcc')     | two-sided     |  39 |      0.394 |      0.420 |        -0.008 | 0.101  |         -0.072 | negligible  |
| ('Kamei', 'weight vs none', 'recall')     | greater       |  18 |      0.680 |      0.699 |        -0.022 | 0.932  |         -0.090 | negligible  |
| ('Kamei', 'under vs none', 'recall')      | greater       |  18 |      0.694 |      0.699 |        -0.026 | 0.985  |         -0.105 | negligible  |
| ('Kamei', 'weight vs none', 'f1')         | greater       |  18 |      0.437 |      0.429 |         0.010 | 0.024  |          0.043 | negligible  |
| ('Kamei', 'under vs none', 'f1')          | greater       |  18 |      0.453 |      0.429 |         0.012 | 0.002  |          0.093 | negligible  |
| ('Kamei', 'weight vs none', 'pr_auc')     | two-sided     |  18 |      0.444 |      0.444 |         0.000 | 0.495  |          0.012 | negligible  |
| ('Kamei', 'under vs none', 'pr_auc')      | two-sided     |  18 |      0.430 |      0.444 |        -0.008 | 0.014  |         -0.068 | negligible  |
| ('Kamei', 'weight vs none', 'mcc')        | two-sided     |  18 |      0.272 |      0.255 |         0.012 | 0.018  |          0.117 | negligible  |
| ('Kamei', 'under vs none', 'mcc')         | two-sided     |  18 |      0.274 |      0.255 |         0.014 | 0.007  |          0.198 | small       |

Friedman PR-AUC [All] chi2=68.00 p=<0.001 n=19 CD=1.40 ranks={'XGB': 1.21, 'RF': 2.32, 'LR': 2.47, 'Prevalence': 4.0, 'LT': 5.0}
Friedman PR-AUC [ApacheJIT] chi2=46.28 p=<0.001 n=13 CD=1.69 ranks={'XGB': 1.23, 'LR': 2.31, 'RF': 2.46, 'Prevalence': 4.0, 'LT': 5.0}
Friedman PR-AUC [Kamei] chi2=22.53 p=<0.001 n=6 CD=2.49 ranks={'XGB': 1.17, 'RF': 2.0, 'LR': 2.83, 'Prevalence': 4.0, 'LT': 5.0}
Friedman Recall@20 [All] chi2=109.81 p=<0.001 CD=2.41 ranks={'XGB-density': 1.92, 'RF-density': 2.05, 'LR-density': 2.47, 'LT': 3.55, 'XGB': 6.0, 'RF': 6.32, 'Prevalence': 6.37, 'LR': 7.32}
|                              |   n |   median_a |   median_b |   median_diff | p      |   cliffs_delta | magnitude   |
|:-----------------------------|----:|-----------:|-----------:|--------------:|:-------|---------------:|:------------|
| ('All', 'LR')                |  19 |      0.142 |      0.588 |        -0.397 | 1.000  |         -0.972 | large       |
| ('All', 'RF')                |  19 |      0.229 |      0.588 |        -0.361 | 1.000  |         -0.956 | large       |
| ('All', 'XGB')               |  19 |      0.245 |      0.588 |        -0.303 | 1.000  |         -0.967 | large       |
| ('All', 'LR-density')        |  19 |      0.621 |      0.588 |         0.020 | <0.001 |          0.319 | small       |
| ('All', 'RF-density')        |  19 |      0.630 |      0.588 |         0.023 | <0.001 |          0.310 | small       |
| ('All', 'XGB-density')       |  19 |      0.631 |      0.588 |         0.027 | <0.001 |          0.341 | medium      |
| ('ApacheJIT', 'LR')          |  13 |      0.119 |      0.588 |        -0.438 | 1.000  |         -1.000 | large       |
| ('ApacheJIT', 'RF')          |  13 |      0.158 |      0.588 |        -0.397 | 1.000  |         -0.988 | large       |
| ('ApacheJIT', 'XGB')         |  13 |      0.202 |      0.588 |        -0.363 | 1.000  |         -0.988 | large       |
| ('ApacheJIT', 'LR-density')  |  13 |      0.594 |      0.588 |         0.018 | 0.013  |          0.160 | small       |
| ('ApacheJIT', 'RF-density')  |  13 |      0.594 |      0.588 |         0.011 | 0.008  |          0.118 | negligible  |
| ('ApacheJIT', 'XGB-density') |  13 |      0.587 |      0.588 |         0.020 | 0.016  |          0.148 | small       |
| ('Kamei', 'LR')              |   6 |      0.297 |      0.582 |        -0.293 | 1.000  |         -1.000 | large       |
| ('Kamei', 'RF')              |   6 |      0.350 |      0.582 |        -0.236 | 1.000  |         -0.944 | large       |
| ('Kamei', 'XGB')             |   6 |      0.326 |      0.582 |        -0.254 | 1.000  |         -1.000 | large       |
| ('Kamei', 'LR-density')      |   6 |      0.656 |      0.582 |         0.056 | 0.016  |          0.889 | large       |
| ('Kamei', 'RF-density')      |   6 |      0.665 |      0.582 |         0.064 | 0.016  |          0.889 | large       |
| ('Kamei', 'XGB-density')     |   6 |      0.658 |      0.582 |         0.073 | 0.016  |          0.889 | large       |

Median Recall@20 by dataset:
|                             |   recall20 |   recall20_density |   pr_auc |
|:----------------------------|-----------:|-------------------:|---------:|
| ('ApacheJIT', 'LR')         |      0.119 |              0.594 |    0.689 |
| ('ApacheJIT', 'LT')         |      0.588 |              0.588 |    0.190 |
| ('ApacheJIT', 'Prevalence') |      0.213 |              0.588 |    0.287 |
| ('ApacheJIT', 'RF')         |      0.158 |              0.594 |    0.650 |
| ('ApacheJIT', 'XGB')        |      0.202 |              0.587 |    0.715 |
| ('Kamei', 'LR')             |      0.297 |              0.656 |    0.363 |
| ('Kamei', 'LT')             |      0.582 |              0.582 |    0.130 |
| ('Kamei', 'Prevalence')     |      0.215 |              0.582 |    0.162 |
| ('Kamei', 'RF')             |      0.350 |              0.665 |    0.432 |
| ('Kamei', 'XGB')            |      0.326 |              0.658 |    0.453 |

## RQ3
| dataset   |   rho_top5 |   rho_all |   jacc |
|:----------|-----------:|----------:|-------:|
| ApacheJIT |      0.600 |     0.835 |  0.667 |
| Kamei     |      0.714 |     0.912 |  0.667 |

All: median rho_top5=0.643, share>=0.7=0.431, rho_all=0.854, pairs=476
| feature   |   ApacheJIT |   Kamei |
|:----------|------------:|--------:|
| la        |       1.000 |   0.987 |
| nf        |       0.044 |   0.927 |
| age       |       0.515 |   0.781 |
| aexp      |       0.334 |   0.457 |
| arexp     |       0.221 |   0.424 |
| asexp     |       0.291 |   0.411 |
| fix       |       0.642 |   0.325 |
| ndev      |       0.209 |   0.285 |
| ld        |       0.735 |   0.252 |
| ent       |       0.308 |   0.093 |
| nd        |       0.102 |   0.046 |
| nuc       |       0.593 |   0.013 |
| ns        |       0.006 | nan     |
