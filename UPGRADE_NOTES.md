# Upgrade Notes

## Overview
This document summarises every substantive change made to `main.py`
and explains **why** each change is necessary to meet SCIE/Q1 reviewer expectations.

---

## 1. Evaluation Protocol Upgrades

### 1.1 Repeated Stratified K-Fold CV (`RepeatedStratifiedKFold`)
| Before | After |
|--------|-------|
| Single 5-fold CV | 3 × 5-fold repeated stratified CV (15 scores per model) |

**Why**: A single k-fold run has high variance on small datasets.
Repeated CV reduces the variance of the CV estimate, and stratification
guarantees class-proportion balance in every fold — both required by
top-tier journals (IEEE TNNLS, Sensors, etc.).

### 1.2 Bootstrap 95% Confidence Intervals on Accuracy
Added `bootstrap_ci()` (n=1000, non-parametric percentile method).
Every model's reported accuracy now carries a 95% CI, e.g., `0.9333 [0.8667–0.9667]`.

**Why**: Reviewers of SCIE journals routinely require uncertainty quantification.
Reporting a single point estimate without CI is flagged as a weakness.

---

## 2. Statistical Significance Tests

### 2.1 McNemar's Test (per pair, on the validation set)
`mcnemar_test(y_true, pred_a, pred_b)` — outputs χ² and p-value.

**Why**: McNemar's test is the correct test for comparing two classifiers on
the **same** test set (paired, non-independent errors).
Reference: Dietterich (1998), Neural Computation.

### 2.2 Wilcoxon Signed-Rank Test (on CV fold scores)
`wilcoxon_cv_test(scores_a, scores_b)` — tests whether the CV score distributions differ.

**Why**: For repeated CV scores, the Wilcoxon test is distribution-free and
more powerful than a paired t-test when normality cannot be assumed.

Both tests are saved to `stats/significance_tests.csv` (Model A, Model B,
McNemar p, Wilcoxon p, Significant flag).

---

## 3. New Metrics

| Metric | Implementation | Rationale |
|--------|---------------|-----------|
| Cohen's κ | `cohen_kappa_score` | Accounts for chance agreement; required in medical/food-quality papers |
| AUC (macro OvR) | `roc_auc_score` | Standard in multi-class papers |
| Brier Score | `brier_score_loss` | Probabilistic calibration quality |
| Per-class Sensitivity & Specificity | Confusion-matrix derived | ISO 5725 compliance |

---

## 4. ROC / AUC Curves
`plot_roc_curves()` generates per-class and macro-average ROC curves for the
top-5 models, rendered in a single publication-ready figure (`roc_curves.png`).

---

## 5. Calibration Analysis
`plot_calibration()` produces:
- Reliability diagrams (fraction-of-positives vs. mean predicted probability)
- ECE (Expected Calibration Error) per model
- Brier score

Saved as `calibration_reliability.png` and `stats/calibration_metrics.csv`.

**Why**: Many sensors/IoT papers deploy models in real systems.
A well-calibrated model's confidence scores are actionable.
Q1 journals in this domain increasingly ask for calibration analysis.

---

## 6. SHAP Feature Attribution
`shap_analysis()`:
- `shap.TreeExplainer` for RF and Gradient Boosting (exact Shapley values)
- Permutation importance for all other models as a proxy
- Consensus heatmap across all classical models

Saves per-model SHAP bar charts and `data/permutation_importance.csv`.

**Why**: SHAP provides model-agnostic, theoretically grounded feature attribution
(Shapley values from cooperative game theory). This is the current gold standard
for feature importance in ML papers (Lundberg & Lee, NeurIPS 2017).

---

## 7. Manifold Projections (t-SNE + UMAP + PCA)
`manifold_visualizations()` generates a 3-panel figure:
- PCA (linear)
- t-SNE (perplexity auto-scaled)
- UMAP (if `umap-learn` installed)

**Why**: PCA alone is insufficient for non-linear structure.
t-SNE and UMAP are standard in sensor-array and e-nose papers.

---

## 8. Ablation Study
`ablation_study()`: Leave-one-sensor-out accuracy drop using RF as surrogate.
Quantifies the unique contribution of each of the 14 sensor channels.

Saved as `data/ablation_study.csv` and `plots/ablation_study.png`.

**Why**: Ablation studies validate that all reported sensor channels are
necessary. Reviewers ask "can you remove sensors without performance loss?".

---

## 9. Ensemble Methods
| Model | Description |
|-------|-------------|
| Soft Voting Ensemble | Averages `predict_proba` outputs of all probabilistic models |
| Stacking Ensemble | RF + SVM + GB + KNN → Logistic Regression meta-learner |

**Why**: Ensemble methods routinely outperform individual models.
Including them strengthens the contribution claim.

---

## 10. Additional Classical Baselines
Added `Gradient Boosting` and `Logistic Regression` to the baseline set
(original had 5 classical models; now 7 + 2 ensembles + 4 DL = 13 total).

**Why**: More comprehensive baseline comparison strengthens the argument for
the proposed/best model.

---

## 11. Per-Class Metrics Table (ISO 5725-style)
`per_class_metrics_table()` outputs a full table:
`Model × Class → TP, FP, FN, TN, Sensitivity, Specificity, Precision, F1`

Saved as `data/per_class_metrics.csv`.

---

## 12. Publication-Ready Figure Aesthetics
Global `rcParams` updated:
- DejaVu Sans font (IEEE/Nature compatible)
- 300 dpi output
- No top/right spines
- Subtle grid (α=0.3)
- Row-normalised confusion matrices (counts + heatmap)

---

## 13. Reproducibility
- `RANDOM_SEED` constant at top (used in all random operations)
- `np.random.seed` + `torch.manual_seed` set at import time
- All config parameters in a single block at the top of the file

---

## New Output Structure
```
enose_run_XX/
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_violin_per_class.png
│   ├── 04_anova_f_statistics.png
│   ├── manifold_projections.png        ← NEW
│   ├── confusion_matrices.png
│   ├── roc_curves.png                  ← NEW
│   ├── calibration_reliability.png     ← NEW
│   ├── accuracy_comparison_ci.png      ← NEW (with CI bars)
│   ├── radar_top5_models.png           ← NEW
│   ├── dl_training_curves.png
│   ├── shap_*.png                      ← NEW
│   ├── permutation_importance_heatmap.png ← NEW
│   └── ablation_study.png              ← NEW
├── data/
│   ├── full_performance_table.csv      ← NEW (13 metrics per model)
│   ├── per_class_metrics.csv           ← NEW
│   ├── ablation_study.csv              ← NEW
│   ├── permutation_importance.csv      ← NEW
│   ├── anova_results.csv
│   ├── hyperparameter_tuning.csv
│   └── unlabelled_predictions.csv
├── stats/                              ← NEW directory
│   ├── significance_tests.csv          ← NEW
│   └── calibration_metrics.csv         ← NEW
└── logs/
    └── analysis_log.txt
```

---

## New Dependencies
```
shap>=0.44        # SHAP feature attribution
umap-learn>=0.5   # UMAP manifold projection (optional but recommended)
```

Install: `pip install shap umap-learn`
