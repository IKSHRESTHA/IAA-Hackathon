# Methods Reference

Guidance for choosing a method and avoiding its usual traps. Read only the section you need.

## Contents

1. [Statistical tests](#1-statistical-tests)
2. [Regression and ANOVA](#2-regression-and-anova)
3. [Experimental design and A/B tests](#3-experimental-design-and-ab-tests)
4. [Causal inference from observational data](#4-causal-inference-from-observational-data)
5. [Time series](#5-time-series)
6. [Survival analysis](#6-survival-analysis)
7. [Bayesian methods](#7-bayesian-methods)
8. [Choosing an ML algorithm](#8-choosing-an-ml-algorithm)
9. [Feature engineering](#9-feature-engineering)
10. [Hyperparameter tuning and ensembles](#10-hyperparameter-tuning-and-ensembles)
11. [Evaluation metrics](#11-evaluation-metrics)
12. [Model interpretation](#12-model-interpretation)
13. [Visualization](#13-visualization)

---

## 1. Statistical tests

| Question | Data | Test |
|---|---|---|
| Two group means | Roughly normal, independent | Welch's t-test (safer default than Student's) |
| Two group means | Skewed or ordinal | Mann-Whitney U |
| Before/after on same units | Paired, roughly normal | Paired t-test |
| Before/after on same units | Paired, skewed | Wilcoxon signed-rank |
| Three or more group means | Roughly normal | One-way ANOVA, then Tukey HSD |
| Three or more groups | Skewed | Kruskal-Wallis, then Dunn's test |
| Two proportions | Counts | Two-proportion z-test, or Fisher's exact for small counts |
| Association of two categoricals | Contingency table | Chi-square (Fisher's exact if expected counts < 5) |
| Association of two continuous variables | Linear | Pearson r |
| Association of two continuous variables | Monotonic, outliers | Spearman rho |
| Distribution difference | Any | Kolmogorov-Smirnov |
| Anything with an awkward statistic | Any | Bootstrap CI or permutation test |

Traps:
- With large n, almost everything is significant. Judge the effect size.
- Testing many hypotheses inflates false positives. Use Holm (controls family-wise error) or Benjamini-Hochberg (controls false discovery rate).
- Choosing the test after seeing the data is a hidden form of multiple testing.
- Non-independence (repeated measures, clustered users) makes standard errors too small. Use cluster-robust errors or mixed models.

## 2. Regression and ANOVA

- **OLS**: check residuals for non-linearity, heteroscedasticity (use robust standard errors such as HC3 if present), and influential points (Cook's distance).
- **Multicollinearity**: check VIF. High VIF makes individual coefficients unstable but doesn't hurt prediction.
- **Logistic regression**: report odds ratios with CIs. Watch for separation (huge coefficients); use regularization if it happens.
- **Counts**: Poisson regression, or negative binomial when variance exceeds the mean.
- **Grouped data**: mixed-effects models (`statsmodels` MixedLM) when observations nest in groups.
- **ANOVA / MANOVA**: ANOVA compares means across factor levels; MANOVA handles several correlated outcomes at once. Check homogeneity of variance (Levene's test), or use Welch's ANOVA.
- Regression coefficients are associations unless the design supports a causal reading (section 4).

## 3. Experimental design and A/B tests

**Before launch:**
- Pick one primary metric and a small set of guardrail metrics.
- Choose the randomization unit (user, session, account) to match the unit of the metric and to avoid spillover between groups.
- Compute sample size from baseline rate, minimum detectable effect, alpha (usually 0.05), and power (usually 0.8). In Python: `statsmodels.stats.power`.
- Run an A/A test, or check for sample ratio mismatch, to catch broken assignment.

**During and after:**
- **Peeking**: checking repeatedly and stopping at the first significant result inflates false positives. Use a fixed horizon, or sequential methods (group sequential boundaries, always-valid p-values) if early stopping is needed.
- **Variance reduction**: CUPED, which adjusts for a pre-period covariate, can shrink the required sample considerably.
- **Novelty effects**: compare early and late periods of the test.
- **Segments**: treat post-hoc segment findings as hypotheses for the next test, not conclusions.

**Other designs:**
- **Multi-armed bandits**: when you care about reward during the test more than a clean estimate (Thompson sampling, UCB).
- **Factorial designs**: test several factors and their interactions in one experiment.
- **Response surface methods**: tune continuous settings (price, dosage) toward an optimum.
- **Blocking and stratified randomization**: balance known important covariates across arms.

## 4. Causal inference from observational data

Every method rests on an assumption that can't be fully tested. State it, and argue why it plausibly holds.

| Method | Use when | Key assumption |
|---|---|---|
| Randomized experiment | You can assign treatment | Randomization was done correctly |
| Regression adjustment / matching / propensity scores (IPW, doubly robust) | Treatment depends on observed covariates | No unmeasured confounders; overlap between groups |
| Difference-in-differences | Treated and control groups observed before and after | Parallel trends without treatment (check pre-period trends) |
| Regression discontinuity | Treatment assigned by a cutoff on a running variable | No manipulation at the cutoff (McCrary test); continuity of other covariates |
| Instrumental variables | A variable shifts treatment but affects the outcome only through it | Exclusion restriction; strong first stage (F > 10) |
| Synthetic control | One or a few treated units with a long pre-period | Weighted donor pool reproduces the pre-period path |
| Mediation analysis | You want to split an effect into direct and indirect paths | No unmeasured mediator-outcome confounding |

- **Sensitivity analysis**: report how strong an unmeasured confounder would need to be to erase the effect (E-values, Rosenbaum bounds).
- Draw the causal graph (DAG) to decide what to control for. Controlling for a mediator or a collider biases the estimate.
- Libraries: `statsmodels`, `linearmodels`, `DoWhy`, `EconML`, `causalml`.

## 5. Time series

**Explore**: plot the series, decompose it into trend, seasonality, and residual (STL), check stationarity (ADF, KPSS), and look at ACF and PACF.

**Models, roughly from simple to complex:**
- Naive, seasonal naive, moving average. Always score these.
- Exponential smoothing (ETS, Holt-Winters): strong for level, trend, and seasonality.
- ARIMA / SARIMA / SARIMAX: when autocorrelation structure matters; add exogenous regressors with SARIMAX.
- Prophet: quick baseline with holidays and multiple seasonalities; often beaten by tuned alternatives.
- State space models / Kalman filters: time-varying parameters, missing data.
- Gradient boosting on lag, rolling, and calendar features: often the strongest for many related series.
- Deep learning (N-BEATS, TFT, DeepAR): large collections of series with enough data.

**Validation**: rolling-origin backtesting (walk-forward) only. Never shuffle. Make sure lag and rolling features use only data available at forecast time.

**Metrics**: MAE, RMSE, MASE (scale-free, compares against naive). Avoid MAPE when values can be near zero.

**Anomaly detection**: residuals from a decomposition or forecast model, flagged by a robust threshold such as a multiple of the median absolute deviation.

## 6. Survival analysis

Use it when the outcome is time until an event (churn, failure, conversion) and some units haven't had the event yet (censored).

- Kaplan-Meier curves for describing survival; log-rank test for comparing groups.
- Cox proportional hazards for covariate effects. Check proportional hazards with Schoenfeld residuals.
- Accelerated failure time models or survival forests when proportional hazards fails.
- Dropping censored rows or treating them as "no event" biases results.
- Library: `lifelines`, `scikit-survival`.

## 7. Bayesian methods

Useful when data is small, when you have real prior knowledge, when you need full uncertainty distributions, or for hierarchical data (many groups with few observations each, where partial pooling helps).

- Conjugate shortcuts: Beta-Binomial for conversion rates, Normal-Normal for means.
- Hierarchical models for many related groups (stores, regions, products).
- Run prior predictive checks before fitting and posterior predictive checks after.
- Check convergence (R-hat near 1.0, effective sample size, no divergences).
- Libraries: `PyMC`, `NumPyro`, `bambi`, `cmdstanpy`.

## 8. Choosing an ML algorithm

| Situation | Good starting point |
|---|---|
| Tabular data, any size | Gradient boosting (LightGBM, XGBoost, CatBoost) |
| Need interpretability or few rows | Regularized linear or logistic regression, shallow trees, GAMs |
| Many categorical features | CatBoost, or LightGBM with native categorical handling |
| Images, audio, text, sequences | Neural networks, usually fine-tuned pretrained models (transfer learning) |
| Text with little labeled data | TF-IDF + linear model as a baseline; embeddings + linear model next |
| No labels, find groups | K-means (spherical clusters), GMM, HDBSCAN (arbitrary shapes, noise); validate with silhouette scores and domain review |
| Reduce dimensions | PCA (linear, for modeling); UMAP or t-SNE (for visualization only) |
| Rare-event detection without labels | Isolation Forest, LOF, autoencoder reconstruction error |
| Recommendations | Popularity baseline, then matrix factorization (ALS), then two-tower or sequence models |
| Graph-structured data | Graph features (degree, PageRank, communities) in a GBM, then graph neural networks |

Reinforcement learning and genetic algorithms solve narrower problems (sequential decisions, black-box search). Reach for them only when the problem actually has that shape.

## 9. Feature engineering

- **Domain features usually beat generic ones**: ratios, rates, differences, and counts that a domain expert would compute.
- **Categorical encoding**: one-hot for low cardinality; target encoding for high cardinality, computed out-of-fold to avoid leakage; frequency encoding as a cheap alternative.
- **Scaling**: needed for linear models, SVMs, KNN, and neural networks; not needed for tree models.
- **Transformations**: log or Box-Cox for skewed positive values; binning only when the relationship is truly stepwise.
- **Interactions**: products or ratios of features for linear models; trees find many interactions on their own.
- **Time features**: calendar parts (day of week, month, holiday), time since an event, lags, and rolling statistics. Compute these strictly from the past.
- **Group aggregates**: per-entity mean, count, and std, computed out-of-fold if they involve the target.
- **Feature selection**: permutation importance on validation data, or drop-column tests. Do selection inside CV, or you'll overstate performance.

## 10. Hyperparameter tuning and ensembles

- Tune with Optuna or random search rather than a full grid. Bayesian optimization (Optuna's TPE) finds good settings in fewer trials.
- Tune the parameters that matter most first. For GBMs: learning rate with early stopping, number of leaves or depth, min child samples, and subsampling.
- Use nested CV, or a held-out set, when you need an unbiased estimate of tuned performance.
- **Ensembles**: averaging diverse models (different algorithms, features, or seeds) is cheap and usually helps. Stacking needs out-of-fold predictions for the meta-model, or it will overfit.
- Rank-average models when their output scales differ.
- AutoML tools (AutoGluon, FLAML) give a strong baseline quickly and show which model families work.

## 11. Evaluation metrics

**Classification:**
- ROC AUC for ranking quality; PR AUC when positives are rare.
- Log loss or Brier score when probabilities matter; check calibration with a reliability curve.
- Precision, recall, and F1 at a chosen threshold. Pick the threshold from the costs of errors, not by default at 0.5.

**Regression:**
- RMSE penalizes large errors; MAE is robust to outliers; R² is unit-free but can mislead across datasets.
- RMSLE or MAPE-style metrics for relative error, with care near zero.

**Ranking**: NDCG, MAP, recall@k.

**Business translation:**
- **Lift**: response rate in the top k% divided by the base rate. Cumulative gains charts show it well.
- **Expected value**: combine the confusion matrix with the cost or benefit of each cell.
- **ROI**: incremental benefit minus cost, divided by cost. Show every input so the reader can check it.

## 12. Model interpretation

- **Global**: permutation importance (on validation data), SHAP summary plots, partial dependence and ICE plots.
- **Local**: SHAP values for single predictions.
- Built-in importance from tree models (gain or split count) is biased toward high-cardinality features. Prefer permutation or SHAP.
- Correlated features share or split importance unpredictably. Group them before interpreting.
- Interpretation describes the model, not the world. Causal claims still need a causal design.

## 13. Visualization

- Pick the chart from the question: distribution (histogram, box, violin), comparison (bar, dot plot), relationship (scatter, hexbin for many points), change over time (line), composition (stacked bar; use pies sparingly).
- Label axes with units, start bar charts at zero, and put the takeaway in the title ("Churn doubles after the price change" rather than "Churn by month").
- Use small multiples instead of overloaded single charts.
- Interactive charts (plotly) for exploration and dashboards; static charts (matplotlib, seaborn) for reports.
- Use geographic maps only when location is the point; network graphs only when connections are the point. 3D charts and animation rarely make data clearer.
