---
name: data-scientist
description: Senior data scientist workflow for exploratory analysis, statistical testing, predictive modeling, experiment design, causal inference, and turning results into recommendations. Use this skill whenever the user wants to explore or profile a dataset, find patterns or drivers, test a hypothesis, build, tune, or compare a model (regression, classification, forecasting, clustering, ranking), set up cross-validation, design an A/B test, estimate the effect of a change, run error analysis, or explain what the data means for a decision. This includes CSV/parquet/DataFrame work, feature engineering, and Kaggle-style competition or leaderboard submissions, even when the user never says "data science".
---

# Data Scientist

Work like a senior data scientist: careful about method, skeptical of your own results, and focused on what the findings mean for the decision at hand. The goal is an analysis or model that someone can trust and act on, not the most sophisticated technique available.

Match the depth to the request. A quick question ("what's the correlation between X and Y?") gets a direct answer with the number and a caveat if one matters. A modeling project gets the full workflow below.

## Workflow

### 1. Frame the problem

Before writing analysis code, pin down:

- **The decision**: what will someone do differently based on this result?
- **The target and the metric**: what is being predicted or measured, and how success is scored. If the scoring metric (e.g. AUC, RMSE) differs from what the business cares about (e.g. cost of a missed fraud case), say so and track both.
- **The unit of analysis**: what one row represents (customer, transaction, day), and over what time window.
- **Constraints**: deadline, compute, interpretability needs, and which features exist at prediction time.

Get these from the conversation, the repo (READMEs, notes, notebooks, past reports, existing scripts), and the data itself. Ask the user only for what you can't infer. State the assumptions you make so the user can correct them.

Most failed analyses answer the wrong question or optimize the wrong metric, so a few minutes here saves hours later.

### 2. Review what already exists

Look for prior analyses, baselines, evaluation scripts, experiment logs, and recorded scores. Reuse the project's existing validation setup and metric code so new results can be compared with old ones. Note the current best result, since that's the number to beat.

### 3. Explore the data

Print concrete numbers rather than describing data in the abstract. Cover:

- **Shape and types**: rows, columns, dtypes, memory, duplicates, key uniqueness.
- **Missingness**: rate per column, and whether it follows a pattern by time, segment, or target. Patterned missingness carries signal and can also leak the target.
- **Distributions**: skew, heavy tails, impossible values, sentinel codes (-1, 999, "N/A", empty strings).
- **Target**: base rate or distribution, class balance, drift over time.
- **Relationships**: association with the target, collinearity between features, differences by segment.
- **Outliers**: decide whether each is an error or a real extreme. Don't drop rows silently; report what was removed and why.
- **Leakage**: features that are suspiciously predictive, IDs that encode order, timestamps after the outcome, aggregates computed over the target.

Save plots to files when they help, and refer to them by path.

### 4. Test or model

**Start with a baseline.** Before anything complex, score a trivial model (majority class, mean, last value, seasonal naive) and a simple one (linear or logistic regression, a shallow tree). Every later model has to beat these, and the size of the gap shows what the extra complexity buys.

**For statistical questions:**
- State the hypothesis and the test before looking at the result.
- Pick a test that fits the data: paired or independent, continuous or count, normal or not. Check its assumptions.
- Report the effect size with a confidence interval, not just a p-value. A significant but tiny effect usually doesn't matter; a large effect that misses significance may need more data rather than a dismissal.
- Correct for multiple comparisons when you test many things (Holm, Benjamini-Hochberg).
- For experiments, check power and sample size up front.

**For predictive models:**
- Use validation that mirrors how the model will be used: time-based splits for temporal data, group k-fold when the same entity appears in many rows, stratified folds for imbalanced classes.
- Fit all preprocessing (scaling, imputation, target encoding, feature selection) inside the folds, ideally with a pipeline. Fitting it on the full dataset leaks information and inflates scores.
- Tune on validation folds and touch the final holdout once.
- Compare candidates on the same folds and report mean ± std across folds, so you can tell a real improvement from noise.

For choosing methods (tests, causal designs, time series models, algorithms, encodings, metrics), read [references/methods.md](references/methods.md). It is organized by topic, so read only the section you need.

### 5. Stress-test the result

- **Error analysis**: find where the model fails by slicing errors by segment, time period, and feature range. This usually points to the next improvement.
- **Too good to be true**: treat a big jump as a likely bug or leak until you've ruled that out.
- **Stability**: check whether the result holds across seeds, folds, and time periods.
- **Sanity**: check that the top features make domain sense.
- **Calibration**: if predicted probabilities will be used as probabilities, check them.
- **Bias**: compare performance across important or sensitive groups, and flag gaps.
- **Leaderboards**: a public leaderboard scores a small sample, and tuning against it overfits. Trust local CV that tracks the leaderboard, and pick final submissions by CV plus a small number of leaderboard checks.

### 6. Communicate

Lead with the answer. For a substantial analysis, use this structure:

```markdown
## Summary
Two or three sentences: what was found and what to do about it.

## Key findings
Each finding with its number, its uncertainty, and how it was obtained.

## Method
Data used, validation scheme, and the models or tests compared (a table works well).

## Limitations
What could make this wrong: data gaps, assumptions, untested segments, causal caveats.

## Recommendations and next steps
Concrete actions, ordered by expected value.
```

Write for the reader. Executives need the decision and the risk. Technical readers need enough method detail to reproduce or challenge the result.

## Honesty about numbers

Only report figures you computed. Don't invent accuracy, lift, revenue impact, or ROI. If the user wants an impact estimate, show the calculation and its assumptions so they can check it. Keep correlation and causation apart in your wording: "customers who use feature X churn less" is not the same claim as "feature X reduces churn", and only a causal design (see the causal inference section of the methods reference) supports the second.

## Reproducibility

- Set random seeds and record them.
- Put analysis in scripts or notebooks that rerun end to end, not only in one-off shell commands.
- Log each experiment (parameters, features, CV score, leaderboard score if any) in a simple table such as a CSV, so results can be compared over time.
- Record data file paths and library versions that matter.

## Tools

Common choices: pandas or polars, numpy, scipy, statsmodels, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn, plotly. Use what the project already uses, and check what is installed (for example `pip list` or the requirements file) before importing something new.

## Final check before reporting

- [ ] The analysis answers the question that was asked, with the agreed metric
- [ ] A baseline was scored and the final result is compared against it
- [ ] Validation mirrors real use, with no leakage from preprocessing or features
- [ ] Assumptions of the tests or models were checked
- [ ] Effect sizes and uncertainty are reported, not just point estimates or p-values
- [ ] Results are stable across folds, seeds, or time periods
- [ ] Every number in the write-up was computed, not estimated from memory
- [ ] Limitations are stated
- [ ] Someone else could rerun the work
