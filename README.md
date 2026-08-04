# Healthcare Insurance Fraud Detection Model

Predicting fraudulent healthcare claims using logistic regression and random forest, with a
strong focus on catching data leakage and validating results honestly rather than optimizing
for a single flashy metric.

## Project Overview

This project builds a binary classifier to flag potentially fraudulent healthcare claims from
a synthetic dataset of ~10,000 claims across 300 providers (2021–2025). The dataset includes
patient demographics, claim details (amount, diagnosis, procedure), provider information, and
timing data.

**Baseline fraud rate:** ~8.3% of claims are fraudulent a meaningfully imbalanced classification
problem that shaped most of the modeling decisions below.

**Goal:** maximize recall on the fraud class (catch as many real fraud cases as possible) while
keeping the model honest  i.e., not relying on artifacts of the synthetic data generation process
that wouldn't hold up on real claims.

## Data Cleaning & Feature Engineering

- Handled missing values in `Insurance_Type` and `Provider_Specialty` (filled based on provider
  lookup, or `'Unknown'`), and `Prior_Visits_12m` (filled with median)  each checked first for
  whether missingness correlated with the fraud label before deciding on an imputation strategy.
- Engineered date-based features from `Claim_Submission_Date`: `Claim_Month`, `Claim_Day`
  (day-of-week), `Is_Weekend`, `hour_Sub`.
- Investigated provider-level and specialty-level fraud rates, filtering for minimum claim volume
  (≥10–20 claims) to avoid trusting rates built on tiny, noisy samples (e.g., a provider with
  2 claims and 1 fraud case is not a "50% fraud rate" pattern worth trusting).

## Data Leakage: Investigation & Findings

A significant part of this project was identifying and removing features that leak information
not actually available at prediction time. The guiding test used throughout: *"Is this value
known at the moment a new claim comes in, before anyone has determined fraud or not?"*

**Dropped as leakage:**
- `Claim_Status`, `Approved_Amount`  both are set only after a claim has been processed and a
  decision made, which in most workflows happens at or after the fraud determination itself.
  Including them would let the model learn "rejected → fraud" instead of real fraud patterns.
- `Claim_ID`  a unique row identifier with no predictive meaning.

**The critical finding — `Days_Between_Service_and_Claim`:**

EDA showed a near-perfect split: claims filed within 0–7 days had a 29.3% fraud rate, while
every other filing-delay bucket (8–14, 15–21, 22–30 days) had **exactly 0% fraud**. This kind
of clean cliff is not realistic for real-world fraud behavior and was identified as a synthetic
data leakage artifact.

An initial logistic regression model trained *with* this feature produced a suspiciously strong
result (0.96 ROC-AUC, 0.93 recall). Inspecting the model's coefficients confirmed the suspicion:
`Days_Between_Service_and_Claim` had a coefficient of **-5.74**, roughly 6x larger than the next
strongest feature (`Claim_Amount` at 0.96) the model was almost entirely riding this one
shortcut rather than learning generalizable patterns.

**Before vs. after removing the leaking feature:**

| Metric | With leakage | Without leakage |
|---|---|---|
| Recall (fraud) | 0.93 | 0.69 |
| Precision (fraud) | 0.37 | 0.22 |
| ROC-AUC | 0.96 | 0.81 |
| PR-AUC | 0.70 | 0.36 |

All reported results below use the leakage-free feature set.

## Encoding & Preprocessing

- **Binary mapping** for two-category columns (`Patient_Gender`, `Chronic_Condition_Flag`).
- **One-hot encoding** for low-cardinality nominal categories confirmed via `.nunique()` checks
  (`Insurance_Type`, `Provider_Specialty`, `Visit_Type`, `Patient_State`, `Diagnosis_Code`,
  `Procedure_Code`, `Claim_Month`, `Claim_Day`) — including columns that are numerically stored
  (e.g., CPT procedure codes) but carry no real numeric/ordinal meaning, and cyclical time features
  (month, day-of-week) where a raw linear encoding would misrepresent adjacency (e.g., December
  and January).
- **Standard scaling** for genuinely continuous numeric columns (`Patient_Age`, `Claim_Amount`,
  `Number_of_Claims_Per_Provider_Monthly`, `Length_of_Stay`, `Prior_Visits_12m`).
- **`Claim_Year` was dropped** after checking claim volume and fraud rate per year — four years
  (2021–2024) showed a flat ~8% fraud rate with no real trend, and 2025 had only 99 claims
  (vs. ~2,400–2,570 for other years), making its apparent fraud rate unreliable due to small
  sample size.
- All encoders/scalers were **fit on the training set only** and applied (transform-only) to the
  test set, to avoid leaking test-set distribution information into training.

## Handling Class Imbalance

Two approaches were tested and compared:

1. **`class_weight='balanced'`** in the model itself.
2. **SMOTE** oversampling on the training set only.

| Approach | Recall (fraud) | Precision (fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `class_weight='balanced'` | 0.69 | 0.22 | 0.81 | 0.36 |
| SMOTE | 0.69 | 0.22 | 0.80 | 0.365 |

The two approaches performed essentially identically. Given no measurable benefit from SMOTE's
added complexity — and the reasonable concern that oversampling a synthetic dataset mostly
reinforces whatever pattern (real or artificial) is already present rather than adding new
information — **`class_weight='balanced'` was kept as the simpler, equally effective approach.**

## Model Comparison

Two model types were trained and compared: Logistic Regression and Random Forest.

**Initial single train/test split results:**

| Model | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.69 | 0.22 | 0.81 | 0.36 |
| Random Forest (max_depth=10) | 0.43 | 0.31 | 0.80 | 0.29 |
| Random Forest (max_depth=5) | 0.71 | 0.20 | 0.80 | 0.34 |

The initial Random Forest (depth=10) underperformed badly on recall. A train-vs-test recall
check confirmed overfitting (train recall 0.76 vs. test recall 0.43) — the deeper trees were
memorizing training-specific fraud patterns rather than generalizing. Reducing `max_depth` to 5
closed most of that gap and improved test recall to 0.71.

**5-fold cross-validation** (to check whether the single-split comparison above was trustworthy,
given only ~166 fraud cases in the test set):

| Model | CV mean recall | CV std |
|---|---|---|
| Logistic Regression | 0.596 | 0.036 |
| Random Forest (max_depth=5) | 0.635 | 0.007 |

Cross-validation changed the conclusion from the single split: Random Forest scored both **higher**
and **far more stable** (5x lower standard deviation) recall across folds. The single-split
comparison had made the two models look nearly tied — cross-validation revealed that was partly
due to a favorable split for Random Forest and an unfavorable one for Logistic Regression.

## Final Model

**Random Forest** (`n_estimators=1000`, `max_depth=5`, `class_weight='balanced'`) was selected
as the final model, based on its higher and more stable cross-validated recall.

| Metric | Value |
|---|---|
| Recall (fraud) | 0.71 (single split) / 0.635 (5-fold CV mean) |
| Precision (fraud) | 0.20 |
| ROC-AUC | 0.80 |
| PR-AUC | 0.34 |

**Threshold:** the default 0.5 decision threshold was evaluated against 0.3/0.4/0.6 alternatives.
Lower thresholds substantially increase recall at the cost of precision collapsing below usable
levels (e.g., 0.3 → 1.00 recall but only 0.08 precision — the model flags nearly every claim as
fraud, making this threshold unusable). 0.5 was kept as a reasonable default; 0.4 is noted as a
viable alternative if the business priority shifts further toward maximizing fraud capture over
investigator workload.

## Key Takeaways

- A model that looks unusually good (0.96 ROC-AUC) on a first pass is a signal to investigate,
  not celebrate — feature coefficient inspection traced the result back to a synthetic leakage
  artifact.
- More complex models are not automatically better: an initial Random Forest underperformed a
  simple Logistic Regression due to overfitting, and only became competitive after tuning depth.
- A single train/test split can be misleading with a small positive class (166 fraud cases in
  test); cross-validation was necessary to make a trustworthy final model choice.
- Precision/recall tradeoffs were made explicit via threshold analysis rather than defaulting
  silently to the 0.5 cutoff.

## Limitations

- Trained on synthetic data; some patterns (e.g., filing-delay leakage) reflect artifacts of
  data generation rather than real-world fraud behavior, and overall model performance may not
  transfer directly to real claims data.
- `Provider_ID` was not used directly as a feature (to avoid one-hot cardinality issues and
  leakage from naively-computed provider fraud rates); a cross-fitted target-encoded or
  behavioral provider feature is a natural next step.
- Precision remains low (0.20–0.22) at the chosen threshold, meaning a majority of flagged
  claims would be false alarms in a real investigative workflow — a limitation to state
  explicitly rather than obscure.

## Next Steps

- Cross-fitted target encoding or engineered behavioral features (claim volume, average claim
  amount) for `Provider_ID`, which showed strong raw signal in EDA (one provider at 42.1% fraud
  vs. 8.3% baseline) but was excluded from the final feature set.
- Gradient boosting comparison (e.g., XGBoost/LightGBM) as a third model family.
- Precision-focused refinement (e.g., cost-sensitive threshold selection tied to investigator
  review capacity).
