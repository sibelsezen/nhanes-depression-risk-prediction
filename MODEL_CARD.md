# Model Card

## Model overview

- **Model:** Tuned class-balanced Logistic Regression
- **Task:** Binary classification of elevated depressive-symptom risk
- **Data:** NHANES 2017–2018
- **Inputs:** 29 demographic, body-measurement, sleep, smoking, alcohol-use, and physical-activity predictors
- **Decision threshold:** 0.4125, selected from out-of-fold training probabilities by maximizing F2
- **Primary selection metric:** Recall
- **Secondary metric:** ROC-AUC

## Intended use

The model supports an educational demonstration of screening-oriented machine learning. It may be used to illustrate preprocessing, imbalance handling, cross-validation, threshold selection, evaluation, and deployment.

It is **not** intended to diagnose depression, recommend treatment, replace a validated clinical instrument, or make decisions about eligibility, employment, insurance, or access to care.

## Evaluation design

The project uses a stratified 80/20 train-test split. Hyperparameter tuning, model selection, overfitting checks, and threshold selection use training data only. The final selected pipeline is evaluated once on the untouched test set.

## Untouched test performance

| Metric | Value |
|---|---:|
| Recall | 0.7609 |
| ROC-AUC | 0.7866 |
| Specificity | 0.7017 |
| Balanced accuracy | 0.7313 |
| Precision | 0.2029 |
| F1 | 0.3204 |
| F2 | 0.4909 |
| PR-AUC | 0.2862 |

The test set contains 92 elevated-risk cases. The model identifies 70 and misses 22. It predicts 345 positives, including 275 false positives. This tradeoff reflects a screening threshold designed to emphasize recall.

## Limitations and risks

- The analysis is based on one cross-sectional NHANES cycle.
- Survey responses may contain recall, measurement, or reporting bias.
- The model has not undergone external, prospective, subgroup-fairness, or clinical validation.
- Low positive-class prevalence limits positive predictive value.
- Performance may shift in other populations or time periods.
- Predictions must be interpreted as statistical screening scores, not clinical conclusions.

## Reproducibility

The serialized artifact stores the fitted preprocessing pipeline, classifier, predictor order, positive class, imbalance treatment, threshold strategy, and selected threshold. Deployment dependencies are pinned in `requirements.txt`; modeling dependencies are pinned in `requirements-modeling.txt`.
