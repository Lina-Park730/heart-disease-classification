# Heart Disease Classification

A Python implementation of a coursework classification project, comparing Logistic Regression, K-Nearest Neighbors, and Decision Tree models using five demographic and clinical predictors.

The analysis covers data validation, duplicate removal, feature selection, hyperparameter tuning, and evaluation on a held-out test set. Results below are from this implementation.

## Dataset

The dataset contains 300 observations and five predictors: `age`, `sex`, `chest.pain`, `bp`, and `chol`. The target is `disease`, encoded as `no = 0` and `yes = 1`.

One exact duplicate is removed before splitting, leaving 299 observations. There are no missing values in this file. Median and most-frequent imputation are included in the pipeline to handle missing predictors in compatible inputs. Details and provenance limitations are recorded in [data/README.md](data/README.md).

## Method

- Stratified 80/20 split with random seed 42: 239 training observations and 60 test observations.
- Five outer stratified folds compare models on the training partition. Three inner folds select hyperparameters and either the three highest ANOVA F-score features or all five features.
- Numeric predictors are standardized. Binary predictors retain their 0/1 coding. Imputation, scaling, and feature selection are fitted within each training fold.
- Model selection uses mean outer-fold ROC-AUC. The selected family is tuned again using the complete training partition, then evaluated on the held-out test set.
- Accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC are exported. Classification uses each estimator's default decision rule; the threshold is not tuned on the test set.

| Model | Tuning grid |
| --- | --- |
| Logistic Regression | C = 0.1, 1, 10 |
| KNN | Neighbors = 5, 11, 21; uniform or distance weighting |
| Decision Tree | Maximum depth = 2, 4, unrestricted; minimum leaf size = 5, 15 |

## Results

### Nested cross-validation

| Model | ROC-AUC, mean ± SD | Mean accuracy | Mean F1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.797 ± 0.055 | 0.720 | 0.696 |
| KNN | 0.761 ± 0.061 | 0.707 | 0.693 |
| Decision Tree | 0.736 ± 0.072 | 0.682 | 0.656 |

The standard deviations describe variation across the five outer folds, not confidence intervals.

![Model comparison](figures/model_comparison.png)

### Held-out test set

Logistic Regression was selected, with `C = 10` and all five predictors retained. Feature reduction was evaluated but did not improve the final training CV selection.

| Metric | Logistic Regression | Majority-class baseline |
| --- | --- | --- |
| Accuracy | 0.650 | 0.533 |
| Balanced accuracy | 0.654 | 0.500 |
| Precision | 0.606 | 0.000 |
| Recall | 0.714 | 0.000 |
| F1 | 0.656 | 0.000 |
| ROC-AUC | 0.732 | 0.500 |

The baseline always predicts the training majority class and uses its class prior as the probability. Precision is reported as zero when no positive predictions are made.

![Test evaluation](figures/test_evaluation.png)

## Reproduction

Tested with Python 3.12.14. Install the pinned dependencies and run from the repository root:

```bash
python -m pip install -r requirements.txt
python src/train.py
python -m unittest discover -s tests -v
```

Alternative input and output locations:

```bash
python src/train.py --data data/heart_disease_midterm.csv --output runs/experiment --seed 42 --jobs 1
```

`results/` contains the data audit, split membership, fold-level metrics, comparison summary, selected parameters, test metrics, and test predictions. `source_row` is the zero-based row position in the original CSV, excluding its header. `figures/` contains the model comparison and test evaluation plots. Rerunning the same configuration regenerates these outputs.

## Limitations

This is a small coursework dataset with five predictors and a single held-out split. The 60-observation test set gives an imprecise estimate of generalization, and small cross-validation differences do not establish a reliable ranking. The dataset lacks patient identifiers, so repeated patients cannot be checked beyond exact feature duplicates. Source provenance and label definitions are unverified. The models classify the supplied labels and are not validated for clinical use.
