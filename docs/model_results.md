# Model Evaluation

The system predicts expected learning gain from study-related features.

## Dataset

The current dataset is synthetic and is used for educational prototyping. Results should not be interpreted as evidence of real-world student performance.

## Models Compared

1. Mean Prediction Baseline
2. Linear Regression
3. Random Forest Regressor

## Metrics

- **MAE:** Mean Absolute Error. Lower is better.
- **RMSE:** Root Mean Squared Error. Lower is better.
- **R²:** Proportion of variance explained by the model. Higher is better.

## Results

Evaluation uses a fixed 80/20 train-test split with `random_state=42`.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 13.0000 | 14.7851 | -2.1336 |
| Linear Regression | 0.7185 | 0.8271 | 0.9902 |
| Random Forest | 1.8080 | 2.0471 | 0.9399 |

## Interpretation

On this small synthetic dataset, Linear Regression performs better than both the mean baseline and Random Forest on the fixed hold-out split. Therefore, Linear Regression remains the production model for this prototype.

Because the dataset is small and synthetic, these metrics should be treated as an educational demonstration of the evaluation workflow, not as evidence that the model will generalize to real students.
