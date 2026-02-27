# Day 5 Design: XGBoost + Model Comparison

**Date**: 2026-02-27
**Status**: Approved
**Scope**: 05_xgboost.ipynb + 06_model_comparison.ipynb + src/models/xgboost_model.py

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Granularity | Daily (not weekly) | Consistent with Day 2-4; gap analysis in comparison |
| Model approach | Single Global XGBoost | Family as feature; more data, less overfitting risk |
| Optuna trials | 50 per run | Sufficient for portfolio; ~5-10 min runtime |
| SHAP | Summary + Bar plot | Portfolio-attractive, interpretable |
| Ensemble | Weighted average (1/MAPE) | Simple, explainable, CLAUDE.md spec |

---

## 05_xgboost.ipynb

### Feature Engineering

| Group | Features | Source |
|-------|----------|--------|
| Lag | sales_lag_7, 14, 28, 365 | feature_engineering.py |
| Rolling | sales_rolling_mean_7/30, std_7/30 | feature_engineering.py |
| EWM | sales_ewm_7/30 | feature_engineering.py |
| Date | day_of_week, month, quarter, week_of_year, is_weekend, day_of_year | data_loader.py |
| Exogenous | oil_price, is_holiday, onpromotion, total_transactions | data_loader.py |
| Category | family (label encoded) | New |

- NaN rows from lag_365 dropped → train starts ~2014-01-01
- Rolling features already use shift(1) to prevent future leakage

### Optuna Search Space

```
n_estimators: [100, 1000]
max_depth: [3, 10]
learning_rate: [0.01, 0.3] (log scale)
subsample: [0.6, 1.0]
colsample_bytree: [0.6, 1.0]
min_child_weight: [1, 10]
reg_alpha: [0, 10]
reg_lambda: [0, 10]
```

- Cross-validation: TimeSeriesSplit(n_splits=5)
- Objective: minimize MAPE on validation fold
- 50 trials total

### SHAP Analysis

- TreeExplainer on validation set
- Summary plot (beeswarm): feature impact direction + magnitude
- Bar plot: top 15 feature importance ranking

### Notebook Structure (~30 cells)

1. Introduction markdown
2. Data load + feature generation
3. NaN handling + train/val split
4. Baseline XGBoost (default params)
5. Optuna tuning (50 trials)
6. Best model retrain + validation prediction
7. Per-family MAPE calculation
8. SHAP Summary + Bar plots
9. Forecast visualization (actual vs predicted, by family)
10. Results export (CSV + PNG)

### Output Files

- `outputs/results/xgboost_results.csv` — per-family metrics
- `outputs/figures/18_xgboost_forecast.png`
- `outputs/figures/19_shap_summary.png`
- `outputs/figures/20_shap_bar.png`

---

## 06_model_comparison.ipynb

### Comparison Table

| Dimension | Details |
|-----------|---------|
| Models | Baseline(Naive_7d), SARIMA/SARIMAX, Prophet_Tuned, XGBoost_Tuned |
| Metrics | MAPE, RMSE, MAE |
| Extra | Train time, inference time |
| Visualization | MAPE heatmap (family × model), bar chart |

### Per-Family Best Model Analysis

- Auto-select best model per family by MAPE
- Interpretation in markdown cells: why this model fits this family's characteristics

### Weighted Average Ensemble

```
weight_i = 1 / MAPE_val_i (normalized)
ensemble_pred = sum(w_i * pred_i) / sum(w_i)
```

- Models: SARIMAX, Prophet_Tuned, XGBoost_Tuned (exclude Baseline)
- Compare ensemble vs single best model per family

### Test Set Final Evaluation

- Best config from validation applied to test set (2017-07 ~ 2017-08)
- Final MAPE report
- Gap analysis vs 12% target with explanation

### Notebook Structure (~25 cells)

1. Overview markdown
2. Load all result CSVs
3. Unified comparison table
4. MAPE heatmap
5. Model bar chart
6. Per-family best model analysis
7. Weighted average ensemble implementation
8. Ensemble vs single model comparison
9. Test set final evaluation
10. Gap analysis + conclusions

### Output Files

- `outputs/results/model_comparison.csv` — unified comparison
- `outputs/results/ensemble_results.csv` — ensemble metrics
- `outputs/results/test_final_results.csv` — test set evaluation
- `outputs/figures/21_mape_heatmap.png`
- `outputs/figures/22_model_comparison_bar.png`
- `outputs/figures/23_ensemble_vs_single.png`

---

## src/models/xgboost_model.py

Wrapper class matching existing SARIMA/Prophet model interfaces:

```python
class XGBoostModel:
    def __init__(self, params: dict | None = None)
    def fit(self, X_train, y_train, X_val, y_val) -> None
    def predict(self, X) -> np.ndarray
    def get_feature_importance(self) -> pd.DataFrame
    def get_shap_values(self, X) -> shap.Explanation
    def summary(self) -> dict
```

---

## Success Criteria

- XGBoost MAPE improves over Prophet_Tuned for at least 3/5 families
- SHAP plots clearly show meaningful feature importance patterns
- Ensemble provides marginal improvement over single best
- All results saved consistently with prior notebooks
- Gap analysis clearly explains distance to 12% target
