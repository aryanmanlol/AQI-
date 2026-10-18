# Empirical Modeling Results & Ranked Summary

## 1. Rolling-Origin Cross-Validation Ranking (Horizon = 1 Day)

| Rank | Model | MAE | RMSE | MAPE (%) | MASE | 80% Coverage | 95% Coverage | Mean Width (95%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **SARIMAX** | 17.96 | 24.01 | 25.3% | 0.69 | 45.8% | 70.8% | 48.2 |
| 2 | **Naive** | 17.11 | 24.83 | 22.4% | 0.65 | 58.3% | 77.1% | 57.0 |
| 3 | **VAR** | 19.01 | 25.08 | 27.1% | 0.73 | 43.8% | 72.9% | 50.1 |
| 4 | **HW_ETS** | 17.83 | 25.22 | 23.2% | 0.68 | 47.9% | 72.9% | 48.4 |
| 5 | **LightGBM** | 20.92 | 25.94 | 31.8% | 0.80 | 45.8% | 70.8% | 51.2 |
| 6 | **SeasonalNaive** | 24.39 | 32.75 | 32.1% | 0.93 | 43.8% | 64.6% | 57.0 |

## 2. Hypothesis Testing: Diebold-Mariano Test

- **Best Model**: `SARIMAX`
- **Second-Best Model**: `Naive`
- **DM Statistic**: `-0.6682`
- **Two-Tailed p-value**: `0.5072`
- **Conclusion**: No statistically significant difference in forecast accuracy at the 5% level.

## 3. CPCB AQI Category Classification Performance

- **Category Accuracy**: `45.83%`

### Confusion Matrix (Actual vs. Predicted AQI Category)

```
                   Pred_Satisfactory  Pred_Moderate  Pred_Poor  Pred_Very Poor  Pred_Unknown
True_Satisfactory                  6              8          0               0             0
True_Moderate                      2              9          4               1             2
True_Poor                          1              2          4               1             2
True_Very Poor                     0              0          1               3             0
True_Unknown                       0              1          1               0             0
```

## 4. Multi-Step Horizon Performance Breakdown

| Horizon (Days) | Top Performing Model | MAE | RMSE | 95% Interval Coverage |
|:---:|:---|:---:|:---:|:---:|
| 1 | SARIMAX | 17.96 | 24.01 | 70.8% |
| 3 | VAR | 25.23 | 32.05 | 52.1% |
| 7 | SARIMAX | 27.46 | 34.16 | 52.1% |
