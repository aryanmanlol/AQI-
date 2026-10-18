# Multivariate Air Quality (AQI) Forecasting with VAR/VECM, SARIMAX, and Bootstrapped Uncertainty
## Applied Time Series Analysis — Capstone Project Report Outline

---

### Abstract
- **Context & Motivation**: Urban air pollution in India (e.g., National Capital Region, Indo-Gangetic Plain) exhibits severe seasonal volatility and complex cross-atmospheric dependencies.
- **Objectives**: Develop and validate a multivariate econometric framework comparing Vector Autoregression (VAR), Vector Error Correction Models (VECM), SARIMAX with weather exogeneity, and modern ML baselines (LightGBM).
- **Key Methodologies**: Residual bootstrap and moving block bootstrap uncertainty quantification; rolling-origin cross-validation; official CPCB NAQI conversion.
- **Key Findings**: Summary of ranking models, optimal lag structures, cointegration ranks, and interval coverage performance.

---

### 1. Introduction and Atmospheric Domain Background
- **1.1 The Urban Pollution Crisis**: Particulate matter ($PM_{2.5}$, $PM_{10}$), gaseous precursors ($NO_2, SO_2, CO$), and secondary oxidants ($O_3$).
- **1.2 Meteorological Coupling Mechanisms**: Atmospheric boundary layer dynamics, thermal inversions in winter, wind speed dispersion, and monsoon wet scavenging.
- **1.3 Episodic Exogenous Shocks**: Fireworks during Diwali and agricultural biomass stubble burning in northwestern India.
- **1.4 Research Objectives and Contributions**:
  - Econometric evaluation of bidirectional feedback loops.
  - Comparing classical cointegration (VECM) with differenced VAR.
  - Honest multi-step forecasting without future covariate leakage.

---

### 2. Data Acquisition, Engineering, and Quality Assurance
- **2.1 Data Sources & Provenance**:
  - Open-Meteo Air Quality API (Copernicus Atmosphere Monitoring Service - CAMS reanalysis).
  - Open-Meteo Historical Weather Archive API.
  - User-supplied CPCB ground-station CSV ingestion schema.
- **2.2 Data Quality Audit**:
  - Missingness percentage thresholds, continuous gap distribution.
  - Linear vs. seasonal spline imputation sensitivity analysis.
- **2.3 Temporal Aggregation**:
  - Hourly to daily resampling protocols (daily mean for concentrations; cumulative sum for precipitation).
- **2.4 Feature Engineering**:
  - Variance-stabilizing logarithmic transformations: $\ln(x + \epsilon)$.
  - Calendar markers: Diwali window ($[-2, +5]$ days), seasonal dummy variables (winter, monsoon), weekend indicators.

---

### 3. Exploratory Data Analysis & Time Series Properties
- **3.1 Trend & Multi-Frequency Seasonality**:
  - STL decomposition (Seasonal-Trend decomposition using Loess).
  - Quantifying seasonal and trend strength coefficients.
- **3.2 Autocorrelation and Cross-Correlation Dynamics**:
  - ACF and PACF decay patterns.
  - Distributed lag cross-correlations between $PM_{2.5}$ and boundary layer parameters.

---

### 4. Econometric Methodology & Model Formulations
- **4.1 Stationarity & Integration Order Diagnostics**:
  - Augmented Dickey-Fuller (ADF) unit root testing vs. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity testing.
  - Resolving testing discordance and justifying $I(0)$ vs. $I(1)$ classifications.
- **4.2 Vector Autoregression (VAR) & Lag Specification**:
  - Lag selection criteria comparison: AIC, BIC, HQIC, FPE.
  - Stability verification via companion matrix eigenvalues and polynomial roots on the complex plane.
- **4.3 Johansen Cointegration & Vector Error Correction (VECM)**:
  - Maximum Eigenvalue and Trace test statistics.
  - Estimating cointegration vectors ($\beta$) and error-correction speed-of-adjustment parameters ($\alpha$).
- **4.4 Univariate Baselines & SARIMAX**:
  - Random walk / naive and seasonal naive benchmarks.
  - Holt-Winters Exponential Smoothing (ETS).
  - SARIMAX with lagged exogenous weather regressors.
- **4.5 Machine Learning Benchmark (LightGBM)**:
  - Recursive multi-step forecasting with strictly causal time-ordered splits.
- **4.6 Uncertainty Quantification**:
  - VAR residual bootstrap intervals.
  - Moving Block Bootstrap (MBB) paired with bagged ETS ensembles.

---

### 5. Empirical Results & Comparative Evaluation
- **5.1 Cross-Validation Strategy**:
  - Expanding-window rolling-origin cross-validation (7-day step across horizons $h \in \{1, 3, 7\}$).
- **5.2 Numerical Metrics**:
  - $MAE, RMSE, MAPE$, and Mean Absolute Scaled Error ($MASE$).
  - Diebold-Mariano hypothesis testing of predictive superiority.
  - Empirical coverage of 80% and 95% prediction intervals and mean interval width.
- **5.3 CPCB Air Quality Index Classification**:
  - Conversion of pollutant predictions to CPCB sub-indices.
  - Categorical accuracy and multi-class confusion matrix across health risk categories (Good to Severe).

---

### 6. Structural Econometric Insights
- **6.1 Granger Causality Analysis**:
  - Pairwise F-test matrix interpretation (predictive precedence vs. physical causality).
- **6.2 Orthogonalized Impulse Response Functions (OIRF)**:
  - Dynamic responses of $PM_{2.5}$ to orthogonalized 1-S.D. shocks in wind speed, temperature, and $PM_{10}$.
- **6.3 Forecast Error Variance Decomposition (FEVD)**:
  - Short-run vs. long-run variance contributions of exogenous drivers.

---

### 7. Limitations & Discussion
- **7.1 Gridded Reanalysis vs. In-Situ Monitoring**: CAMS reanalysis smoothing effects compared to local micro-environment sensor spikes.
- **7.2 Epistemological Caveat on Granger Causality**: Information precedence does not equate to causal physical mechanisms.
- **7.3 Structural Breaks & Regime Shifts**: COVID-19 lockdown anomalies (2020) and long-term vehicular policy interventions.

---

### 8. Conclusions & Operational Recommendations
- Key takeaways on model selection for urban forecasting centers.
- Value of integrating bootstrapped prediction intervals in public health advisory systems.

---

### References
1. CPCB (2014). *National Air Quality Index*. Central Pollution Control Board, Ministry of Environment, Forest and Climate Change, Govt. of India.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
3. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
4. Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). *Bagging exponential smoothing methods using STL decomposition and Box-Cox transformation*. International Journal of Forecasting.
