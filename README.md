# Multivariate Air Quality (AQI) Forecasting with VAR/VECM, SARIMAX, Prophet and Bootstrapped Uncertainty

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A complete, runnable, final-year-level capstone project for an **Applied Time Series Analysis** course. The project automatically builds its own dataset from atmospheric APIs, fits multivariate econometric and modern machine learning models, produces multi-step forecasts with bootstrapped prediction intervals, and benchmarks performance using rolling-origin cross-validation.

---

## 📌 Project Overview & Highlights

- **Multi-Source Autonomous Ingestion**: Downloads hourly pollutant data (CAMS model output via Open-Meteo Air Quality API) and meteorological covariates (Open-Meteo Historical Archive API) with exponential backoff and yearly chunking. Also accepts user-provided CPCB CSVs.
- **Multivariate Econometric Modeling**:
  - Vector Autoregression (VAR) with AIC/BIC lag selection and polynomial stability checks.
  - Johansen Cointegration Test with Vector Error Correction Models (VECM).
  - SARIMAX with lagged weather regressors (honest handling without look-ahead bias).
  - Structural Econometrics: Pairwise Granger causality matrix, Orthogonalized Impulse Response Functions (OIRF), and Forecast Error Variance Decomposition (FEVD).
- **Machine Learning & Baselines**:
  - LightGBM with recursive multi-step forecasting and strict chronological splits.
  - Holt-Winters Exponential Smoothing (ETS), Seasonal Naive, and Naive baselines.
  - Optional Facebook Prophet module with clean degradation if uninstalled.
- **Bootstrapped Uncertainty**:
  - VAR residual bootstrap prediction intervals (80% and 95%).
  - Moving Block Bootstrap (MBB) paired with bagged ETS ensembles.
- **Rigorous Time-Series Evaluation**:
  - Expanding-window rolling-origin cross-validation (7-day step, horizons 1, 3, 7 days).
  - Metrics: MAE, RMSE, MAPE (zero-safe), MASE (seasonal naive benchmark), interval coverage, and interval width.
  - Diebold-Mariano statistical significance hypothesis test.
  - Regulatory conversion to Indian National Air Quality Index (NAQI) sub-indices and confusion matrix.
- **Interactive 7-Tab Dashboard**: Streamlit interface with interactive Plotly fan charts and structural exploration.

---

## 🛠️ Environment Setup (Windows 10 / 11)

Open **PowerShell** in the project directory:

```powershell
# 1. Create a virtual environment using Python 3.10+
py -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\activate

# 3. Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Prophet (Optional):** Prophet is intentionally made optional. The pipeline automatically checks whether Prophet is installed; if not, it logs a clean notice and proceeds without interruption. To enable Prophet: `pip install prophet`.

---

## 🚀 Running the Project

### 1. End-to-End Pipeline Execution (One Command)
Run the entire pipeline end-to-end for Delhi:
```powershell
py run_all.py --city Delhi --allow-synthetic
```
The pipeline automatically:
1. Downloads ~3 years of pollutant and weather data (or generates synthetic data if offline).
2. Cleans data, caps outliers, and constructs log transforms and calendar flags.
3. Performs EDA, STL decomposition, and stationarity tests (ADF/KPSS).
4. Fits VAR, VECM, SARIMAX, LightGBM, and baseline models.
5. Performs Granger causality, IRF, and FEVD structural analysis.
6. Computes bootstrapped prediction intervals and runs rolling-origin CV.
7. Writes figures (`outputs/figures/`), tables (`outputs/tables/`), models (`outputs/models/`), and auto-generates `outputs/results.md` with real computed metrics.

### 2. Single-Stage Execution
You can execute individual stages of the modeling pipeline:
```powershell
# Run only Exploratory Data Analysis (generates ACF, PACF, STL plots)
py run_all.py --stage eda

# Run only stationarity tests (ADF and KPSS)
py run_all.py --stage stationarity

# Run only VAR/VECM modeling
py run_all.py --stage var

# Run only Granger causality and IRF structural analysis
py run_all.py --stage analysis

# Run only rolling-origin cross-validation
py run_all.py --stage eval
```

### 3. Launching the Interactive Streamlit Dashboard
```powershell
streamlit run app/streamlit_app.py
```
Explore the 7 interactive tabs:
- **Overview & Quality**: City selector, data quality reports, and mandatory CAMS/Synthetic attribution banners.
- **EDA & Seasonality**: Interactive Plotly time series, monthly and weekly box plots, and cross-correlations.
- **Stationarity**: Interactive ADF and KPSS tables with econometric decisions.
- **Forecasts & Fan Charts**: Interactive model selector and horizon slider with 80% and 95% fan charts and official CPCB AQI category forecasts.
- **VAR Econometrics**: Granger causality heatmap, companion polynomial roots, interactive IRF, and FEVD charts.
- **Model Comparison**: Benchmark metrics table and rolling-origin comparison charts.
- **Residual Diagnostics**: Ljung-Box, Jarque-Bera, ARCH-LM tests, and QQ plots.

---

## 🧪 Running the Offline Test Suite

Run pytest offline to verify AQI calculations, metric formulas, and pipeline integrity:
```powershell
pytest -v
```

---

## 📂 Repository Structure

```
aqi-var-forecasting/
├── README.md                          # Setup, commands, methodology, limitations
├── requirements.txt                   # Production dependencies
├── config.yaml                        # Central configuration (cities, lags, dates, seeds)
├── run_all.py                         # Single-command end-to-end pipeline runner
├── src/
│   ├── __init__.py
│   ├── config.py                      # Config loader, RNG seed, path helpers
│   ├── data_fetch.py                  # Open-Meteo API fetcher, CPCB loader, synthetic generator
│   ├── data_clean.py                  # Regular index, interpolation, outlier capping
│   ├── features.py                    # Log transforms, event flags (Diwali, seasons)
│   ├── aqi_utils.py                   # CPCB AQI breakpoints and sub-index mapping
│   ├── eda.py                         # Time plots, seasonal subseries, ACF/PACF, STL
│   ├── stationarity.py                # ADF and KPSS unit root test decisions
│   ├── baselines.py                   # Naive, Seasonal Naive, HW-ETS, SARIMA AIC grid
│   ├── sarimax_model.py               # SARIMAX with lagged weather regressors
│   ├── var_model.py                   # VAR lag order, stability check, Johansen test, VECM
│   ├── var_analysis.py                # Granger causality matrix, IRF, FEVD
│   ├── prophet_model.py               # Optional Prophet with weather and holiday regressors
│   ├── ml_baseline.py                 # LightGBM with recursive multi-step forecasting
│   ├── uncertainty.py                 # VAR residual bootstrap, Moving Block Bootstrap (MBB)
│   ├── diagnostics.py                 # Ljung-Box, Jarque-Bera, ARCH-LM, QQ plots
│   ├── evaluation.py                  # Rolling-origin CV, DM test, CPCB confusion matrix
│   ├── regimes.py                     # Winter vs. Monsoon vs. Diwali regime analysis
│   └── plotting.py                    # Shared plotting helpers and interactive fan charts
├── app/
│   └── streamlit_app.py               # Interactive 7-tab dashboard
├── notebooks/
│   └── walkthrough.ipynb              # Guided laboratory notebook
├── tests/
│   ├── test_aqi_utils.py              # Unit tests for CPCB breakpoints and categories
│   ├── test_metrics.py                # Unit tests for hand-computed metrics (MASE, MAE, etc.)
│   └── test_data_pipeline.py          # Data pipeline tests using synthetic data
├── data/
│   ├── raw/                           # Raw downloaded and synthetic parquets/CSVs
│   ├── interim/                       # Intermediate clean files and gap masks
│   └── processed/                     # Final featured datasets
├── outputs/
│   ├── figures/                       # Saved plots (ACF, STL, roots, IRF, FEVD, QQ)
│   ├── tables/                        # CSV tables (data quality, stationarity, metrics)
│   ├── models/                        # Serialized fitted model pickles
│   └── results.md                     # Auto-generated ranked summary with real numbers
└── reports/
    ├── report_outline.md              # Academic paper-style report outline
    └── viva_questions.md              # 25 in-depth viva questions with model answers
```

---

## ⚠️ Important Methodological Assumptions & Limitations

1. **Model-Based Pollutant Data (CAMS Reanalysis)**:
   - Pollutant concentrations from the Open-Meteo Air Quality API originate from the **Copernicus Atmosphere Monitoring Service (CAMS)** European atmospheric reanalysis models (~40 km spatial grid), **not direct in-situ ground monitoring stations**.
   - Gridded reanalysis smooths micro-scale emission spikes (e.g., direct street-level vehicular exhaust). For local regulatory deployment, ingest ground-station data by placing CSVs in `data/raw/cpcb_<city>.csv`.
2. **Granger Causality vs. True Physical Causality**:
   - Granger causality evaluates **predictive precedence** (whether past lags of $X$ contain statistically significant information to forecast $Y$). It does **not** demonstrate physical, chemical, or etiological causality and can be confounded by unobserved atmospheric drivers.
3. **Missing Data Handling & Imputation**:
   - Short gaps ($\le 3$ days) are linearly interpolated. Longer gaps are flagged and excluded from estimation to prevent fabricating synthetic correlation structures.
4. **Structural Breaks (COVID-19 Lockdown Risk)**:
   - If historical date ranges span early 2020, extreme anomalies associated with nationwide industrial and transport lockdowns introduce severe structural breaks.
5. **Regulatory Breakpoints Verification**:
   - The breakpoint table in `src/aqi_utils.py` is configured from the 2014 CPCB National Air Quality Index (NAQI). Always verify breakpoints against the current Central Pollution Control Board notification before utilizing outputs for statutory compliance.
6. **Diwali & Holiday Dates**:
   - Holiday windows in `config.yaml` are based on the lunisolar Hindu calendar and must be verified for future calendar years.
