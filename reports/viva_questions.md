# Viva Voce Examination: 25 Master-Level Questions & Model Answers
## Subject: Applied Time Series Analysis — Multivariate AQI Forecasting

---

### Q1: Why do we model air quality as a multivariate system (VAR/VECM) rather than training independent univariate ARIMA models for each pollutant?
**Model Answer:**  
Pollutants participate in coupled chemical and meteorological processes. For example, nitrogen dioxide ($NO_2$) undergoes photolysis to generate ground-level ozone ($O_3$), while wind speed and temperature simultaneously disperse or concentrate both particulate matter ($PM_{2.5}, PM_{10}$) and gaseous pollutants. Univariate ARIMA models assume that the target variable depends solely on its own past history and random shocks, discarding valuable cross-lagged explanatory power. A Vector Autoregression (VAR) or VECM models bidirectional dynamic interactions simultaneously, improving forecast accuracy and enabling structural policy analysis like Impulse Response Functions (IRF).

---

### Q2: Why is the number of endogenous variables in a VAR typically capped at 6 or 7?
**Model Answer:**  
VAR models suffer from the **curse of dimensionality** (parameter explosion). In a standard $VAR(p)$ system with $K$ endogenous variables, each of the $K$ equations estimates $1 + K \times p$ parameters (plus exogenous coefficients). For $K=7$ and lag order $p=14$, each equation has $1 + 7 \times 14 = 99$ parameters, requiring $99 \times 7 = 693$ parameters overall. With only ~1,000 daily observations, estimating that many parameters severely inflates estimator variance, depletes degrees of freedom, and causes severe overfitting.

---

### Q3: What is the fundamental difference between the null hypotheses of the ADF test and the KPSS test?
**Model Answer:**  
- **Augmented Dickey-Fuller (ADF)**:
  - $H_0$: The series possesses a unit root (is non-stationary, $I(1)$).
  - $H_1$: The series is stationary ($I(0)$).
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**:
  - $H_0$: The series is level- or trend-stationary ($I(0)$).
  - $H_1$: The series possesses a unit root (non-stationary, $I(1)$).
Testing both provides confirmatory analysis. If ADF fails to reject $H_0$ and KPSS rejects $H_0$, we confidently classify the series as $I(1)$.

---

### Q4: If the ADF test and KPSS test disagree, how do you decide whether to difference the variable?
**Model Answer:**  
When ADF rejects the unit root ($p < 0.05$) but KPSS also rejects stationarity ($p < 0.05$), the series is often **trend-stationary** or exhibits structural breaks. In applied forecasting with VAR, where strict covariance-stationarity is required to ensure polynomial roots lie strictly inside the unit circle, conservative practice dictates applying first differencing ($\Delta y_t$) or removing deterministic trends to guard against spurious regression.

---

### Q5: How is the optimal lag length $p$ selected in a VAR model, and why might AIC and BIC suggest different lags?
**Model Answer:**  
Lag selection evaluates $VAR(m)$ models over $m \in \{1, \dots, p_{\max}\}$ using information criteria:
$$\text{AIC}(m) = \ln |\hat{\Sigma}_u| + \frac{2 m K^2}{T}$$
$$\text{BIC}(m) = \ln |\hat{\Sigma}_u| + \frac{m K^2 \ln T}{T}$$
AIC prioritizes predictive accuracy and has a lower penalty for degrees of freedom, often selecting longer lags. BIC applies a heavier penalty proportional to $\ln T$, selecting more parsimonious models. In small samples, BIC is consistent and protects against overparameterization, whereas AIC can overfit.

---

### Q6: What condition must be satisfied for a $VAR(p)$ process to be stable, and what happens if this condition is violated?
**Model Answer:**  
The companion matrix eigenvalues or, equivalently, the roots of the reverse characteristic polynomial:
$$\det(I_K - A_1 z - A_2 z^2 - \dots - A_p z^p) = 0$$
must all lie **strictly outside the complex unit circle** ($|z| > 1$). In statsmodels root plots, roots of the polynomial are displayed directly, so all moduli must satisfy $|z| < 1$. If violated, the system is non-stationary or explosive: shocks do not dissipate over time, and forecast variance diverges to infinity.

---

### Q7: Explain the concept of Cointegration and what the Johansen Cointegration Test determines.
**Model Answer:**  
Two or more non-stationary series $y_t \sim I(1)$ are cointegrated if a linear combination $\beta' y_t$ is stationary ($I(0)$). This implies a long-run equilibrium relationship binding the variables together despite wandering independently in the short run. The Johansen test estimates the rank $r$ of the matrix $\Pi = \alpha \beta'$ in the VECM representation using Maximum Eigenvalue and Trace tests. If $0 < r < K$, cointegration exists, necessitating a VECM instead of differenced VAR.

---

### Q8: Why is differencing an $I(1)$ cointegrated system into a standard VAR suboptimal compared to estimating a VECM?
**Model Answer:**  
Differencing cointegrated variables removes non-stationarity, but it completely discards the **long-run equilibrium information** contained in the level relationship ($\beta' y_{t-1}$). A Vector Error Correction Model (VECM) explicitly incorporates both: short-run dynamics via differenced terms ($\Delta y_{t-k}$) and the speed of mean-reversion toward long-run equilibrium via the error correction term $\alpha (\beta' y_{t-1})$.

---

### Q9: Does Granger causality prove that one pollutant physically causes another?
**Model Answer:**  
**No.** Granger causality is strictly an econometric measure of **predictive precedence and incremental information containment**. Variable $X$ "Granger-causes" $Y$ if historical values of $X$ provide statistically significant information for predicting $Y_t$ beyond what is already contained in the past history of $Y_t$ alone. It does not establish physical, chemical, or etiological causality and can be confounded by omitted common third variables (e.g., wind speed driving both $PM_{2.5}$ and $NO_2$).

---

### Q10: What are Orthogonalized Impulse Response Functions (OIRF), and why is Cholesky ordering critical?
**Model Answer:**  
OIRFs trace the dynamic response of each endogenous variable to an exogenous one-standard-deviation shock in another variable over future horizons. Because VAR reduced-form error terms $u_t$ have a non-diagonal covariance matrix $\Sigma_u$ (contemporaneous correlation), a shock to one equation implies simultaneous shocks to others. Cholesky factorization decomposes $\Sigma_u = P P'$, orthogonalizing the shocks. However, Cholesky decomposition is lower-triangular, meaning the ordering of variables in the vector matters: variables placed earlier exert contemporaneous effects on later variables, but later variables only affect earlier variables with a lag.

---

### Q11: What does Forecast Error Variance Decomposition (FEVD) reveal about a time series?
**Model Answer:**  
FEVD quantifies the percentage of the $h$-step-ahead forecast error variance of variable $i$ that is attributable to exogenous orthogonal shocks to variable $j$ versus shocks to itself. In air quality, FEVD answers questions such as: *"At a 7-day horizon, what proportion of the forecast uncertainty in $PM_{2.5}$ stems from wind speed fluctuations versus auto-particulate inertia?"*

---

### Q12: How do you handle exogenous weather regressors in SARIMAX without introducing look-ahead bias (data leakage)?
**Model Answer:**  
Using contemporaneous weather observations ($W_t$) to forecast pollutant levels ($Y_t$) at horizon $h$ implies having perfect foresight of future weather. In operational forecasting, this is addressed either by:
1. Lagging meteorological regressors by at least 1 day ($W_{t-1}$), so only historical weather is required; or
2. Using actual numerical weather prediction (NWP) forecasts as the exogenous inputs.
In this project, weather regressors are lagged by 1 day, while calendar/event features (such as Diwali or weekends) are deterministic and legitimately known in advance.

---

### Q13: Why is MASE (Mean Absolute Scaled Error) superior to MAPE for environmental time series?
**Model Answer:**  
$MAPE = \frac{1}{n}\sum |\frac{y_t - \hat{y}_t}{y_t}|$ has severe flaws:
1. It divides by $y_t$. When pollutant levels or precipitation approach zero, MAPE explodes or produces division-by-zero errors.
2. It penalizes positive errors more heavily than negative errors.
$MASE$ scales the forecast MAE by the in-sample Mean Absolute Error of a seasonal naive baseline:
$$MASE = \frac{\text{MAE}}{\frac{1}{T-s}\sum_{t=s+1}^T |y_t - y_{t-s}|}$$
A $MASE < 1$ indicates that the model outperforms the naive seasonal benchmark, and it is invariant to scale and well-defined near zero.

---

### Q14: What is the purpose of the Diebold-Mariano test in model evaluation?
**Model Answer:**  
Even if Model A has a lower sample RMSE than Model B, the difference might be due to random sampling variability. The Diebold-Mariano (DM) test formally tests the null hypothesis $H_0: E[d_t] = 0$, where $d_t = L(e_{A,t}) - L(e_{B,t})$ is the loss differential. It accounts for contemporaneous correlation and $h-1$ order serial correlation in multi-step forecast errors, determining whether Model A's superiority is statistically significant.

---

### Q15: Why is Rolling-Origin Cross-Validation mandatory for time series instead of K-Fold cross-validation?
**Model Answer:**  
Standard $K$-Fold cross-validation randomly shuffles observations, which destroys temporal dependencies and uses future observations to predict the past (severe data leakage / look-ahead bias). Rolling-origin (expanding window) CV strictly preserves temporal ordering: the model is trained on data up to time $t$, forecasts are generated for $t+1, \dots, t+h$, and the origin $t$ rolls forward by a fixed step, faithfully simulating real-world operational deployment.

---

### Q16: How does the Residual Bootstrap generate prediction intervals for VAR models?
**Model Answer:**  
Analytical intervals assume that errors are Gaussian white noise. When residuals exhibit skewness, kurtosis, or non-normality, the residual bootstrap resamples the centered empirical residual vectors $\hat{u}_t$ with replacement:
1. Sample a sequence of residual vectors $u^*_{t+1}, \dots, u^*_{t+h}$.
2. Recursively generate simulated forecast trajectories using the estimated VAR coefficients.
3. Repeat $B=1000$ times to obtain an empirical predictive distribution.
4. Compute the 10th/90th and 2.5th/97.5th percentiles to obtain empirical 80% and 95% prediction intervals.

---

### Q17: What is the Moving Block Bootstrap (MBB), and why is it required for Bagged ETS?
**Model Answer:**  
Standard i.i.d. bootstrap resampling assumes observations are independent, which fails if the remainder series retains autocorrelation or heteroskedasticity. The Moving Block Bootstrap resamples continuous blocks of length $l$ (e.g., $l=14$ days) with replacement. Resampling overlapping contiguous blocks preserves local temporal dependence and autocorrelation structures within each block.

---

### Q18: What is Bagged ETS (Bergmeir et al.), and how does it improve forecast stability?
**Model Answer:**  
1. Decompose the series via STL into trend, seasonal, and remainder: $Y_t = T_t + S_t + R_t$.
2. Bootstrap the remainder $R_t$ using the Moving Block Bootstrap to produce $B$ synthetic remainder series $R^*_t$.
3. Reconstruct $B$ synthetic series $Y^*_t = T_t + S_t + R^*_t$.
4. Fit an Exponential Smoothing (ETS) model to each simulated series and forecast $h$ steps ahead.
5. Average the $B$ forecasts.
Ensemble averaging reduces model estimation variance, guards against parameter selection instability, and yields robust forecasts.

---

### Q19: Which diagnostic tests are performed on model residuals, and what are their null hypotheses?
**Model Answer:**  
1. **Ljung-Box Test**: $H_0$: Residuals exhibit no autocorrelation up to lag $m$. (Desire $p > 0.05$).
2. **Jarque-Bera Test**: $H_0$: Residuals are normally distributed with zero skewness and zero excess kurtosis. (Desire $p > 0.05$).
3. **Engle ARCH-LM Test**: $H_0$: Residuals exhibit no autoregressive conditional heteroskedasticity (constant variance). (Desire $p > 0.05$).

---

### Q20: How does LightGBM recursively generate multi-step forecasts?
**Model Answer:**  
A single LightGBM model is trained to predict $y_{t+1}$ using features constructed up to time $t$. To forecast step $t+2$, the model uses the predicted value $\hat{y}_{t+1}$ as a substituted input for the lag-1 feature, recursively propagating predictions forward up to horizon $h$. While error accumulation can occur over long horizons, recursive forecasting allows arbitrary multi-horizon generation with a single trained model.

---

### Q21: How is the official CPCB National Air Quality Index (NAQI) calculated from individual pollutant values?
**Model Answer:**  
For each monitored pollutant $p$, the sub-index $I_p$ is computed via piecewise linear interpolation across regulatory breakpoints:
$$I_p = I_{\text{lo}} + \frac{I_{\text{hi}} - I_{\text{lo}}}{C_{\text{hi}} - C_{\text{lo}}} (C_p - C_{\text{lo}})$$
The overall CPCB AQI is the **maximum of all sub-indices**:
$$\text{AQI} = \max_p \{I_p\}$$
The overall index is then categorized into one of six health classes: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401–500+).

---

### Q22: What is the meteorological reason for extreme pollution spikes in northern India during November?
**Model Answer:**  
The November spike (post-monsoon/early winter) is driven by three compounding factors:
1. **Atmospheric Inversion**: Surface cooling creates a shallow planetary boundary layer, trapping emissions near the ground.
2. **Low Wind Speed**: Calm synoptic conditions impede horizontal atmospheric ventilation and dispersion.
3. **Episodic Emissions**: Agricultural paddy-residue burning (stubble burning) in Punjab/Haryana coincides with Diwali fireworks, introducing massive particulate pulses into stagnant air.

---

### Q23: Why do we apply a log transformation ($\ln(x + 1)$) to pollutant concentrations in VAR models?
**Model Answer:**  
1. **Variance Stabilization**: Pollutant concentrations exhibit heteroskedasticity where variance scales with the mean level (higher variance during high-pollution winter months).
2. **Positivity Constraint**: Taking logs ensures that when predictions are exponentiated ($\exp(\hat{y}) - 1$), the forecast concentrations are strictly non-negative.
3. **Linear Elasticity**: In logarithmic form, linear VAR coefficients represent constant elasticities / percentage responses to shocks.

---

### Q24: What are the primary limitations of using Open-Meteo CAMS data instead of continuous ambient air quality monitoring stations (CAAQMS)?
**Model Answer:**  
CAMS (Copernicus Atmosphere Monitoring Service) provides gridded atmospheric reanalysis combining satellite observations with numerical chemical transport models at ~0.4° resolution (~40 km). Consequently:
- It smooths localized micro-environmental spikes (e.g., roadside traffic emissions or industrial point sources).
- Ground-station measurements reflect direct in-situ optical/gravimetric sampling at breathing height, whereas CAMS represents a column/volume spatial average.
Thus, models trained on CAMS reflect regional atmospheric background trends rather than localized street-level exposure.

---

### Q25: What is a structural break, and why is the year 2020 a significant modeling risk in Indian air quality datasets?
**Model Answer:**  
A structural break is an unexpected, permanent or long-lasting shift in the parameters of a time series process (e.g., shifts in mean, trend, or covariance). In 2020, nationwide COVID-19 lockdown restrictions abruptly halted vehicular transport, construction, and industrial manufacturing, causing historic drops in $NO_2$ and $PM_{2.5}$. If training data spans 2020 without intervention dummy variables, models will estimate distorted parameters that underpredict post-pandemic baseline emissions.
