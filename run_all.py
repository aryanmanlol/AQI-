"""
Master Pipeline Runner for Multivariate AQI Forecasting.

Executes the complete pipeline end-to-end or runs individual stages:
  python run_all.py --city Delhi --allow-synthetic
  python run_all.py --stage eda
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aqi_utils import compute_aqi_series
from src.baselines import run_baselines
from src.config import DATA_PROCESSED, FIGURES, TABLES, init_project
from src.data_clean import clean_data
from src.data_fetch import fetch_all_data
from src.diagnostics import aggregate_diagnostics, compute_model_diagnostics
from src.eda import run_eda
from src.evaluation import rolling_origin_cv
from src.features import prepare_features
from src.ml_baseline import run_lightgbm_pipeline
from src.prophet_model import run_prophet_pipeline
from src.regimes import run_regimes_analysis
from src.sarimax_model import run_sarimax
from src.stationarity import run_stationarity_tests
from src.uncertainty import bagged_ets_forecast, residual_bootstrap_var
from src.var_analysis import run_var_analysis
from src.var_model import run_var_pipeline

logger = logging.getLogger("run_all")


def parse_args():
    parser = argparse.ArgumentParser(description="Multivariate AQI Forecasting Pipeline")
    parser.add_argument("--city", type=str, default="Delhi", help="Target city (default: Delhi)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all", "data", "clean", "features", "eda", "stationarity",
            "baselines", "sarimax", "var", "analysis", "prophet",
            "ml", "regimes", "uncertainty", "diagnostics", "eval",
        ],
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic data fallback if Open-Meteo network request fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = init_project(city=args.city)
    city = args.city
    stage = args.stage
    allow_synth = args.allow_synthetic

    logger.info("==================================================================")
    logger.info("Starting Pipeline: City=%s | Stage=%s | AllowSynthetic=%s", city, stage, allow_synth)
    logger.info("==================================================================")

    # 1. DATA STAGE
    is_synth = False
    if stage in ["all", "data"]:
        logger.info(">>> Stage 1: Data Fetching / Acquisition <<<")
        hourly_df, daily_df, is_synth = fetch_all_data(cfg, city, allow_synthetic=allow_synth)
    else:
        # Load from disk
        daily_path = PROJECT_ROOT / "data" / "raw" / f"daily_{city}.parquet"
        if not daily_path.exists():
            daily_path = PROJECT_ROOT / "data" / "raw" / f"synthetic_{city}.parquet"
            is_synth = True
        daily_df = pd.read_parquet(daily_path)

    # 2. CLEAN STAGE
    if stage in ["all", "clean"]:
        logger.info(">>> Stage 2: Data Cleaning & Preprocessing <<<")
        cleaned_df = clean_data(daily_df, cfg, city, freq="D")
    else:
        clean_path = DATA_PROCESSED / f"processed_{city}_D.parquet"
        cleaned_df = pd.read_parquet(clean_path)

    # 3. FEATURES STAGE
    if stage in ["all", "features"]:
        logger.info(">>> Stage 3: Feature Engineering <<<")
        featured_df = prepare_features(cleaned_df, cfg)
    else:
        feat_path = DATA_PROCESSED / f"featured_{city}.parquet"
        featured_df = pd.read_parquet(feat_path)

    # 4. EDA STAGE
    if stage in ["all", "eda"]:
        logger.info(">>> Stage 4: Exploratory Data Analysis & STL <<<")
        run_eda(featured_df, cfg, is_synthetic=is_synth)

    # 5. STATIONARITY STAGE
    if stage in ["all", "stationarity"]:
        logger.info(">>> Stage 5: Stationarity & Unit Root Tests (ADF / KPSS) <<<")
        run_stationarity_tests(featured_df)

    # 6. BASELINES STAGE
    if stage in ["all", "baselines"]:
        logger.info(">>> Stage 6: Univariate Baselines (Naive, SN, HW-ETS, SARIMA) <<<")
        n_obs = len(featured_df)
        train_cutoff = featured_df.index[int(n_obs * 0.7)]
        run_baselines(featured_df, ["pm2_5", "pm10"], cfg, train_end=train_cutoff, horizons=[1, 3, 7], is_synthetic=is_synth)

    # 7. SARIMAX STAGE
    sarimax_res = None
    if stage in ["all", "sarimax"]:
        logger.info(">>> Stage 7: SARIMAX with Weather & Event Exogenous Regressors <<<")
        sarimax_res = run_sarimax(featured_df, cfg, target="pm2_5", horizon=7)

    # 8. VAR / VECM STAGE
    var_res = None
    if stage in ["all", "var"]:
        logger.info(">>> Stage 8: Multivariate VAR / VECM Modeling <<<")
        var_res = run_var_pipeline(featured_df, cfg, horizon=7)

    # 9. VAR STRUCTURAL ANALYSIS STAGE
    if stage in ["all", "analysis"]:
        logger.info(">>> Stage 9: Structural VAR Analysis (Granger, IRF, FEVD) <<<")
        if var_res is None:
            # Quick fit for analysis
            var_res = run_var_pipeline(featured_df, cfg, horizon=7)
        endo = var_res["endo_vars"]
        diff_endo = featured_df[endo].diff().dropna()
        run_var_analysis(var_res["var_model"], diff_endo, cfg, is_synthetic=is_synth)

    # 10. PROPHET STAGE
    if stage in ["all", "prophet"]:
        logger.info(">>> Stage 10: Prophet Modeling (Optional) <<<")
        run_prophet_pipeline(featured_df, cfg, target_col="pm2_5", horizon=7)

    # 11. MACHINE LEARNING BASELINE STAGE
    lgbm_res = None
    if stage in ["all", "ml"]:
        logger.info(">>> Stage 11: Machine Learning Baseline (LightGBM) <<<")
        lgbm_res = run_lightgbm_pipeline(featured_df, cfg, target_col="pm2_5", horizon=7)

    # 12. REGIMES STAGE
    if stage in ["all", "regimes"]:
        logger.info(">>> Stage 12: Atmospheric Regime Analysis <<<")
        run_regimes_analysis(featured_df, cfg, target_var="pm2_5", is_synthetic=is_synth)

    # 13. UNCERTAINTY QUANTIFICATION STAGE
    if stage in ["all", "uncertainty"]:
        logger.info(">>> Stage 13: Bootstrapped Uncertainty Quantification <<<")
        if var_res is None:
            var_res = run_var_pipeline(featured_df, cfg, horizon=7)
        residual_bootstrap_var(var_res["var_model"], steps=7, n_bootstrap=cfg.get("n_bootstrap", 100))
        bagged_ets_forecast(featured_df["pm2_5"], horizon=7, n_bootstrap=cfg.get("n_bootstrap", 100))

    # 14. RESIDUAL DIAGNOSTICS STAGE
    if stage in ["all", "diagnostics"]:
        logger.info(">>> Stage 14: Model Residual Diagnostics (Ljung-Box, JB, ARCH, QQ) <<<")
        diagnostics_records = []
        if var_res is not None:
            # Check VAR residuals for first target
            var_resid = var_res["var_model"].resid["log_pm2_5"] if "log_pm2_5" in var_res["var_model"].resid else var_res["var_model"].resid.iloc[:, 0]
            d_var = compute_model_diagnostics(var_resid, model_name="VAR", target_var="PM2.5", is_synthetic=is_synth)
            diagnostics_records.append(d_var)

        if sarimax_res is not None and "residuals" in sarimax_res:
            d_sarimax = compute_model_diagnostics(sarimax_res["residuals"], model_name="SARIMAX", target_var="PM2.5", is_synthetic=is_synth)
            diagnostics_records.append(d_sarimax)

        # Baseline residuals check
        naive_resid = featured_df["pm2_5"].diff().dropna()
        d_naive = compute_model_diagnostics(naive_resid, model_name="Naive", target_var="PM2.5", is_synthetic=is_synth)
        diagnostics_records.append(d_naive)

        if diagnostics_records:
            aggregate_diagnostics(diagnostics_records)

    # 15. EVALUATION & RESULTS GENERATION
    if stage in ["all", "eval"]:
        logger.info(">>> Stage 15: Rolling-Origin CV & Results Generation (outputs/results.md) <<<")
        rolling_origin_cv(
            featured_df,
            cfg,
            target_col="pm2_5",
            horizons=(1, 3, 7),
            step_days=cfg.get("eval_step_days", 7),
            initial_train_frac=cfg.get("eval_initial_train_frac", 0.70),
        )

    logger.info("==================================================================")
    logger.info("Pipeline Execution Completed Successfully.")
    logger.info("Artifacts saved to: outputs/figures/, outputs/tables/, outputs/models/, outputs/results.md")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
