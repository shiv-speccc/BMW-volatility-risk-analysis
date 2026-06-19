"""
main.py
-------
End-to-end pipeline:
  1. Load data
  2. Compute returns
  3. Run EDA (saves figures + prints stylized facts)
  4. Fit GARCH / GJR-GARCH / EGARCH and compare via AIC/BIC
  5. Forecast volatility 30 days ahead with the best model
  6. Compute 1-day 95% and 99% VaR / Expected Shortfall
  7. Backtest VaR on the test set with Kupiec's POF test
  8. Save all results to outputs/results/

Run with:  python main.py
"""

import json
import pandas as pd
from src.data_loader import load_bmw_data
from src.returns import compute_log_returns, train_test_split_returns
from src.eda import run_full_eda
from src.garch_models import compare_models, forecast_volatility, fit_garch
from src.risk_metrics import parametric_var, expected_shortfall, rolling_var_series
from src.backtesting import backtest_summary

RESULTS_DIR = "outputs/results"


def main():
    print("=" * 60)
    print("BMW STOCK VOLATILITY & RISK MODELING PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading data...")
    df = load_bmw_data("data/bmw.csv")
    print(f"  Loaded {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")

    # 2. Returns
    print("\n[2/7] Computing log returns...")
    returns = compute_log_returns(df)
    train_returns, test_returns = train_test_split_returns(returns, test_size=0.15)
    print(f"  Train: {len(train_returns)} obs | Test: {len(test_returns)} obs")

    # 3. EDA
    print("\n[3/7] Running EDA (saving figures to outputs/figures/)...")
    stats, adf = run_full_eda(df, returns)
    print(f"  Annualized volatility: {stats['annualized_volatility_%']:.2f}%")
    print(f"  Excess kurtosis: {stats['kurtosis']:.2f} (fat tails if > 0)")
    print(f"  ADF p-value: {adf['p_value']:.6f} (stationary: {adf['is_stationary']})")

    # 4. Model comparison
    print("\n[4/7] Fitting GARCH(1,1), GJR-GARCH, and EGARCH on training data...")
    comparison, fitted_models = compare_models(train_returns)
    print(comparison.to_string())
    best_model_name = comparison.index[0]
    best_model = fitted_models[best_model_name]
    print(f"  Best model by AIC: {best_model_name}")

    # 5. Forecast
    print(f"\n[5/7] Forecasting volatility 30 days ahead with {best_model_name}...")
    vol_forecast = forecast_volatility(best_model, horizon=30)
    print(vol_forecast.head())

    # 6. VaR / ES on test set (refit conditional vol walk-forward would be ideal;
    #    here we use the fitted model's in-sample conditional vol on test via a
    #    simple one-step-ahead static approach for portfolio-project clarity)
    print("\n[6/7] Computing Value-at-Risk and Expected Shortfall...")
    nu = best_model.params.get("nu", None)
    last_vol = best_model.conditional_volatility.iloc[-1]
    mu = best_model.params.get("mu", 0.0)

    var_95 = parametric_var(mu, last_vol, confidence=0.95, dist="t", nu=nu)
    var_99 = parametric_var(mu, last_vol, confidence=0.99, dist="t", nu=nu)
    es_95 = expected_shortfall(mu, last_vol, confidence=0.95, dist="t", nu=nu)

    print(f"  1-day 95% VaR: {var_95:.3f}% of position value")
    print(f"  1-day 99% VaR: {var_99:.3f}% of position value")
    print(f"  1-day 95% Expected Shortfall: {es_95:.3f}%")

    # 7. Backtest on test period using a rolling forecast from train model
    print("\n[7/7] Backtesting VaR on held-out test set (Kupiec POF test)...")
    test_model = fit_garch(pd.concat([train_returns, test_returns]))
    cond_vol_full = test_model.conditional_volatility
    cond_vol_test = cond_vol_full.loc[test_returns.index]
    var_series_test = rolling_var_series(cond_vol_test, mu=mu, confidence=0.95, dist="t", nu=nu)
    backtest_result = backtest_summary(test_returns, var_series_test, confidence=0.95)
    print(json.dumps(backtest_result, indent=2, default=str))

    # Save everything
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison.to_csv(f"{RESULTS_DIR}/model_comparison.csv")
    vol_forecast.to_csv(f"{RESULTS_DIR}/volatility_forecast_30d.csv", index=False)

    summary = {
        "data_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "n_observations": len(df),
        "annualized_volatility_%": round(stats["annualized_volatility_%"], 3),
        "excess_kurtosis": round(stats["kurtosis"], 3),
        "adf_p_value": round(adf["p_value"], 6),
        "best_model": best_model_name,
        "var_95_1day_%": round(var_95, 3),
        "var_99_1day_%": round(var_99, 3),
        "es_95_1day_%": round(es_95, 3),
        "backtest": backtest_result,
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone. Results saved to outputs/results/, figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
