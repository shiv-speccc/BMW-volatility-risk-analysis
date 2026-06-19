"""
dashboard/app.py
-----------------
Interactive Streamlit dashboard for BMW volatility & risk analysis.

Run with:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_bmw_data
from src.returns import compute_log_returns, compute_realized_volatility, train_test_split_returns
from src.garch_models import fit_garch, fit_gjr_garch, fit_egarch, forecast_volatility, compare_models
from src.risk_metrics import parametric_var, expected_shortfall

st.set_page_config(page_title="BMW Volatility & Risk Dashboard", layout="wide")

st.title("BMW Stock — Volatility & Risk Dashboard")
st.caption("GARCH-family volatility modeling and Value-at-Risk estimation on BMW daily OHLCV data (1996-2026)")

# ---- Load data ----
@st.cache_data
def get_data():
    df = load_bmw_data("data/bmw.csv")
    returns = compute_log_returns(df)
    return df, returns

df, returns = get_data()

# ---- Sidebar controls ----
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Volatility model", ["GARCH(1,1)", "GJR-GARCH(1,1,1)", "EGARCH(1,1,1)"])
confidence = st.sidebar.slider("VaR confidence level", 0.90, 0.99, 0.95, step=0.01)
horizon = st.sidebar.slider("Forecast horizon (days)", 5, 60, 30)

@st.cache_resource
def fit_selected_model(model_name: str, _returns: pd.Series):
    if model_name == "GARCH(1,1)":
        return fit_garch(_returns)
    elif model_name == "GJR-GARCH(1,1,1)":
        return fit_gjr_garch(_returns)
    else:
        return fit_egarch(_returns)

with st.spinner(f"Fitting {model_choice}..."):
    fitted = fit_selected_model(model_choice, returns)

# ---- Top metrics ----
nu = fitted.params.get("nu", None)
mu = fitted.params.get("mu", 0.0)
last_vol = fitted.conditional_volatility.iloc[-1]
var_val = parametric_var(mu, last_vol, confidence=confidence, dist="t", nu=nu)
es_val = expected_shortfall(mu, last_vol, confidence=confidence, dist="t", nu=nu)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close (EUR)", f"{df['Close'].iloc[-1]:.2f}")
col2.metric("Annualized Volatility", f"{returns.std() * (252 ** 0.5):.2f}%")
col3.metric(f"1-Day VaR ({int(confidence*100)}%)", f"{var_val:.2f}%")
col4.metric(f"1-Day Expected Shortfall", f"{es_val:.2f}%")

st.divider()

# ---- Price chart ----
st.subheader("Price History")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="#1f4e79")))
fig_price.update_layout(height=350, margin=dict(t=20, b=20))
st.plotly_chart(fig_price, use_container_width=True)

# ---- Volatility chart ----
st.subheader("Conditional Volatility (Fitted)")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=fitted.conditional_volatility.index,
    y=fitted.conditional_volatility,
    mode="lines", name="Conditional Volatility (%)", line=dict(color="#c0392b")
))
fig_vol.update_layout(height=350, margin=dict(t=20, b=20))
st.plotly_chart(fig_vol, use_container_width=True)

# ---- Forecast ----
st.subheader(f"{horizon}-Day Volatility Forecast")
vol_forecast = forecast_volatility(fitted, horizon=horizon)
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=vol_forecast["day_ahead"], y=vol_forecast["annualized_volatility_%"],
    mode="lines+markers", name="Forecasted Annualized Volatility (%)", line=dict(color="#27ae60")
))
fig_forecast.update_layout(height=350, margin=dict(t=20, b=20), xaxis_title="Days ahead", yaxis_title="Annualized Volatility (%)")
st.plotly_chart(fig_forecast, use_container_width=True)

# ---- Model comparison (cached, runs once) ----
st.subheader("Model Comparison (AIC / BIC)")
with st.spinner("Comparing models..."):
    comparison, _ = compare_models(returns)
st.dataframe(comparison.style.highlight_min(axis=0, color="#d4f7d4"), use_container_width=True)

st.caption("Lower AIC/BIC indicates better fit. Built by Shivarchan Coomaran — GARCH-family volatility & risk modeling on BMW stock.")
