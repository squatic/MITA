"""
Sugar Price Monte Carlo Risk Model — with integrated Parameter Estimator + SugarBot
Run with: streamlit run sugar_app.py
Requires: pip install streamlit plotly numpy scipy matplotlib pandas anthropic
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import stats
import anthropic

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sugar Price Risk Model",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1923;
    color: #e8dcc8;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] p:not(button p) {
    color: #c9bfac !important;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
  }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: #d4a843; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #e8dcc8;
  }
  div[data-testid="metric-container"] label { color: #8a9ab0 !important; font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e8dcc8; font-size: 1.6rem; font-weight: 600; }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.8rem; }

  /* Main area */
  .main .block-container { padding-top: 1.5rem; }
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #4a7fa5;
    border-bottom: 1px solid #1e2d3d;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
  }
  .est-section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #2e3347;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }
  .info-pill {
    display: inline-block;
    background: #1e2d3d;
    color: #a8c4d8;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px 3px;
  }
  .applied-pill {
    display: inline-block;
    background: #122d1e;
    color: #52c87a;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px 3px;
  }
  .alert-danger { background: #2d1a1a; border-left: 3px solid #e05252; padding: 0.6rem 1rem; border-radius: 6px; font-size: 0.85rem; color: #f5a0a0; }
  .alert-safe   { background: #122d1e; border-left: 3px solid #52c87a; padding: 0.6rem 1rem; border-radius: 6px; font-size: 0.85rem; color: #8de8a8; }
  .info-box {
    background: #0f1923;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 16px;
  }
  .warn-box {
    background: #0f1923;
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 16px;
  }
  .stButton > button {
    background: #FFF600;
    color: #0f1923 !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    padding: 0.55rem 1.8rem;
    letter-spacing: 0.04em;
    width: 100%;
  }
  .stButton > button:hover { background: #F7DC6F; color: #0f1923 !important; }
  section[data-testid="stSidebar"] .stButton > button { color: #4A4747 !important; }
  section[data-testid="stSidebar"] .stButton > button p { color: #4A4747 !important; }
  section[data-testid="stSidebar"] .stButton > button:hover { color: #4A4747 !important; }
  .apply-btn > button {
    background: #4A4747 !important;
    color: #52c87a !important;
    border: 1px solid #52c87a !important;
  }
  .apply-btn > button:hover { background: #255c37 !important; }

  /* ── SugarBot Chat Styles ── */
  .sugarbot-container {
    background: #0a1520;
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
  }
  .sugarbot-header {
    background: linear-gradient(135deg, #0f1923 0%, #162030 100%);
    border-bottom: 1px solid #1e2d3d;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .chat-bubble-user {
    background: #1e3a2f;
    border: 1px solid #2a5040;
    color: #b8e8c8;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px;
    margin: 4px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.88rem;
    line-height: 1.5;
  }
  .chat-bubble-bot {
    background: #0f1f2e;
    border: 1px solid #1e2d3d;
    color: #c9bfac;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 16px;
    margin: 4px 0;
    max-width: 85%;
    font-size: 0.88rem;
    line-height: 1.5;
  }
  .chat-message-row {
    padding: 6px 0;
  }
  .stChatMessage {
    background: transparent !important;
  }
  /* Style the chat input */
  .stChatInputContainer {
    background: #0f1923 !important;
    border-top: 1px solid #1e2d3d !important;
  }
  .stChatInputContainer textarea {
    background: #0a1520 !important;
    color: #c9bfac !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  /* Suggestion chips */
  .suggestion-chip {
    display: inline-block;
    background: #0f1923;
    border: 1px solid #2a3d50;
    color: #7ab8d4;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    margin: 3px 3px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
  }
  .suggestion-chip:hover {
    background: #1e2d3d;
    border-color: #4a9fb5;
    color: #a8d8ea;
  }
</style>
""", unsafe_allow_html=True)


# ── Parameter Estimation Helpers ───────────────────────────────────────────────

def annualization_factor(freq: str) -> int:
    return {"Daily": 252, "Weekly": 52, "Monthly": 12, "Yearly": 1}[freq]

def dt_value(freq: str) -> float:
    return {"Daily": 1/252, "Weekly": 1/52, "Monthly": 1/12, "Yearly": 1.0}[freq]


def compute_gbm_params(prices: np.ndarray, freq: str) -> dict:
    N            = annualization_factor(freq)
    log_returns  = np.diff(np.log(prices))
    mu_period    = np.mean(log_returns)
    sigma_period = np.std(log_returns, ddof=1)
    mu_annual    = mu_period * N
    sigma_annual = sigma_period * np.sqrt(N)
    mu_ito       = mu_annual - 0.5 * sigma_annual**2
    return {
        "log_returns":   log_returns,
        "mu_period":     mu_period,
        "sigma_period":  sigma_period,
        "mu_annual":     mu_annual,
        "sigma_annual":  sigma_annual,
        "mu_ito":        mu_ito,
        "n_obs":         len(log_returns),
    }


def compute_ou_params(prices: np.ndarray, freq: str) -> dict:
    dt         = dt_value(freq)
    N          = annualization_factor(freq)
    log_prices = np.log(prices)
    d_logP     = np.diff(log_prices)
    logP_lag   = log_prices[:-1]
    slope, intercept, r_value, p_value, se = stats.linregress(logP_lag, d_logP)
    beta      = slope
    alpha     = intercept
    k         = -beta / dt
    theta     = np.exp(-alpha / beta) if beta != 0 else np.nan
    residuals = d_logP - (alpha + beta * logP_lag)
    sigma_ou  = np.std(residuals, ddof=2) / np.sqrt(dt)
    half_life_years   = np.log(2) / k if k > 0 else np.nan
    half_life_periods = half_life_years * N if k > 0 else np.nan
    return {
        "k":                 k,
        "theta":             theta,
        "sigma_ou":          sigma_ou,
        "half_life_years":   half_life_years,
        "half_life_periods": half_life_periods,
        "r_squared":         r_value**2,
        "p_value":           p_value,
        "residuals":         residuals,
        "beta":              beta,
        "alpha":             alpha,
        "dP":                d_logP,
        "P_lag":             logP_lag,
    }


# ── Simulation Engine ──────────────────────────────────────────────────────────

def run_gbm_terminal(S0, mu, sigma, T, N, seed):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(N)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def run_mean_revert_terminal(S0, kappa, theta, sigma, T, N, steps_per_year, seed):
    rng   = np.random.default_rng(seed)
    steps = max(1, int(T * steps_per_year))
    dt    = T / steps
    ln_S  = np.full(N, np.log(S0))
    ln_theta = np.log(theta) - sigma**2 / (2 * kappa)
    sqrt_dt  = np.sqrt(dt)
    for _ in range(steps):
        Z    = rng.standard_normal(N)
        ln_S += kappa * (ln_theta - ln_S) * dt + sigma * sqrt_dt * Z
    return np.exp(ln_S)


def run_gbm_paths(S0, mu, sigma, T, steps_per_year, K, seed):
    rng   = np.random.default_rng(seed)
    steps = max(1, int(T * steps_per_year))
    dt    = T / steps
    paths = np.zeros((steps + 1, K))
    paths[0] = S0
    for t in range(1, steps + 1):
        Z        = rng.standard_normal(K)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return np.linspace(0, T, steps + 1), paths


def run_mean_revert_paths(S0, kappa, theta, sigma, T, steps_per_year, K, seed):
    rng      = np.random.default_rng(seed)
    steps    = max(1, int(T * steps_per_year))
    dt       = T / steps
    ln_theta = np.log(theta) - sigma**2 / (2 * kappa)
    sqrt_dt  = np.sqrt(dt)
    ln_paths = np.zeros((steps + 1, K))
    ln_paths[0] = np.log(S0)
    for t in range(1, steps + 1):
        Z          = rng.standard_normal(K)
        ln_paths[t] = ln_paths[t - 1] + kappa * (ln_theta - ln_paths[t - 1]) * dt + sigma * sqrt_dt * Z
    return np.linspace(0, T, steps + 1), np.exp(ln_paths)


# ── Weekly Prediction Engine ───────────────────────────────────────────────────

def run_weekly_gbm(S0, mu, sigma, n_weeks, N_sim, seed):
    rng = np.random.default_rng(seed)
    dt  = 1 / 52
    prices = np.full(N_sim, S0, dtype=float)
    weekly_stats = []
    for w in range(1, n_weeks + 1):
        Z      = rng.standard_normal(N_sim)
        prices = prices * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        weekly_stats.append({
            "week":   w,
            "mean":   float(np.mean(prices)),
            "median": float(np.median(prices)),
            "p05":    float(np.percentile(prices, 5)),
            "p25":    float(np.percentile(prices, 25)),
            "p75":    float(np.percentile(prices, 75)),
            "p95":    float(np.percentile(prices, 95)),
        })
    return pd.DataFrame(weekly_stats)


def run_weekly_ou(S0, kappa, theta, sigma, n_weeks, N_sim, seed):
    rng      = np.random.default_rng(seed)
    dt       = 1 / 52
    ln_theta = np.log(theta) - sigma**2 / (2 * kappa)
    sqrt_dt  = np.sqrt(dt)
    ln_prices = np.full(N_sim, np.log(S0), dtype=float)
    weekly_stats = []
    for w in range(1, n_weeks + 1):
        Z         = rng.standard_normal(N_sim)
        ln_prices = ln_prices + kappa * (ln_theta - ln_prices) * dt + sigma * sqrt_dt * Z
        prices    = np.exp(ln_prices)
        weekly_stats.append({
            "week":   w,
            "mean":   float(np.mean(prices)),
            "median": float(np.median(prices)),
            "p05":    float(np.percentile(prices, 5)),
            "p25":    float(np.percentile(prices, 25)),
            "p75":    float(np.percentile(prices, 75)),
            "p95":    float(np.percentile(prices, 95)),
        })
    return pd.DataFrame(weekly_stats)


# ── Session State Defaults ─────────────────────────────────────────────────────

_defaults = {
    "param_mu":    0.03,
    "param_sigma": 0.18,
    "param_kappa": 0.60,
    "param_theta": 2400.0,
    "params_applied": False,
    "applied_from": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# SugarBot chat history
if "sugarbot_messages" not in st.session_state:
    st.session_state["sugarbot_messages"] = []


# ── SugarBot System Prompt ─────────────────────────────────────────────────────

SUGARBOT_SYSTEM = """You are SugarBot 🍬, a friendly and knowledgeable assistant embedded inside the Sugar Price Monte Carlo Risk Model — a Streamlit app for Philippine mill-gate raw sugar price analysis and risk forecasting.

Your job is to help users understand and use this application. You speak in a warm, clear, professional tone. Keep responses concise but thorough. Use bullet points or short paragraphs. Avoid jargon unless you explain it.

## About the App

The app has 3 main tabs plus a chatbot (you):

### 📊 Tab 1 — Parameter Estimator
- Users upload a historical CSV of sugar prices
- The app computes GBM and OU parameters automatically
- Outputs: annual drift μ, volatility σ, mean reversion speed κ, long-run mean θ, half-life
- Users can click "Apply Parameters" to send these directly to the simulation

### 🎲 Tab 2 — Monte Carlo Simulation
- Users click "▶ Run Simulation" in the sidebar to generate results
- Shows terminal price distribution at the forecast horizon
- Key outputs: Mean, Median, VaR 95%, P(price ≤ break-even), Expected Shortfall
- Sub-tabs: Distribution histogram, Price Paths chart, Percentile Table

### 📅 Tab 3 — Weekly Price Prediction
- Shows week-by-week forecasted prices for up to 104 weeks
- Bar chart coloured by risk zone: 🔴 at/below break-even, 🟡 within 5%, 🟢 safe
- Users can choose Median or Mean as the central estimate, and P05–P95 or P25–P75 confidence bands

## Sidebar Controls

**Model Setup:**
- Price model: GBM (Lognormal) — good for trending markets; Mean-Reverting (OU) — good for commodities that snap back to a long-run price
- Current spot price in ₱/Lkg
- Forecast horizon: in Weeks, Months, or Years

**Historical Data (optional):**
- Upload a CSV with a price column
- Select data frequency (Daily/Weekly/Monthly/Yearly)
- Estimated parameters appear and can be applied with one click

**Model Parameters (manual entry):**
- GBM: Annual drift μ (use Itô-corrected value), Annual volatility σ
- Mean-Reverting: κ (reversion speed), θ (long-run mean in ₱/Lkg), σ (volatility)

**Risk & Volume:**
- Break-even price: triggers risk alerts when price may fall below this
- Annual volume: multiplied by VaR to compute Revenue at Risk

**Simulation Settings:**
- N: number of terminal simulations (5,000 default)
- K: number of sample paths to draw on chart (30 default)
- Random seed: for reproducibility

**Weekly Prediction Settings:**
- Weeks to forecast (4–104)
- Bar shows: Median or Mean
- Confidence interval: P05–P95 (90%) or P25–P75 (50%)

## Key Concepts

**GBM (Geometric Brownian Motion):** Assumes prices grow/fall with a trend plus random shocks. Good when you expect a directional trend.

**OU (Ornstein-Uhlenbeck / Mean-Reverting):** Assumes prices drift back toward a long-run average (θ). Common in commodity markets. The speed κ controls how fast — higher κ means faster snap-back. Half-life = ln(2)/κ.

**VaR 95% (Value at Risk):** The worst expected loss in the bottom 5% of outcomes. Calculated as: Spot Price − P05.

**Expected Shortfall (ES):** The average price across all scenarios at or below P05. A more conservative risk measure than VaR.

**Itô-Corrected Drift:** In GBM, the drift input should be μ − σ²/2 (Itô correction) to match the expected log-price growth. The estimator does this automatically.

**P05, P25, P50, P75, P95:** Percentiles of the simulated price distribution at the forecast horizon.

## CSV Format
The uploaded CSV should have:
- One column with historical prices (numeric, in ₱/Lkg)
- Optionally a date column
- At least 10 rows; 36+ recommended for monthly data

## Common Questions

**"How do I get started?"**
→ Enter your spot price and horizon in the sidebar, choose a model, set parameters (or upload CSV to auto-estimate), then click ▶ Run Simulation.

**"Which model should I use?"**
→ Use GBM if prices have a clear upward/downward trend. Use Mean-Reverting if sugar prices tend to stabilize around a known level (typical for regulated/mill-gate prices).

**"What does κ mean?"**
→ κ is mean-reversion speed. κ = 0.6 means prices revert to θ with a half-life of about 1.15 years. Higher κ = faster snap-back.

**"My κ is negative — what does that mean?"**
→ Negative κ means the data shows no mean reversion — prices are actually diverging. Switch to the GBM model instead.

**"What CSV should I upload?"**
→ A simple spreadsheet with a price column (e.g., monthly mill-gate prices in ₱/Lkg). Just export from Excel as CSV.

Always be helpful, encouraging, and specific. If you don't know something, say so honestly."""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  🍬 Sugar Price\nMonte Carlo Risk Model")
    st.markdown("---")

    st.markdown("### Model Setup")
    model = st.selectbox(
        "Price model",
        ["GBM (Lognormal)", "Mean-Reverting"],
        help="GBM: prices drift with random shocks. Mean-Reverting: prices gravitate to a long-run average."
    )
    S0 = st.number_input("Current spot price (₱/Lkg)", min_value=100.0, max_value=20000.0, value=2400.0, step=50.0)

    horizon_unit = st.selectbox("Horizon unit", ["Weeks", "Months", "Years"])
    if horizon_unit == "Weeks":
        horizon_val    = st.number_input("Forecast horizon (weeks)", min_value=1, value=52, step=1)
        T              = horizon_val / 52.0
        steps_per_year = 52
        horizon_label  = f"{horizon_val} week{'s' if horizon_val > 1 else ''}"
    elif horizon_unit == "Months":
        horizon_val    = st.number_input("Forecast horizon (months)", min_value=1, value=12, step=1)
        T              = horizon_val / 12.0
        steps_per_year = 12
        horizon_label  = f"{horizon_val} month{'s' if horizon_val > 1 else ''}"
    else:
        horizon_val    = st.number_input("Forecast horizon (years)", min_value=1, value=1, step=1)
        T              = float(horizon_val)
        steps_per_year = 52
        horizon_label  = f"{horizon_val} year{'s' if horizon_val > 1 else ''}"

    st.markdown("---")

    st.markdown("### 📁 Historical Data (optional)")
    st.markdown(
        '<div style="font-size:11px;color:#6b7280;font-family:\'IBM Plex Mono\',monospace;margin-bottom:8px">'
        'Upload a price CSV to auto-estimate model parameters.</div>',
        unsafe_allow_html=True
    )

    est_freq = st.selectbox(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly", "Yearly"],
        index=2,
        help="Match to how often your price data is recorded."
    )

    uploaded_file = st.file_uploader("CSV with price column", type=["csv"])

    price_col = None
    date_col  = None
    df_raw    = None
    gbm_est   = None
    ou_est    = None

    if uploaded_file:
        df_raw       = pd.read_csv(uploaded_file)
        numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        all_cols     = df_raw.columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns found in the CSV.")
        else:
            price_col = st.selectbox("Price Column", numeric_cols)
            date_col  = st.selectbox("Date Column (optional)", ["None"] + all_cols)

            _prices = df_raw[price_col].dropna().values
            if len(_prices) >= 10:
                gbm_est = compute_gbm_params(_prices, est_freq)
                ou_est  = compute_ou_params(_prices, est_freq)

                st.markdown(
                    '<div style="font-size:11px;color:#000000;font-family:\'IBM Plex Mono\',monospace;margin:8px 0 4px 0">'
                    'ESTIMATED PARAMS</div>', unsafe_allow_html=True
                )
                if "GBM" in model:
                    st.markdown(
                        f'<div style="font-size:12px;color:#c9bfac;font-family:\'IBM Plex Mono\',monospace;line-height:1.8">'
                        f'μ (Itô) = <b style="color:#d4a843">{gbm_est["mu_ito"]*100:.2f}%</b><br>'
                        f'σ annual = <b style="color:#4a9fb5">{gbm_est["sigma_annual"]*100:.2f}%</b>'
                        f'</div>', unsafe_allow_html=True
                    )
                    apply_label = "✅ Apply GBM Parameters"
                else:
                    k_val  = ou_est["k"]
                    th_val = ou_est["theta"]
                    sg_val = ou_est["sigma_ou"]
                    hl_val = ou_est["half_life_years"]
                    hl_str = f"{hl_val:.2f} yr" if not np.isnan(hl_val) else "N/A"
                    k_ok   = k_val > 0
                    st.markdown(
                        f'<div style="font-size:12px;color:#c9bfac;font-family:\'IBM Plex Mono\',monospace;line-height:1.8">'
                        f'κ = <b style="color:{"#d4a843" if k_ok else "#e05252"}">{k_val:.4f}</b><br>'
                        f'θ = <b style="color:#d4a843">₱{th_val:,.0f}</b><br>'
                        f'σ_ou = <b style="color:#4a9fb5">{sg_val*100:.2f}%</b><br>'
                        f'Half-life = <b style="color:#9ca3af">{hl_str}</b>'
                        f'</div>', unsafe_allow_html=True
                    )
                    if not k_ok:
                        st.markdown(
                            '<div class="warn-box" style="margin-top:6px">⚠️ κ ≤ 0 — no mean reversion detected. GBM may suit this data better.</div>',
                            unsafe_allow_html=True
                        )
                    apply_label = "✅ Apply OU Parameters"

                st.markdown("")
                if st.button(apply_label, use_container_width=True):
                    if "GBM" in model:
                        st.session_state["param_mu"]    = float(round(gbm_est["mu_ito"], 4))
                        st.session_state["param_sigma"] = float(round(gbm_est["sigma_annual"], 4))
                        st.session_state["applied_from"] = "GBM"
                    else:
                        st.session_state["param_kappa"] = float(round(max(ou_est["k"], 0.001), 4))
                        st.session_state["param_theta"] = float(round(ou_est["theta"], 2))
                        st.session_state["param_sigma"] = float(round(ou_est["sigma_ou"], 4))
                        st.session_state["applied_from"] = "OU"
                    st.session_state["params_applied"] = True
                    st.rerun()

            else:
                st.warning("Need at least 10 data points to estimate parameters.")

    st.markdown("---")

    st.markdown("### Model Parameters")
    if st.session_state["params_applied"]:
        st.markdown(
            f'<span class="applied-pill">✔ Parameters loaded from {st.session_state["applied_from"]} estimation</span>',
            unsafe_allow_html=True
        )
        if st.button("↩ Reset to defaults", use_container_width=False):
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v
            st.rerun()

    if "GBM" in model:
        mu    = st.number_input(
            "Annual drift μ", value=float(st.session_state["param_mu"]),
            step=0.001, format="%.4f",
            help="Expected annual price growth (Itô-corrected). Use the value from the estimator."
        )
        sigma = st.number_input(
            "Annual volatility σ", min_value=0.001, value=float(st.session_state["param_sigma"]),
            step=0.001, format="%.4f",
            help="Annualised standard deviation of log returns."
        )
        st.session_state["param_mu"]    = mu
        st.session_state["param_sigma"] = sigma
    else:
        kappa = st.number_input(
            "Mean-reversion speed κ", min_value=0.001, value=float(st.session_state["param_kappa"]),
            step=0.01, format="%.4f",
            help="How fast prices return to the long-run mean. Higher = faster snap-back."
        )
        theta = st.number_input(
            "Long-run mean θ (₱/Lkg)", min_value=0.01, value=float(st.session_state["param_theta"]),
            step=50.0,
            help="The price level the model gravitates toward."
        )
        sigma = st.number_input(
            "Annual volatility σ", min_value=0.001, value=float(st.session_state["param_sigma"]),
            step=0.001, format="%.4f",
        )
        st.session_state["param_kappa"] = kappa
        st.session_state["param_theta"] = theta
        st.session_state["param_sigma"] = sigma

    st.markdown("---")

    st.markdown("### Risk & Volume")
    breakeven = st.number_input("Break-even / alert price (₱/Lkg)", min_value=0.0, value=2000.0, step=50.0)
    volume    = st.number_input("Annual volume (Lkg, 0 = ignore)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    st.markdown("### Simulation Settings")
    N_sim = st.number_input("Terminal simulations (N)", min_value=100, value=5000, step=1000)
    K     = st.number_input("Sample paths to display", min_value=1, value=30, step=5)
    seed  = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.markdown("---")
    st.markdown("### 📅 Weekly Prediction Settings")
    weekly_n_weeks = st.number_input(
        "Weeks to forecast", min_value=4, max_value=104, value=26, step=4,
        help="Number of future weeks shown in the Weekly Prediction bar chart."
    )
    weekly_display  = st.selectbox(
        "Bar shows",
        ["Median (P50)", "Mean"],
        help="Central estimate displayed as bar height."
    )
    weekly_interval = st.selectbox(
        "Confidence interval",
        ["P05–P95 (90%)", "P25–P75 (50%)"],
        help="Error bars around the weekly bar."
    )

    run = st.button("▶  Run Simulation", use_container_width=True)


# ── Title ──────────────────────────────────────────────────────────────────────
st.title("Sugar Price Monte Carlo Risk Model")
col_model, col_spot, col_horizon = st.columns(3)
with col_model:
    st.markdown(f'<span class="info-pill">Model: {model}</span>', unsafe_allow_html=True)
with col_spot:
    st.markdown(f'<span class="info-pill">Spot: ₱{S0:,.0f}/Lkg</span>', unsafe_allow_html=True)
with col_horizon:
    st.markdown(f'<span class="info-pill">Horizon: {horizon_label}</span>', unsafe_allow_html=True)


# ── Main Tabs ──────────────────────────────────────────────────────────────────
DARK_BG  = "#0f1923"
GRID_CLR = "#1e2d3d"
TEXT_CLR = "#c9bfac"
GOLD     = "#d4a843"
TEAL     = "#4a9fb5"
RED_CLR  = "#e05252"
GREEN_OK = "#52c87a"
AMBER    = "#f59e0b"

tab_est, tab_sim, tab_weekly, tab_bot = st.tabs([
    "📊 Parameter Estimator",
    "🎲 Monte Carlo Simulation",
    "📅 Weekly Price Prediction",
    "🍬 SugarBot",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Parameter Estimator
# ════════════════════════════════════════════════════════════════════════════════
with tab_est:
    if df_raw is None or price_col is None:
        st.markdown("### Upload a CSV in the sidebar to estimate model parameters.")
        sample = pd.DataFrame({
            "date":  ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
            "price": [2450, 2510, 2480, 2530, 2495]
        })
        st.markdown('<div class="info-box">Upload a CSV of historical mill-gate prices in the sidebar. '
                    'The estimator will compute <b>GBM drift (μ)</b>, <b>Volatility (σ)</b>, and '
                    '<b>OU mean reversion parameters (κ, θ)</b> — then apply them directly to the simulation.</div>',
                    unsafe_allow_html=True)
        st.markdown("#### Expected CSV format")
        st.dataframe(sample, use_container_width=True)
    else:
        prices_est = df_raw[price_col].dropna().values

        if len(prices_est) < 10:
            st.error("Need at least 10 data points.")
            st.stop()

        min_recommended = {"Daily": 500, "Weekly": 104, "Monthly": 36, "Yearly": 5}
        min_rec = min_recommended[est_freq]
        if len(prices_est) < min_rec:
            st.warning(
                f"⚠️ Only {len(prices_est)} observations detected. "
                f"For {est_freq.lower()} data, at least {min_rec} rows are recommended for reliable estimates."
            )

        if date_col and date_col != "None":
            try:
                dates_est = pd.to_datetime(df_raw[date_col].dropna().values[:len(prices_est)])
            except Exception:
                dates_est = np.arange(len(prices_est))
        else:
            dates_est = np.arange(len(prices_est))

        gbm = compute_gbm_params(prices_est, est_freq)
        ou  = compute_ou_params(prices_est, est_freq)
        N_ann = annualization_factor(est_freq)
        dt    = dt_value(est_freq)

        st.markdown('<div class="est-section-header">📊 Price History</div>', unsafe_allow_html=True)
        col_prev, col_chart = st.columns([1, 2])

        with col_prev:
            st.markdown(f"**{len(prices_est)} observations** · {est_freq} · `{price_col}`")
            st.dataframe(df_raw[[price_col]].head(10), use_container_width=True)

        with col_chart:
            fig_px, ax_px = plt.subplots(figsize=(7, 3))
            fig_px.patch.set_facecolor(DARK_BG)
            ax_px.set_facecolor(DARK_BG)
            ax_px.plot(dates_est, prices_est, color=TEAL, linewidth=1.5)
            ax_px.axhline(ou["theta"], color=GOLD, linewidth=1, linestyle="--",
                          label=f"Long-run mean θ: {ou['theta']:.1f}")
            ax_px.set_ylabel(price_col, color="#9ca3af", fontsize=9)
            ax_px.tick_params(colors="#6b7280", labelsize=8)
            for sp in ax_px.spines.values(): sp.set_edgecolor(GRID_CLR)
            ax_px.legend(fontsize=8, facecolor="#1a1d27", labelcolor="#9ca3af", edgecolor=GRID_CLR)
            plt.tight_layout()
            st.pyplot(fig_px)
            plt.close()

        st.markdown('<div class="est-section-header">📈 GBM Parameters</div>', unsafe_allow_html=True)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Annual Drift μ",        f"{gbm['mu_annual']*100:.2f}%",   help="Average yearly price trend.")
        g2.metric("Itô-Corrected Drift",   f"{gbm['mu_ito']*100:.2f}%",      help="Use this as your GBM drift input.")
        g3.metric("Annual Volatility σ",   f"{gbm['sigma_annual']*100:.2f}%", help="Annualised std dev of log returns.")
        g4.metric(f"{est_freq} Volatility σ", f"{gbm['sigma_period']*100:.2f}%")

        with st.expander("Show GBM calculation detail"):
            st.markdown(f"""
| Step | Value |
|---|---|
| Observations (n) | {gbm['n_obs']} log returns |
| Mean log return per {est_freq.lower()} | `{gbm['mu_period']:.6f}` |
| Std dev log return per {est_freq.lower()} | `{gbm['sigma_period']:.6f}` |
| Annualization factor (N) | `{N_ann}` |
| **Annual Drift** = mean × N | `{gbm['mu_annual']:.6f}` → **{gbm['mu_annual']*100:.2f}%** |
| **Annual Volatility** = std × √N | `{gbm['sigma_annual']:.6f}` → **{gbm['sigma_annual']*100:.2f}%** |
| **Itô-Corrected Drift** = μ − σ²/2 | `{gbm['mu_ito']:.6f}` → **{gbm['mu_ito']*100:.2f}%** |
""")

        fig_lr, ax_lr = plt.subplots(figsize=(8, 2.8))
        fig_lr.patch.set_facecolor(DARK_BG)
        ax_lr.set_facecolor(DARK_BG)
        ax_lr.hist(gbm['log_returns'], bins=25, color="#3b82f6", alpha=0.7, edgecolor="#1e293b")
        ax_lr.axvline(gbm['mu_period'], color=GREEN_OK, linewidth=1.5, linestyle="--",
                      label=f"Mean: {gbm['mu_period']:.4f}")
        ax_lr.set_title("Distribution of Log Returns", color="#9ca3af", fontsize=10)
        ax_lr.tick_params(colors="#6b7280", labelsize=8)
        for sp in ax_lr.spines.values(): sp.set_edgecolor(GRID_CLR)
        ax_lr.legend(fontsize=8, facecolor="#1a1d27", labelcolor="#9ca3af", edgecolor=GRID_CLR)
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close()

        st.markdown('<div class="est-section-header">🔄 Ornstein-Uhlenbeck Parameters</div>', unsafe_allow_html=True)

        if ou["k"] <= 0:
            st.markdown(
                '<div class="warn-box">⚠️ k ≤ 0 — prices are NOT mean-reverting in this dataset. GBM may be the more appropriate model.</div>',
                unsafe_allow_html=True
            )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Mean Reversion Speed κ", f"{ou['k']:.4f}",
                  help="Higher = faster price snap-back to long-run mean.")
        o2.metric("Long-Run Mean θ",        f"{ou['theta']:,.2f}",
                  help=f"Price level the model gravitates toward ({price_col}).")
        o3.metric("OU Volatility σ",        f"{ou['sigma_ou']*100:.2f}%",
                  help="Annualised OU volatility.")
        hl_label = f"{ou['half_life_years']:.2f} yrs" if not np.isnan(ou['half_life_years']) else "N/A"
        hl_delta = f"≈ {ou['half_life_periods']:.1f} {est_freq.lower()} periods" if not np.isnan(ou.get('half_life_periods', np.nan)) else None
        o4.metric("Half-Life", hl_label, delta=hl_delta, delta_color="off")

        with st.expander("Show OU calculation detail"):
            st.markdown(f"""
**Method:** OLS regression on Δln(P) = α + β·ln(P_{{t-1}}) + ε

| Parameter | Raw | Annualised |
|---|---|---|
| OLS slope β | `{ou['beta']:.6f}` | — |
| OLS intercept α | `{ou['alpha']:.6f}` | — |
| **Reversion Speed κ** = −β / dt | — | **{ou['k']:.4f}** |
| **Long-run Mean θ** = exp(−α / β) | **{ou['theta']:,.2f}** | — |
| **OU Volatility σ** = std(residuals) / √dt | — | **{ou['sigma_ou']*100:.2f}%** |
| **Half-life** = ln(2) / κ | — | **{ou['half_life_years']:.4f} years** |
| R² of regression | `{ou['r_squared']:.4f}` | — |
| p-value of slope | `{ou['p_value']:.4f}` | — |
| dt per {est_freq.lower()} period | `{dt:.6f}` | — |
""")
            if ou['p_value'] > 0.05:
                st.markdown('<div class="warn-box">⚠️ p-value > 0.05 — mean reversion is not statistically significant at the 95% level.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">✅ p-value < 0.05 — mean reversion is statistically significant.</div>', unsafe_allow_html=True)

        fig_reg, ax_reg = plt.subplots(figsize=(8, 2.8))
        fig_reg.patch.set_facecolor(DARK_BG)
        ax_reg.set_facecolor(DARK_BG)
        ax_reg.scatter(ou["P_lag"], ou["dP"], color=TEAL, alpha=0.4, s=12, label="Δln(P) observations")
        x_line = np.linspace(ou["P_lag"].min(), ou["P_lag"].max(), 100)
        ax_reg.plot(x_line, ou["alpha"] + ou["beta"] * x_line, color=RED_CLR, linewidth=1.5, label="OLS fit")
        ax_reg.axhline(0, color="#4b5563", linewidth=0.8)
        ax_reg.set_xlabel("ln(P(t-1))", color="#9ca3af", fontsize=9)
        ax_reg.set_ylabel("Δln(P)", color="#9ca3af", fontsize=9)
        ax_reg.set_title("OLS Regression: Δln(P) vs ln(P(t-1))", color="#9ca3af", fontsize=10)
        ax_reg.tick_params(colors="#6b7280", labelsize=8)
        for sp in ax_reg.spines.values(): sp.set_edgecolor(GRID_CLR)
        ax_reg.legend(fontsize=8, facecolor="#1a1d27", labelcolor="#9ca3af", edgecolor=GRID_CLR)
        plt.tight_layout()
        st.pyplot(fig_reg)
        plt.close()

        st.markdown('<div class="est-section-header">📋 Summary — Values to Use in Your Simulation</div>', unsafe_allow_html=True)

        N_ann_est = annualization_factor(est_freq)
        summary = pd.DataFrame({
            "Parameter": [
                "Annual Drift μ (GBM)", "Itô-Corrected Drift (GBM input)", "Annual Volatility σ (GBM)",
                f"{est_freq} Volatility σ (GBM)", "Mean Reversion Speed κ (OU)", "Long-Run Mean θ (OU)",
                "OU Volatility σ (annualised)", "Half-Life (years)", f"Half-Life ({est_freq.lower()} periods)",
            ],
            "Value": [
                f"{gbm['mu_annual']*100:.4f}%", f"{gbm['mu_ito']*100:.4f}%",
                f"{gbm['sigma_annual']*100:.4f}%", f"{gbm['sigma_period']*100:.4f}%",
                f"{ou['k']:.4f}", f"{ou['theta']:,.4f}", f"{ou['sigma_ou']*100:.4f}%",
                f"{ou['half_life_years']:.4f}" if not np.isnan(ou['half_life_years']) else "N/A",
                f"{ou['half_life_periods']:.2f}" if not np.isnan(ou.get('half_life_periods', np.nan)) else "N/A",
            ],
            "Use In": [
                "GBM (raw drift)", "GBM (recommended input)", "GBM (annual steps)",
                f"GBM ({est_freq.lower()} steps)", "OU simulation", "OU simulation",
                "OU simulation", "Interpretation", "Interpretation",
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        dl_col, apply_col = st.columns(2)
        with dl_col:
            st.download_button(
                "⬇️ Download Summary CSV",
                data=summary.to_csv(index=False),
                file_name="sugar_model_parameters.csv",
                mime="text/csv",
                use_container_width=True
            )
        with apply_col:
            apply_target = "GBM" if "GBM" in model else "OU"
            if st.button(f"✅ Apply {apply_target} Parameters to Simulation →", use_container_width=True):
                if "GBM" in model:
                    st.session_state["param_mu"]    = float(round(gbm["mu_ito"], 4))
                    st.session_state["param_sigma"] = float(round(gbm["sigma_annual"], 4))
                    st.session_state["applied_from"] = "GBM"
                else:
                    st.session_state["param_kappa"] = float(round(max(ou["k"], 0.001), 4))
                    st.session_state["param_theta"] = float(round(ou["theta"], 2))
                    st.session_state["param_sigma"] = float(round(ou["sigma_ou"], 4))
                    st.session_state["applied_from"] = "OU"
                st.session_state["params_applied"] = True
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monte Carlo Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    if not run:
        st.info("👈  Configure the sidebar and click **Run Simulation** to generate results.", icon="💡")
        st.stop()

    with st.spinner("Running Monte Carlo simulation…"):
        N_sim, K = int(N_sim), int(K)
        if "GBM" in model:
            terminal        = run_gbm_terminal(S0, mu, sigma, T, N_sim, seed)
            times, paths    = run_gbm_paths(S0, mu, sigma, T, steps_per_year, K, seed + 1)
        else:
            terminal        = run_mean_revert_terminal(S0, kappa, theta, sigma, T, N_sim, steps_per_year, seed)
            times, paths    = run_mean_revert_paths(S0, kappa, theta, sigma, T, steps_per_year, K, seed + 1)

    mean_p   = float(np.mean(terminal))
    median_p = float(np.median(terminal))
    std_p    = float(np.std(terminal))
    p05      = float(np.percentile(terminal, 5))
    p25      = float(np.percentile(terminal, 25))
    p75      = float(np.percentile(terminal, 75))
    p95      = float(np.percentile(terminal, 95))
    var95    = S0 - p05
    es_vals  = terminal[terminal <= p05]
    es95     = float(np.mean(es_vals)) if len(es_vals) > 0 else p05
    prob_be  = float(np.mean(terminal <= breakeven))
    rev_risk = var95 * volume if volume > 0 else None

    st.markdown('<div class="section-header">Key Statistics at Horizon</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean Price",              f"₱{mean_p:,.0f}",  f"{(mean_p/S0-1)*100:+.1f}% vs spot")
    k2.metric("Median Price",            f"₱{median_p:,.0f}", f"{(median_p/S0-1)*100:+.1f}% vs spot")
    k3.metric("Std Deviation",           f"₱{std_p:,.0f}")
    k4.metric("VaR 95%",                 f"₱{var95:,.0f}",   "Max likely downside (1-in-20)")
    k5.metric("P(Price ≤ Break-even)",   f"{prob_be*100:.1f}%")

    st.markdown("")
    if prob_be > 0.30:
        st.markdown(f'<div class="alert-danger">⚠️ High risk: <b>{prob_be*100:.1f}%</b> probability of finishing at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)
    elif prob_be > 0.10:
        st.markdown(f'<div class="alert-danger">⚠️ Moderate risk: <b>{prob_be*100:.1f}%</b> probability of finishing at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-safe">✅ Low risk: only <b>{prob_be*100:.1f}%</b> probability of finishing at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)

    if rev_risk is not None:
        st.markdown(f'<div class="alert-danger" style="margin-top:6px">Revenue at Risk (VaR × Volume): <b>₱{rev_risk:,.0f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Price Distribution at Horizon</div>', unsafe_allow_html=True)
    tab_dist, tab_paths, tab_pct = st.tabs(["📊 Distribution", "📈 Price Paths", "🔢 Percentile Table"])

    with tab_dist:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=terminal, nbinsx=80, name="Simulated prices",
                                   marker_color=TEAL, opacity=0.75,
                                   hovertemplate="Price: ₱%{x:,.0f}<br>Count: %{y}<extra></extra>"))
        hist_vals, bin_edges = np.histogram(terminal, bins=80)
        below_mask = bin_edges[:-1] <= p05
        fig.add_trace(go.Bar(x=bin_edges[:-1][below_mask], y=hist_vals[below_mask],
                             width=np.diff(bin_edges)[0], marker_color=RED_CLR, opacity=0.6,
                             name="Below P05 (VaR zone)",
                             hovertemplate="Price: ₱%{x:,.0f}<br>Count: %{y}<extra></extra>"))
        for val, label, color in [
            (p05, f"P05  ₱{p05:,.0f}", RED_CLR),
            (p95, f"P95  ₱{p95:,.0f}", GREEN_OK),
            (mean_p, f"Mean ₱{mean_p:,.0f}", GOLD),
            (breakeven, f"Break-even ₱{breakeven:,.0f}", "#e8a0a0"),
        ]:
            fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5,
                          annotation_text=label, annotation_font_color=color,
                          annotation_font_size=11, annotation_position="top")
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color=TEXT_CLR,
            xaxis=dict(title="Terminal Price (₱/Lkg)", gridcolor=GRID_CLR, zeroline=False),
            yaxis=dict(title="Number of simulations", gridcolor=GRID_CLR),
            legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR, borderwidth=1),
            margin=dict(t=30, b=50, l=50, r=30), height=420, barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P05 (worst 5%)",  f"₱{p05:,.0f}",  f"{(p05/S0-1)*100:+.1f}% vs spot")
        c2.metric("P25",             f"₱{p25:,.0f}",  f"{(p25/S0-1)*100:+.1f}% vs spot")
        c3.metric("P75",             f"₱{p75:,.0f}",  f"{(p75/S0-1)*100:+.1f}% vs spot")
        c4.metric("P95 (best 5%)",   f"₱{p95:,.0f}",  f"{(p95/S0-1)*100:+.1f}% vs spot")
        st.caption(f"Expected Shortfall (avg price when ≤ P05): **₱{es95:,.0f}/Lkg**  —  Based on {N_sim:,} simulations.")

    with tab_paths:
        if horizon_unit == "Weeks":
            times_display = times * 52
        elif horizon_unit == "Months":
            times_display = times * 12
        else:
            times_display = times

        fig2 = go.Figure()
        path_at_t = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        fig2.add_trace(go.Scatter(
            x=np.concatenate([times_display, times_display[::-1]]),
            y=np.concatenate([path_at_t[4], path_at_t[0][::-1]]),
            fill="toself", fillcolor="rgba(74,159,181,0.1)", line_color="rgba(0,0,0,0)",
            name="P05–P95 range", hoverinfo="skip"))
        fig2.add_trace(go.Scatter(
            x=np.concatenate([times_display, times_display[::-1]]),
            y=np.concatenate([path_at_t[3], path_at_t[1][::-1]]),
            fill="toself", fillcolor="rgba(74,159,181,0.2)", line_color="rgba(0,0,0,0)",
            name="P25–P75 range", hoverinfo="skip"))
        display_k = min(K, 25)
        for i in range(display_k):
            fig2.add_trace(go.Scatter(
                x=times_display, y=paths[:, i], mode="lines",
                line=dict(color=TEAL, width=0.6), opacity=0.35,
                showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(
            x=times_display, y=path_at_t[2], mode="lines",
            line=dict(color=GOLD, width=2.5), name="Median path"))
        fig2.add_hline(y=breakeven, line_dash="dot", line_color=RED_CLR, line_width=1.5,
                       annotation_text=f"Break-even ₱{breakeven:,.0f}", annotation_font_color=RED_CLR)
        fig2.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color=TEXT_CLR,
            xaxis=dict(title=f"{horizon_unit} from now", gridcolor=GRID_CLR),
            yaxis=dict(title="Price (₱/Lkg)", gridcolor=GRID_CLR),
            legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR, borderwidth=1),
            margin=dict(t=30, b=50, l=50, r=30), height=430,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Showing {display_k} sample paths with P05–P95 and P25–P75 confidence bands.")

    with tab_pct:
        pcts = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]
        vals = np.percentile(terminal, pcts)
        rows = []
        for p, v in zip(pcts, vals):
            chg = (v / S0 - 1) * 100
            rows.append({
                "Percentile": f"P{p:02d}",
                "Price (₱/Lkg)": f"₱{v:,.0f}",
                "Change vs Spot": f"{chg:+.1f}%",
                "Below Break-even?": "❌ Yes" if v <= breakeven else "✅ No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=520)
        st.caption(f"Spot: ₱{S0:,.0f}/Lkg | Break-even: ₱{breakeven:,.0f}/Lkg | Model: {model}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Weekly Price Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab_weekly:
    if not run:
        st.info("👈  Configure the sidebar and click **Run Simulation** to generate the weekly prediction chart.", icon="💡")
        st.stop()

    with st.spinner("Computing week-by-week predictions…"):
        n_weeks_int = int(weekly_n_weeks)
        N_sim_int   = int(N_sim)

        if "GBM" in model:
            wdf = run_weekly_gbm(S0, mu, sigma, n_weeks_int, N_sim_int, seed + 99)
        else:
            wdf = run_weekly_ou(S0, kappa, theta, sigma, n_weeks_int, N_sim_int, seed + 99)

    bar_col   = "median" if weekly_display == "Median (P50)" else "mean"
    bar_label = "Median" if weekly_display == "Median (P50)" else "Mean"
    if weekly_interval == "P05–P95 (90%)":
        lo_col, hi_col = "p05", "p95"
        int_label = "P05–P95"
    else:
        lo_col, hi_col = "p25", "p75"
        int_label = "P25–P75"

    bar_vals = wdf[bar_col].values
    lo_vals  = wdf[lo_col].values
    hi_vals  = wdf[hi_col].values
    weeks    = wdf["week"].values

    err_plus  = hi_vals - bar_vals
    err_minus = bar_vals - lo_vals

    def bar_color(price, be):
        if price <= be:
            return RED_CLR
        elif price <= be * 1.05:
            return AMBER
        else:
            return GREEN_OK

    colors = [bar_color(v, breakeven) for v in bar_vals]

    st.markdown('<div class="section-header">Weekly Price Prediction</div>', unsafe_allow_html=True)

    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Week 1 Forecast",      f"₱{bar_vals[0]:,.0f}",  f"{(bar_vals[0]/S0-1)*100:+.1f}% vs spot")
    mid_idx = n_weeks_int // 2 - 1
    w2.metric(f"Week {mid_idx+1} Forecast", f"₱{bar_vals[mid_idx]:,.0f}", f"{(bar_vals[mid_idx]/S0-1)*100:+.1f}% vs spot")
    w3.metric(f"Week {n_weeks_int} Forecast", f"₱{bar_vals[-1]:,.0f}", f"{(bar_vals[-1]/S0-1)*100:+.1f}% vs spot")
    weeks_below = int(np.sum(bar_vals <= breakeven))
    w4.metric("Weeks Below Break-even", f"{weeks_below} / {n_weeks_int}")
    peak_wk = int(np.argmax(bar_vals)) + 1
    w5.metric("Peak Forecast Week", f"Week {peak_wk}", f"₱{bar_vals[peak_wk-1]:,.0f}")

    st.markdown("")
    st.markdown(
        f'<span class="info-pill" style="background:#2d1a1a;color:{RED_CLR}">🔴 At / below break-even</span>'
        f'<span class="info-pill" style="background:#2b2000;color:{AMBER}">🟡 Within 5% of break-even</span>'
        f'<span class="info-pill" style="background:#122d1e;color:{GREEN_OK}">🟢 Above break-even +5%</span>'
        f'<span class="info-pill">Error bars: {int_label}</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    fig_w = go.Figure()

    fig_w.add_trace(go.Scatter(
        x=np.concatenate([weeks, weeks[::-1]]),
        y=np.concatenate([hi_vals, lo_vals[::-1]]),
        fill="toself",
        fillcolor="rgba(74,159,181,0.12)",
        line_color="rgba(0,0,0,0)",
        name=f"{int_label} band",
        hoverinfo="skip",
    ))

    fig_w.add_trace(go.Bar(
        x=weeks,
        y=bar_vals,
        name=f"{bar_label} Price",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
        opacity=0.85,
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_plus,
            arrayminus=err_minus,
            color="#5a7a90",
            thickness=1.2,
            width=3,
        ),
        customdata=np.stack([lo_vals, hi_vals, wdf["p05"].values, wdf["p95"].values,
                             wdf["mean"].values, wdf["median"].values], axis=-1),
        hovertemplate=(
            "<b>Week %{x}</b><br>"
            f"{bar_label}: ₱%{{y:,.0f}}<br>"
            f"{int_label} Low: ₱%{{customdata[0]:,.0f}}<br>"
            f"{int_label} High: ₱%{{customdata[1]:,.0f}}<br>"
            "P05: ₱%{customdata[2]:,.0f}<br>"
            "P95: ₱%{customdata[3]:,.0f}<br>"
            "Mean: ₱%{customdata[4]:,.0f}<br>"
            "Median: ₱%{customdata[5]:,.0f}<extra></extra>"
        ),
    ))

    fig_w.add_hline(
        y=breakeven,
        line_dash="dash", line_color=RED_CLR, line_width=1.5,
        annotation_text=f"Break-even ₱{breakeven:,.0f}",
        annotation_font_color=RED_CLR,
        annotation_position="top right",
    )
    fig_w.add_hline(
        y=S0,
        line_dash="dot", line_color=GOLD, line_width=1,
        annotation_text=f"Spot ₱{S0:,.0f}",
        annotation_font_color=GOLD,
        annotation_position="bottom right",
    )

    if "Mean-Reverting" in model:
        fig_w.add_hline(
            y=theta,
            line_dash="dashdot", line_color="#a78bfa", line_width=1,
            annotation_text=f"Long-run mean θ ₱{theta:,.0f}",
            annotation_font_color="#a78bfa",
            annotation_position="top left",
        )

    fig_w.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color=TEXT_CLR,
        font_family="DM Sans, sans-serif",
        xaxis=dict(
            title="Week from Today", gridcolor=GRID_CLR, zeroline=False,
            tickmode="linear", tick0=1, dtick=max(1, n_weeks_int // 13),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Predicted Price (₱/Lkg)", gridcolor=GRID_CLR, zeroline=False,
            tickprefix="₱", tickformat=",",
        ),
        legend=dict(
            bgcolor=DARK_BG, bordercolor=GRID_CLR, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        bargap=0.25, margin=dict(t=60, b=60, l=70, r=30), height=500,
        title=dict(
            text=f"Week-by-Week {bar_label} Sugar Price Forecast · {n_weeks_int} Weeks · {model}",
            font=dict(family="DM Serif Display, serif", size=16, color="#8ab4cc"),
            x=0.0, xanchor="left",
        ),
    )

    st.plotly_chart(fig_w, use_container_width=True)

    trend_pct = (bar_vals[-1] / S0 - 1) * 100
    trend_dir = "📈 upward" if trend_pct > 1 else ("📉 downward" if trend_pct < -1 else "➡️ flat")
    st.caption(
        f"Overall trend over {n_weeks_int} weeks: **{trend_dir}** · "
        f"Start ₱{S0:,.0f} → Week {n_weeks_int} {bar_label} ₱{bar_vals[-1]:,.0f} "
        f"({trend_pct:+.1f}%) · "
        f"Error bars show {int_label} confidence interval · "
        f"Based on {N_sim_int:,} Monte Carlo paths."
    )

    with st.expander("📋 View detailed weekly forecast table"):
        tbl_rows = []
        for _, row in wdf.iterrows():
            wk   = int(row["week"])
            med  = row["median"]
            mn   = row["mean"]
            p5   = row["p05"]
            p25r = row["p25"]
            p75r = row["p75"]
            p95r = row["p95"]
            chg  = (med / S0 - 1) * 100
            risk_flag = "❌ Below" if med <= breakeven else ("⚠️ Near" if med <= breakeven * 1.05 else "✅ Safe")
            tbl_rows.append({
                "Week": wk,
                "Median (₱)": f"₱{med:,.0f}",
                "Mean (₱)": f"₱{mn:,.0f}",
                "P05 (₱)": f"₱{p5:,.0f}",
                "P25 (₱)": f"₱{p25r:,.0f}",
                "P75 (₱)": f"₱{p75r:,.0f}",
                "P95 (₱)": f"₱{p95r:,.0f}",
                "vs Spot": f"{chg:+.1f}%",
                "Break-even Risk": risk_flag,
            })
        tbl_df = pd.DataFrame(tbl_rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True, height=400)
        st.download_button(
            "⬇️ Download Weekly Forecast CSV",
            data=tbl_df.to_csv(index=False),
            file_name="sugar_weekly_forecast.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — SugarBot Chatbot
# ════════════════════════════════════════════════════════════════════════════════
with tab_bot:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f1923,#162030);border:1px solid #1e2d3d;
                border-radius:16px;padding:20px 24px;margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
        <span style="font-size:2rem">🍬</span>
        <div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#e8dcc8">SugarBot</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#4a9fb5;
                      text-transform:uppercase;letter-spacing:0.1em">
            Your Sugar Price Model Assistant
          </div>
        </div>
        <div style="margin-left:auto;background:#122d1e;border:1px solid #52c87a;border-radius:20px;
                    padding:3px 12px;font-size:11px;color:#52c87a;font-family:'IBM Plex Mono',monospace">
          ● Online
        </div>
      </div>
      <div style="font-size:13px;color:#7a8fa8;font-family:'DM Sans',sans-serif;line-height:1.5">
        Ask me anything about how to use this app — model selection, uploading CSV data,
        reading results, interpreting risk metrics, and more.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API key input ─────────────────────────────────────────────────────────
    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com. It is not stored anywhere.",
    )

    if not api_key:
        st.markdown("""
        <div class="info-box">
          Enter your <b>Anthropic API key</b> above to activate SugarBot.<br>
          Get one free at <a href="https://console.anthropic.com" target="_blank"
          style="color:#4a9fb5">console.anthropic.com</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Suggestion chips ──────────────────────────────────────────────────────
    st.markdown("**💡 Quick questions — click to ask:**")

    suggestions = [
        "How do I get started?",
        "GBM vs Mean-Reverting — which should I use?",
        "How do I upload a CSV?",
        "What does VaR 95% mean?",
        "What is κ (kappa)?",
        "How do I read the weekly forecast chart?",
        "What is Expected Shortfall?",
        "What does the break-even price do?",
        "How do I interpret P05 and P95?",
        "What is the Itô-corrected drift?",
    ]

    # Render chips as buttons in rows of 3
    chip_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with chip_cols[i % 3]:
            if st.button(suggestion, key=f"chip_{i}", use_container_width=True):
                st.session_state["sugarbot_messages"].append({
                    "role": "user",
                    "content": suggestion
                })
                st.session_state["_sugarbot_trigger"] = True
                st.rerun()

    st.markdown("---")

    # ── Chat history display ──────────────────────────────────────────────────
    if not st.session_state["sugarbot_messages"]:
        st.markdown("""
        <div style="text-align:center;padding:32px;color:#4a5568;">
          <div style="font-size:2.5rem;margin-bottom:12px">🍬</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#6b7280;margin-bottom:6px">
            No messages yet
          </div>
          <div style="font-size:13px;color:#4a5568">
            Click a suggestion above or type your question below
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["sugarbot_messages"]:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🍬"):
                    st.markdown(msg["content"])

    # ── Auto-respond if chip was clicked ─────────────────────────────────────
    if st.session_state.get("_sugarbot_trigger") and api_key:
        st.session_state["_sugarbot_trigger"] = False
        try:
            client = anthropic.Anthropic(api_key=api_key)
            with st.chat_message("assistant", avatar="🍬"):
                with st.spinner("SugarBot is thinking…"):
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        system=SUGARBOT_SYSTEM,
                        messages=st.session_state["sugarbot_messages"],
                    )
                    reply = response.content[0].text
                    st.markdown(reply)
            st.session_state["sugarbot_messages"].append({
                "role": "assistant",
                "content": reply
            })
            st.rerun()
        except Exception as e:
            st.error(f"❌ API error: {e}")

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask SugarBot anything about this app…",
        disabled=not api_key,
    )

    if user_input:
        st.session_state["sugarbot_messages"].append({
            "role": "user",
            "content": user_input
        })

        if not api_key:
            st.warning("Please enter your Anthropic API key above to chat with SugarBot.")
        else:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                with st.chat_message("assistant", avatar="🍬"):
                    with st.spinner("SugarBot is thinking…"):
                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1000,
                            system=SUGARBOT_SYSTEM,
                            messages=st.session_state["sugarbot_messages"],
                        )
                        reply = response.content[0].text
                        st.markdown(reply)
                st.session_state["sugarbot_messages"].append({
                    "role": "assistant",
                    "content": reply
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ API error: {e}")
                st.session_state["sugarbot_messages"].pop()

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state["sugarbot_messages"]:
        st.markdown("")
        if st.button("🗑️ Clear conversation", use_container_width=False):
            st.session_state["sugarbot_messages"] = []
            st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mill-gate raw sugar price model. "
    "GBM assumes lognormally distributed returns. "
    "Mean-Reverting uses an Ornstein–Uhlenbeck process on log-prices. "
    "Results are probabilistic estimates, not forecasts."
)
