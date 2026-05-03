"""
Sugar Price Monte Carlo Risk Model — with integrated Parameter Estimator
Run with: streamlit run sugar_app.py
Requires: conda install streamlit plotly numpy scipy matplotlib pandas supabase json os
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
from datetime import datetime

# ── Supabase Client ────────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    _SUPABASE_URL = st.secrets["supabase"]["url"]
    _SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False
    supabase = None


# ── Auth Helpers ───────────────────────────────────────────────────────────────

def auth_login(email: str, password: str):
    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    return res

def auth_signup(email: str, password: str):
    res = supabase.auth.sign_up({"email": email, "password": password})
    return res

def auth_logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state["user"]          = None
    st.session_state["access_token"]  = None
    st.session_state["refresh_token"] = None
    # FIX (Bug 2): Do NOT clear URL params that contain tokens — we no longer
    # store tokens in the URL at all, so there is nothing to clear here.

def get_current_user():
    return st.session_state.get("user", None)


# ── DB Helpers ─────────────────────────────────────────────────────────────────

def _try_set_session(client, token, refresh):
    """Warn instead of silently swallowing stale session errors."""
    if not (token and refresh):
        return
    try:
        client.auth.set_session(token, refresh)
    except Exception as e:
        st.warning(f"Session expired — please sign in again. ({e})")
        auth_logout()
        st.rerun()


def save_simulation(user_id: str, params: dict, results: dict, token: str = None, refresh: str = None):
    try:
        client = supabase
        _try_set_session(client, token, refresh)
        client.table("simulation_runs").insert({
            "user_id":    user_id,
            "model":      params.get("model"),
            "spot_price": params.get("S0"),
            "horizon":    params.get("horizon_label"),
            "params":     json.dumps(params),
            "results":    json.dumps(results),
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    except Exception as e:
        st.warning(f"Could not save simulation: {e}")
        return False

def load_simulations(user_id: str, token: str = None, refresh: str = None):
    try:
        _try_set_session(supabase, token, refresh)
        res = supabase.table("simulation_runs") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        return res.data or []
    except Exception as e:
        st.warning(f"Could not load simulations: {e}")
        return []


# ── Login / Signup Wall ────────────────────────────────────────────────────────

def render_auth_page():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Outfit:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
      html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: #0a0f0d !important; }
      .stApp { background: #0a0f0d !important; }
      header[data-testid="stHeader"] { display: none !important; }
      section[data-testid="stSidebar"] { display: none !important; }
      .main .block-container { padding-top: 0 !important; max-width: 100% !important; }
      .stTextInput input {
        background: rgba(10,20,14,0.9) !important;
        border: 1px solid rgba(52,120,70,0.4) !important;
        border-radius: 10px !important;
        color: #e8dcc8 !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s, box-shadow 0.2s;
      }
      .stTextInput input:focus {
        border-color: rgba(212,168,67,0.7) !important;
        box-shadow: 0 0 0 3px rgba(212,168,67,0.12) !important;
        outline: none !important;
      }
      .stTextInput label { color: #6a8f72 !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
      .stTabs [data-baseweb="tab-list"] { background: rgba(10,20,14,0.7) !important; border-radius: 10px !important; padding: 4px !important; border: 1px solid rgba(52,120,70,0.25) !important; }
      .stTabs [data-baseweb="tab"] { border-radius: 7px !important; color: #4a6b52 !important; font-weight: 500 !important; }
      .stTabs [aria-selected="true"] { background: rgba(52,120,70,0.3) !important; color: #a8d4b0 !important; }
      .stButton > button {
        background: linear-gradient(135deg, #1e4d2a 0%, #2d7040 100%) !important;
        color: #d4f5dc !important; font-weight: 600 !important; font-size: 0.9rem !important;
        border: 1px solid rgba(52,180,70,0.3) !important; border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important; width: 100% !important;
        box-shadow: 0 4px 20px rgba(34,85,47,0.35) !important; margin-top: 0.5rem !important;
        transition: all 0.2s !important;
      }
      .stButton > button:hover { background: linear-gradient(135deg, #265e34 0%, #368a4e 100%) !important; transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(34,85,47,0.5) !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
      min-height: 30vh;
      background: radial-gradient(ellipse 80% 60% at 20% 80%, rgba(34,85,47,0.4) 0%, transparent 60%),
                  radial-gradient(ellipse 60% 80% at 80% 20%, rgba(212,168,67,0.15) 0%, transparent 55%),
                  #0a0f0d;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 3rem 1rem 2rem; text-align: center;
    ">
      <div style="font-size:4rem; margin-bottom:1rem; filter: drop-shadow(0 0 30px rgba(52,200,80,0.5));">🍬</div>
      <div style="font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:700;
           color:#e8dcc8; letter-spacing:-0.02em; margin-bottom:0.4rem;">
        Montecarlo Risk Model | Price Prediction
      </div>
      <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#3a6b45;
           letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.5rem;">
        Monte Carlo Risk Model | Sugar Price Prediction
      </div>
      <div style="width:60px; height:2px; background:linear-gradient(90deg,transparent,#d4a843,transparent); margin:0.5rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="background:rgba(15,25,18,0.97); border:1px solid rgba(52,120,70,0.3);
             border-radius:20px; padding:2rem 2rem 1.5rem;
             box-shadow: 0 0 0 1px rgba(212,168,67,0.06), 0 32px 80px rgba(0,0,0,0.7),
                         0 0 60px rgba(34,85,47,0.12);">
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            email    = st.text_input("Email address", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            if st.button("Sign In →", key="btn_login", width='stretch'):
                if not SUPABASE_OK:
                    st.error("Supabase is not configured. Add credentials to `.streamlit/secrets.toml`.")
                elif not email or not password:
                    st.warning("Please enter email and password.")
                else:
                    try:
                        res = auth_login(email, password)
                        # FIX (Bug 2): Store tokens ONLY in session_state — never in the URL.
                        # The URL ?tok= approach exposes long-lived refresh tokens in browser
                        # history, proxy logs, and referrer headers even when Fernet-encrypted,
                        # because the decryption key is derivable from semi-public Supabase creds.
                        st.session_state["user"]          = res.user
                        st.session_state["access_token"]  = res.session.access_token
                        st.session_state["refresh_token"] = res.session.refresh_token
                        supabase.auth.set_session(res.session.access_token, res.session.refresh_token)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")

        with tab_signup:
            email2 = st.text_input("Email address", key="signup_email", placeholder="you@example.com")
            pw2    = st.text_input("Password", type="password", key="signup_pw", placeholder="Min. 6 characters")
            pw3    = st.text_input("Confirm password", type="password", key="signup_pw2", placeholder="••••••••")
            if st.button("Create Account →", key="btn_signup", width='stretch'):
                if not SUPABASE_OK:
                    st.error("Supabase is not configured. Add credentials to `.streamlit/secrets.toml`.")
                elif not email2 or not pw2:
                    st.warning("Please fill in all fields.")
                elif pw2 != pw3:
                    st.error("Passwords do not match.")
                elif len(pw2) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        res = auth_signup(email2, pw2)
                        st.success("✅ Account created! Check your email to confirm, then sign in.")
                    except Exception as e:
                        msg = str(e)
                        if "already registered" in msg.lower() or "already exists" in msg.lower():
                            st.error("An account with this email already exists. Please sign in instead.")
                        else:
                            st.error(f"Sign-up failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-top:1.2rem; font-size:0.7rem;
             color:#3a6b45; font-family:'Space Mono',monospace; letter-spacing:0.06em;">
          Mill-gate Sugar · Philippines · Probabilistic estimates only
        </div>
        """, unsafe_allow_html=True)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sugar Pricing Forecasting",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ───────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state["user"] = None
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "refresh_token" not in st.session_state:
    st.session_state["refresh_token"] = None

# FIX (Bug 2): Removed all URL ?tok= token persistence logic.
# Tokens now live exclusively in st.session_state. Consequence: a hard browser
# refresh will require the user to sign in again — this is the correct and safe
# behaviour. To enable persistent sessions properly, use Supabase's
# PKCE/cookie flow or a server-side session table, not the URL.

_user = get_current_user()
if _user is None:
    render_auth_page()
    st.stop()

_access_token  = st.session_state.get("access_token")
_refresh_token = st.session_state.get("refresh_token")
if _access_token and _refresh_token and SUPABASE_OK:
    _try_set_session(supabase, _access_token, _refresh_token)

# ── Main App CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #080e0b !important;
    color: #d4c9b4;
  }
  .stApp { background: #080e0b !important; }
  .main .block-container { padding-top: 1.5rem; background: #080e0b; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1410 0%, #080e0b 100%);
    border-right: 1px solid rgba(52,120,70,0.2);
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] p:not(button p) {
    color: #6a8f72 !important;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
  }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: #d4a843; }
  section[data-testid="stSidebar"] input {
    background: rgba(10,20,14,0.8) !important;
    border: 1px solid rgba(52,120,70,0.3) !important;
    color: #d4c9b4 !important;
    border-radius: 8px !important;
  }

  div[data-testid="metric-container"] {
    background: rgba(12,22,16,0.9);
    border: 1px solid rgba(52,120,70,0.25);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(212,168,67,0.06);
    transition: border-color 0.2s;
  }
  div[data-testid="metric-container"]:hover { border-color: rgba(212,168,67,0.3); }
  div[data-testid="metric-container"] label {
    color: #4a6b52 !important;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8dcc8;
    font-size: 1.55rem;
    font-weight: 600;
    font-family: 'Playfair Display', serif;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.78rem; }

  .section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-style: italic;
    color: #6aab7a;
    border-bottom: 1px solid rgba(52,120,70,0.25);
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1.1rem 0;
    letter-spacing: 0.01em;
  }
  .est-section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #3a6b45;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid rgba(52,120,70,0.2);
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  .info-pill {
    display: inline-block;
    background: rgba(12,30,18,0.9);
    color: #7aab86;
    border: 1px solid rgba(52,120,70,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px 3px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
  }
  .applied-pill {
    display: inline-block;
    background: rgba(18,45,30,0.9);
    color: #52c87a;
    border: 1px solid rgba(52,200,80,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px 3px;
  }

  .alert-danger {
    background: rgba(45,26,26,0.8);
    border-left: 3px solid #e05252;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.85rem;
    color: #f5a0a0;
    backdrop-filter: blur(4px);
  }
  .alert-safe {
    background: rgba(18,45,30,0.8);
    border-left: 3px solid #52c87a;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.85rem;
    color: #8de8a8;
    backdrop-filter: blur(4px);
  }
  .info-box {
    background: rgba(12,22,30,0.9);
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #7a9aaf;
    margin-bottom: 16px;
  }
  .warn-box {
    background: rgba(25,18,8,0.9);
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #c4935a;
    margin-bottom: 16px;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1a4226 0%, #256336 100%);
    color: #c8f0d0 !important;
    font-weight: 600;
    border: 1px solid rgba(52,160,70,0.35);
    border-radius: 10px;
    font-size: 0.88rem;
    padding: 0.55rem 1.5rem;
    letter-spacing: 0.04em;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 3px 16px rgba(34,85,47,0.3);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #205230 0%, #2e7a41 100%);
    box-shadow: 0 5px 24px rgba(34,85,47,0.45);
    transform: translateY(-1px);
    color: #d8f8e0 !important;
  }
  section[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: linear-gradient(135deg, #6b4800 0%, #a06c00 100%) !important;
    color: #fff3cc !important;
    border: 1px solid rgba(212,168,67,0.4) !important;
    box-shadow: 0 3px 16px rgba(160,108,0,0.35) !important;
  }
  section[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: linear-gradient(135deg, #7d5500 0%, #b87c00 100%) !important;
    box-shadow: 0 5px 24px rgba(160,108,0,0.5) !important;
  }
  .apply-btn > button {
    background: rgba(18,45,30,0.9) !important;
    color: #52c87a !important;
    border: 1px solid rgba(52,200,80,0.35) !important;
  }
  .apply-btn > button:hover { background: rgba(25,60,38,0.9) !important; }

  .stTabs [data-baseweb="tab-list"] {
    background: rgba(10,20,14,0.6);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(52,120,70,0.2);
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #3a6b45;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(52,120,70,0.25) !important;
    color: #8fd4a0 !important;
  }

  .stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(10,20,14,0.8) !important;
    border: 1px solid rgba(52,120,70,0.3) !important;
    color: #d4c9b4 !important;
    border-radius: 8px !important;
  }
  .stTextInput input:focus, .stNumberInput input:focus {
    border-color: rgba(212,168,67,0.5) !important;
    box-shadow: 0 0 0 2px rgba(212,168,67,0.1) !important;
  }

  .page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8dcc8;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
  }
  .page-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #3a6b45;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


def annualization_factor(freq: str) -> int:
    return {"Daily": 252, "Weekly": 52, "Monthly": 12, "Yearly": 1}[freq]

def dt_value(freq: str) -> float:
    return {"Daily": 1/252, "Weekly": 1/52, "Monthly": 1/12, "Yearly": 1.0}[freq]


def clean_price_series(prices: np.ndarray, dates=None):
    """
    Returns (clean_prices, clean_dates, n_dropped) after:
      - Removing NaNs
      - Removing zero and negative values (log() is undefined for these)
      - Sorting by date if dates are provided
    """
    mask = np.isfinite(prices) & (prices > 0)
    n_dropped = int(np.sum(~mask))
    clean_prices = prices[mask]
    clean_dates  = dates[mask] if dates is not None else None

    if clean_dates is not None:
        try:
            order = np.argsort(clean_dates)
            clean_prices = clean_prices[order]
            clean_dates  = clean_dates[order]
        except Exception:
            pass

    return clean_prices, clean_dates, n_dropped


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
    """
    Guard against zero-variance (constant) price series.
    """
    dt         = dt_value(freq)
    N          = annualization_factor(freq)
    log_prices = np.log(prices)
    d_logP     = np.diff(log_prices)
    logP_lag   = log_prices[:-1]

    if np.std(logP_lag) < 1e-12 or np.std(d_logP) < 1e-12:
        return {
            "k":                 np.nan,
            "theta":             float(np.exp(np.mean(log_prices))),
            "sigma_ou":          0.0,
            "half_life_years":   np.nan,
            "half_life_periods": np.nan,
            "r_squared":         0.0,
            "p_value":           1.0,
            "residuals":         np.zeros(len(d_logP)),
            "beta":              0.0,
            "alpha":             0.0,
            "dP":                d_logP,
            "P_lag":             logP_lag,
            "_constant_series":  True,
        }

    slope, intercept, r_value, p_value, se = stats.linregress(logP_lag, d_logP)
    beta      = slope
    alpha     = intercept

    if abs(beta) < 1e-12:
        k     = 0.0
        theta = float(np.exp(np.mean(log_prices)))
    else:
        k     = -beta / dt
        theta = float(np.exp(-alpha / beta))

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
        "_constant_series":  False,
    }


# ── Simulation Engine ──────────────────────────────────────────────────────────

def run_gbm_terminal(S0, mu, sigma, T, N, seed):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(N)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def run_mean_revert_terminal(S0, kappa, theta, sigma, T, N, steps_per_year, seed):
    """
    FIX (Bug 4): Use the exact OU solution instead of Euler-Maruyama to eliminate
    numerical instability when κ·dt ≥ 1 (e.g. κ=100 with weekly steps).

    Exact solution for log-OU:
        ln_S_t = ln_theta_adj + (ln_S_0 - ln_theta_adj)*exp(-κ·dt)
                 + σ·sqrt((1 - exp(-2κ·dt)) / (2κ)) · Z

    where ln_theta_adj = ln(θ) - σ²/(2κ)  (Itô correction for log-price mean).

    This is unconditionally stable for all κ > 0 and dt > 0.
    """
    rng      = np.random.default_rng(seed)
    steps    = max(1, int(T * steps_per_year))
    dt       = T / steps
    ln_theta_adj = np.log(theta) - sigma**2 / (2 * kappa)
    ln_S     = np.full(N, np.log(S0), dtype=np.float64)

    decay       = np.exp(-kappa * dt)
    noise_scale = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    for _ in range(steps):
        Z    = rng.standard_normal(N)
        ln_S = ln_theta_adj + (ln_S - ln_theta_adj) * decay + noise_scale * Z
        # Safety clamp: prevent exp() overflow/underflow (±30 in log-space)
        np.clip(ln_S, -30.0, 30.0, out=ln_S)

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
    """
    FIX (Bug 4): Use the exact OU solution for path generation as well.
    Same reasoning as run_mean_revert_terminal above.
    """
    rng          = np.random.default_rng(seed)
    steps        = max(1, int(T * steps_per_year))
    dt           = T / steps
    ln_theta_adj = np.log(theta) - sigma**2 / (2 * kappa)
    decay        = np.exp(-kappa * dt)
    noise_scale  = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    ln_paths = np.zeros((steps + 1, K))
    ln_paths[0] = np.log(S0)

    for t in range(1, steps + 1):
        Z = rng.standard_normal(K)
        ln_paths[t] = ln_theta_adj + (ln_paths[t - 1] - ln_theta_adj) * decay + noise_scale * Z
        # Safety clamp
        np.clip(ln_paths[t], -30.0, 30.0, out=ln_paths[t])

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
    """
    FIX (Bug 4): Exact OU solution applied to weekly simulation as well.
    """
    rng          = np.random.default_rng(seed)
    dt           = 1 / 52
    ln_theta_adj = np.log(theta) - sigma**2 / (2 * kappa)
    decay        = np.exp(-kappa * dt)
    noise_scale  = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    ln_prices = np.full(N_sim, np.log(S0), dtype=float)
    weekly_stats = []
    for w in range(1, n_weeks + 1):
        Z         = rng.standard_normal(N_sim)
        ln_prices = ln_theta_adj + (ln_prices - ln_theta_adj) * decay + noise_scale * Z
        np.clip(ln_prices, -30.0, 30.0, out=ln_prices)
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
    "param_mu":       0.03,
    "param_sigma":    0.18,
    "param_kappa":    0.60,
    "param_theta":    2400.0,
    "params_applied": False,
    "applied_from":   None,
    "wdf":            None,
    "wdf_cache_key":  None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:700;
         color:#ccfa34; margin-bottom:0.2rem; line-height:1.2;">
      🍬 Sugar Price<br><span style="font-size:0.9rem;color:#ffffff;font-family:'Space Mono',monospace;
      font-style:normal;font-weight:400;letter-spacing:0.05em;">Montecarlo Risk Model | Price Prediction</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px;color:#6b7280;font-family:\'IBM Plex Mono\',monospace;'
        f'margin-bottom:4px">Signed in as<br>'
        f'<span style="color:#d4a843">{_user.email}</span></div>',
        unsafe_allow_html=True
    )
    if st.button("🚪 Sign Out", width='stretch'):
        auth_logout()
        st.rerun()
    st.markdown("---")

    st.markdown("### Model Setup")
    model = st.selectbox(
        "Price model",
        ["GBM (Lognormal)", "Mean-Reverting"],
        help="GBM: prices drift with random shocks. Mean-Reverting: prices gravitate to a long-run average."
    )

    if st.session_state.get("params_applied") and st.session_state.get("applied_from"):
        applied_from = st.session_state["applied_from"]
        current_is_gbm = "GBM" in model
        applied_was_gbm = applied_from == "GBM"
        if current_is_gbm != applied_was_gbm:
            st.markdown(
                '<div class="warn-box">⚠️ <b>Parameter mismatch:</b> You applied '
                f'<b>{applied_from}</b> parameters but switched to '
                f'<b>{"GBM" if current_is_gbm else "OU"}</b>. '
                'The σ values use different scales. Please re-estimate or reset parameters.</div>',
                unsafe_allow_html=True
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

            _raw_prices = df_raw[price_col].values
            _raw_dates  = None
            if date_col and date_col != "None":
                try:
                    _raw_dates = pd.to_datetime(df_raw[date_col]).values
                except Exception:
                    _raw_dates = None

            _prices, _, _n_dropped = clean_price_series(_raw_prices, _raw_dates)

            if _n_dropped > 0:
                st.markdown(
                    f'<div class="warn-box">⚠️ Removed {_n_dropped} invalid row(s) '
                    f'(zero, negative, or non-finite prices) before estimation.</div>',
                    unsafe_allow_html=True
                )

            if len(_prices) >= 10:
                gbm_est = compute_gbm_params(_prices, est_freq)
                ou_est  = compute_ou_params(_prices, est_freq)

                lr = gbm_est["log_returns"]
                if len(lr) > 0:
                    z_scores = np.abs((lr - np.mean(lr)) / (np.std(lr) + 1e-12))
                    n_outliers = int(np.sum(z_scores > 4))
                    if n_outliers > 0:
                        st.markdown(
                            f'<div class="warn-box">⚠️ {n_outliers} extreme return(s) detected '
                            f'(|z| &gt; 4). These may inflate σ significantly. '
                            f'Verify your data for data-entry errors.</div>',
                            unsafe_allow_html=True
                        )

                if ou_est.get("_constant_series"):
                    st.markdown(
                        '<div class="warn-box">⚠️ Price series is constant (zero variance). '
                        'OU parameters cannot be estimated. Please upload data with price variation.</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div style="font-size:11px;color:#de901b;font-family:\'IBM Plex Mono\',monospace;margin:8px 0 4px 0">'
                    'ESTIMATED PARAMS</div>', unsafe_allow_html=True
                )
                if "GBM" in model:
                    st.markdown(
                        f'<div style="font-size:12px;color:#d4a843;font-family:\'IBM Plex Mono\',monospace;line-height:1.8">'
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

                    # FIX (Bug 4): Warn when κ·dt is large enough to cause instability
                    # in naive Euler (now moot for the exact solution, but still useful
                    # as a sanity signal for the user's parameter choices).
                    _dt_sidebar = dt_value(est_freq)
                    if k_ok and k_val * _dt_sidebar > 0.5:
                        st.markdown(
                            f'<div class="warn-box">⚠️ κ·dt = {k_val * _dt_sidebar:.2f} &gt; 0.5 '
                            f'(κ={k_val:.2f}, dt={_dt_sidebar:.4f}). The exact OU solution is used '
                            f'automatically, so results are stable — but such a high reversion speed '
                            f'may not be economically meaningful.</div>',
                            unsafe_allow_html=True
                        )

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
                            '<div class="warn-box">⚠️ κ ≤ 0 — no mean reversion detected. GBM may suit this data better.</div>',
                            unsafe_allow_html=True
                        )
                    apply_label = "✅ Apply OU Parameters"

                st.markdown("")
                _constant   = ou_est.get("_constant_series", False)
                _ou_k_valid = ou_est.get("k", 0) > 0
                _can_apply_ou  = (not _constant) and _ou_k_valid
                _can_apply_gbm = not _constant

                can_apply = _can_apply_gbm if "GBM" in model else _can_apply_ou

                if can_apply:
                    if st.button(apply_label, width='stretch'):
                        if "GBM" in model:
                            st.session_state["param_mu"]    = float(round(gbm_est["mu_ito"], 4))
                            st.session_state["param_sigma"] = float(round(gbm_est["sigma_annual"], 4))
                            st.session_state["applied_from"] = "GBM"
                        else:
                            st.session_state["param_kappa"] = float(round(ou_est["k"], 4))
                            st.session_state["param_theta"] = float(round(ou_est["theta"], 2))
                            st.session_state["param_sigma"] = float(round(ou_est["sigma_ou"], 4))
                            st.session_state["applied_from"] = "OU"
                        st.session_state["params_applied"] = True
                        st.session_state["sim_ran"] = False
                        st.rerun()
                elif "Mean-Reverting" in model and not _ou_k_valid and not _constant:
                    st.markdown(
                        '<div class="warn-box">Apply blocked: κ ≤ 0 means no mean reversion. '
                        'Switch to GBM or use a dataset that shows reversion.</div>',
                        unsafe_allow_html=True
                    )

            else:
                st.warning("Need at least 10 data points to estimate parameters.")

    st.markdown("---")

    st.markdown("### Model Parameters")
    if st.session_state["params_applied"]:
        st.markdown(
            f'<span class="applied-pill">✔ Parameters loaded from {st.session_state["applied_from"]} estimation</span>',
            unsafe_allow_html=True
        )
        if st.button("↩ Reset to defaults", width='content'):
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v
            st.session_state["sim_ran"] = False
            st.rerun()

    if "GBM" in model:
        mu    = st.number_input(
            "Annual drift μ", value=float(st.session_state["param_mu"]),
            step=0.001, format="%.4f",
        )
        sigma = st.number_input(
            "Annual volatility σ", min_value=0.001, max_value=5.0,
            value=float(st.session_state["param_sigma"]),
            step=0.001, format="%.4f",
            help="Max 5.0 (500%). Values above ~1.0 produce very wide distributions."
        )
        if mu != st.session_state["param_mu"] or sigma != st.session_state["param_sigma"]:
            st.session_state["sim_ran"] = False
        st.session_state["param_mu"]    = mu
        st.session_state["param_sigma"] = sigma
    else:
        kappa = st.number_input(
            "Mean-reversion speed κ", min_value=0.001, max_value=100.0,
            value=float(st.session_state["param_kappa"]),
            step=0.01, format="%.4f",
            help="κ=0.001 → half-life ≈ 693 yrs (no reversion). Typical commodity values: 0.3–3.0."
        )
        if kappa < 0.05:
            hl_warn = np.log(2) / kappa
            st.markdown(
                f'<div class="warn-box">⚠️ κ={kappa:.4f} implies a half-life of '
                f'<b>{hl_warn:.1f} years</b> — effectively no mean reversion at this horizon. '
                f'Consider using GBM instead.</div>',
                unsafe_allow_html=True
            )

        # FIX (Bug 4): Warn user when κ·dt is large (though exact solution handles it)
        _dt_sim = dt_value("Weekly")   # simulation always runs at weekly granularity internally
        if kappa * _dt_sim > 0.5:
            st.markdown(
                f'<div class="warn-box">ℹ️ High κ·dt = {kappa * _dt_sim:.2f}. '
                f'The exact OU solver is used, so simulation is numerically stable. '
                f'However, κ={kappa:.2f} means mean reversion completes in '
                f'~{np.log(2)/kappa:.2f} years — verify this is realistic.</div>',
                unsafe_allow_html=True
            )

        theta = st.number_input(
            "Long-run mean θ (₱/Lkg)", min_value=0.01, value=float(st.session_state["param_theta"]),
            step=50.0,
        )
        sigma = st.number_input(
            "Annual volatility σ", min_value=0.001, max_value=5.0,
            value=float(st.session_state["param_sigma"]),
            step=0.001, format="%.4f",
            help="Max 5.0 (500%). Values above ~1.0 produce very wide distributions."
        )
        if (kappa != st.session_state["param_kappa"] or
                theta != st.session_state["param_theta"] or
                sigma != st.session_state["param_sigma"]):
            st.session_state["sim_ran"] = False
        st.session_state["param_kappa"] = kappa
        st.session_state["param_theta"] = theta
        st.session_state["param_sigma"] = sigma

    st.markdown("---")

    st.markdown("### Risk & Volume")
    breakeven = st.number_input("Break-even / alert price (₱/Lkg)", min_value=0.0, value=2000.0, step=50.0)
    volume    = st.number_input("Annual volume (Lkg, 0 = ignore)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    st.markdown("### Simulation Settings")

    # FIX (Bug 3): Enforce N_sim ≥ 1000 to ensure ES/CVaR has enough tail samples.
    N_sim = st.number_input(
        "Terminal simulations (N)", min_value=1000, max_value=100_000,
        value=5000, step=1000,
        help="Minimum 1,000 required for reliable Expected Shortfall (ES/CVaR). Capped at 100,000."
    )
    K     = st.number_input("Sample paths to display", min_value=1, value=30, step=5)
    seed  = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.markdown("---")
    st.markdown("### 📅 Weekly Prediction Settings")
    weekly_n_weeks = st.number_input(
        "Weeks to forecast", min_value=4, max_value=104, value=26, step=4,
    )
    weekly_display  = st.selectbox("Bar shows", ["Median (P50)", "Mean"])
    weekly_interval = st.selectbox("Confidence interval", ["P05–P95 (90%)", "P25–P75 (50%)"])

    run = st.button("▶  Run Simulation", width='stretch')


# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="page-title">🍬 Sugar Pricing Forecasting </div>
<div class="page-subtitle">Montecarlo Risk Model | Price Prediction</div>
''', unsafe_allow_html=True)
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

tab_est, tab_sim, tab_weekly, tab_saved = st.tabs([
    "📊 Parameter Estimator",
    "🎲 Monte Carlo Simulation",
    "📅 Weekly Price Prediction",
    "💾 Saved Runs",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Parameter Estimator
# FIX (Bug 1): Replaced all st.stop() calls inside this tab with a flag-based
# early-exit pattern. st.stop() halts the entire Streamlit script, which
# prevents all subsequent tabs from rendering. Instead, we use an
# `est_data_ok` boolean gate: content only renders inside `if est_data_ok:`
# blocks, so control always falls through to Tab 2, 3, and 4.
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
        st.dataframe(sample, width='stretch')
    else:
        _raw_prices_est = df_raw[price_col].values
        _raw_dates_est  = None
        if date_col and date_col != "None":
            try:
                _raw_dates_est = pd.to_datetime(df_raw[date_col]).values
            except Exception:
                _raw_dates_est = None

        prices_est, dates_est_clean, n_dropped_est = clean_price_series(_raw_prices_est, _raw_dates_est)

        if n_dropped_est > 0:
            st.markdown(
                f'<div class="warn-box">⚠️ {n_dropped_est} invalid price(s) removed '
                f'(zero, negative, or non-finite) before estimation.</div>',
                unsafe_allow_html=True
            )

        # FIX (Bug 1): Guard with a flag instead of st.stop().
        # All content below is wrapped in `if est_data_ok:` so the tab renders
        # an informative error but does NOT halt the script for other tabs.
        est_data_ok = len(prices_est) >= 10

        if not est_data_ok:
            st.error(
                f"⚠️ Only {len(prices_est)} valid data point(s) found — at least 10 are required "
                f"to estimate model parameters. Please upload a CSV with more rows, or check that "
                f"the correct price column is selected and that values are positive numbers."
            )
            # Tab content stops here; script continues to Tab 2 / 3 / 4.
        else:
            if dates_est_clean is not None:
                dates_est = dates_est_clean
            else:
                dates_est = np.arange(len(prices_est))

            min_recommended = {"Daily": 500, "Weekly": 104, "Monthly": 36, "Yearly": 5}
            min_rec = min_recommended[est_freq]
            if len(prices_est) < min_rec:
                st.warning(
                    f"⚠️ Only {len(prices_est)} observations detected. "
                    f"For {est_freq.lower()} data, at least {min_rec} rows are recommended."
                )

            gbm = compute_gbm_params(prices_est, est_freq)
            ou  = compute_ou_params(prices_est, est_freq)
            N_ann = annualization_factor(est_freq)
            dt    = dt_value(est_freq)

            lr_est = gbm["log_returns"]
            if len(lr_est) > 0:
                z_scores_est = np.abs((lr_est - np.mean(lr_est)) / (np.std(lr_est) + 1e-12))
                n_out_est = int(np.sum(z_scores_est > 4))
                if n_out_est > 0:
                    st.markdown(
                        f'<div class="warn-box">⚠️ {n_out_est} extreme return(s) detected '
                        f'(|z| &gt; 4). These may inflate σ. Verify your source data.</div>',
                        unsafe_allow_html=True
                    )

            if ou.get("_constant_series"):
                st.markdown(
                    '<div class="warn-box">⚠️ <b>Constant price series detected.</b> All prices are identical, '
                    'so OU parameters (κ, θ, σ) cannot be estimated. GBM parameters are shown but σ = 0. '
                    'Please upload data with price variation.</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="est-section-header">📊 Price History</div>', unsafe_allow_html=True)
            col_prev, col_chart = st.columns([1, 2])

            with col_prev:
                st.markdown(f"**{len(prices_est)} observations** · {est_freq} · `{price_col}`")
                st.dataframe(df_raw[[price_col]].head(10), width='stretch')

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
            g1.metric("Annual Drift μ",        f"{gbm['mu_annual']*100:.2f}%")
            g2.metric("Itô-Corrected Drift",   f"{gbm['mu_ito']*100:.2f}%")
            g3.metric("Annual Volatility σ",   f"{gbm['sigma_annual']*100:.2f}%")
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

            if ou.get("_constant_series"):
                st.markdown(
                    '<div class="warn-box">⚠️ Cannot display OU parameters — constant price series.</div>',
                    unsafe_allow_html=True
                )
            elif ou["k"] <= 0:
                st.markdown(
                    '<div class="warn-box">⚠️ k ≤ 0 — prices are NOT mean-reverting in this dataset. GBM may be more appropriate.</div>',
                    unsafe_allow_html=True
                )

            if not ou.get("_constant_series"):
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Mean Reversion Speed κ", f"{ou['k']:.4f}")
                o2.metric("Long-Run Mean θ",        f"{ou['theta']:,.2f}")
                o3.metric("OU Volatility σ",        f"{ou['sigma_ou']*100:.2f}%")
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
                        st.markdown('<div class="warn-box">⚠️ p-value > 0.05 — mean reversion is not statistically significant.</div>', unsafe_allow_html=True)
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

            hl_y_str  = f"{ou['half_life_years']:.4f}" if not np.isnan(ou['half_life_years']) else "N/A"
            hl_p_str  = f"{ou['half_life_periods']:.2f}" if not np.isnan(ou.get('half_life_periods', np.nan)) else "N/A"
            summary = pd.DataFrame({
                "Parameter": [
                    "Annual Drift μ (GBM)", "Itô-Corrected Drift (GBM input)", "Annual Volatility σ (GBM)",
                    f"{est_freq} Volatility σ (GBM)", "Mean Reversion Speed κ (OU)", "Long-Run Mean θ (OU)",
                    "OU Volatility σ (annualised)", "Half-Life (years)", f"Half-Life ({est_freq.lower()} periods)",
                ],
                "Value": [
                    f"{gbm['mu_annual']*100:.4f}%", f"{gbm['mu_ito']*100:.4f}%",
                    f"{gbm['sigma_annual']*100:.4f}%", f"{gbm['sigma_period']*100:.4f}%",
                    f"{ou['k']:.4f}" if not np.isnan(ou['k']) else "N/A",
                    f"{ou['theta']:,.4f}",
                    f"{ou['sigma_ou']*100:.4f}%",
                    hl_y_str, hl_p_str,
                ],
                "Use In": [
                    "GBM (raw drift)", "GBM (recommended input)", "GBM (annual steps)",
                    f"GBM ({est_freq.lower()} steps)", "OU simulation", "OU simulation",
                    "OU simulation", "Interpretation", "Interpretation",
                ]
            })
            st.dataframe(summary, width='stretch', hide_index=True)

            dl_col, apply_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "⬇️ Download Summary CSV",
                    data=summary.to_csv(index=False),
                    file_name="sugar_model_parameters.csv",
                    mime="text/csv",
                    width='stretch'
                )
            with apply_col:
                apply_target = "GBM" if "GBM" in model else "OU"
                _ou_k_valid_tab = ou.get("k", 0) > 0
                _constant_tab   = ou.get("_constant_series", False)
                can_apply_tab = (not _constant_tab) and (_ou_k_valid_tab if "Mean-Reverting" in model else True)
                if can_apply_tab:
                    if st.button(f"✅ Apply {apply_target} Parameters to Simulation →", width='stretch'):
                        if "GBM" in model:
                            st.session_state["param_mu"]    = float(round(gbm["mu_ito"], 4))
                            st.session_state["param_sigma"] = float(round(gbm["sigma_annual"], 4))
                            st.session_state["applied_from"] = "GBM"
                        else:
                            st.session_state["param_kappa"] = float(round(ou["k"], 4))
                            st.session_state["param_theta"] = float(round(ou["theta"], 2))
                            st.session_state["param_sigma"] = float(round(ou["sigma_ou"], 4))
                            st.session_state["applied_from"] = "OU"
                        st.session_state["params_applied"] = True
                        st.session_state["sim_ran"] = False
                        st.rerun()
                else:
                    st.markdown(
                        f'<div class="warn-box">Apply {apply_target} blocked: '
                        + ("κ ≤ 0 — no reversion detected." if not _ou_k_valid_tab else "Constant series.")
                        + '</div>',
                        unsafe_allow_html=True
                    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monte Carlo Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    if not run:
        st.info("👈  Configure the sidebar and click **Run Simulation** to generate results.", icon="💡")
    else:
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
        p05_v    = float(np.percentile(terminal, 5))
        p25_v    = float(np.percentile(terminal, 25))
        p75_v    = float(np.percentile(terminal, 75))
        p95_v    = float(np.percentile(terminal, 95))
        var95_v  = S0 - p05_v

        # FIX (Bug 3): Distinguish between a well-estimated ES and an unreliable one.
        # With N_sim ≥ 1000 (enforced in sidebar), the 5% tail has at least ~50 samples.
        # We still check explicitly and flag any edge case with a clear "unreliable" label.
        es_vals = terminal[terminal <= p05_v]
        _ES_MIN_SAMPLES = 30   # below this the ES estimate is meaningless
        if len(es_vals) >= _ES_MIN_SAMPLES:
            es95_v      = float(np.mean(es_vals))
            es95_label  = f"₱{es95_v:,.0f}"
            es95_delta  = None
            es95_reliable = True
        else:
            # Extremely unlikely with N_sim ≥ 1000 but guard it anyway
            es95_v      = p05_v
            es95_label  = "N/A"
            es95_delta  = f"< {_ES_MIN_SAMPLES} tail samples ({len(es_vals)} found)"
            es95_reliable = False

        prob_be_v = float(np.mean(terminal <= breakeven))

        st.session_state["last_sim_results"] = {
            "mean_p": mean_p, "median_p": median_p, "std_p": std_p,
            "p05": p05_v, "p25": p25_v, "p75": p75_v, "p95": p95_v,
            "var95": var95_v,
            "es95": es95_v if es95_reliable else None,
            "prob_be": prob_be_v,
        }
        st.session_state["last_sim_params"] = {
            "model": model, "S0": S0, "horizon_label": horizon_label,
            "T": T, "N_sim": int(N_sim), "seed": int(seed),
            "breakeven": breakeven, "volume": volume,
            **({"mu": mu, "sigma": sigma} if "GBM" in model else
               {"kappa": kappa, "theta": theta, "sigma": sigma}),
        }
        st.session_state["last_sim_terminal"] = terminal
        st.session_state["last_sim_times"]    = times
        st.session_state["last_sim_paths"]    = paths
        st.session_state["last_es_reliable"]  = es95_reliable
        st.session_state["last_es_label"]     = es95_label
        st.session_state["last_es_delta"]     = es95_delta
        st.session_state["sim_ran"] = True

    if st.session_state.get("sim_ran"):
        terminal = st.session_state["last_sim_terminal"]
        times    = st.session_state["last_sim_times"]
        paths    = st.session_state["last_sim_paths"]
        _r       = st.session_state["last_sim_results"]
        mean_p   = _r["mean_p"]
        median_p = _r["median_p"]
        std_p    = _r["std_p"]
        p05      = _r["p05"]
        p25      = _r["p25"]
        p75      = _r["p75"]
        p95      = _r["p95"]
        var95    = _r["var95"]
        es95     = _r.get("es95") or p05   # fallback for display only
        prob_be  = _r["prob_be"]
        rev_risk = var95 * volume if volume > 0 else None

        # Restore ES display labels from session state
        es95_reliable = st.session_state.get("last_es_reliable", True)
        es95_label    = st.session_state.get("last_es_label", f"₱{es95:,.0f}")
        es95_delta    = st.session_state.get("last_es_delta", None)

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

        st.markdown("")
        save_col, _ = st.columns([1, 3])
        with save_col:
            if st.button("💾 Save This Run", width='stretch'):
                if not SUPABASE_OK:
                    st.warning("Supabase not configured — cannot save.")
                elif not st.session_state.get("sim_ran"):
                    st.warning("Run a simulation first before saving.")
                else:
                    ok = save_simulation(
                        _user.id,
                        st.session_state["last_sim_params"],
                        st.session_state["last_sim_results"],
                        token=st.session_state.get("access_token"),
                        refresh=st.session_state.get("refresh_token"),
                    )
                    if ok:
                        st.success("✅ Simulation saved! View it in the 💾 Saved Runs tab.")

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
            st.plotly_chart(fig, width='stretch')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P05 (worst 5%)",  f"₱{p05:,.0f}",  f"{(p05/S0-1)*100:+.1f}% vs spot")
            c2.metric("P25",             f"₱{p25:,.0f}",  f"{(p25/S0-1)*100:+.1f}% vs spot")
            c3.metric("P75",             f"₱{p75:,.0f}",  f"{(p75/S0-1)*100:+.1f}% vs spot")
            c4.metric("P95 (best 5%)",   f"₱{p95:,.0f}",  f"{(p95/S0-1)*100:+.1f}% vs spot")

            # FIX (Bug 3): Show ES/CVaR with an explicit reliability indicator.
            # If there are fewer than _ES_MIN_SAMPLES tail observations, the metric
            # shows "N/A" with the sample count so the user understands why.
            if es95_reliable:
                st.caption(
                    f"Expected Shortfall / CVaR (avg price when ≤ P05): "
                    f"**₱{es95:,.0f}/Lkg**  —  Based on {int(N_sim):,} simulations "
                    f"({len(terminal[terminal <= p05])} tail samples)."
                )
            else:
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>Expected Shortfall (ES/CVaR): N/A</b> — '
                    f'only {len(terminal[terminal <= p05])} tail samples below P05 '
                    f'(minimum {_ES_MIN_SAMPLES} needed for a reliable estimate). '
                    f'Increase N or reduce the spot price relative to break-even.</div>',
                    unsafe_allow_html=True
                )

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
            st.plotly_chart(fig2, width='stretch')
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
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=520)
            st.caption(f"Spot: ₱{S0:,.0f}/Lkg | Break-even: ₱{breakeven:,.0f}/Lkg | Model: {model}")
    else:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Weekly Price Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab_weekly:
    _sim_has_run = st.session_state.get("sim_ran", False)

    if not _sim_has_run and not run:
        st.info(
            "👈  Configure the sidebar and click **Run Simulation** to generate the weekly prediction chart.",
            icon="💡"
        )
    else:
        if "GBM" in model:
            _wdf_key = ("GBM", S0, mu, sigma, int(weekly_n_weeks), int(N_sim), int(seed))
        else:
            _wdf_key = ("OU", S0, kappa, theta, sigma, int(weekly_n_weeks), int(N_sim), int(seed))

        _need_recompute = (
            run
            or st.session_state.get("wdf") is None
            or st.session_state.get("wdf_cache_key") != _wdf_key
        )

        if _need_recompute:
            with st.spinner("Computing week-by-week predictions…"):
                n_weeks_int = int(weekly_n_weeks)
                N_sim_int   = int(N_sim)
                if "GBM" in model:
                    wdf = run_weekly_gbm(S0, mu, sigma, n_weeks_int, N_sim_int, seed + 99)
                else:
                    wdf = run_weekly_ou(S0, kappa, theta, sigma, n_weeks_int, N_sim_int, seed + 99)
                st.session_state["wdf"]           = wdf
                st.session_state["wdf_cache_key"] = _wdf_key
        else:
            wdf = st.session_state["wdf"]
            n_weeks_int = int(weekly_n_weeks)
            N_sim_int   = int(N_sim)

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
        mid_idx = min(n_weeks_int // 2, n_weeks_int - 1)
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
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font_color=TEXT_CLR,
            font_family="DM Sans, sans-serif",
            xaxis=dict(
                title="Week from Today",
                gridcolor=GRID_CLR,
                zeroline=False,
                tickmode="linear",
                tick0=1,
                dtick=max(1, n_weeks_int // 13),
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="Predicted Price (₱/Lkg)",
                gridcolor=GRID_CLR,
                zeroline=False,
                tickprefix="₱",
                tickformat=",",
            ),
            legend=dict(
                bgcolor=DARK_BG,
                bordercolor=GRID_CLR,
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            bargap=0.25,
            margin=dict(t=60, b=60, l=70, r=30),
            height=500,
            title=dict(
                text=f"Week-by-Week {bar_label} Sugar Price Forecast · {n_weeks_int} Weeks · {model}",
                font=dict(family="DM Serif Display, serif", size=16, color="#8ab4cc"),
                x=0.0,
                xanchor="left",
            ),
        )
        st.plotly_chart(fig_w, width='stretch')

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
            st.dataframe(tbl_df, width='stretch', hide_index=True, height=400)
            st.download_button(
                "⬇️ Download Weekly Forecast CSV",
                data=tbl_df.to_csv(index=False),
                file_name="sugar_weekly_forecast.csv",
                mime="text/csv",
                width='content',
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Saved Runs
# ════════════════════════════════════════════════════════════════════════════════
with tab_saved:
    st.markdown('<div class="section-header">Your Saved Simulation Runs</div>', unsafe_allow_html=True)

    if not SUPABASE_OK:
        st.warning("Supabase is not configured. Add your credentials to `.streamlit/secrets.toml` to enable saving.")
    else:
        if st.button("🔄 Refresh", key="refresh_saved"):
            st.rerun()

        runs = load_simulations(
            _user.id,
            token=st.session_state.get("access_token"),
            refresh=st.session_state.get("refresh_token"),
        )

        if not runs:
            st.info("No saved runs yet. Run a simulation and click **💾 Save This Run** to save it here.")
        else:
            st.caption(
                f"{len(runs)} saved run{'s' if len(runs) != 1 else ''} shown "
                f"(most recent 50) · {_user.email}"
            )

            for i, run_row in enumerate(runs):
                try:
                    p = json.loads(run_row.get("params", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    p = {}
                try:
                    r = json.loads(run_row.get("results", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    r = {}

                created = (run_row.get("created_at", "") or "")[:19].replace("T", " ")

                with st.expander(
                    f"🕒 {created}  ·  {run_row.get('model','?')}  ·  "
                    f"Spot ₱{run_row.get('spot_price') or 0:,.0f}  ·  Horizon {run_row.get('horizon','?')}",
                    expanded=(i == 0),
                ):
                    if not p and not r:
                        st.warning("⚠️ This saved run has corrupted data and cannot be displayed.")
                        continue

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Mean Price",   f"₱{r.get('mean_p', 0):,.0f}")
                    rc2.metric("Median Price", f"₱{r.get('median_p', 0):,.0f}")
                    rc3.metric("VaR 95%",      f"₱{r.get('var95', 0):,.0f}")
                    rc4.metric("P(≤ Break-even)", f"{r.get('prob_be', 0)*100:.1f}%")

                    st.markdown(
                        f"**Model:** {p.get('model','?')}  |  "
                        f"**Simulations:** {p.get('N_sim', '?'):,}  |  "
                        f"**Break-even:** ₱{p.get('breakeven', 0):,.0f}/Lkg  |  "
                        f"**Seed:** {p.get('seed','?')}"
                    )

                    if "GBM" in str(p.get("model", "")):
                        st.markdown(f"μ = `{p.get('mu', '?')}` · σ = `{p.get('sigma', '?')}`")
                    else:
                        st.markdown(f"κ = `{p.get('kappa','?')}` · θ = `₱{p.get('theta',0):,.0f}` · σ = `{p.get('sigma','?')}`")

                    # FIX (Bug 3): Show ES as N/A in saved runs when it wasn't computed reliably.
                    es_saved = r.get("es95")
                    es_display = f"₱{es_saved:,.0f}" if es_saved is not None else "N/A"

                    pct_tbl = pd.DataFrame({
                        "Metric": ["P05", "P25", "Median", "Mean", "P75", "P95", "ES95 (CVaR)"],
                        "Price (₱)": [
                            f"₱{r.get('p05',0):,.0f}", f"₱{r.get('p25',0):,.0f}",
                            f"₱{r.get('median_p',0):,.0f}", f"₱{r.get('mean_p',0):,.0f}",
                            f"₱{r.get('p75',0):,.0f}", f"₱{r.get('p95',0):,.0f}",
                            es_display,
                        ]
                    })
                    st.dataframe(pct_tbl, width='stretch', hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#ffffff; font-size:0.8rem;">'
    "Mill-gate raw sugar price model. "
    "GBM assumes lognormally distributed returns. "
    "Mean-Reverting uses an Ornstein–Uhlenbeck process on log-prices (exact solution). "
    "Results are probabilistic estimates, not forecasts."
    "</p>",
    unsafe_allow_html=True,
)
