"""
Sugar Price Monte Carlo Risk Model — Neon (PostgreSQL) backend
Replaces Supabase with Neon free-tier Postgres + simple password auth.

Run with: streamlit run sugar_app_neon.py
Install:  pip install streamlit plotly numpy scipy matplotlib pandas psycopg2-binary bcrypt
Neon setup: https://neon.tech  (free, never pauses)

secrets.toml:
  [neon]
  dsn = "postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require"
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import stats
import json
import hashlib
import os
from datetime import datetime

# ── DB Client ──────────────────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _NEON_DSN = "postgresql://neondb_owner:npg_mDTNIvgV58Qz@ep-little-wind-aq9fvk28-pooler.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    DB_OK = True
except Exception:
    DB_OK = False

def get_conn():
    return psycopg2.connect(_NEON_DSN)

def init_db():
    """Create tables if they don't exist yet."""
    if not DB_OK:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        pw_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS simulation_runs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        model TEXT,
                        spot_price NUMERIC,
                        horizon TEXT,
                        params TEXT,
                        results TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
            conn.commit()
    except Exception as e:
        st.warning(f"DB init error: {e}")


# ── Auth Helpers ───────────────────────────────────────────────────────────────
def _hash_pw(password: str) -> str:
    """SHA-256 hash. For production swap with bcrypt."""
    return hashlib.sha256(password.encode()).hexdigest()

def db_signup(email: str, password: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, pw_hash) VALUES (%s, %s) RETURNING id, email",
                (email.lower().strip(), _hash_pw(password))
            )
            row = cur.fetchone()
        conn.commit()
    return {"id": row[0], "email": row[1]}

def db_login(email: str, password: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, pw_hash FROM users WHERE email = %s",
                (email.lower().strip(),)
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError("No account found with that email.")
    if row[2] != _hash_pw(password):
        raise ValueError("Incorrect password.")
    return {"id": row[0], "email": row[1]}

def save_simulation(user_id: int, params: dict, results: dict) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO simulation_runs
                        (user_id, model, spot_price, horizon, params, results)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    params.get("model"),
                    params.get("S0"),
                    params.get("horizon_label"),
                    json.dumps(params),
                    json.dumps(results),
                ))
            conn.commit()
        return True
    except Exception as e:
        st.warning(f"Could not save simulation: {e}")
        return False

def load_simulations(user_id: int):
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM simulation_runs
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 50
                """, (user_id,))
                return cur.fetchall()
    except Exception as e:
        st.warning(f"Could not load simulations: {e}")
        return []

def do_logout():
    st.session_state["user"] = None


# ── Auth Page ──────────────────────────────────────────────────────────────────
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
      }
      .stTextInput label { color: #6a8f72 !important; font-size: 0.75rem !important;
        letter-spacing: 0.08em !important; text-transform: uppercase !important; }
      .stTabs [data-baseweb="tab-list"] { background: rgba(10,20,14,0.7) !important;
        border-radius: 10px !important; border: 1px solid rgba(52,120,70,0.25) !important; }
      .stTabs [data-baseweb="tab"] { color: #4a6b52 !important; }
      .stTabs [aria-selected="true"] { background: rgba(52,120,70,0.3) !important; color: #a8d4b0 !important; }
      .stButton > button {
        background: linear-gradient(135deg, #1e4d2a 0%, #2d7040 100%) !important;
        color: #d4f5dc !important; font-weight: 600 !important; border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important; width: 100% !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="min-height:30vh; background:radial-gradient(ellipse 80% 60% at 20% 80%,
      rgba(34,85,47,0.4) 0%,transparent 60%), #0a0f0d;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
      padding:3rem 1rem 2rem; text-align:center;">
      <div style="font-size:4rem; margin-bottom:1rem;">🎋</div>
      <div style="font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:700;
           color:#e8dcc8; margin-bottom:0.4rem;">Monte Carlo Risk Model</div>
      <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#3a6b45;
           letter-spacing:0.2em; text-transform:uppercase;">Sugar Price Prediction</div>
      <div style="width:60px; height:2px; background:linear-gradient(90deg,transparent,#d4a843,transparent); margin:0.8rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div style="background:rgba(15,25,18,0.97); border:1px solid rgba(52,120,70,0.3); border-radius:20px; padding:2rem;">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            email    = st.text_input("Email address", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            if st.button("Sign In →", key="btn_login"):
                if not DB_OK:
                    st.error("Database not configured. Add [neon] dsn to .streamlit/secrets.toml")
                elif not email or not password:
                    st.warning("Please enter email and password.")
                else:
                    try:
                        user = db_login(email, password)
                        st.session_state["user"] = user
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with tab_signup:
            email2 = st.text_input("Email address", key="signup_email", placeholder="you@example.com")
            pw2    = st.text_input("Password", type="password", key="signup_pw", placeholder="Min. 6 characters")
            pw3    = st.text_input("Confirm password", type="password", key="signup_pw2", placeholder="••••••••")
            if st.button("Create Account →", key="btn_signup"):
                if not DB_OK:
                    st.error("Database not configured.")
                elif not email2 or not pw2:
                    st.warning("Please fill in all fields.")
                elif pw2 != pw3:
                    st.error("Passwords do not match.")
                elif len(pw2) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        db_signup(email2, pw2)
                        st.success("✅ Account created! Sign in now.")
                    except Exception as e:
                        msg = str(e)
                        if "unique" in msg.lower() or "duplicate" in msg.lower():
                            st.error("An account with this email already exists.")
                        else:
                            st.error(f"Sign-up failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; margin-top:1rem; font-size:0.7rem; color:#3a6b45;">Mill-gate Sugar · Philippines · Probabilistic estimates only</div>', unsafe_allow_html=True)


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sugar Pricing Forecasting",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    if DB_OK:
        init_db()

    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None:
        render_auth_page()
        st.stop()

    _user = st.session_state["user"]

    # ── Main App CSS ───────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
      html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: #080e0b !important; color: #d4c9b4; }
      .stApp { background: #080e0b !important; }
      .main .block-container { padding-top: 1.5rem; background: #080e0b; }
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1410 0%, #080e0b 100%);
        border-right: 1px solid rgba(52,120,70,0.2);
      }
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p:not(button p) {
        color: #6a8f72 !important; font-size: 0.75rem; font-weight: 500;
        letter-spacing: 0.07em; text-transform: uppercase; font-family: 'Space Mono', monospace !important;
      }
      div[data-testid="metric-container"] {
        background: rgba(12,22,16,0.9); border: 1px solid rgba(52,120,70,0.25);
        border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
      }
      div[data-testid="metric-container"] label { color: #4a6b52 !important; font-size: 0.7rem;
        letter-spacing: 0.1em; text-transform: uppercase; font-family: 'Space Mono', monospace; }
      div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e8dcc8; font-size: 1.55rem; font-weight: 600; font-family: 'Playfair Display', serif; }
      .section-header {
        font-family: 'Playfair Display', serif; font-size: 1.05rem; font-style: italic;
        color: #6aab7a; border-bottom: 1px solid rgba(52,120,70,0.25);
        padding-bottom: 0.4rem; margin: 1.8rem 0 1.1rem 0;
      }
      .info-pill {
        display: inline-block; background: rgba(12,30,18,0.9); color: #7aab86;
        border: 1px solid rgba(52,120,70,0.3); border-radius: 20px;
        padding: 3px 14px; font-size: 0.72rem; margin: 2px 3px;
        font-family: 'Space Mono', monospace;
      }
      .applied-pill {
        display: inline-block; background: rgba(18,45,30,0.9); color: #52c87a;
        border: 1px solid rgba(52,200,80,0.3); border-radius: 20px;
        padding: 3px 14px; font-size: 0.72rem; margin: 2px 3px;
      }
      .alert-danger { background: rgba(45,26,26,0.8); border-left: 3px solid #e05252;
        border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.85rem; color: #f5a0a0; }
      .alert-safe { background: rgba(18,45,30,0.8); border-left: 3px solid #52c87a;
        border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.85rem; color: #8de8a8; }
      .info-box { background: rgba(12,22,30,0.9); border-left: 3px solid #3b82f6;
        border-radius: 6px; padding: 12px 16px; font-size: 13px; color: #7a9aaf; margin-bottom: 16px; }
      .warn-box { background: rgba(25,18,8,0.9); border-left: 3px solid #f59e0b;
        border-radius: 6px; padding: 12px 16px; font-size: 13px; color: #c4935a; margin-bottom: 16px; }
      .stButton > button {
        background: linear-gradient(135deg, #1a4226 0%, #256336 100%);
        color: #c8f0d0 !important; font-weight: 600; border: 1px solid rgba(52,160,70,0.35);
        border-radius: 10px; font-size: 0.88rem; width: 100%; transition: all 0.2s;
      }
      .signout-btn > button {
        background: linear-gradient(135deg, #6b4800 0%, #a06c00 100%) !important;
        color: #fff3cc !important; border: 1px solid rgba(212,168,67,0.4) !important;
      }
      .stTabs [data-baseweb="tab-list"] { background: rgba(10,20,14,0.6); border-radius: 10px;
        padding: 4px; border: 1px solid rgba(52,120,70,0.2); }
      .stTabs [data-baseweb="tab"] { color: #3a6b45; font-weight: 500; }
      .stTabs [aria-selected="true"] { background: rgba(52,120,70,0.25) !important; color: #8fd4a0 !important; }
      .page-title { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700;
        color: #e8dcc8; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
      .page-subtitle { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #3a6b45;
        letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def annualization_factor(freq):
        return {"Daily": 252, "Weekly": 52, "Monthly": 12, "Yearly": 1}[freq]

    def dt_value(freq):
        return {"Daily": 1/252, "Weekly": 1/52, "Monthly": 1/12, "Yearly": 1.0}[freq]

    def clean_price_series(prices, dates=None):
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

    def compute_gbm_params(prices, freq):
        N = annualization_factor(freq)
        lr = np.diff(np.log(prices))
        mu_p = np.mean(lr); sig_p = np.std(lr, ddof=1)
        mu_a = mu_p * N; sig_a = sig_p * np.sqrt(N)
        return {"log_returns": lr, "mu_period": mu_p, "sigma_period": sig_p,
                "mu_annual": mu_a, "sigma_annual": sig_a, "mu_ito": mu_a - 0.5*sig_a**2,
                "n_obs": len(lr)}

    def compute_ou_params(prices, freq):
        dt = dt_value(freq); N = annualization_factor(freq)
        lp = np.log(prices); dlp = np.diff(lp); lp_lag = lp[:-1]
        if np.std(lp_lag) < 1e-12 or np.std(dlp) < 1e-12:
            return {"k": np.nan, "theta": float(np.exp(np.mean(lp))), "sigma_ou": 0.0,
                    "half_life_years": np.nan, "half_life_periods": np.nan, "r_squared": 0.0,
                    "p_value": 1.0, "residuals": np.zeros(len(dlp)), "beta": 0.0, "alpha": 0.0,
                    "dP": dlp, "P_lag": lp_lag, "_constant_series": True}
        slope, intercept, r_val, p_val, _ = stats.linregress(lp_lag, dlp)
        k = -slope / dt if abs(slope) > 1e-12 else 0.0
        theta = float(np.exp(-intercept / slope)) if abs(slope) > 1e-12 else float(np.exp(np.mean(lp)))
        residuals = dlp - (intercept + slope * lp_lag)
        sigma_ou = np.std(residuals, ddof=2) / np.sqrt(dt)
        hl_y = np.log(2) / k if k > 0 else np.nan
        hl_p = hl_y * N if k > 0 else np.nan
        return {"k": k, "theta": theta, "sigma_ou": sigma_ou,
                "half_life_years": hl_y, "half_life_periods": hl_p,
                "r_squared": r_val**2, "p_value": p_val, "residuals": residuals,
                "beta": slope, "alpha": intercept, "dP": dlp, "P_lag": lp_lag,
                "_constant_series": False}

    # ── Simulation Engine ──────────────────────────────────────────────────────
    def run_gbm_terminal(S0, mu, sigma, T, N, seed):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(N)
        return S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    def run_gbm_paths(S0, mu, sigma, T, steps_per_year, K, seed):
        rng = np.random.default_rng(seed)
        steps = max(1, int(T*steps_per_year)); dt = T/steps
        paths = np.zeros((steps+1, K)); paths[0] = S0
        for t in range(1, steps+1):
            Z = rng.standard_normal(K)
            paths[t] = paths[t-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        return np.linspace(0, T, steps+1), paths

    def run_mean_revert_terminal(S0, kappa, theta, sigma, T, N, steps_per_year, seed):
        rng = np.random.default_rng(seed)
        steps = max(1, int(T*steps_per_year)); dt = T/steps
        lta = np.log(theta) - sigma**2/(2*kappa)
        decay = np.exp(-kappa*dt)
        ns = sigma * np.sqrt((1.0 - np.exp(-2.0*kappa*dt))/(2.0*kappa))
        lnS = np.full(N, np.log(S0), dtype=np.float64)
        for _ in range(steps):
            Z = rng.standard_normal(N)
            lnS = lta + (lnS-lta)*decay + ns*Z
            np.clip(lnS, -30, 30, out=lnS)
        return np.exp(lnS)

    def run_mean_revert_paths(S0, kappa, theta, sigma, T, steps_per_year, K, seed):
        rng = np.random.default_rng(seed)
        steps = max(1, int(T*steps_per_year)); dt = T/steps
        lta = np.log(theta) - sigma**2/(2*kappa)
        decay = np.exp(-kappa*dt)
        ns = sigma * np.sqrt((1.0 - np.exp(-2.0*kappa*dt))/(2.0*kappa))
        lnp = np.zeros((steps+1, K)); lnp[0] = np.log(S0)
        for t in range(1, steps+1):
            Z = rng.standard_normal(K)
            lnp[t] = lta + (lnp[t-1]-lta)*decay + ns*Z
            np.clip(lnp[t], -30, 30, out=lnp[t])
        return np.linspace(0, T, steps+1), np.exp(lnp)

    def run_weekly_gbm(S0, mu, sigma, n_weeks, N_sim, seed):
        rng = np.random.default_rng(seed); dt = 1/52
        prices = np.full(N_sim, S0, dtype=float); rows = []
        for w in range(1, n_weeks+1):
            Z = rng.standard_normal(N_sim)
            prices = prices * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
            rows.append({"week": w, "mean": float(np.mean(prices)),
                "median": float(np.median(prices)),
                "p05": float(np.percentile(prices,5)), "p25": float(np.percentile(prices,25)),
                "p75": float(np.percentile(prices,75)), "p95": float(np.percentile(prices,95))})
        return pd.DataFrame(rows)

    def run_weekly_ou(S0, kappa, theta, sigma, n_weeks, N_sim, seed):
        rng = np.random.default_rng(seed); dt = 1/52
        lta = np.log(theta) - sigma**2/(2*kappa)
        decay = np.exp(-kappa*dt)
        ns = sigma * np.sqrt((1.0 - np.exp(-2.0*kappa*dt))/(2.0*kappa))
        lnp = np.full(N_sim, np.log(S0), dtype=float); rows = []
        for w in range(1, n_weeks+1):
            Z = rng.standard_normal(N_sim)
            lnp = lta + (lnp-lta)*decay + ns*Z
            np.clip(lnp, -30, 30, out=lnp)
            prices = np.exp(lnp)
            rows.append({"week": w, "mean": float(np.mean(prices)),
                "median": float(np.median(prices)),
                "p05": float(np.percentile(prices,5)), "p25": float(np.percentile(prices,25)),
                "p75": float(np.percentile(prices,75)), "p95": float(np.percentile(prices,95))})
        return pd.DataFrame(rows)

    _ES_MIN_SAMPLES = 30

    # ── Session defaults ───────────────────────────────────────────────────────
    _defaults = {"param_mu": 0.03, "param_sigma": 0.18, "param_kappa": 0.60,
                 "param_theta": 2400.0, "params_applied": False, "applied_from": None,
                 "wdf": None, "wdf_cache_key": None}
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ────────────────────────────────────────────────────────────────
    DARK_BG = "#0f1923"; GRID_CLR = "#1e2d3d"; TEXT_CLR = "#c9bfac"
    GOLD = "#d4a843"; TEAL = "#4a9fb5"; RED_CLR = "#e05252"
    GREEN_OK = "#52c87a"; AMBER = "#f59e0b"

    with st.sidebar:
        st.markdown('<div style="font-family:\'Playfair Display\',serif; font-size:1.3rem; font-weight:700; color:#ccfa34; margin-bottom:0.2rem;">🎋 Sugar Price<br><span style="font-size:0.9rem;color:#ccfa34;font-family:\'Space Mono\',monospace;font-weight:400;">Monte Carlo Risk Model</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">Signed in as<br><span style="color:#d4a843">{_user["email"]}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="btn_signout"):
            do_logout(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### Model Setup")
        model = st.selectbox("Price model", ["GBM (Lognormal)", "Mean-Reverting"])
        S0 = st.number_input("Current spot price (₱/Lkg)", min_value=100.0, max_value=20000.0, value=2400.0, step=50.0)

        horizon_unit = st.selectbox("Horizon unit", ["Weeks", "Months", "Years"])
        if horizon_unit == "Weeks":
            horizon_val = st.number_input("Forecast horizon (weeks)", min_value=1, value=52, step=1)
            T = horizon_val/52.0; steps_per_year = 52
            horizon_label = f"{horizon_val} week{'s' if horizon_val>1 else ''}"
        elif horizon_unit == "Months":
            horizon_val = st.number_input("Forecast horizon (months)", min_value=1, value=12, step=1)
            T = horizon_val/12.0; steps_per_year = 12
            horizon_label = f"{horizon_val} month{'s' if horizon_val>1 else ''}"
        else:
            horizon_val = st.number_input("Forecast horizon (years)", min_value=1, value=1, step=1)
            T = float(horizon_val); steps_per_year = 52
            horizon_label = f"{horizon_val} year{'s' if horizon_val>1 else ''}"

        st.markdown("---")
        st.markdown("### 📁 Historical Data (optional)")
        est_freq = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly", "Yearly"], index=2)
        uploaded_file = st.file_uploader("CSV with price column", type=["csv"])

        price_col = None; date_col = None; df_raw = None
        gbm_est = None; ou_est = None

        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file)
            numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            all_cols = df_raw.columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
            else:
                price_col = st.selectbox("Price Column", numeric_cols)
                date_col = st.selectbox("Date Column (optional)", ["None"] + all_cols)
                _raw = df_raw[price_col].values
                _dates = None
                if date_col and date_col != "None":
                    try: _dates = pd.to_datetime(df_raw[date_col]).values
                    except: pass
                _prices, _, _nd = clean_price_series(_raw, _dates)
                if _nd > 0:
                    st.markdown(f'<div class="warn-box">⚠️ Removed {_nd} invalid row(s).</div>', unsafe_allow_html=True)
                if len(_prices) >= 10:
                    gbm_est = compute_gbm_params(_prices, est_freq)
                    ou_est  = compute_ou_params(_prices, est_freq)
                    if "GBM" in model:
                        st.markdown(f'<div style="font-size:12px;color:#d4a843;font-family:\'IBM Plex Mono\',monospace;line-height:1.8">μ (Itô) = <b>{gbm_est["mu_ito"]*100:.2f}%</b><br>σ annual = <b style="color:#4a9fb5">{gbm_est["sigma_annual"]*100:.2f}%</b></div>', unsafe_allow_html=True)
                        if st.button("✅ Apply GBM Parameters"):
                            st.session_state["param_mu"] = float(round(gbm_est["mu_ito"], 4))
                            st.session_state["param_sigma"] = float(round(gbm_est["sigma_annual"], 4))
                            st.session_state["params_applied"] = True; st.session_state["applied_from"] = "GBM"
                            st.session_state["sim_ran"] = False; st.rerun()
                    else:
                        k_ok = ou_est.get("k", 0) > 0
                        st.markdown(f'<div style="font-size:12px;color:#c9bfac;font-family:\'IBM Plex Mono\',monospace;line-height:1.8">κ = <b style="color:{"#d4a843" if k_ok else "#e05252"}">{ou_est["k"]:.4f}</b><br>θ = <b style="color:#d4a843">₱{ou_est["theta"]:,.0f}</b><br>σ_ou = <b style="color:#4a9fb5">{ou_est["sigma_ou"]*100:.2f}%</b></div>', unsafe_allow_html=True)
                        if k_ok and st.button("✅ Apply OU Parameters"):
                            st.session_state["param_kappa"] = float(round(ou_est["k"], 4))
                            st.session_state["param_theta"] = float(round(ou_est["theta"], 2))
                            st.session_state["param_sigma"] = float(round(ou_est["sigma_ou"], 4))
                            st.session_state["params_applied"] = True; st.session_state["applied_from"] = "OU"
                            st.session_state["sim_ran"] = False; st.rerun()
                else:
                    st.warning("Need at least 10 data points.")

        st.markdown("---")
        st.markdown("### Model Parameters")
        if st.session_state["params_applied"]:
            st.markdown(f'<span class="applied-pill">✔ From {st.session_state["applied_from"]} estimation</span>', unsafe_allow_html=True)
            if st.button("↩ Reset to defaults"):
                for k, v in _defaults.items(): st.session_state[k] = v
                st.session_state["sim_ran"] = False; st.rerun()

        if "GBM" in model:
            mu    = st.number_input("Annual drift μ", value=float(st.session_state["param_mu"]), step=0.001, format="%.4f")
            sigma = st.number_input("Annual volatility σ", min_value=0.001, max_value=5.0, value=float(st.session_state["param_sigma"]), step=0.001, format="%.4f")
            st.session_state["param_mu"] = mu; st.session_state["param_sigma"] = sigma
        else:
            kappa = st.number_input("Mean-reversion speed κ", min_value=0.001, max_value=100.0, value=float(st.session_state["param_kappa"]), step=0.01, format="%.4f")
            theta = st.number_input("Long-run mean θ (₱/Lkg)", min_value=0.01, value=float(st.session_state["param_theta"]), step=50.0)
            sigma = st.number_input("Annual volatility σ", min_value=0.001, max_value=5.0, value=float(st.session_state["param_sigma"]), step=0.001, format="%.4f")
            st.session_state["param_kappa"] = kappa; st.session_state["param_theta"] = theta; st.session_state["param_sigma"] = sigma

        st.markdown("---")
        st.markdown("### Risk & Volume")
        breakeven = st.number_input("Break-even / alert price (₱/Lkg)", min_value=0.0, value=2000.0, step=50.0)
        volume    = st.number_input("Annual volume (Lkg, 0 = ignore)", min_value=0.0, value=0.0, step=1000.0)

        st.markdown("---")
        st.markdown("### Simulation Settings")
        N_sim = int(st.number_input("Terminal simulations (N)", min_value=1000, max_value=100_000, value=5000, step=1000))
        K     = int(st.number_input("Sample paths to display", min_value=1, value=30, step=5))
        seed  = int(st.number_input("Random seed", min_value=0, value=42, step=1))

        st.markdown("---")
        st.markdown("### 📅 Weekly Prediction Settings")
        weekly_n_weeks  = int(st.number_input("Weeks to forecast", min_value=4, max_value=104, value=26, step=4))
        weekly_display  = st.selectbox("Bar shows", ["Median (P50)", "Mean"])
        weekly_interval = st.selectbox("Confidence interval", ["P05–P95 (90%)", "P25–P75 (50%)"])

        run = st.button("▶  Run Simulation")

    # ── Title ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="page-title">🎋 Sugar Pricing Forecasting</div><div class="page-subtitle">Monte Carlo Risk Model | Price Prediction</div>', unsafe_allow_html=True)
    for pill, val in [("Model", model), ("Spot", f"₱{S0:,.0f}/Lkg"), ("Horizon", horizon_label)]:
        st.markdown(f'<span class="info-pill">{pill}: {val}</span>', unsafe_allow_html=True)

    tab_sim, tab_weekly, tab_saved = st.tabs([
        "🎲 Monte Carlo Simulation",
        "📅 Weekly Price Prediction",
        "💾 Saved Runs",
    ])

    # ── Tab: Monte Carlo ───────────────────────────────────────────────────────
    with tab_sim:
        if not run:
            st.info("👈  Configure the sidebar and click **Run Simulation** to generate results.", icon="💡")
        else:
            with st.spinner("Running Monte Carlo simulation…"):
                if "GBM" in model:
                    terminal     = run_gbm_terminal(S0, mu, sigma, T, N_sim, seed)
                    times, paths = run_gbm_paths(S0, mu, sigma, T, steps_per_year, K, seed+1)
                else:
                    terminal     = run_mean_revert_terminal(S0, kappa, theta, sigma, T, N_sim, steps_per_year, seed)
                    times, paths = run_mean_revert_paths(S0, kappa, theta, sigma, T, steps_per_year, K, seed+1)

            mean_p=float(np.mean(terminal)); median_p=float(np.median(terminal))
            std_p=float(np.std(terminal))
            p05_v=float(np.percentile(terminal,5)); p25_v=float(np.percentile(terminal,25))
            p75_v=float(np.percentile(terminal,75)); p95_v=float(np.percentile(terminal,95))
            var95_v = S0 - p05_v
            es_vals = terminal[terminal <= p05_v]
            es95_reliable = len(es_vals) >= _ES_MIN_SAMPLES
            es95_v = float(np.mean(es_vals)) if es95_reliable else p05_v
            prob_be_v = float(np.mean(terminal <= breakeven))

            st.session_state.update({
                "last_sim_results": {"mean_p": mean_p, "median_p": median_p, "std_p": std_p,
                    "p05": p05_v, "p25": p25_v, "p75": p75_v, "p95": p95_v, "var95": var95_v,
                    "es95": es95_v if es95_reliable else None, "prob_be": prob_be_v},
                "last_sim_params": {"model": model, "S0": S0, "horizon_label": horizon_label,
                    "T": T, "N_sim": N_sim, "seed": seed, "breakeven": breakeven, "volume": volume,
                    **({"mu": mu, "sigma": sigma} if "GBM" in model else {"kappa": kappa, "theta": theta, "sigma": sigma})},
                "last_sim_terminal": terminal, "last_sim_times": times, "last_sim_paths": paths,
                "last_es_reliable": es95_reliable, "sim_ran": True,
            })

        if st.session_state.get("sim_ran"):
            terminal = st.session_state["last_sim_terminal"]
            times    = st.session_state["last_sim_times"]
            paths    = st.session_state["last_sim_paths"]
            _r       = st.session_state["last_sim_results"]
            mean_p=_r["mean_p"]; median_p=_r["median_p"]; std_p=_r["std_p"]
            p05=_r["p05"]; p25=_r["p25"]; p75=_r["p75"]; p95=_r["p95"]
            var95=_r["var95"]; es95=_r.get("es95") or p05; prob_be=_r["prob_be"]
            rev_risk = var95 * volume if volume > 0 else None
            es95_reliable = st.session_state.get("last_es_reliable", True)

            st.markdown('<div class="section-header">Key Statistics at Horizon</div>', unsafe_allow_html=True)
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Mean Price",    f"₱{mean_p:,.0f}",  f"{(mean_p/S0-1)*100:+.1f}% vs spot")
            k2.metric("Median Price",  f"₱{median_p:,.0f}", f"{(median_p/S0-1)*100:+.1f}% vs spot")
            k3.metric("Std Deviation", f"₱{std_p:,.0f}")
            k4.metric("VaR 95%",       f"₱{var95:,.0f}")
            k5.metric("P(Price ≤ BE)", f"{prob_be*100:.1f}%")

            if prob_be > 0.30:
                st.markdown(f'<div class="alert-danger">⚠️ High risk: <b>{prob_be*100:.1f}%</b> probability at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)
            elif prob_be > 0.10:
                st.markdown(f'<div class="alert-danger">⚠️ Moderate risk: <b>{prob_be*100:.1f}%</b> probability at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ Low risk: <b>{prob_be*100:.1f}%</b> probability at or below ₱{breakeven:,.0f}/Lkg.</div>', unsafe_allow_html=True)

            if rev_risk:
                st.markdown(f'<div class="alert-danger" style="margin-top:6px">Revenue at Risk: <b>₱{rev_risk:,.0f}</b></div>', unsafe_allow_html=True)

            save_col, _ = st.columns([1,3])
            with save_col:
                if st.button("💾 Save This Run"):
                    if not DB_OK:
                        st.warning("Database not configured.")
                    else:
                        ok = save_simulation(_user["id"], st.session_state["last_sim_params"], st.session_state["last_sim_results"])
                        if ok: st.success("✅ Saved! View in 💾 Saved Runs tab.")

            st.markdown('<div class="section-header">Price Distribution at Horizon</div>', unsafe_allow_html=True)
            td, tp, tpc = st.tabs(["📊 Distribution", "📈 Price Paths", "🔢 Percentile Table"])

            with td:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=terminal, nbinsx=80, name="Simulated prices", marker_color=TEAL, opacity=0.75))
                hv, be = np.histogram(terminal, bins=80)
                bm = be[:-1] <= p05
                fig.add_trace(go.Bar(x=be[:-1][bm], y=hv[bm], width=np.diff(be)[0], marker_color=RED_CLR, opacity=0.6, name="Below P05"))
                for val, label, color in [(p05,f"P05 ₱{p05:,.0f}",RED_CLR),(p95,f"P95 ₱{p95:,.0f}",GREEN_OK),(mean_p,f"Mean ₱{mean_p:,.0f}",GOLD),(breakeven,f"BE ₱{breakeven:,.0f}","#e8a0a0")]:
                    fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5, annotation_text=label, annotation_font_color=color)
                fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color=TEXT_CLR,
                    xaxis=dict(title="Terminal Price (₱/Lkg)", gridcolor=GRID_CLR),
                    yaxis=dict(gridcolor=GRID_CLR), height=420, barmode="overlay", margin=dict(t=30,b=50,l=50,r=30))
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("P05",f"₱{p05:,.0f}",f"{(p05/S0-1)*100:+.1f}%")
                c2.metric("P25",f"₱{p25:,.0f}",f"{(p25/S0-1)*100:+.1f}%")
                c3.metric("P75",f"₱{p75:,.0f}",f"{(p75/S0-1)*100:+.1f}%")
                c4.metric("P95",f"₱{p95:,.0f}",f"{(p95/S0-1)*100:+.1f}%")

            with tp:
                td_disp = times*52 if horizon_unit=="Weeks" else (times*12 if horizon_unit=="Months" else times)
                fig2 = go.Figure()
                pp = np.percentile(paths,[5,25,50,75,95],axis=1)
                fig2.add_trace(go.Scatter(x=np.concatenate([td_disp,td_disp[::-1]]),y=np.concatenate([pp[4],pp[0][::-1]]),fill="toself",fillcolor="rgba(74,159,181,0.1)",line_color="rgba(0,0,0,0)",name="P05–P95"))
                fig2.add_trace(go.Scatter(x=np.concatenate([td_disp,td_disp[::-1]]),y=np.concatenate([pp[3],pp[1][::-1]]),fill="toself",fillcolor="rgba(74,159,181,0.2)",line_color="rgba(0,0,0,0)",name="P25–P75"))
                for i in range(min(K,25)):
                    fig2.add_trace(go.Scatter(x=td_disp,y=paths[:,i],mode="lines",line=dict(color=TEAL,width=0.6),opacity=0.35,showlegend=False))
                fig2.add_trace(go.Scatter(x=td_disp,y=pp[2],mode="lines",line=dict(color=GOLD,width=2.5),name="Median"))
                fig2.add_hline(y=breakeven,line_dash="dot",line_color=RED_CLR,annotation_text=f"BE ₱{breakeven:,.0f}",annotation_font_color=RED_CLR)
                fig2.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor=DARK_BG,font_color=TEXT_CLR,
                    xaxis=dict(title=f"{horizon_unit}",gridcolor=GRID_CLR),yaxis=dict(title="Price (₱/Lkg)",gridcolor=GRID_CLR),height=430,margin=dict(t=30,b=50,l=50,r=30))
                st.plotly_chart(fig2, use_container_width=True)

            with tpc:
                pcts=[1,5,10,15,20,25,30,40,50,60,70,75,80,85,90,95,99]
                vals=np.percentile(terminal,pcts)
                rows=[{"Percentile":f"P{p:02d}","Price (₱/Lkg)":f"₱{v:,.0f}",
                    "Change vs Spot":f"{(v/S0-1)*100:+.1f}%",
                    "Below Break-even?":"❌ Yes" if v<=breakeven else "✅ No"} for p,v in zip(pcts,vals)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=520)

    # ── Tab: Weekly ────────────────────────────────────────────────────────────
    with tab_weekly:
        if not st.session_state.get("sim_ran") and not run:
            st.info("👈  Click **Run Simulation** first.", icon="💡")
        else:
            if "GBM" in model:
                _wkey = ("GBM", S0, mu, sigma, weekly_n_weeks, N_sim, seed)
            else:
                _wkey = ("OU", S0, kappa, theta, sigma, weekly_n_weeks, N_sim, seed)
            if run or st.session_state.get("wdf") is None or st.session_state.get("wdf_cache_key") != _wkey:
                with st.spinner("Computing weekly predictions…"):
                    if "GBM" in model:
                        wdf = run_weekly_gbm(S0, mu, sigma, weekly_n_weeks, N_sim, seed+99)
                    else:
                        wdf = run_weekly_ou(S0, kappa, theta, sigma, weekly_n_weeks, N_sim, seed+99)
                    st.session_state["wdf"] = wdf; st.session_state["wdf_cache_key"] = _wkey
            else:
                wdf = st.session_state["wdf"]

            bar_col  = "median" if weekly_display == "Median (P50)" else "mean"
            bar_label= "Median" if weekly_display == "Median (P50)" else "Mean"
            lo_col, hi_col = ("p05","p95") if "P05" in weekly_interval else ("p25","p75")
            int_label = "P05–P95" if "P05" in weekly_interval else "P25–P75"
            bar_vals=wdf[bar_col].values; lo_vals=wdf[lo_col].values; hi_vals=wdf[hi_col].values; weeks=wdf["week"].values
            colors = [RED_CLR if v<=breakeven else (AMBER if v<=breakeven*1.05 else GREEN_OK) for v in bar_vals]

            st.markdown('<div class="section-header">Weekly Price Prediction</div>', unsafe_allow_html=True)
            w1,w2,w3,w4,w5 = st.columns(5)
            mid_idx = min(weekly_n_weeks//2, weekly_n_weeks-1)
            w1.metric("Week 1",              f"₱{bar_vals[0]:,.0f}",  f"{(bar_vals[0]/S0-1)*100:+.1f}%")
            w2.metric(f"Week {mid_idx+1}",   f"₱{bar_vals[mid_idx]:,.0f}", f"{(bar_vals[mid_idx]/S0-1)*100:+.1f}%")
            w3.metric(f"Week {weekly_n_weeks}",f"₱{bar_vals[-1]:,.0f}", f"{(bar_vals[-1]/S0-1)*100:+.1f}%")
            w4.metric("Weeks Below BE",      f"{int(np.sum(bar_vals<=breakeven))} / {weekly_n_weeks}")
            pk=int(np.argmax(bar_vals))+1
            w5.metric("Peak Week", f"Week {pk}", f"₱{bar_vals[pk-1]:,.0f}")

            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=np.concatenate([weeks,weeks[::-1]]),y=np.concatenate([hi_vals,lo_vals[::-1]]),fill="toself",fillcolor="rgba(74,159,181,0.12)",line_color="rgba(0,0,0,0)",name=f"{int_label} band"))
            fig_w.add_trace(go.Bar(x=weeks, y=bar_vals, name=f"{bar_label} Price", marker_color=colors, opacity=0.85,
                error_y=dict(type="data",symmetric=False,array=hi_vals-bar_vals,arrayminus=bar_vals-lo_vals,color="#5a7a90",thickness=1.2,width=3)))
            fig_w.add_hline(y=breakeven,line_dash="dash",line_color=RED_CLR,annotation_text=f"BE ₱{breakeven:,.0f}",annotation_font_color=RED_CLR)
            fig_w.add_hline(y=S0,line_dash="dot",line_color=GOLD,annotation_text=f"Spot ₱{S0:,.0f}",annotation_font_color=GOLD)
            if "Mean-Reverting" in model:
                fig_w.add_hline(y=theta,line_dash="dashdot",line_color="#a78bfa",annotation_text=f"θ ₱{theta:,.0f}",annotation_font_color="#a78bfa")
            fig_w.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor=DARK_BG,font_color=TEXT_CLR,
                xaxis=dict(title="Week from Today",gridcolor=GRID_CLR),
                yaxis=dict(title="Predicted Price (₱/Lkg)",gridcolor=GRID_CLR,tickprefix="₱",tickformat=","),
                height=500,bargap=0.25,margin=dict(t=60,b=60,l=70,r=30))
            st.plotly_chart(fig_w, use_container_width=True)

            with st.expander("📋 View detailed weekly forecast table"):
                tbl=[{"Week":int(r["week"]),"Median (₱)":f"₱{r['median']:,.0f}","Mean (₱)":f"₱{r['mean']:,.0f}",
                    "P05":f"₱{r['p05']:,.0f}","P95":f"₱{r['p95']:,.0f}",
                    "vs Spot":f"{(r[bar_col]/S0-1)*100:+.1f}%",
                    "BE Risk":"❌ Below" if r[bar_col]<=breakeven else ("⚠️ Near" if r[bar_col]<=breakeven*1.05 else "✅ Safe")} for _,r in wdf.iterrows()]
                tdf=pd.DataFrame(tbl)
                st.dataframe(tdf, use_container_width=True, hide_index=True, height=400)
                st.download_button("⬇️ Download Weekly Forecast CSV", data=tdf.to_csv(index=False), file_name="sugar_weekly_forecast.csv", mime="text/csv")

    # ── Tab: Saved Runs ────────────────────────────────────────────────────────
    with tab_saved:
        st.markdown('<div class="section-header">Your Saved Simulation Runs</div>', unsafe_allow_html=True)
        if not DB_OK:
            st.warning("Database not configured.")
        else:
            if st.button("🔄 Refresh", key="refresh_saved"):
                st.rerun()
            runs = load_simulations(_user["id"])
            if not runs:
                st.info("No saved runs yet. Run a simulation and click **💾 Save This Run**.")
            else:
                st.caption(f"{len(runs)} saved run(s) · {_user['email']}")
                for i, row in enumerate(runs):
                    try: p = json.loads(row.get("params","{}") or "{}")
                    except: p = {}
                    try: r = json.loads(row.get("results","{}") or "{}")
                    except: r = {}
                    created = str(row.get("created_at",""))[:19].replace("T"," ")
                    with st.expander(f"🕒 {created}  ·  {row.get('model','?')}  ·  Spot ₱{row.get('spot_price') or 0:,.0f}  ·  {row.get('horizon','?')}", expanded=(i==0)):
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Mean",f"₱{r.get('mean_p',0):,.0f}")
                        c2.metric("Median",f"₱{r.get('median_p',0):,.0f}")
                        c3.metric("VaR 95%",f"₱{r.get('var95',0):,.0f}")
                        c4.metric("P(≤ BE)",f"{r.get('prob_be',0)*100:.1f}%")
                        st.markdown(f"**Model:** {p.get('model','?')} | **N:** {p.get('N_sim','?'):,} | **Seed:** {p.get('seed','?')}")
                        pct_tbl=pd.DataFrame({"Metric":["P05","P25","Median","Mean","P75","P95"],
                            "Price":[f"₱{r.get(k,0):,.0f}" for k in ["p05","p25","median_p","mean_p","p75","p95"]]})
                        st.dataframe(pct_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<p style="color:#6b7280;font-size:0.8rem;">Mill-gate raw sugar price model. GBM = lognormal returns. Mean-Reverting = Ornstein–Uhlenbeck on log-prices (exact solution). Probabilistic estimates only.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
