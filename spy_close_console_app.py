# SPY Close Console with ML Overlay
# Deploy on Streamlit Cloud

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LogisticRegression

SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"

VWAP_THRESHOLD = 0.0015
RANGE_ACCEPTANCE = 0.80
LOW_ACCEPTANCE = 0.20

st.set_page_config(page_title="SPY Close Console", layout="wide")

# ---------------- DARK STYLE ----------------
st.markdown("""
<style>
body {background-color: #0e1117; color: #e6e6e6;}
</style>
""", unsafe_allow_html=True)

st.title("SPY 3:30pm Close Console")

# ---------------- AUTO REFRESH ----------------
eastern = pytz.timezone("US/Eastern")
now = datetime.now(eastern)

if now.hour > 15 or (now.hour == 15 and now.minute >= 25):
    st_autorefresh(interval=30 * 1000, key="late_refresh")

# ---------------- FUNCTIONS ----------------
import time

@st.cache_data(ttl=30)
def get_intraday(symbol):
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                period="1d",
                interval="1m",
                progress=False,
                threads=False
            )
            if df is not None and len(df) > 50:
                return df
        except:
            pass
        time.sleep(1)

    return None

def calculate_vwap(df):
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    return df

def build_model(days=45):
    spy = yf.download(SYMBOL, period=f"{days}d", interval="1m", progress=False)
    spy["Date"] = spy.index.date
    rows = []

    for d in spy["Date"].unique():
        day = spy[spy["Date"] == d]
        if len(day) < 350:
            continue
        try:
            snap = day.iloc[330]
            close = day.iloc[-1]
            vwap = (day["Close"] * day["Volume"]).cumsum() / day["Volume"].cumsum()
            vwap_330 = vwap.iloc[330]
            day_high = day["High"].max()
            day_low = day["Low"].min()

            vwap_dist = (snap["Close"] - vwap_330) / vwap_330
            range_pos = (snap["Close"] - day_low) / (day_high - day_low)
            momentum = (snap["Close"] - day.iloc[300]["Close"]) / day.iloc[300]["Close"]
            ret = (close["Close"] - snap["Close"]) / snap["Close"]

            rows.append([vwap_dist, range_pos, momentum, int(ret > 0)])
        except:
            continue

    df = pd.DataFrame(rows, columns=["vwap_dist", "range_pos", "momentum", "target"])
    if len(df) < 10:
        return None

    X = df[["vwap_dist", "range_pos", "momentum"]]
    y = df["target"]

    model = LogisticRegression()
    model.fit(X, y)
    return model

def classify(spy_df, vix_df):
    # Defensive checks
    if spy_df is None or len(spy_df) < 10:
        return 0, 0, 0.5, 0, 0, "NO DATA", "→", "#ffaa00", 50

    latest = spy_df.iloc[-1]
    price = latest["Close"]
    vwap = latest["VWAP"]

    vwap_dist = (price - vwap) / vwap

    day_high = spy_df["High"].max()
    day_low = spy_df["Low"].min()

    if day_high == day_low:
        range_pos = 0.5
    else:
        range_pos = (price - day_low) / (day_high - day_low)

    if vix_df is None or len(vix_df) < 30:
        vix_change = 0
    else:
        vix_change = (
            vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-30]
        ) / vix_df["Close"].iloc[-30]

    score = 0
    if vwap_dist > VWAP_THRESHOLD: score += 1
    if range_pos > RANGE_ACCEPTANCE: score += 1
    if vix_change < 0: score += 1
    if vwap_dist < -VWAP_THRESHOLD: score -= 1
    if range_pos < LOW_ACCEPTANCE: score -= 1
    if vix_change > 0: score -= 1

    if score >= 2:
        bias, arrow, color = "CONTINUATION UP", "↑", "#00ff99"
    elif score <= -2:
        bias, arrow, color = "CONTINUATION DOWN", "↓", "#ff4d4d"
    else:
        bias, arrow, color = "MIXED / CHOP", "→", "#ffaa00"

    confidence = int((score + 3) / 6 * 100)

    return price, vwap, range_pos, vwap_dist, score, bias, arrow, color, confidence

# ---------------- LOAD DATA ----------------
with st.spinner("Loading live data..."):
    spy = calculate_vwap(get_intraday(SYMBOL))
    vix = get_intraday(VIX_SYMBOL)

    price, vwap, range_pos, vwap_dist, score, bias, arrow, color, confidence = classify(spy, vix)

# ---------------- ML PROBABILITY ----------------
model = build_model(45)

if model is not None:
    momentum = (spy["Close"].iloc[-1] - spy["Close"].iloc[-30]) / spy["Close"].iloc[-30]
    X_live = np.array([[vwap_dist, range_pos, momentum]])
    ml_prob = round(model.predict_proba(X_live)[0][1] * 100, 1)
else:
    ml_prob = None

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("SPY", round(price, 2))
col2.metric("VWAP", round(vwap, 2))
col3.metric("Range %", f"{round(range_pos*100,1)}%")

# ---------------- DIRECTION PANEL ----------------
st.markdown(f"""
<div style='padding:20px; background-color:#1c1f26; border-radius:8px; text-align:center;'>
<h1 style='color:{color}; font-size:48px;'>{arrow}</h1>
<h2 style='color:{color};'>{bias}</h2>
<h3 style='color:white;'>Confidence: {confidence}%</h3>
</div>
""", unsafe_allow_html=True)

if ml_prob is not None:
    st.markdown("### ML Probability Close Higher")
    if ml_prob > 60:
        st.success(f"{ml_prob}%")
    elif ml_prob < 40:
        st.error(f"{ml_prob}%")
    else:
        st.warning(f"{ml_prob}%")

# ---------------- INTRADAY CHART ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=spy.index, y=spy["Close"], mode='lines', name="SPY", line=dict(color="#00b3ff")))
fig.add_trace(go.Scatter(x=spy.index, y=spy["VWAP"], mode='lines', name="VWAP", line=dict(color="#ffffff", dash="dash")))
fig.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

st.caption("Auto-refresh every 30 seconds after 3:25pm ET.")
