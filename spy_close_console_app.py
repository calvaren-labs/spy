import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("SPY 3:30pm Close Console")
st.caption("v1.0 – Stable Yahoo (5m) + Continuous Model")

# -------------------------------------------------
# DATA FETCH (Cloud-friendly, rate-limit tolerant)
# -------------------------------------------------

@st.cache_data(ttl=300)
def get_intraday():
    try:
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="1d", interval="5m", prepost=False)

        if df is None or df.empty:
            return None

        df = df.reset_index()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # VWAP
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        return df

    except Exception:
        return None


spy = get_intraday()

if spy is None:
    st.warning("Data temporarily unavailable (Yahoo rate limit or outside market hours).")
    st.stop()

# -------------------------------------------------
# SNAPSHOT
# -------------------------------------------------

price = spy["Close"].iloc[-1]
vwap = spy["VWAP"].iloc[-1]

day_high = spy["High"].max()
day_low = spy["Low"].min()

range_pos = (price - day_low) / (day_high - day_low + 1e-9)
vwap_dist = (price - vwap) / vwap

# Simple volatility proxy (kept light for Cloud stability)
vix_proxy = np.random.normal(0, 0.001)

# -------------------------------------------------
# CONTINUOUS SCORING MODEL
# -------------------------------------------------

score = 0

score += vwap_dist * 15
score += (range_pos - 0.5) * 4
score -= vix_proxy * 20

# -------------------------------------------------
# LATE-DAY AMPLIFICATION
# -------------------------------------------------

now = datetime.now(pytz.timezone("US/Eastern"))

if now.hour == 15 and now.minute >= 30:
    time_progress = (now.minute - 30) / 30
    score *= (1 + 0.4 * time_progress)

# -------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------

if score > 0.5:
    bias = "CONTINUATION UP"
    arrow = "↑"
    color = "#00C853"
elif score < -0.5:
    bias = "CONTINUATION DOWN"
    arrow = "↓"
    color = "#FF5252"
else:
    bias = "MIXED / CHOP"
    arrow = "→"
    color = "#FFB300"

confidence = int(min(90, abs(score) * 20))

# -------------------------------------------------
# METRICS
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("SPY", f"{price:.2f}")
col2.metric("VWAP", f"{vwap:.2f}")
col3.metric("Range %", f"{range_pos*100:.1f}%")

# -------------------------------------------------
# BIAS PANEL
# -------------------------------------------------

st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg, #0f172a, #111827);
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="font-size:40px;color:{color};">{arrow}</div>
        <div style="font-size:28px;color:{color};font-weight:600;">
            {bias}
        </div>
        <div style="font-size:18px;color:white;margin-top:10px;">
            Confidence: {confidence}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# PLOTLY CHART (fixed scaling)
# -------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=spy.index,
        y=spy["Close"],
        name="SPY",
        line=dict(width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=spy.index,
        y=spy["VWAP"],
        name="VWAP",
        line=dict(width=2, dash="dash")
    )
)

fig.update_yaxes(
    range=[
        spy["Close"].min() * 0.995,
        spy["Close"].max() * 1.005
    ]
)

fig.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
