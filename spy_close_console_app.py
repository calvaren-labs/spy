import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# -------------------------------
# CONFIG
# -------------------------------

SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"
VWAP_THRESHOLD = 0.002  # 0.2%
MODEL_VERSION = "v1.0 – Continuous Score + Late-Day Amplification"

# -------------------------------
# DATA FUNCTIONS
# -------------------------------

@st.cache_data(ttl=60)
def get_intraday(symbol):
    df = yf.download(symbol, period="2d", interval="1m", progress=False)

    if df is None or len(df) == 0:
        return None

    df = df.dropna()

    # Keep only today
    df = df[df.index.date == df.index[-1].date()]

    return df


def calculate_vwap(df):
    df = df.copy()

    # Ensure numeric columns
    for col in ["High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["High", "Low", "Close", "Volume"])

    df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3

    cumulative_volume = df["Volume"].cumsum()
    cumulative_tp_volume = (df["TP"] * df["Volume"]).cumsum()

    df["VWAP"] = cumulative_tp_volume / cumulative_volume

    return df


# -------------------------------
# CLASSIFICATION ENGINE
# -------------------------------

def classify(spy_df, vix_df):

    if spy_df is None or len(spy_df) < 30:
        return 0, 0, 0.5, 0, 0, "NO DATA", "→", "#ffaa00", 0

    latest = spy_df.iloc[-1]

    price = latest["Close"]
    vwap = latest["VWAP"]

    vwap_dist = (price - vwap) / vwap

    day_high = spy_df["High"].max()
    day_low = spy_df["Low"].min()
    range_pos = 0.5 if day_high == day_low else (price - day_low) / (day_high - day_low)

    # VIX change (30 min lookback)
    if vix_df is None or len(vix_df) < 30:
        vix_change = 0
    else:
        vix_change = (
            vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-30]
        ) / vix_df["Close"].iloc[-30]

    # -------------------------------
    # CONTINUOUS SCORING
    # -------------------------------

    score = 0

    # VWAP contribution
    score += vwap_dist / VWAP_THRESHOLD

    # Range contribution (centered at 0.5)
    score += (range_pos - 0.5) * 2

    # VIX contribution (rising VIX = bearish)
    score -= vix_change * 5

    # -------------------------------
    # LATE-DAY AMPLIFICATION
    # -------------------------------

    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.hour == 15 and now.minute >= 30:
        time_progress = (now.minute - 30) / 30
        score *= (1 + 0.4 * time_progress)

    # -------------------------------
    # CLASSIFICATION
    # -------------------------------

    if score > 1:
        bias = "CONTINUATION UP"
        arrow = "↑"
        color = "#00cc66"
    elif score < -1:
        bias = "CONTINUATION DOWN"
        arrow = "↓"
        color = "#ff4444"
    else:
        bias = "MIXED / CHOP"
        arrow = "→"
        color = "#ffaa00"

    confidence = int(min(90, abs(score) * 20))

    # Internal logging for testing
    print({
        "score": round(score, 3),
        "vwap_dist": round(vwap_dist, 4),
        "range_pos": round(range_pos, 3),
        "vix_change": round(vix_change, 4),
        "confidence": confidence
    })

    return price, vwap, range_pos, vwap_dist, score, bias, arrow, color, confidence


# -------------------------------
# STREAMLIT APP
# -------------------------------

st.set_page_config(layout="wide")
st.title("SPY 3:30pm Close Console")
st.caption(MODEL_VERSION)

spy_raw = get_intraday(SYMBOL)
vix_raw = get_intraday(VIX_SYMBOL)

if spy_raw is not None:
    spy = calculate_vwap(spy_raw)
    result = classify(spy, vix_raw)

    price, vwap, range_pos, vwap_dist, score, bias, arrow, color, confidence = result

    col1, col2, col3 = st.columns(3)
    col1.metric("SPY", f"{price:.2f}")
    col2.metric("VWAP", f"{vwap:.2f}")
    col3.metric("Range %", f"{range_pos * 100:.1f}%")

    st.markdown(
        f"""
        <div style="background-color:#111822;padding:40px;border-radius:10px;text-align:center;">
            <h1 style="color:{color};">{arrow}</h1>
            <h2 style="color:{color};">{bias}</h2>
            <h4 style="color:white;">Confidence: {confidence}%</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.line_chart(spy["Close"])
else:
    st.warning("No data available.")
