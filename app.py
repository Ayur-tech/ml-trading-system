import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="ML Trading System", layout="wide")

st.title("📈 ML Trading System — Functional Version")

# -----------------------------
# SAFE SYMBOL SELECTION
# -----------------------------
SYMBOLS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA"
}

label = st.selectbox("Select Asset", list(SYMBOLS.keys()))
symbol = SYMBOLS[label]  # always string

# HARD SAFETY (prevents tuple crash)
if isinstance(symbol, tuple):
    symbol = symbol[0]
symbol = str(symbol)

confidence = st.slider("Confidence Threshold", 0.50, 0.95, 0.60)

st.write("Selected Symbol:", symbol)

# -----------------------------
# DATA FETCH
# -----------------------------
@st.cache_data(ttl=300)
def load_data(sym):
    df = yf.download(sym, period="6mo", interval="1d")
    return df

try:
    data = load_data(symbol)

    if data.empty:
        st.error("No data returned. Try another asset.")
        st.stop()

    st.subheader("Market Data")
    st.dataframe(data.tail())

    st.subheader("Price Chart")
    st.line_chart(data["Close"])

    # -----------------------------
    # SIMPLE SIGNAL LOGIC (functional placeholder)
    # -----------------------------
    last = data.iloc[-1]
    prev = data.iloc[-2]

    if last["Close"] > prev["Close"] * (1 + confidence / 100):
        signal = "BUY"
    elif last["Close"] < prev["Close"] * (1 - confidence / 100):
        signal = "SELL"
    else:
        signal = "HOLD"

    st.subheader("Trading Signal")
    st.success(f"Signal: {signal}")

except Exception as e:
    st.error("System error. Data source or pipeline failed.")
    st.code(str(e))




