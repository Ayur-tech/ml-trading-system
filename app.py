import streamlit as st
import yfinance as yf
import pandas as pd

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(page_title="ML Trading System", layout="wide")

st.title("📈 ML Trading System — Functional Version")
st.write("Clean functional market dashboard (stable base)")

# --------------------
# USER INPUT
# --------------------
assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA"
}

asset_name = st.selectbox("Select Asset", list(assets.keys()))
symbol = assets[asset_name]

confidence = st.slider("Confidence Threshold", 0.50, 0.95, 0.60)

st.info(f"Selected Symbol: {symbol}")

# --------------------
# DATA FETCH
# --------------------
@st.cache_data(ttl=300)
def load_data(sym):
    df = yf.download(sym, period="6mo", interval="1d")
    return df

try:
    df = load_data(symbol)

    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

except Exception as e:
    st.error("System error while loading data.")
    st.code(str(e))
    st.stop()

# --------------------
# DISPLAY DATA
# --------------------
st.subheader("Price Chart")
st.line_chart(df["Close"])

st.subheader("Latest Market Data")
st.dataframe(df.tail())

# --------------------
# SIMPLE SIGNAL ENGINE (TEMP BASE)
# --------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

latest = df.iloc[-1]

signal = "HOLD"
if latest["MA20"] > latest["MA50"]:
    signal = "BUY"
elif latest["MA20"] < latest["MA50"]:
    signal = "SELL"

st.subheader("System Signal")

if signal == "BUY":
    st.success("🟢 BUY Signal")
elif signal == "SELL":
    st.error("🔴 SELL Signal")
else:
    st.warning("🟡 HOLD Signal")

# --------------------
# RISK PREVIEW
# --------------------
price = float(latest["Close"])
stop_loss = price * 0.97
take_profit = price * 1.05

st.subheader("Risk Levels (example)")

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"{price:.2f}")
col2.metric("Stop Loss", f"{stop_loss:.2f}")
col3.metric("Take Profit", f"{take_profit:.2f}")

st.caption("This is a functional system base. ML, backtesting, and automation can now be safely added.")




