import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="ML Trading System", layout="wide")
st.title("📈 ML Trading System — Functional Version")

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

# -----------------------------
# DATA LOAD
# -----------------------------
@st.cache_data(ttl=300)
def load_data(sym):
    df = yf.download(sym, period="6mo", interval="1d", auto_adjust=True)

    # ✅ Flatten columns if Yahoo returns multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

# -----------------------------
# DISPLAY
# -----------------------------
st.subheader("Price Chart")
st.line_chart(df["Close"])

st.subheader("Latest Market Data")
st.dataframe(df.tail())

# -----------------------------
# INDICATORS
# -----------------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df = df.dropna()

latest = df.iloc[-1]

ma20 = float(latest["MA20"])
ma50 = float(latest["MA50"])
price = float(latest["Close"])

# -----------------------------
# SIGNAL ENGINE
# -----------------------------
signal = "HOLD"
if ma20 > ma50:
    signal = "BUY"
elif ma20 < ma50:
    signal = "SELL"

st.subheader("System Signal")

if signal == "BUY":
    st.success("🟢 BUY Signal")
elif signal == "SELL":
    st.error("🔴 SELL Signal")
else:
    st.warning("🟡 HOLD Signal")

# -----------------------------
# RISK LEVELS
# -----------------------------
stop_loss = price * 0.97
take_profit = price * 1.05

st.subheader("Risk Levels")

c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"{price:.2f}")
c2.metric("Stop Loss", f"{stop_loss:.2f}")
c3.metric("Take Profit", f"{take_profit:.2f}")

st.caption("Stable functional base. ML, backtesting, and automation can now be added safely.")





