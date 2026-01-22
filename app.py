import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Indian Stock Trading System", page_icon="📊", layout="wide")
st.title("📊 Indian Stock Market Trading System (NSE)")
st.caption("Live Indian market dashboard with Buy / Sell / Hold engine")

# ---------------- INPUT ----------------
symbol = st.text_input("Enter NSE stock symbol (example: RELIANCE, TCS, INFY)", "RELIANCE")
days = st.slider("Select historical period (days)", 30, 365, 120)

ticker = symbol.upper().strip() + ".NS"
st.info(f"Selected stock: {ticker}")

# ---------------- DATA ----------------
@st.cache_data(ttl=300)
def load_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # ✅ FIX
    return df

df = load_data(ticker, days)

if df.empty or "Close" not in df.columns:
    st.error("No valid market data found. Check the NSE symbol.")
    st.stop()

# ---------------- INDICATORS ----------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

latest = df.iloc[-1]
price = float(latest["Close"])
ma20 = float(latest["MA20"])
ma50 = float(latest["MA50"])
rsi = float(latest["RSI"])

# ---------------- SIGNAL ENGINE ----------------
if ma20 > ma50 and rsi < 70:
    signal = "BUY"
elif ma20 < ma50 and rsi > 30:
    signal = "SELL"
else:
    signal = "HOLD"

confidence = min(abs(ma20 - ma50) / ma50 * 100 + abs(50 - rsi), 100)

# ---------------- UI ----------------
def show_signal(sig):
    if sig == "BUY":
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=80)
        st.success("BUY — Bullish trend detected")
    elif sig == "SELL":
        st.image("https://cdn-icons-png.flaticon.com/512/190/190406.png", width=80)
        st.error("SELL — Bearish trend detected")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/190/190422.png", width=80)
        st.warning("HOLD — Market indecision")

col1, col2 = st.columns([1.1, 2])

with col1:
    st.subheader("📌 Decision Panel")
    show_signal(signal)
    st.metric("Live Price", f"₹{price:,.2f}")
    st.metric("MA20", f"₹{ma20:,.2f}")
    st.metric("MA50", f"₹{ma50:,.2f}")
    st.metric("RSI", f"{rsi:.2f}")
    st.progress(confidence / 100)
    st.caption(f"Confidence Score: {confidence:.1f}%")
    st.warning("⚠ Educational system only. Not financial advice.")

with col2:
    st.subheader("📈 Price & Moving Averages")
    chart_df = df[["Close", "MA20", "MA50"]].copy()
    chart_df.columns = ["Price", "MA20", "MA50"]  # extra safety
    st.line_chart(chart_df, use_container_width=True)

with st.expander("📄 Raw NSE Data"):
    st.dataframe(df.tail(25), use_container_width=True)
