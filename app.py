# app.py
# Indian Stock Market Trading Dashboard (NSE)
# Author: Your AI Trading System

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Indian Stock Market Trading System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Indian Stock Market Trading System (NSE)")
st.caption("Professional AI-ready trading dashboard for Indian markets")

# ----------------------------
# USER INPUT
# ----------------------------
symbol = st.text_input("Enter NSE stock symbol (example: RELIANCE, TCS, INFY)", "RELIANCE")
days = st.slider("Select historical period (days)", 30, 365, 120)

ticker = symbol.upper().strip() + ".NS"
st.info(f"Selected stock: {ticker}")

# ----------------------------
# DATA FETCH
# ----------------------------
@st.cache_data(ttl=300)
def load_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    return df

df = load_data(ticker, days)

if df.empty:
    st.error("No data found. Please check the NSE symbol.")
    st.stop()

# ----------------------------
# INDICATORS
# ----------------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

df = df.dropna()

latest = df.iloc[-1]
price = float(latest["Close"])
ma20 = float(latest["MA20"])
ma50 = float(latest["MA50"])
rsi = float(latest["RSI"])

# ----------------------------
# SIGNAL LOGIC
# ----------------------------
if ma20 > ma50 and rsi < 70:
    signal = "BUY"
elif ma20 < ma50 and rsi > 30:
    signal = "SELL"
else:
    signal = "HOLD"

# Confidence score (simple model)
confidence = min(abs(ma20 - ma50) / ma50 * 100 + abs(50 - rsi), 100)

# ----------------------------
# SIGNAL DISPLAY
# ----------------------------
def show_signal(signal):
    if signal == "BUY":
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=90)
        st.success("BUY SIGNAL — Bullish trend detected")

    elif signal == "SELL":
        st.image("https://cdn-icons-png.flaticon.com/512/190/190406.png", width=90)
        st.error("SELL SIGNAL — Bearish trend detected")

    else:
        st.image("https://cdn-icons-png.flaticon.com/512/190/190422.png", width=90)
        st.warning("HOLD — No strong trend detected")

# ----------------------------
# DASHBOARD LAYOUT
# ----------------------------
col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("📌 Decision Panel")
    show_signal(signal)

    st.metric("Live Price", f"₹{price:,.2f}")
    st.metric("MA20", f"₹{ma20:,.2f}")
    st.metric("MA50", f"₹{ma50:,.2f}")
    st.metric("RSI", f"{rsi:.2f}")
    st.progress(confidence / 100)
    st.caption(f"Confidence Score: {confidence:.1f}%")

    st.warning("⚠ This system is for educational purposes only. Not financial advice.")

with col2:
    st.subheader("📈 Price & Moving Averages")
    st.line_chart(df[["Close", "MA20", "MA50"]])

# ----------------------------
# DATA VIEW
# ----------------------------
with st.expander("📄 View Raw Market Data"):
    st.dataframe(df.tail(20), use_container_width=True)

