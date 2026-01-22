# ===============================
# AI Trading Platform (India + Crypto)
# Sidebar-based Professional System
# ===============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Platform")
st.caption("Indian Markets • Crypto • Market Intelligence")
st.info("Educational use only. Not financial advice.")

# -------------------------------
# Sidebar
# -------------------------------
section = st.sidebar.radio(
    "Navigation",
    ["🇮🇳 Indian Stock Market", "🪙 Crypto Market", "🌍 Market Overview"]
)

# -------------------------------
# Data engine (safe)
# -------------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol, days=180):
    df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def market_snapshot(symbols: dict):
    rows = []
    for name, symbol in symbols.items():
        df = fetch_data(symbol, 7)
        if df.empty or len(df) < 2:
            continue
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        change = last - prev
        pct = (change / prev) * 100
        rows.append([name, symbol, round(last,2), round(change,2), round(pct,2)])
    return pd.DataFrame(rows, columns=["Name","Symbol","Price","Change","Change %"])

# -------------------------------
# Symbol universes
# -------------------------------
NSE = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "SBIN": "SBIN.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS",
    "MARUTI": "MARUTI.NS"
}

CRYPTO = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD"
}

# -------------------------------
# Indian Stock Market
# -------------------------------
if section == "🇮🇳 Indian Stock Market":

    st.header("🇮🇳 Indian Stock Market (NSE)")
    st.subheader("Market Snapshot")
    st.dataframe(market_snapshot(NSE), use_container_width=True)

    st.divider()

    stock = st.selectbox("Select Stock", list(NSE.keys()))
    days = st.slider("Historical data (days)", 30, 500, 180)

    df = fetch_data(NSE[stock], days)

    if df.empty:
        st.error("No data found.")
        st.stop()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    last = df.iloc[-1]
    if last["MA20"] > last["MA50"]:
        st.success("🟢 BUY SIGNAL")
    elif last["MA20"] < last["MA50"]:
        st.error("🔴 SELL SIGNAL")
    else:
        st.warning("🟡 HOLD")

# -------------------------------
# Crypto Market
# -------------------------------
elif section == "🪙 Crypto Market":

    st.header("🪙 Crypto Market")
    st.subheader("Market Snapshot")
    st.dataframe(market_snapshot(CRYPTO), use_container_width=True)

    st.divider()

    coin = st.selectbox("Select Crypto", list(CRYPTO.keys()))
    days = st.slider("Historical data (days)", 30, 500, 180)

    df = fetch_data(CRYPTO[coin], days)

    if df.empty:
        st.error("No data found.")
        st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Market Overview
# -------------------------------
else:

    st.header("🌍 Global Market Overview")

    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Gold": "GC=F",
        "Crude Oil": "CL=F"
    }

    st.subheader("World Market Snapshot")
    st.dataframe(market_snapshot(indices), use_container_width=True)
