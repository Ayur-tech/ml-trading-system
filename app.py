import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Trading Terminal", layout="wide")

# ---------------- SAFE DATA ENGINE ----------------
@st.cache_data(ttl=300)
def load_data(symbol, days=180, interval="1d"):
    end = datetime.now()
    start = end - timedelta(days=days)

    symbols_to_try = [symbol, symbol.replace(".NS", ".BO")]

    for sym in symbols_to_try:
        try:
            df = yf.download(
                sym,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, sym
        except:
            pass

    return pd.DataFrame(), None


# ---------------- MARKET LISTS ----------------
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","ITC.NS",
    "LT.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","BAJFINANCE.NS","HINDUNILVR.NS",
    "BHARTIARTL.NS","ASIANPAINT.NS","HCLTECH.NS","MARUTI.NS","SUNPHARMA.NS"
]

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD"]

# ---------------- UI ----------------
st.title("📊 AI Trading Terminal")
st.caption("Indian Market + Crypto | Professional Functional Base")

tab1, tab2, tab3 = st.tabs(["🇮🇳 Indian Market", "🪙 Crypto Market", "📈 Market Browser"])

# ---------------- INDIAN MARKET ----------------
with tab1:
    col1, col2 = st.columns([2,1])

    with col1:
        symbol = st.text_input("Enter NSE Stock (RELIANCE, TCS, INFY)", "RELIANCE")
    with col2:
        days = st.slider("Historical Days", 30, 1000, 180)

    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        symbol = symbol + ".NS"

    df, used = load_data(symbol, days)

    if used:
        st.success(f"Live source: {used}")

    if df.empty:
        st.error("No market data received.")
    else:
        st.subheader("Price Chart")
        st.line_chart(df["Close"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Last Price", round(df["Close"].iloc[-1],2))
        c2.metric("Day High", round(df["High"].iloc[-1],2))
        c3.metric("Day Low", round(df["Low"].iloc[-1],2))

        st.subheader("Raw Market Data")
        st.dataframe(df.tail(20), use_container_width=True)

# ---------------- CRYPTO MARKET ----------------
with tab2:
    crypto = st.selectbox("Select Crypto", CRYPTO)
    days = st.slider("Days", 30, 1000, 180, key="crypto")

    df, used = load_data(crypto, days, "1d")

    if df.empty:
        st.error("Crypto data unavailable.")
    else:
        st.success(f"Source: {used}")
        st.line_chart(df["Close"])
        st.dataframe(df.tail(15), use_container_width=True)

# ---------------- MARKET BROWSER ----------------
with tab3:
    st.subheader("NIFTY 50 Snapshot")

    market_data = {}

    with st.spinner("Fetching market..."):
        for stock in NIFTY50:
            df, _ = load_data(stock, 7)
            if not df.empty:
                market_data[stock] = df["Close"].iloc[-1]

    if market_data:
        market_df = pd.DataFrame(market_data.items(), columns=["Stock", "Last Price"])
        st.dataframe(market_df, use_container_width=True)

    st.subheader("Indices")
    i1, _ = load_data("^NSEI", 30)
    i2, _ = load_data("^NSEBANK", 30)

    if not i1.empty:
        st.write("NIFTY 50")
        st.line_chart(i1["Close"])

    if not i2.empty:
        st.write("BANK NIFTY")
        st.line_chart(i2["Close"])

