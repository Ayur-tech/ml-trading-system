import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Indian Stock Market Dashboard", layout="wide")
st.title("📊 Indian Stock Market Dashboard (NSE)")
st.caption("Stable functional base for a professional trading system")

# ---------------- SYMBOL INPUT ---------------- #
symbol_input = st.text_input("Enter NSE Stock Symbol (example: RELIANCE, TCS, INFY)", "RELIANCE")
symbol = symbol_input.strip().upper() + ".NS"

days = st.slider("Select historical period (days)", 30, 365, 120)

st.info(f"Selected stock: {symbol}")

# ---------------- DATA FETCH ---------------- #
@st.cache_data(ttl=300)
def load_data(sym, days):
    df = yf.download(sym, period=f"{days}d", interval="1d", auto_adjust=True)

    # Defensive fix for multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

try:
    df = load_data(symbol, days)

    if df.empty:
        st.error("No data found. Please check the NSE symbol.")
        st.stop()

except Exception as e:
    st.error("Failed to fetch data.")
    st.code(str(e))
    st.stop()

# ---------------- INDICATORS ---------------- #
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df = df.dropna()

latest = df.iloc[-1]
price = float(latest["Close"])
ma20 = float(latest["MA20"])
ma50 = float(latest["MA50"])

# ---------------- PRICE CHART ---------------- #
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price (INR)")
st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET METRICS ---------------- #
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"₹ {price:,.2f}")
c2.metric("MA20", f"₹ {ma20:,.2f}")
c3.metric("MA50", f"₹ {ma50:,.2f}")

# ---------------- SIGNAL ENGINE ---------------- #
signal = "HOLD"
if ma20 > ma50:
    signal = "BUY"
elif ma20 < ma50:
    signal = "SELL"

st.subheader("📌 System Signal")

if signal == "BUY":
    st.success("🟢 BUY Signal")
elif signal == "SELL":
    st.error("🔴 SELL Signal")
else:
    st.warning("🟡 HOLD")

# ---------------- RISK LEVELS ---------------- #
stop_loss = price * 0.97
take_profit = price * 1.05

st.subheader("⚠ Risk Preview (example)")

r1, r2, r3 = st.columns(3)
r1.metric("Current", f"₹ {price:,.2f}")
r2.metric("Stop Loss", f"₹ {stop_loss:,.2f}")
r3.metric("Take Profit", f"₹ {take_profit:,.2f}")

# ---------------- RAW DATA ---------------- #
with st.expander("View Raw Market Data"):
    st.dataframe(df.tail(30))
