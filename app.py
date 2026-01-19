import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ML Trading System", layout="wide")

st.title("📈 ML Trading System — Functional Version")

# ---------------- UI ----------------
symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD", "SOL-USD"])
# Safety: force symbol to string (kills tuple bug permanently)
if isinstance(symbol, (list, tuple)):
    symbol = symbol[0]
symbol = str(symbol)

threshold = st.slider("Confidence Threshold", 0.50, 0.90, 0.60, 0.01)

# ---------------- DATA ----------------
@st.cache_data(ttl=300)
def fetch_data(symbol: str):
    df = yf.download(symbol, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df

def build_features(df):
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["returns"] = df["close"].pct_change()
    df["target"] = (df["returns"].shift(-1) > 0).astype(int)
    df = df.dropna()
    return df

# ---------------- ML ----------------
def train_model(df):
    features = ["ma10", "ma20", "returns"]
    X = df[features]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# ---------------- PIPELINE ----------------
try:
    df = fetch_data(symbol)
    df = build_features(df)

    model, features = train_model(df)

    latest = df.iloc[-1]
    prob = model.predict_proba([latest[features]])[0][1]

    if prob > threshold:
        signal = "BUY"
    elif prob < 1 - threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

except Exception as e:
    st.error("System error. Data source or pipeline failed.")
    st.code(str(e))
    st.stop()

# ---------------- DASHBOARD ----------------
st.subheader("Price Chart")
st.line_chart(df.set_index("date")["close"])

st.subheader("Moving Averages")
st.line_chart(df.set_index("date")[["close", "ma10", "ma20"]])

c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${latest['close']:.2f}")
c2.metric("ML Confidence (Up)", f"{prob:.2%}")
c3.metric("Signal", signal)

st.subheader("Latest Data")
st.dataframe(df.tail(10))




