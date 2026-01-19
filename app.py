# ================================
# CLEAN FUNCTIONAL ML TRADING SYSTEM
# Price DataFrame is NEVER overwritten
# ML DataFrame is isolated for modeling
# ================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Trading System — Functional", layout="wide")

# -----------------------------
# Data Loader
# -----------------------------
@st.cache_data(ttl=300)
def load_price_data(symbol: str):
    df = yf.download(symbol, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df

# -----------------------------
# Indicator Engine (PRICE DF ONLY)
# -----------------------------
def add_indicators(price_df: pd.DataFrame):
    df = price_df.copy()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["returns"] = df["close"].pct_change()
    return df

# -----------------------------
# ML Pipeline (ISOLATED DF)
# -----------------------------
def ml_pipeline(price_df: pd.DataFrame):
    df = price_df.copy()

    df["returns"] = df["close"].pct_change()
    df["target"] = (df["returns"].shift(-1) > 0).astype(int)
    df = df.dropna()

    features = df[["returns"]]
    target = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    latest_return = scaler.transform([[features.iloc[-1, 0]]])
    prob = model.predict_proba(latest_return)[0][1]

    return float(prob)

# -----------------------------
# UI
# -----------------------------
st.title("📈 ML Trading System — Functional Version")

symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
threshold = st.slider("Confidence Threshold", 0.50, 0.90, 0.60)

try:
    raw_df = load_price_data(symbol)
    price_df = add_indicators(raw_df)
    prob = ml_pipeline(raw_df)

    signal = "BUY" if prob > threshold else "SELL"

except Exception as e:
    st.error("System error. Data source or pipeline failed.")
    st.code(e)
    st.stop()

# -----------------------------
# Metrics
# -----------------------------
latest = price_df.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("Price", round(latest["close"], 2))
c2.metric("Probability Up", f"{prob:.2%}")
c3.metric("Signal", signal)

# -----------------------------
# Charts
# -----------------------------
st.subheader("Price Chart")
st.line_chart(price_df.set_index("date")["close"])

st.subheader("Moving Averages")
required = {"close", "ma10", "ma20"}

if required.issubset(price_df.columns):
    st.line_chart(price_df.set_index("date")[["close", "ma10", "ma20"]])
else:
    st.error(f"Missing columns: {required - set(price_df.columns)}")

# -----------------------------
# Debug (optional)
# -----------------------------
with st.expander("Debug Data"):
    st.write(price_df.tail())



