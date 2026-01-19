# ============================================
# ML Trading System — Clean Functional Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ---------------- #
SYMBOLS = ["BTC-USD", "ETH-USD"]
DEFAULT_THRESHOLD = 0.6

# ---------------- DATA ---------------- #
def fetch_data(symbol):
    df = yf.download(symbol, period="6mo", interval="1d")
    df.dropna(inplace=True)
    return df

# ---------------- FEATURES ---------------- #
def build_features(df):
    df = df.copy()
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

# ---------------- MODEL ---------------- #
def train_model(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["SMA5", "SMA10", "Return"]][:-1]
    y = df["Target"][:-1]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# ---------------- SIGNAL ---------------- #
def predict_signal(df, model, threshold):
    latest = df.iloc[-1]
    X = latest[["SMA5", "SMA10", "Return"]].values.reshape(1, -1)

    prob = model.predict_proba(X)[0][1]

    if prob >= threshold:
        signal = "BUY"
    elif prob <= (1 - threshold):
        signal = "SELL"
    else:
        signal = "NO TRADE"

    return latest["Close"], prob, signal

# ---------------- BACKTEST ---------------- #
def backtest(df, model, threshold):
    wins = 0
    trades = 0

    for i in range(len(df) - 1):
        X = df[["SMA5", "SMA10", "Return"]].iloc[i:i+1]
        prob = model.predict_proba(X)[0][1]

        if prob >= threshold:
            trades += 1
            if df["Close"].iloc[i+1] > df["Close"].iloc[i]:
                wins += 1

        elif prob <= (1 - threshold):
            trades += 1
            if df["Close"].iloc[i+1] < df["Close"].iloc[i]:
                wins += 1

    winrate = (wins / trades * 100) if trades > 0 else 0

    return {
        "Trades": trades,
        "Wins": wins,
        "Winrate %": round(winrate, 2)
    }

# ---------------- PIPELINE ---------------- #
def run_pipeline(symbol, threshold):
    df = fetch_data(symbol)

    if df.empty or len(df) < 60:
        raise ValueError("Not enough market data.")

    df = build_features(df)
    model = train_model(df)
    price, prob, signal = predict_signal(df, model, threshold)
    stats = backtest(df, model, threshold)

    return df, price, prob, signal, stats

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="ML Trading System", layout="wide")
st.title("📈 ML Trading System — Functional Version")

symbol = st.selectbox("Symbol", SYMBOLS)
threshold = st.slider("Confidence Threshold", 0.50, 0.90, DEFAULT_THRESHOLD, 0.01)

try:
    df, price, prob, signal, stats = run_pipeline(symbol, threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${price:,.2f}")
    c2.metric("Probability Up", f"{prob*100:.2f}%")
    c3.metric("Signal", signal)

    st.subheader("Backtest Performance")
    st.json(stats)

    st.subheader("Price Chart")
    st.line_chart(df["Close"])

except Exception as e:
    st.error(str(e))

