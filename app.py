# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ML Trading System", layout="wide")

st.title("ML Trading System — Clean Functional Version")

# -----------------------------
# Sidebar Controls
# -----------------------------
symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "MSFT"])
confidence_threshold = st.slider("Confidence Threshold", 0.50, 0.90, 0.60)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(sym):
    df = yf.download(sym, period="1y", interval="1d")
    df.dropna(inplace=True)
    return df

df = load_data(symbol)

st.subheader("Market Data")
st.dataframe(df.tail())

# -----------------------------
# Feature Engineering
# -----------------------------
df["return"] = df["Close"].pct_change()
df["ma10"] = df["Close"].rolling(10).mean()
df["ma20"] = df["Close"].rolling(20).mean()
df["volatility"] = df["return"].rolling(10).std()

df.dropna(inplace=True)

df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

features = ["return", "ma10", "ma20", "volatility"]
X = df[features]
y = df["target"]

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.subheader("Model Accuracy")
st.write(f"{accuracy:.2%}")

# -----------------------------
# Prediction
# -----------------------------
latest_features = X.iloc[-1:].values
prediction = model.predict(latest_features)[0]
proba = model.predict_proba(latest_features)[0]
confidence = np.max(proba)

# -----------------------------
# Trading Decision (NO Series used in if)
# -----------------------------
st.subheader("AI Trading Signal")

if confidence < confidence_threshold:
    st.warning("HOLD — Low confidence")
else:
    if prediction == 1:
        st.success(f"BUY — Confidence: {confidence:.2%}")
    else:
        st.error(f"SELL — Confidence: {confidence:.2%}")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Price Chart")
st.line_chart(df["Close"])

st.subheader("Moving Averages")
st.line_chart(df[["Close", "ma10", "ma20"]])


