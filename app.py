"""
ML Trading System — Functional Version
------------------------------------
This version adds:
- Risk management (ATR-based Stop Loss / Take Profit)
- Backtesting engine
- Performance metrics
- Streamlit dashboard (if available)
- CLI fallback mode

This makes the system FUNCTIONAL (not just a signal viewer).
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ================= SAFE STREAMLIT IMPORT ================= #
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# ================= DATA FETCH WITH CACHING ================= #
CACHE_FILE = "crypto_cache.csv"

def fetch_data(symbol="BTC", currency="USDT", limit=500):
    try:
        df = pd.read_csv(CACHE_FILE, parse_dates=["time"])
        if not df.empty and df["time"].max() > pd.Timestamp.now() - pd.Timedelta(hours=1):
            return df
    except:
        pass

    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    try:
        r = requests.get(url, params={"fsym": symbol, "tsym": currency, "limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json()["Data"]["Data"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.to_csv(CACHE_FILE, index=False)
        return df
    except:
        return pd.DataFrame()

# ================= INDICATORS ================= #
def ema(series, p): return series.ewm(span=p, adjust=False).mean()

def rsi(series, p=14):
    d = series.diff()
    gain = d.where(d > 0, 0).rolling(p).mean()
    loss = -d.where(d < 0, 0).rolling(p).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series): return ema(series, 12) - ema(series, 26)

def atr(df, p=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# ================= FEATURE ENGINEERING ================= #
def build_features(df):
    if df.empty: return df
    df = df.copy()
    df["EMA20"] = ema(df["close"], 20)
    df["EMA50"] = ema(df["close"], 50)
    df["RSI"] = rsi(df["close"])
    df["MACD"] = macd(df["close"])
    df["ATR"] = atr(df)
    df["TARGET"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

# ================= MODEL ================= #
def train_model(df):
    if df.empty: return None
    X = df[["EMA20", "EMA50", "RSI", "MACD", "ATR"]]
    y = df["TARGET"]
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

# ================= RISK MANAGEMENT ================= #
def compute_sl_tp(entry, atr, sl_mult=1.5, rr=2):
    stop_loss = entry - sl_mult * atr
    take_profit = entry + sl_mult * atr * rr
    return stop_loss, take_profit

# ================= BACKTEST ENGINE ================= #
def backtest(df, model, threshold=0.6, capital=10000, risk_per_trade=0.01):
    balance = capital
    trades = []
    position = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        X = row[["EMA20", "EMA50", "RSI", "MACD", "ATR"]].values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        price = row["close"]

        # ENTRY
        if position is None and prob >= threshold:
            sl, tp = compute_sl_tp(price, row["ATR"])
            risk_amount = balance * risk_per_trade
            qty = risk_amount / (price - sl)
            position = {
                "entry": price,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": row["time"]
            }

        # EXIT
        elif position is not None:
            if row["low"] <= position["sl"]:
                pnl = (position["sl"] - position["entry"]) * position["qty"]
                balance += pnl
                trades.append(pnl)
                position = None

            elif row["high"] >= position["tp"]:
                pnl = (position["tp"] - position["entry"]) * position["qty"]
                balance += pnl
                trades.append(pnl)
                position = None

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    winrate = wins / total * 100 if total else 0
    roi = (balance - capital) / capital * 100

    return {
        "final_balance": round(balance, 2),
        "roi": round(roi, 2),
        "trades": total,
        "winrate": round(winrate, 2)
    }

# ================= CORE PIPELINE ================= #
def run_pipeline(symbol="BTC", threshold=0.6):
    df = fetch_data(symbol)
    df = build_features(df)
    model = train_model(df)
    if df.empty or model is None:
        return None, None, None

    latest = df.iloc[-1]
    X = latest[["EMA20", "EMA50", "RSI", "MACD", "ATR"]].values.reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    signal = "BUY" if prob >= threshold else "NO TRADE"

    stats = backtest(df, model, threshold)

    return latest, prob, signal, stats, df

# ================= STREAMLIT DASHBOARD ================= #
if STREAMLIT_AVAILABLE:
    st.set_page_config("AI Trading System", layout="wide")
    st.title("ML Trading System — Functional Version")

    symbol = st.sidebar.selectbox("Symbol", ["BTC", "ETH"])
    threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.75, 0.6, 0.01)

    latest, prob, signal, stats, df = run_pipeline(symbol, threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"{latest['close']:.2f}")
    c2.metric("Probability Up", f"{prob:.2%}")
    c3.metric("Signal", signal)

    st.subheader("Backtest Performance")
    st.json(stats)

    st.subheader("Price Chart")
    st.line_chart(df.set_index("time")["close"])

# ================= CLI MODE ================= #
else:
    if __name__ == "__main__":
        latest, prob, signal, stats, df = run_pipeline("BTC", 0.6)
        print("Price:", latest["close"])
        print("Probability:", prob)
        print("Signal:", signal)
        print("Backtest:", stats)
        assert 0 <= prob <= 1, "Probability out of bounds"
