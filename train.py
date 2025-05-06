# train.py
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_data(symbol="BTC-USD", period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.rename_axis("timestamp").reset_index()
    df["timestamp"] = df["timestamp"].astype(int) // 10**9
    return df.set_index("timestamp")

def compute_features(df):
    df["atr14"]  = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    df["macd_h"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    up  = df["Close"].diff().clip(lower=0).rolling(14).mean()
    dn  = df["Close"].diff().clip(upper=0).abs().rolling(14).mean()
    df["rsi14"] = 100 - (100/(1 + up/dn))
    return df.dropna()[["atr14","macd_h","rsi14"]]

def compute_labels(df, horizon=60):
    df["future_close"] = df["Close"].shift(-horizon)
    df["signal"] = "HOLD"
    df.loc[df["future_close"] > df["Close"], "signal"] = "BUY"
    df.loc[df["future_close"] < df["Close"], "signal"] = "SELL"
    return df["signal"][:-horizon]

def main():
    df = fetch_data()
    feats = compute_features(df)
    labels= compute_labels(df)
    data = feats.join(labels).dropna()
    X = data[["atr14","macd_h","rsi14"]]
    y = data["signal"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    fname = f"BTC-USD_1m_model.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    print("Saved model to", os.path.join(MODEL_DIR, fname))

if __name__=="__main__":
    main()
