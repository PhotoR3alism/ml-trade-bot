# train_complex.py (overwrite this file)

import os
import time
import joblib
import numpy as np
import pandas as pd
import ccxt
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── CONFIG ──────────────────────────────────────────────────────────────────────
SYMBOL      = "BTC-USD"                   # Will convert to BTC/USDT for ccxt
INTERVAL    = "1m"
HORIZON     = 60                          # Minutes ahead to predict
MODEL_DIR   = "models"
NEWSAPI_KEY = "YOUR_REAL_NEWSAPI_KEY"     # ← Paste your real key here
# ────────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_price(symbol=SYMBOL, interval=INTERVAL, lookback=7*24*60):
    ex   = ccxt.binance()
    pair = symbol.replace("-", "/").replace("USD","USDT")
    bars = ex.fetch_ohlcv(pair, timeframe=interval, limit=lookback)
    df = pd.DataFrame(bars, columns=["timestamp","Open","High","Low","Close","Volume"])
    df["timestamp"] //= 1000
    df.set_index("timestamp", inplace=True)
    return df

def fetch_news_sentiment(start_ts, end_ts):
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    frm = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start_ts))
    to  = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(end_ts))
    try:
        resp = client.get_everything(q="Bitcoin OR BTC", from_param=frm, to=to,
                                     language="en", page_size=100)
        articles = resp.get("articles", [])
    except Exception:
        articles = []
    rows = []
    for a in articles:
        ts = a.get("publishedAt","").rstrip("Z")
        try:
            t = time.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            minute = int(time.mktime(t))//60*60
        except:
            continue
        txt = a.get("title","").lower()
        s = 1 if "bull" in txt else -1 if "bear" in txt else 0
        rows.append({"timestamp":minute,"sentiment":s})
    if not rows:
        return pd.DataFrame([], columns=["sentiment"]).set_index(pd.Index([],name="timestamp"))
    df = pd.DataFrame(rows).groupby("timestamp").mean()
    return df

def engineer_features(df):
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema30"] = df["Close"].ewm(span=30).mean()
    df["macd"]  = df["ema10"] - df["ema30"]
    d = df["Close"].diff()
    up   = d.clip(lower=0).rolling(14).mean()
    down = d.clip(upper=0).abs().rolling(14).mean()
    df["rsi14"] = 100 - (100/(1+up/down))
    m20 = df["Close"].rolling(20).mean(); s20 = df["Close"].rolling(20).std()
    df["bb_up"] = m20 + 2*s20; df["bb_dn"] = m20 - 2*s20
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["vol_chg"] = df["Volume"].pct_change()
    return df.dropna()

def label_data(df):
    df["future"] = df["Close"].shift(-HORIZON)
    df["ret"]    = (df["future"]-df["Close"])/df["Close"]
    df["signal"] = pd.cut(df["ret"], bins=[-np.inf,-0.001,0.001,np.inf],
                          labels=["SELL","HOLD","BUY"])
    return df.dropna(subset=["signal"])

def main():
    price = fetch_price()
    news  = fetch_news_sentiment(price.index[0], price.index[-1])
    price = price.merge(news, left_index=True, right_index=True, how="left").fillna(0)
    df = engineer_features(price)
    df = label_data(df)
    feats = ["ema10","ema30","macd","rsi14","bb_up","bb_dn","atr14","vol_chg","sentiment"]
    X, y = df[feats], df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train,y_train)
    print(classification_report(y_test, model.predict(X_test)))
    out = os.path.join(MODEL_DIR, f"complex_{SYMBOL.replace('-','_')}_{INTERVAL}.pkl")
    joblib.dump(model, out)
    print("Saved model to", out)

if __name__=="__main__":
    main()
