#!/usr/bin/env python3
import os
import pickle
import requests
import pandas as pd
import numpy as np
import logging
import time
import warnings
from datetime import datetime
import ccxt
from colorama import init as colorama_init, Fore, Style
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
print("NEWSAPI_KEY in env:", os.getenv("NEWSAPI_KEY"))

# Initialize colorama & sentiment analyzer
colorama_init(autoreset=True)
sent_analyzer = SentimentIntensityAnalyzer()
# Setup CCXT exchange
exchange      = ccxt.binance()
# Paths
MODEL_DIR     = "models"
LOG_FILE      = "trade_log.txt"
# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")

def print_row(label, value, color=Fore.WHITE):
    print(f"{color}{label:<12}{Style.RESET_ALL} {value}")

# Fetch historical bars
def fetch_historical(ticker, interval, days):
    sym   = ticker.replace('-', '/').replace('USD','USDT')
    limit = days * 24 * (60 if interval.endswith('m') else 1)
    ohlc  = exchange.fetch_ohlcv(sym, timeframe=interval, limit=limit)
    df    = pd.DataFrame(ohlc, columns=['ts','Open','High','Low','Close','Volume'])
    df['timestamp'] = pd.to_datetime(df['ts'],unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Compute entry, stop, target, R/R
def compute_trade(df, rr=2.0):
    c   = df['Close']
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    lower = mid - 2 * std
    entry  = c.iloc[-1]
    stop   = lower.iloc[-1]
    target = entry + (entry - stop) * rr
    rrr    = abs((target - entry) / (entry - stop)) if entry != stop else 0
    return entry, stop, target, rrr

# Generate ML features
def generate_features(df, sent):
    c    = df['Close']
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    std  = c.rolling(20).std().iloc[-1]
    bbw  = (std * 2) / ma20
    hl   = df['High'] - df['Low']
    hc   = (df['High'] - df['Close'].shift()).abs()
    lc   = (df['Low']  - df['Close'].shift()).abs()
    atr  = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    return np.array([ma20, ma50, bbw, atr, sent]).reshape(1, -1)

if __name__ == '__main__':
    # News sentiment
    sent = 0.0
    nk   = os.getenv("NEWSAPI_KEY")
    if nk:
        try:
            js   = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={'apiKey': nk, 'pageSize': 1}, timeout=5
            ).json()
            sent = sent_analyzer.polarity_scores(js['articles'][0]['title'])['compound']
        except:
            pass

    # User inputs
    ticker   = input("Ticker [BTC-USD]: ") or "BTC-USD"
    interval = input("Interval (1m,5m,1h) [1m]: ") or "1m"
    if interval.isdigit():
        interval += 'm'
    days     = int(input("Lookback days [7]: ") or "7")
    size_usd = float(input("Size USD [100]: ") or "100")

    # Load ML model
    mfile = os.path.join(MODEL_DIR, 'complex_model.pkl')
    model = pickle.load(open(mfile, 'rb')) if os.path.exists(mfile) else None

    # Determine sleep interval
    sleep_sec = 60 if interval.endswith('m') else 3600 if interval.endswith('h') else 86400
    print(Style.BRIGHT + f"\nStarting live ML updates every {interval} for {days}d...\n")

    # Main loop
    while True:
        try:
            df = fetch_historical(ticker, interval, days)
        except Exception:
            time.sleep(sleep_sec)
            continue

        entry, stop, target, rr = compute_trade(df)
        units = size_usd / entry if entry else 0
        pl_t  = (target - entry) * units
        pl_s  = (stop - entry) * units

        feats = generate_features(df, sent)
        if model:
            pred = model.predict(feats[:, :4])[0]
            conf = model.predict_proba(feats[:, :4]).max()
        else:
            pred, conf = 'N/A', 0.0

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(Style.BRIGHT + f"{ts} === ML TRADE SIGNAL ===")

        sig_col = Fore.GREEN if pred == 'BUY' else Fore.RED if pred == 'SELL' else Fore.YELLOW
        print_row("ML_SIGNAL", pred, sig_col)
        print_row("ML_CONF",   f"{conf:.0%}", sig_col)

        print_row("ENTRY",     f"{entry:.2f}")
        print_row("STOP",      f"{stop:.2f}")
        print_row("TARGET",    f"{target:.2f}")
        print_row("R/R",       f"{rr:.2f}x")

        print_row("SIZE_USD",  f"{size_usd:.2f}")
        print_row("UNITS",     f"{units:.6f}")

        print_row("P/L_TGT",   f"${pl_t:.2f}", Fore.GREEN if pl_t > 0 else Fore.RED if pl_t < 0 else Fore.YELLOW)
        print_row("P/L_STOP",  f"${pl_s:.2f}", Fore.GREEN if pl_s > 0 else Fore.RED if pl_s < 0 else Fore.YELLOW)

        print_row("NEWS_SNT",  f"{sent:.3f}", Fore.GREEN if sent > 0 else Fore.RED if sent < 0 else Fore.YELLOW)

        time.sleep(sleep_sec)
