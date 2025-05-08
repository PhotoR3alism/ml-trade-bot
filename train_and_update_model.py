#!/usr/bin/env python3
import os
import pickle
import requests
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from polygon import RESTClient
import yfinance as yf
from colorama import init as colorama_init, Fore, Style
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Debug API keys
print("POLYGON_API_KEY in env:", os.getenv("POLYGON_API_KEY"))
print("NEWSAPI_KEY      in env:", os.getenv("NEWSAPI_KEY"))

# Initialize colorama and sentiment analyzer
colorama_init(autoreset=True, convert=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Setup Polygon client
POLY_KEY = os.getenv("POLYGON_API_KEY")
poly = None
if POLY_KEY:
    try:
        poly = RESTClient(POLY_KEY)
    except Exception as e:
        print(f"⚠️ Polygon init error, fallback to yfinance: {e}")

# Constants
MODEL_DIR = "models"
LOG_FILE = "trade_log.txt"

# Configure logging
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")

# Helper to print a labeled row with color
def print_row(label, value, color=Fore.WHITE):
    print(f"{color}{label:<15}{Style.RESET_ALL} {value}")

# Convert ticker to Polygon symbol
def polygon_symbol(ticker):
    t = ticker.upper()
    return "X:" + t.replace("-", "") if t.endswith("-USD") else t

# Fetch historical data
def fetch_historical(ticker, interval, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    # Try Polygon
    if poly:
        symbol = polygon_symbol(ticker)
        tf = 'day'
        if interval.endswith('m'): tf = 'minute'
        if interval.endswith('h'): tf = 'hour'
        try:
            aggs = poly.get_aggs(
                symbol, 1, tf,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d")
            )
            df = pd.DataFrame([{'timestamp': a.t, 'o': a.o, 'h': a.h, 'l': a.l, 'c': a.c, 'v': a.v} for a in aggs])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
            print_row("Bars (Poly)", len(df), Fore.CYAN)
            return df
        except Exception as e:
            print_row("Polygon error, fallback", str(e), Fore.YELLOW)

    # Fallback to yfinance
    iv = interval if not interval.isdigit() else interval + 'm'
    df = yf.download(ticker, period=f"{days}d", interval=iv, progress=False)
    if df.empty and iv.endswith('m'):
        df = yf.download(ticker, period=f"{days}d", interval='5m', progress=False)
    df.rename(columns=str.title, inplace=True)
    print_row("Bars (YF)", len(df), Fore.MAGENTA)
    return df

# Generate features
def generate_features(df, sentiment, lookback_pattern=60):
    c = df['Close']
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    mid = c.rolling(20).mean().iloc[-1]
    std = c.rolling(20).std().iloc[-1]
    bb_width = (std*2)/mid if mid else 0
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    ret = c.pct_change().dropna()
    pattern_score = 0
    if len(ret) > lookback_pattern*2:
        w = ret.values[-lookback_pattern:]
        hist = np.lib.stride_tricks.sliding_window_view(ret.values, lookback_pattern)[:-1]
        pattern_score = -np.linalg.norm(hist - w, axis=1).min()
    tod = datetime.utcnow().hour + datetime.utcnow().minute/60
    return np.array([ma20, ma50, bb_width, atr, pattern_score, sentiment, tod]).reshape(1, -1)

# Compute trade
def compute_trade(df, rr=2.0):
    c = df['Close']
    m20 = c.rolling(20).mean(); m50 = c.rolling(50).mean()
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    lower = mid - 2*std; upper = mid + 2*std
    last = len(df)-1; prev = last-1
    sig = 'HOLD'
    if m20.iloc[prev] <= m50.iloc[prev] < m20.iloc[last]: sig = 'BUY'
    if m20.iloc[prev] >= m50.iloc[prev] > m20.iloc[last]: sig = 'SELL'
    entry = c.iloc[last]
    stop = lower.iloc[last] if sig=='BUY' else upper.iloc[last]
    target = entry + (entry-stop)*rr if sig=='BUY' else entry - (stop-entry)*rr
    rrr = abs((target-entry)/(entry-stop)) if entry!=stop else 0
    return sig, entry, stop, target, rrr

# Main loop
if __name__ == '__main__':
    setup_logging()
    head, sent = None, 0.0
    nk = os.getenv("NEWSAPI_KEY")
    if nk:
        js = requests.get("https://newsapi.org/v2/top-headlines",
                          params={'language':'en','pageSize':1,'apiKey':nk}).json()
        if js.get('articles'):
            head = js['articles'][0]['title']
            sent = sentiment_analyzer.polarity_scores(head)['compound']
    if head:
        print_row("Headline", head, Fore.CYAN)
    print_row("NewsSent", f"{sent:.3f}", Fore.MAGENTA)

    ticker   = input("Ticker [BTC-USD]: ") or "BTC-USD"
    interval = input("Interval (1m,5m,1h) [1m]: ") or "1m"
    days     = int(input("Lookback days [7]: ") or "7")
    size_usd = float(input("Size USD [100]: ") or "100")
    print(Style.BRIGHT + f"\nStarting live updates every {interval} for {days}d...\n")

    sleep_sec = 60 if interval.endswith('m') else 3600 if interval.endswith('h') else 86400
    backoff = 1

    while True:
        try:
            df = fetch_historical(ticker, interval, days)
            backoff = 1
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too many" in msg:
                print_row("Rate-limit, backing off", f"{backoff*sleep_sec}s", Fore.YELLOW)
                time.sleep(backoff*sleep_sec)
                backoff = min(backoff*2, 16)
                continue
            else:
                raise

        if df is None or df.empty:
            print_row("No data fetched, skipping", "", Fore.YELLOW)
        else:
            sig, ent, st, tgt, rr = compute_trade(df)
            units = size_usd/ent if ent else 0
            pl_t = (tgt-ent)*units; pl_s = (st-ent)*units
            feats = generate_features(df, sent)
            ml_sig, ml_conf = None, None
            mfile = os.path.join(MODEL_DIR, 'complex_model.pkl')
            if os.path.exists(mfile):
                m = pickle.load(open(mfile,'rb'))
                X_model = feats[:, :4]
                ml_sig  = m.predict(X_model)[0]
                ml_conf = m.predict_proba(X_model)[0][1]

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(Style.BRIGHT + f"{ts} === Trade Call ===")
            print_row("Call", sig, Fore.GREEN if sig=='BUY' else Fore.RED)
            print_row("Entry", f"{ent:.2f}")
            print_row("Stop", f"{st:.2f}")
            print_row("Target", f"{tgt:.2f}")
            print_row("R/R", f"{rr:.2f}x")
            print_row("SizeUSD", f"{size_usd:.2f}")
            print_row("Units", f"{units:.6f}")
            print_row("P/Lt", f"${pl_t:.2f}")
            print_row("P/Ls", f"${pl_s:.2f}")
            if ml_sig is not None: print_row("MLCall", ml_sig)
            if ml_conf is not None: print_row("MLConf", f"{ml_conf:.0%}")
            print_row("NewsSent", f"{sent:.3f}")

        time.sleep(sleep_sec)
