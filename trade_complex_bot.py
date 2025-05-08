
#!/usr/bin/env python3
import os
import pickle
import requests
import pandas as pd
import numpy as np
import logging
import time
import warnings
from datetime import datetime, timedelta
import ccxt
from colorama import init as colorama_init, Fore, Style
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Debug API key presence
print("NEWSAPI_KEY in env:", os.getenv("NEWSAPI_KEY"))

# Initialize colorama & sentiment analyzer
colorama_init(autoreset=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Setup CCXT exchange for crypto data (no key needed)
exchange = ccxt.binance()

# Constants
MODEL_DIR = "models"
LOG_FILE  = "trade_log.txt"

# Configure logging
def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")

# Helper to print a labeled row with color
def print_row(label, value, color=Fore.WHITE):
    print(f"{color}{label:<12}{Style.RESET_ALL} {value}")

# Fetch historical OHLCV: CCXT → CoinGecko fallback
def fetch_historical(ticker, interval, days):
    symbol = ticker.replace('-', '/').replace('USD', 'USDT')
    limit = days * 24 * (60 if interval.endswith('m') else 1)
    try:
        ohlc = exchange.fetch_ohlcv(symbol, timeframe=interval, since=None, limit=limit)
        df = pd.DataFrame(ohlc, columns=['ts','Open','High','Low','Close','Volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('timestamp', inplace=True)
        print_row("Bars (CCXT)", len(df), Fore.GREEN)
        return df[['Open','High','Low','Close','Volume']]
    except Exception as e:
        print_row("CCXT error", str(e), Fore.YELLOW)
    # CoinGecko fallback for BTC only
    try:
        data = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",
            params={"vs_currency":"usd","days":days}, timeout=5
        ).json()
        df2 = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
        df2.set_index('timestamp', inplace=True)
        df2['Volume'] = np.nan
        print_row("Bars (CoinGecko)", len(df2), Fore.BLUE)
        return df2
    except Exception as e2:
        print_row("CoinGecko err", str(e2), Fore.RED)
        raise RuntimeError("No data available from CCXT or CoinGecko")

# Feature engineering
def generate_features(df, sentiment, lookback_pattern=60):
    c = df['Close']
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    std  = c.rolling(20).std().iloc[-1]
    bbw  = (std*2)/(c.rolling(20).mean().iloc[-1])
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    atr = pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    ret = c.pct_change().dropna()
    pattern = 0
    if len(ret) > lookback_pattern*2:
        w = ret.values[-lookback_pattern:]
        hist = np.lib.stride_tricks.sliding_window_view(ret.values, lookback_pattern)[:-1]
        pattern = -np.linalg.norm(hist - w, axis=1).min()
    tod = datetime.utcnow().hour + datetime.utcnow().minute/60
    return np.array([ma20,ma50,bbw,atr,pattern,sentiment,tod]).reshape(1,-1)

# Compute trade signal & levels
def compute_trade(df, rr=2.0):
    c = df['Close']
    m20 = c.rolling(20).mean(); m50 = c.rolling(50).mean()
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    lower = mid - 2*std; upper = mid + 2*std
    i,len_df = len(df)-1,len(df)
    sig = 'HOLD'
    if m20.iloc[-2] <= m50.iloc[-2] < m20.iloc[-1]: sig='BUY'
    if m20.iloc[-2] >= m50.iloc[-2] > m20.iloc[-1]: sig='SELL'
    entry = c.iloc[-1]
    stop  = lower.iloc[-1] if sig=='BUY' else upper.iloc[-1]
    tgt   = entry + (entry-stop)*rr if sig=='BUY' else entry - (stop-entry)*rr
    rrr   = abs((tgt-entry)/(entry-stop)) if entry!=stop else 0
    return sig,entry,stop,tgt,rrr

# Main loop
if __name__=='__main__':
    setup_logging()
    # News sentiment
    head,sent=None,0.0
    nk=os.getenv("NEWSAPI_KEY")
    if nk:
        try:
            js=requests.get("https://newsapi.org/v2/top-headlines",
                params={'language':'en','pageSize':1,'apiKey':nk},timeout=5).json()
            if js.get('articles'):
                head=js['articles'][0]['title']
                sent=sentiment_analyzer.polarity_scores(head)['compound']
        except: pass
    if head: print_row("Headline",head,Fore.CYAN)
    print_row("NewsSent",f"{sent:.3f}",Fore.MAGENTA)

    # ——— User inputs ———
    ticker   = input("Ticker [BTC-USD]: ") or "BTC-USD"
    interval = input("Interval (1m,5m,1h) [1m]: ") or "1m"
    if interval.isdigit(): interval += 'm'
    days     = int(input("Lookback days [7]: ") or "7")
    size_usd = float(input("Size USD [100]: ") or "100")
    # —————————————————

    print(Style.BRIGHT+f"\nStarting live updates every {interval} for {days}d...\n")
    sleep_sec=60 if interval.endswith('m') else 3600 if interval.endswith('h') else 86400

    while True:
        try:
            df = fetch_historical(ticker,interval,days)
        except:
            time.sleep(sleep_sec)
            continue

        sig,ent,st,tgt,rr = compute_trade(df)
        units=size_usd/ent if ent else 0
        pl_t=(tgt-ent)*units; pl_s=(st-ent)*units
        feats=generate_features(df,sent)
        ml_sig,ml_conf=None,None
        mf=os.path.join(MODEL_DIR,'complex_model.pkl')
        if os.path.exists(mf):
            m=pickle.load(open(mf,'rb'))
            X=feats[:,:4]
            ml_sig=m.predict(X)[0]
            ml_conf=m.predict_proba(X)[0][1]

        ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(Style.BRIGHT+f"{ts} === Trade Call ===")
        col = Fore.GREEN if sig=='BUY' else Fore.RED if sig=='SELL' else Fore.YELLOW
        print_row("Call", sig, col)
        print_row("Entry",f"{ent:.2f}")
        print_row("Stop",f"{st:.2f}")
        print_row("Target",f"{tgt:.2f}")
        print_row("R/R",f"{rr:.2f}x")
        print_row("SizeUSD",f"{size_usd:.2f}")
        print_row("Units",f"{units:.6f}")
        print_row("P/Lt",f"${pl_t:.2f}")
        print_row("P/Ls",f"${pl_s:.2f}")
        if ml_sig: print_row("MLCall",ml_sig)
        if ml_conf is not None: print_row("MLConf",f"{ml_conf:.0%}")
        print_row("NewsSent",f"{sent:.3f}")
        time.sleep(sleep_sec)

