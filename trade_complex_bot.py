```python
#!/usr/bin/env python3
import os
import time
import ccxt
import joblib
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from colorama import Fore, Style

# -- configuration --
HORIZON = 60  # minutes ahead for target/stop calculation
NEWS_KEY = os.getenv('NEWSAPI_KEY')
if not NEWS_KEY:
    raise RuntimeError("Please set NEWSAPI_KEY in your environment before running.")
client = NewsApiClient(api_key=NEWS_KEY)

# -- helper functions --
def fetch_ohlcv(ex, symbol, timeframe, limit):
    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    return df


def fetch_news_sentiment(start, end):
    # convert to UTC ISO strings
    frm = start.strftime('%Y-%m-%dT%H:%M:%S')
    to = end.strftime('%Y-%m-%dT%H:%M:%S')
    resp = client.get_everything(q='bitcoin OR btc', from_param=frm, to=to, language='en')
    articles = resp.get('articles', [])
    times = [datetime.fromisoformat(a['publishedAt'].rstrip('Z')) for a in articles]
    df = pd.DataFrame(times, columns=['timestamp'])
    df['sentiment'] = 0  # placeholder; implement sentiment analysis here
    df = df.set_index('timestamp').resample('1min').mean().fillna(0)
    df.index = df.index.tz_localize(None)
    return df


def compute_features(df):
    # ATR 14
    df['atr14'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    # MACD histogram
    df['macd_h'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    # RSI 14
    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = delta.clip(upper=0).abs().rolling(14).mean()
    df['rsi14'] = 100 - (100/(1 + up/down))
    # Moving avgs
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    # Bollinger Bands
    m = df['close'].rolling(20).mean()
    s = df['close'].rolling(20).std()
    df['bb_upper'] = m + 2*s
    df['bb_lower'] = m - 2*s
    # volume
    df['vol'] = df['volume']
    # engulf
    df['engulf'] = 0
    df.loc[(df['close']>df['open']) & (df['open'].shift(1)>df['close'].shift(1)) &
           (df['open']<df['close'].shift(1)) & (df['close']>df['open'].shift(1)), 'engulf'] = 1
    df.loc[(df['close']<df['open']) & (df['open'].shift(1)<df['close'].shift(1)) &
           (df['open']>df['close'].shift(1)) & (df['close']<df['open'].shift(1)), 'engulf'] = -1
    return df.dropna()


def print_signal(sig, price, tgt, stp, units, pl, rr, conf):
    mapping = {-1: ('SELL', Fore.RED), 0: ('NO TRADE', Fore.YELLOW), 1: ('BUY', Fore.GREEN)}
    label, col = mapping[sig]
    print(f"{col}[{label}]{Style.RESET_ALL} Conf={conf:.0%}")
    print(f"  Entry : {price:.2f}")
    print(f"  Target: {tgt:.2f}")
    print(f"  Stop  : {stp:.2f}")
    print(f"  Units : {units:.6f}")
    print(f"  P/L   : {pl:.2f}")
    print(f"  R/R   : {rr:.2f}")
    action = 'MAKE TRADE' if sig!=0 else 'SKIP TRADE'
    action_col = Fore.GREEN if sig==1 else Fore.RED if sig==-1 else Fore.YELLOW
    print(f"  {action_col}>>> {action} <<<{Style.RESET_ALL}\n")


# -- main loop --
def main():
    symbol = input("Ticker (e.g. BTC/USDT) [BTC/USDT]: ") or 'BTC/USDT'
    interval = input("Interval (e.g. 1m,5m,15m) [1m]: ") or '1m'
    lookback = int(input("Lookback bars [10080]: ") or 10080)
    invest = float(input("Invest per trade ($) [100.0]: ") or 100.0)

    # load model
    fname = f"complex_{symbol.replace('/','_')}_{interval}.pkl"
    model = joblib.load(os.path.join('models', fname))
    print(f"✓ Model loaded: models/{fname}\n")

    ex = ccxt.binance()
    df = fetch_ohlcv(ex, symbol, interval, lookback)
    news = fetch_news_sentiment(df.index[0], df.index[-1])
    df = df.rename(columns=str.title)
    df.index = df.index.tz_localize(None)
    merged = df.merge(news, left_index=True, right_index=True, how='left').fillna(0)

    while True:
        df = fetch_ohlcv(ex, symbol, interval, lookback+1)
        df = compute_features(df)
        df.index = df.index.tz_localize(None)
        merged = df.rename(columns=str.title).merge(news, left_index=True, right_index=True, how='left').fillna(0)
        feats = merged[['rsi14','macd_h','atr14','engulf','sentiment']]
        price = float(df['close'].iloc[-1])
        probs = model.predict_proba(feats.tail(1))[0]
        sig = int(model.classes_[probs.argmax()])
        tgt = price * (1 + HORIZON/price/100)
        stp = price * (1 - HORIZON/price/100)
        units = invest/price
        pl = (tgt-price)*units if sig==1 else (price-stp)*units if sig==-1 else 0
        rr = (tgt-price)/(price-stp) if sig!=0 else 0
        conf = max(probs)
        print_signal(sig, price, tgt, stp, units, pl, rr, conf)
        time.sleep(60)

if __name__ == '__main__':
    main()
```
