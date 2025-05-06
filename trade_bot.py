# trade_bot.py
import ccxt
import pandas as pd
import joblib
import time
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

MODEL_DIR        = "models"
DEFAULT_SYMBOL   = "BTC-USD"
DEFAULT_INTERVAL = "1m"
DEFAULT_HISTORY  = "7d"
DEFAULT_LOOKBACK = 360
DEFAULT_HORIZON  = 60
DEFAULT_INVEST   = 100.0
DEFAULT_LOGFILE  = "trade_log.txt"

def prompt_inputs():
    sym      = input(f"Ticker (e.g. BTC-USD or BTC/USDT) [{DEFAULT_SYMBOL}]: ").strip() or DEFAULT_SYMBOL
    interval = input(f"Interval (e.g. 1m,5m,15m) [{DEFAULT_INTERVAL}]: ").strip() or DEFAULT_INTERVAL
    hist     = input(f"History (e.g. 7d,1h) [{DEFAULT_HISTORY}]: ").strip() or DEFAULT_HISTORY
    lb       = input(f"Lookback bars (overrides history) [{DEFAULT_LOOKBACK}]: ").strip()
    horizon  = input(f"Horizon minutes ahead [{DEFAULT_HORIZON}]: ").strip()
    invest   = input(f"Invest per trade ($) [{DEFAULT_INVEST}]: ").strip()
    logfile  = input(f"Log file name [{DEFAULT_LOGFILE}]: ").strip() or DEFAULT_LOGFILE

    lookback = int(lb) if lb else None
    horizon  = int(horizon) if horizon else DEFAULT_HORIZON
    invest   = float(invest) if invest else DEFAULT_INVEST

    if lookback is None:
        num, unit = int(hist[:-1]), hist[-1]
        mins = num*24*60 if unit=="d" else num*60
        lookback = mins // int(interval[:-1])

    return sym, interval, lookback, horizon, invest, logfile

def normalize_symbol(raw, ex):
    s = raw.replace("-", "/").upper()
    if s.endswith("/USD"):
        s = s.replace("/USD","/USDT")
    if s in ex.symbols:
        return s
    raise ValueError(f"Symbol {raw} → {s} not on exchange")

def find_model_path(symbol, interval):
    import glob, os
    norm = symbol.replace("/", "_").replace("-", "_").lower()
    for f in glob.glob(os.path.join(MODEL_DIR,"*.pkl")):
        bn = os.path.basename(f).lower().replace("-", "_")
        if norm in bn and interval in bn:
            return f
    raise FileNotFoundError(f"No model for {symbol}@{interval} in {MODEL_DIR}")

def load_model(symbol, interval):
    path = find_model_path(symbol, interval)
    print(Fore.GREEN + "✓ Loaded model:", path)
    return joblib.load(path)

def fetch_ohlcv(ex, symbol, tf, limit):
    df = pd.DataFrame(
        ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit),
        columns=["timestamp","open","high","low","close","volume"]
    ).set_index("timestamp")
    return df

def compute_features(df):
    df["atr14"]  = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["macd_h"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    up  = df["close"].diff().clip(lower=0).rolling(14).mean()
    dn  = df["close"].diff().clip(upper=0).abs().rolling(14).mean()
    df["rsi14"] = 100 - (100/(1 + up/dn))
    return df.dropna()

def print_signal(rec, price, tgt, stp, units, pl, rr, conf):
    col   = {"BUY":Fore.GREEN, "SELL":Fore.RED, "HOLD":Fore.YELLOW}[rec]
    ccol  = Fore.GREEN if conf>=.6 else Fore.YELLOW if conf>=.5 else Fore.RED
    plcol = Fore.GREEN if pl>=0 else Fore.RED
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Signal: {col}{rec}{Style.RESET_ALL}  "
          f"P={price:.2f} T={tgt:.2f} S={stp:.2f} U={units:.6f}  "
          f"P/L={plcol}{pl:.2f}{Style.RESET_ALL}  R/R={rr:.2f}  Conf={ccol}{conf:.0%}{Style.RESET_ALL}")

def main():
    symbol, interval, lookback, horizon, invest, logfile = prompt_inputs()

    ex = ccxt.binance()
    ex.load_markets()
    symbol = normalize_symbol(symbol, ex)
    print(f"\n🚀 Live bot for {symbol} @ {interval}, lookback={lookback}, horizon={horizon}")

    model = load_model(symbol, interval)

    df = fetch_ohlcv(ex, symbol, interval, lookback+1)
    print(Fore.GREEN + f"✓ Seeded {len(df)} bars\n")
    last_ts = df.index[-1]

    with open(logfile, "a") as f:
        f.write("timestamp,symbol,signal,ml_conf,price,target,stop,units,pl,rr\n")

    while True:
        try:
            new = fetch_ohlcv(ex, symbol, interval, 2).iloc[-1:]
            if new.index[0] > last_ts:
                df = pd.concat([df, new]).iloc[-(lookback+1):]
                last_ts = new.index[0]

                feats = compute_features(df).iloc[[-1]]
                probs = model.predict_proba(feats)[0]
                rec   = model.classes_[probs.argmax()]
                conf  = probs.max()

                price = feats["close"].iloc[0]
                tgt   = price * (1 + 0.001)
                stp   = price * (1 - 0.001)
                units = invest/price
                pl    = (tgt - price) * units
                rr    = (tgt - price) / (price - stp)

                print_signal(rec, price, tgt, stp, units, pl, rr, conf)

                with open(logfile, "a") as f:
                    f.write(f"{datetime.now()},{symbol},{rec},{conf:.2f},"
                            f"{price:.2f},{tgt:.2f},{stp:.2f},{units:.6f},{pl:.2f},{rr:.2f}\n")

            time.sleep(60)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Fetch error: {e}; retrying in 60s…")
            time.sleep(60)

if __name__ == "__main__":
    main()
