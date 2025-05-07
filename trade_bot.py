#!/usr/bin/env python3
import ccxt
import pandas as pd
import joblib
import time
import sys
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

# Configuration
MODEL_DIR = "models"
DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_INTERVAL = "1m"
DEFAULT_HISTORY = "7d"
DEFAULT_LOOKBACK = 360
DEFAULT_INVEST = 100.0
DEFAULT_LOGFILE = "trade_log.txt"

# Prompt user for inputs
def prompt_inputs():
    sym = input(f"Ticker [{DEFAULT_SYMBOL}]: ") or DEFAULT_SYMBOL
    interval = input(f"Interval [{DEFAULT_INTERVAL}]: ") or DEFAULT_INTERVAL
    hist = input(f"History (e.g. 7d,1h) [{DEFAULT_HISTORY}]: ") or DEFAULT_HISTORY
    lb = input(f"Lookback bars (overrides history) [{DEFAULT_LOOKBACK}]: ")
    invest = input(f"Invest per trade ($) [{DEFAULT_INVEST}]: ")
    logfile = input(f"Log file name [{DEFAULT_LOGFILE}]: ") or DEFAULT_LOGFILE

    if interval.isdigit():
        interval += "m"
    lookback = int(lb) if lb else None
    invest = float(invest) if invest else DEFAULT_INVEST
    if lookback is None:
        num, unit = int(hist[:-1]), hist[-1]
        minutes = num * (1440 if unit == 'd' else 60)
        lookback = minutes // int(interval[:-1])
    return sym, interval, lookback, invest, logfile

# Normalize symbol to exchange format
def normalize_symbol(raw, ex):
    s = raw.replace("-", "/").upper()
    if s.endswith("/USD"):
        s = s[:-4] + "/USDT"
    if s in ex.symbols:
        return s
    raise ValueError(f"Symbol {raw} -> {s} not supported on exchange")

# Find the model pickle file
def find_model_path(symbol, interval):
    import glob, os
    pattern = os.path.join(MODEL_DIR, f"*{symbol.replace('/','_').lower()}*{interval}*.pkl")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model matching {symbol}@{interval}")
    return files[0]

# Load the model
def load_model(symbol, interval):
    path = find_model_path(symbol, interval)
    print(Fore.GREEN + "Loaded model: " + path)
    return joblib.load(path)

# Fetch OHLCV from exchange
def fetch_ohlcv(ex, symbol, timeframe, limit):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    return df

# Compute features for model
def compute_features(df):
    df["atr14"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["macd_h"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = delta.clip(upper=0).abs().rolling(14).mean()
    df["rsi14"] = 100 - (100/(1+up/down))
    m20 = df["close"].rolling(20).mean()
    s20 = df["close"].rolling(20).std()
    df["bb_upper"] = m20 + 2*s20
    df["bb_lower"] = m20 - 2*s20
    df["vol"] = df["volume"]
    df["engulf"] = 0
    # bullish engulf
    df.loc[(df["close"]>df["open"]) &
           (df["open"].shift(1)>df["close"].shift(1)) &
           (df["open"]<df["close"].shift(1)) &
           (df["close"]>df["open"].shift(1)), "engulf"] = 1
    # bearish engulf
    df.loc[(df["close"]<df["open"]) &
           (df["open"].shift(1)<df["close"].shift(1)) &
           (df["open"]>df["close"].shift(1)) &
           (df["close"]<df["open"].shift(1)), "engulf"] = -1
    return df.dropna()

# Print formatted signal
def print_signal(sig, entry, target, stop, units, pl, rr, conf):
    colors = {"BUY":Fore.GREEN, "SHORT":Fore.RED, "NO TRADE":Fore.RED}
    c = colors.get(sig, Fore.WHITE)
    confc = Fore.GREEN if conf>=0.6 else Fore.YELLOW if conf>=0.5 else Fore.RED
    plc = Fore.GREEN if pl>=0 else Fore.RED
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Signal: {c}{sig}{Style.RESET_ALL}  Conf={confc}{conf:.0%}{Style.RESET_ALL}")
    print(f"  Entry : {entry:.2f}")
    print(f"  Target: {target:.2f}")
    print(f"  Stop  : {stop:.2f}")
    print(f"  Units : {units:.6f}")
    print(f"  P/L   : {plc}{pl:.2f}{Style.RESET_ALL}")
    print(f"  R/R   : {rr:.2f}\n")
    action = "MAKE TRADE" if sig in ("BUY","SHORT") else "NO TRADE"
    print(f"  >>> {action} <<<\n")

# Main loop
def main():
    sym, interval, lookback, invest, logfile = prompt_inputs()
    ex = ccxt.binance()
    ex.load_markets()
    symbol = normalize_symbol(sym, ex)
    print(f"Live bot for {symbol} @ {interval}, lookback={lookback}")

    model = load_model(symbol, interval)
    df = fetch_ohlcv(ex, symbol, interval, lookback+1)
    print(Fore.GREEN + f"Seeded {len(df)} bars\n")
    last = df.index[-1]

    with open(logfile, "a") as f:
        f.write("timestamp,symbol,signal,conf,entry,target,stop,units,pl,rr\n")

    try:
        while True:
            new = fetch_ohlcv(ex, symbol, interval, 2).iloc[-1:]
            if new.index[0] > last:
                df = pd.concat([df, new]).iloc[-(lookback+1):]
                last = new.index[0]

                feats_all = compute_features(df).iloc[[-1]]
                feats = feats_all[model.feature_names_in_]

                probs = model.predict_proba(feats)[0]
                raw = model.classes_[probs.argmax()]
                mapping = {1:"BUY", 0:"NO TRADE", -1:"SHORT"}
                sig = mapping.get(raw, "NO TRADE")
                conf = probs.max()

                entry = feats_all["close"].iloc[0]
                atr = feats_all["atr14"].iloc[0]
                target = entry + 2 * atr
                stop = entry - atr
                units = invest / entry
                pl = (target - entry) * units if sig=="BUY" else (entry - stop) * units if sig=="SHORT" else 0
                rr = (target - entry) / (entry - stop)

                print_signal(sig, entry, target, stop, units, pl, rr, conf)
                with open(logfile, "a") as f:
                    f.write(f"{datetime.now()},{symbol},{sig},{conf:.4f},{entry:.2f},{target:.2f},{stop:.2f},{units:.6f},{pl:.2f},{rr:.2f}\n")

            time.sleep(60)
    except KeyboardInterrupt:
        print("Exiting... goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
