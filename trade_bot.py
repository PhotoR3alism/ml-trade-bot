#!/usr/bin/env python3
import ccxt
import pandas as pd
import joblib
import time
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR        = "models"
DEFAULT_SYMBOL   = "BTC-USD"
DEFAULT_INTERVAL = "1m"
DEFAULT_HISTORY  = "7d"
DEFAULT_LOOKBACK = 360
DEFAULT_HORIZON  = 60
DEFAULT_INVEST   = 100.0
DEFAULT_LOGFILE  = "trade_log.txt"
# ────────────────────────────────────────────────────────────────────────────────

def prompt_inputs():
    sym      = input(f"Ticker (e.g. BTC-USD or BTC/USDT) [{DEFAULT_SYMBOL}]: ") or DEFAULT_SYMBOL
    interval = input(f"Interval (e.g. 1m,5m,15m) [{DEFAULT_INTERVAL}]: ") or DEFAULT_INTERVAL
    hist     = input(f"History (e.g. 7d,1h) [{DEFAULT_HISTORY}]: ") or DEFAULT_HISTORY
    lb       = input(f"Lookback bars (overrides history) [{DEFAULT_LOOKBACK}]: ")
    invest   = input(f"Invest per trade ($) [{DEFAULT_INVEST}]: ")
    logfile  = input(f"Log file name [{DEFAULT_LOGFILE}]: ") or DEFAULT_LOGFILE

    if interval.isdigit(): interval += "m"
    lookback = int(lb) if lb else None
    invest   = float(invest) if invest else DEFAULT_INVEST

    if lookback is None:
        num, unit = int(hist[:-1]), hist[-1]
        minutes = num * (1440 if unit=="d" else 60)
        lookback = minutes // int(interval[:-1])

    return sym, interval, lookback, invest, logfile


def normalize_symbol(raw, ex):
    s = raw.replace("-", "/").upper()
    if s.endswith("/USD"): s = s[:-4] + "/USDT"
    if s in ex.symbols: return s
    raise ValueError(f"Symbol {raw} → {s} not found")


def find_model_path(symbol, interval):
    import glob, os
    pat = os.path.join(MODEL_DIR, f"*{symbol.replace('/','_').lower()}*{interval}*.pkl")
    files = glob.glob(pat)
    if not files: raise FileNotFoundError(pat)
    return files[0]


def fetch_ohlcv(ex, sym, tf, limit):
    df = pd.DataFrame(ex.fetch_ohlcv(sym, timeframe=tf, limit=limit),
                      columns=["timestamp","open","high","low","close","volume"])
    df.set_index("timestamp", inplace=True)
    return df


def compute_features(df):
    df["atr14"]   = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["macd_h"]  = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    d = df["close"].diff()
    up   = d.clip(lower=0).rolling(14).mean()
    down = d.clip(upper=0).abs().rolling(14).mean()
    df["rsi14"]  = 100 - (100/(1+up/down))
    df["ma_10"]  = df["close"].rolling(10).mean()
    df["ma_30"]  = df["close"].rolling(30).mean()
    m20 = df["close"].rolling(20).mean(); s20 = df["close"].rolling(20).std()
    df["bb_upper"] = m20 + 2*s20; df["bb_lower"] = m20 - 2*s20
    df["vol"]    = df["volume"]
    df["engulf"] = 0
    df.loc[(df["close"]>df["open"]) & (df["open"].shift(1)>df["close"].shift(1)) &
           (df["open"]<df["close"].shift(1)) & (df["close"]>df["open"].shift(1)), "engulf"] = 1
    df.loc[(df["close"]<df["open"]) & (df["open"].shift(1)<df["close"].shift(1)) &
           (df["open"]>df["close"].shift(1)) & (df["close"]<df["open"].shift(1)), "engulf"] = -1
    return df.dropna()


def print_signal(sig, price, tgt, stp, units, pl, rr, conf):
    cols = {"BUY":Fore.GREEN,"SELL":Fore.RED,"HOLD":Fore.YELLOW}
    c = cols.get(sig, Fore.WHITE)
    confc = Fore.GREEN if conf>=.6 else Fore.YELLOW if conf>=.5 else Fore.RED
    plc   = Fore.GREEN if pl>=0 else Fore.RED
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Signal: {c}{sig:<4}{Style.RESET_ALL} Conf={confc}{conf:.0%}{Style.RESET_ALL}")
    # each on its own line:
    print(f"  Entry : {price:.2f}")
    print(f"  Target: {tgt:.2f}")
    print(f"  Stop  : {stp:.2f}")
    print(f"  Units : {units:.6f}")
    print(f"  P/L   : {plc}{pl:.2f}{Style.RESET_ALL}")
    print(f"  R/R   : {rr:.2f}")
    # explicit action:
    action = "MAKE TRADE" if sig in ("BUY","SELL") else "SKIP TRADE"
    print(f"  Action: {action}\n")


def main():
    sym, interval, lookback, invest, logfile = prompt_inputs()
    ex = ccxt.binance(); ex.load_markets()
    symbol = normalize_symbol(sym, ex)
    print(f"\n🚀 Live bot for {symbol} @ {interval}, lookback={lookback}")

    model = joblib.load(find_model_path(symbol, interval))
    df    = fetch_ohlcv(ex, symbol, interval, lookback+1)
    print(Fore.GREEN + f"✓ Seeded {len(df)} bars\n")
    last_ts = df.index[-1]

    with open(logfile,"a") as f:
        f.write("ts,sym,signal,conf,price,target,stop,units,pl,rr\n")

    while True:
        try:
            new = fetch_ohlcv(ex, symbol, interval, 2).iloc[-1:]
            if new.index[0] > last_ts:
                df = pd.concat([df,new]).iloc[-(lookback+1):]
                last_ts = new.index[0]

                feats_all = compute_features(df).iloc[[-1]]
                feats     = feats_all[model.feature_names_in_]
                print("DEBUG columns after filter:", list(feats.columns))

                probs = model.predict_proba(feats)[0]
                raw   = model.classes_[probs.argmax()]
                # raw might be numeric -1/0/1 or string; normalize:
                mapping = {1:"BUY", 0:"HOLD", -1:"SELL", "BUY":"BUY", "SELL":"SELL", "HOLD":"HOLD"}
                sig     = mapping.get(raw, "HOLD")
                conf    = probs.max()

                price = feats_all["close"].iloc[0]
                atr   = feats_all["atr14"].iloc[0]
                tgt   = price + 2*atr    # 2×ATR target
                stp   = price - 1*atr    # 1×ATR stop
                units = invest/price
                pl    = (tgt-price)*units
                rr    = (tgt-price)/(price-stp)

                print_signal(sig, price, tgt, stp, units, pl, rr, conf)
                with open(logfile,"a") as f:
                    f.write(f"{datetime.now()},{symbol},{sig},{conf:.4f},"
                            f"{price:.2f},{tgt:.2f},{stp:.2f},{units:.6f},{pl:.2f},{rr:.2f}\n")

            time.sleep(60)

        except Exception as e:
            print(Fore.YELLOW+f"⚠️ Fetch error: {e}; retrying in 60s…")
            time.sleep(60)

if __name__=="__main__":
    main()
