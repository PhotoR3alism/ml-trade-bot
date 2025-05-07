def compute_features(df):
    # ATR 14
    df["atr14"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    # MACD histogram
    df["macd_h"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    # RSI 14
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = delta.clip(upper=0).abs().rolling(14).mean()
    df["rsi14"] = 100 - (100/(1 + up/down))
    # Moving averages
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_30"] = df["close"].rolling(30).mean()
    # Bollinger Bands (20)
    m = df["close"].rolling(20).mean()
    s = df["close"].rolling(20).std()
    df["bb_upper"] = m + 2*s
    df["bb_lower"] = m - 2*s
    # Volume feature
    df["vol"] = df["volume"]
    # Engulfing pattern
    df["engulf"] = 0
    # bullish engulf
    df.loc[
      (df["close"] > df["open"]) &
      (df["open"].shift(1) > df["close"].shift(1)) &
      (df["open"] < df["close"].shift(1)) &
      (df["close"] > df["open"].shift(1)),
      "engulf"
    ] = 1
    # bearish engulf
    df.loc[
      (df["close"] < df["open"]) &
      (df["open"].shift(1) < df["close"].shift(1)) &
      (df["open"] > df["close"].shift(1)) &
      (df["close"] < df["open"].shift(1)),
      "engulf"
    ] = -1

    return df.dropna()
