from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    high=df["high"]; low=df["low"]; close=df["close"]
    prev_close=close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(window)

def adx(df: pd.DataFrame, period: int=14) -> pd.Series:
    high=df["high"]; low=df["low"]; close=df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / (atr_+1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / (atr_+1e-9)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di)+1e-9)
    return dx.rolling(period, min_periods=period).mean()
