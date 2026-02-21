from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from .indicators import ema, atr, realized_vol, adx

FEATURES = [
    "ret_5m","ret_30m",
    "ema_align","adx14",
    "atr_pct","rv_30m",
    "rvol","obv_slope",
    "vwap_loc","dist_to_high",
    "btc_ret_30m",
    "time_to_horizon_frac",
    "range_so_far_atr",
    "dist_to_target_atr",
]

def candles_to_df(candles: List[List[float]]) -> pd.DataFrame:
    # [time, low, high, open, close, volume]
    df = pd.DataFrame(candles, columns=["ts","low","high","open","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    return df

def compute_features(df5: pd.DataFrame, df1: pd.DataFrame, btc5: Optional[pd.DataFrame], now: datetime, horizon_end: datetime, target_pct: float) -> Dict[str, float]:
    close=df5["close"]
    ret5 = close.pct_change(1)
    ret_5m = float(ret5.iloc[-1]) if len(ret5) else 0.0
    ret_30m = float(close.pct_change(6).iloc[-1]) if len(close) > 6 else 0.0

    e9=ema(close,9); e21=ema(close,21); e55=ema(close,55)
    ema_align = float((e9.iloc[-1]>e21.iloc[-1]) and (e21.iloc[-1]>e55.iloc[-1])) if len(close)>=55 else 0.0
    adx14 = float(adx(df5,14).iloc[-1]) if len(df5)>=30 else 0.0

    atr14 = atr(df5,14)
    atr_last = float(atr14.iloc[-1]) if len(atr14.dropna()) else float((df5["high"].iloc[-1]-df5["low"].iloc[-1]))
    atr_pct = float(atr_last/(close.iloc[-1]+1e-9))

    rv_30m = float(realized_vol(ret5,6).iloc[-1]) if len(ret5)>=6 else 0.0

    # volume relative: current 5m vol vs median of last 78 (6.5h)
    vol = df5["volume"]
    med = float(vol.tail(78).median()) if len(vol) else 0.0
    rvol = float(vol.iloc[-1]/(med+1e-9)) if med>0 else 1.0

    # OBV slope
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * vol).cumsum()
    obv_slope = float((obv.iloc[-1]-obv.iloc[-7])/(abs(obv.iloc[-7])+1e-9)) if len(obv)>7 else 0.0

    # VWAP (today-ish over df1 window)
    tp = (df1["high"]+df1["low"]+df1["close"])/3.0
    vwap = float((tp*df1["volume"]).sum()/(df1["volume"].sum()+1e-9))
    vwap_loc = float((df1["close"].iloc[-1]-vwap)/(atr_last+1e-9))

    # dist to high last 12 bars (1h)
    recent_high = float(df5["high"].tail(12).max())
    dist_to_high = float((recent_high - close.iloc[-1])/(atr_last+1e-9))

    btc_ret_30m = 0.0
    if btc5 is not None and len(btc5)>=7:
        btc_ret_30m = float(btc5["close"].pct_change(6).iloc[-1])

    # time to horizon
    tfrac = float(max(0.0, (horizon_end-now).total_seconds()) / (5*3600))
    tfrac = float(min(1.0, tfrac))

    # range so far in ATR (past 1h)
    rng = float((df5["high"].tail(12).max()-df5["low"].tail(12).min())/(atr_last+1e-9))

    dist_to_target_atr = float((target_pct/100.0)/(atr_pct+1e-9))

    return {
        "ret_5m": ret_5m,
        "ret_30m": ret_30m,
        "ema_align": ema_align,
        "adx14": adx14,
        "atr_pct": atr_pct,
        "rv_30m": rv_30m,
        "rvol": rvol,
        "obv_slope": obv_slope,
        "vwap_loc": vwap_loc,
        "dist_to_high": dist_to_high,
        "btc_ret_30m": btc_ret_30m,
        "time_to_horizon_frac": tfrac,
        "range_so_far_atr": rng,
        "dist_to_target_atr": dist_to_target_atr,
        "vwap": vwap,
        "price": float(df1["close"].iloc[-1]),
    }
