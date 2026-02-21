from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Alpaca uses compact keys: t,o,h,l,c,v
    if "t" not in df.columns:
        return pd.DataFrame()
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df.empty or len(df) < period + 1:
        return float("nan")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float("nan")


def compute_features_from_5m(
    bars_5m: List[Dict[str, Any]],
    price: float,
    target_pcts: List[int],
) -> Dict[str, float]:
    """Compute a compact feature set for a single symbol at scan time.

    `price` should be the point-in-time P0 (midquote if fresh, else last 1m close).
    """

    df = _to_df(bars_5m)
    out: Dict[str, float] = {}
    if df.empty or len(df) < 20 or not np.isfinite(price) or price <= 0:
        return out

    close = df["close"].astype(float)
    vol = df["volume"].astype(float).fillna(0.0)

    # Returns
    ret_5m = np.log(close / close.shift(1))
    out["ret_5m"] = float(ret_5m.iloc[-1]) if pd.notna(ret_5m.iloc[-1]) else float("nan")
    out["ret_30m"] = float(np.log(close.iloc[-1] / close.iloc[-7])) if len(close) >= 7 else float("nan")
    out["ret_2h"] = float(np.log(close.iloc[-1] / close.iloc[-25])) if len(close) >= 25 else float("nan")

    # Realized volatility (annualization not needed for relative features)
    out["rv_1h"] = float(ret_5m.iloc[-12:].std()) if len(ret_5m) >= 13 else float("nan")
    out["rv_5h"] = float(ret_5m.iloc[-60:].std()) if len(ret_5m) >= 61 else float("nan")

    # Vol-of-vol (std of rolling 1h vol)
    if len(ret_5m) >= 120:
        rolling_rv = ret_5m.rolling(12).std()
        out["vol_of_vol"] = float(rolling_rv.iloc[-60:].std())
    else:
        out["vol_of_vol"] = float("nan")

    # ATR
    atr14 = compute_atr(df, 14)
    out["atr"] = float(atr14)
    out["atr_pct"] = float(atr14 / price) if np.isfinite(atr14) and atr14 > 0 else float("nan")

    # IMPORTANT anchor: distance-to-target in ATR units
    # m is decimal (e.g., 5% => 0.05). ATR% is also decimal (e.g., 1% => 0.01).
    atr_pct = out.get("atr_pct", float("nan"))
    for pct in target_pcts:
        m = float(pct) / 100.0
        key = f"dist_to_target_atr_{pct}"
        if np.isfinite(atr_pct) and atr_pct > 0:
            out[key] = float(m / atr_pct)
        else:
            out[key] = float("nan")

    # EMAs and slopes/spacings
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    ema50 = _ema(close, 50)
    out["ema12_ema26"] = float((ema12.iloc[-1] - ema26.iloc[-1]) / atr14) if np.isfinite(atr14) and atr14 > 0 else float("nan")
    out["ema12_ema50"] = float((ema12.iloc[-1] - ema50.iloc[-1]) / atr14) if np.isfinite(atr14) and atr14 > 0 else float("nan")
    out["ema_slope_12"] = float((ema12.iloc[-1] - ema12.iloc[-4]) / atr14) if len(ema12) >= 4 and np.isfinite(atr14) and atr14 > 0 else float("nan")

    # VWAP (2h) and location
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap2h = (typical * vol).rolling(24).sum() / (vol.rolling(24).sum() + 1e-12)
    out["vwap_loc"] = float((price - vwap2h.iloc[-1]) / atr14) if pd.notna(vwap2h.iloc[-1]) and np.isfinite(atr14) and atr14 > 0 else float("nan")

    # Participation / RVOL
    vol_30m = float(vol.iloc[-6:].sum()) if len(vol) >= 6 else float("nan")
    out["vol_30m"] = vol_30m
    if len(vol) >= 72:
        # compare 30m bucket to median bucket over last 6h
        buckets = vol.rolling(6).sum().dropna().iloc[-72:]
        med = float(buckets.median()) if len(buckets) else float("nan")
        out["rvol_30m"] = float(vol_30m / (med + 1e-12)) if np.isfinite(med) else float("nan")
    else:
        out["rvol_30m"] = float("nan")

    # Volume pressure proxy: close position within bar
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["close_pos"] = float(((df["close"] - df["low"]) / rng).iloc[-1]) if pd.notna(rng.iloc[-1]) else float("nan")

    # Notional 6h
    if len(df) >= 72:
        out["notional_6h"] = float((df["close"].iloc[-72:] * vol.iloc[-72:]).sum())
    else:
        out["notional_6h"] = float((df["close"] * vol).sum())

    return out
