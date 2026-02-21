from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _extract_ohlcv(bars: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not bars:
        z = np.array([], dtype=float)
        return z, z, z, z, z
    # Assume bars already filtered/sorted by time.
    o = np.array([float(b.get("o", np.nan)) for b in bars], dtype=float)
    h = np.array([float(b.get("h", np.nan)) for b in bars], dtype=float)
    l = np.array([float(b.get("l", np.nan)) for b in bars], dtype=float)
    c = np.array([float(b.get("c", np.nan)) for b in bars], dtype=float)
    v = np.array([float(b.get("v", 0.0)) for b in bars], dtype=float)
    return o, h, l, c, v


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_features_5m_fast(
    bars_5m: List[Dict[str, Any]],
    price: float,
    target_pcts: List[int],
) -> Dict[str, float]:
    """Fast feature computation from a small lookback slice (e.g., last 6h of 5m bars)."""
    out: Dict[str, float] = {}
    if not bars_5m or not np.isfinite(price) or price <= 0:
        return out

    # Ensure sorted
    bars = sorted(bars_5m, key=lambda b: b.get("t", ""))
    o, h, l, c, v = _extract_ohlcv(bars)
    n = len(c)
    if n < 20 or not np.isfinite(c[-1]):
        return out

    # Log returns
    ret = np.empty(n)
    ret[:] = np.nan
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))

    out["ret_5m"] = float(ret[-1]) if np.isfinite(ret[-1]) else float("nan")
    out["ret_30m"] = float(np.log(c[-1] / c[-7])) if n >= 7 and np.isfinite(c[-7]) else float("nan")
    out["ret_2h"] = float(np.log(c[-1] / c[-25])) if n >= 25 and np.isfinite(c[-25]) else float("nan")

    # Realized vols
    out["rv_1h"] = float(np.nanstd(ret[-12:])) if n >= 13 else float("nan")
    out["rv_5h"] = float(np.nanstd(ret[-60:])) if n >= 61 else float("nan")

    # Vol-of-vol
    if n >= 120:
        # rolling 1h vol over last 5h
        rvs = []
        for i in range(n - 60, n + 1):
            j0 = max(0, i - 12)
            rvs.append(np.nanstd(ret[j0:i]))
        out["vol_of_vol"] = float(np.nanstd(np.array(rvs)))
    else:
        out["vol_of_vol"] = float("nan")

    # ATR14
    if n >= 15:
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
        atr14 = np.nanmean(tr[-14:])
    else:
        atr14 = np.nan

    out["atr"] = float(atr14) if np.isfinite(atr14) else float("nan")
    atr_pct = float(atr14 / price) if np.isfinite(atr14) and atr14 > 0 else float("nan")
    out["atr_pct"] = atr_pct

    for pct in target_pcts:
        m = float(pct) / 100.0
        if np.isfinite(atr_pct) and atr_pct > 0:
            out[f"dist_to_target_atr_{pct}"] = float(m / atr_pct)
        else:
            out[f"dist_to_target_atr_{pct}"] = float("nan")

    # EMA spacings / slopes
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    ema50 = _ema(c, 50)
    if np.isfinite(atr14) and atr14 > 0:
        out["ema12_ema26"] = float((ema12[-1] - ema26[-1]) / atr14)
        out["ema12_ema50"] = float((ema12[-1] - ema50[-1]) / atr14)
        out["ema_slope_12"] = float((ema12[-1] - ema12[-4]) / atr14) if n >= 4 else float("nan")
    else:
        out["ema12_ema26"] = float("nan")
        out["ema12_ema50"] = float("nan")
        out["ema_slope_12"] = float("nan")

    # VWAP 2h location
    tp = (h + l + c) / 3.0
    win = 24
    if n >= win and np.nansum(v[-win:]) > 0:
        vwap = float(np.nansum(tp[-win:] * v[-win:]) / (np.nansum(v[-win:]) + 1e-12))
        out["vwap_loc"] = float((price - vwap) / atr14) if np.isfinite(atr14) and atr14 > 0 else float("nan")
    else:
        out["vwap_loc"] = float("nan")

    # RVOL 30m
    if n >= 6:
        vol_30m = float(np.nansum(v[-6:]))
    else:
        vol_30m = float("nan")
    out["vol_30m"] = vol_30m

    if n >= 72:
        bucket = np.array([np.nansum(v[i - 6 : i]) for i in range(6, n + 1)], dtype=float)
        bucket_last = bucket[-1] if bucket.size else np.nan
        med = float(np.nanmedian(bucket[-72:])) if bucket.size else np.nan
        out["rvol_30m"] = float(bucket_last / (med + 1e-12)) if np.isfinite(med) else float("nan")
    else:
        out["rvol_30m"] = float("nan")

    # Close position
    rng = h[-1] - l[-1]
    if np.isfinite(rng) and rng > 0:
        out["close_pos"] = float((c[-1] - l[-1]) / rng)
    else:
        out["close_pos"] = float("nan")

    # Notional 6h
    out["notional_6h"] = float(np.nansum(c * v))

    return out
