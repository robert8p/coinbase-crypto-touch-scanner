from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .coinbase import CoinbaseClient
from .alpaca_crypto import AlpacaCryptoClient
from .config import Settings
from .features import compute_features
from .modeling import load_artifacts, ModelArtifacts

logger = logging.getLogger(__name__)

def parse_pcts(pcts: str) -> list[int]:
    out=[]
    for part in str(pcts or "").split(","):
        part=part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return sorted(set(out))


@dataclass
class ProductScore:
    product_id: str
    base_price: float
    pct_move_to_target: float
    probs: Dict[int, float]
    rvol: float
    atr_pct: float
    ret_30m: float
    vwap_loc: float


def _candles_to_df(candles: List[List[float]]) -> pd.DataFrame:
    """Convert Coinbase candles response to a DataFrame.

    Coinbase candles are returned as [time, low, high, open, close, volume]
    with `time` as Unix seconds.
    """
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time")
    df = df.set_index("time")
    return df


async def _fetch_5m_history(
    cb: CoinbaseClient,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
) -> Optional[pd.DataFrame]:
    try:
        candles = await cb.get_candles(product_id, start=start, end=end, granularity=granularity)
    except Exception:
        # Most common cause is Coinbase rate limiting (HTTP 429) or transient
        # network errors. We treat it as a missing series for this cycle.
        return None
    if not candles:
        return None
    df = _candles_to_df(candles)
    if df.empty:
        return None
    # Ensure we have enough bars for indicators
    if len(df) < 40:
        return None
    return df


async def load_models(settings: Settings) -> Dict[int, Optional[dict]]:
    """Load model artifacts for each threshold.

    We store models in subdirs like {model_dir}/pt2, {model_dir}/pt5, etc.
    """
    base = Path(settings.model_dir)
    models: Dict[int, Optional[dict]] = {}
    for pct in sorted(set([int(round(x)) for x in settings.thresholds()])):
        d = base / f"pt{pct}"
        art = load_artifacts(d)
        # Backward compatibility: allow artifacts directly in model_dir
        if art is None:
            art = load_artifacts(base)
        models[pct] = art
    return models


def _fallback_prob(features: Dict[str, float], pct: int) -> float:
    """Heuristic probability fallback when no trained model exists.

    This is intentionally conservative; it should not emit tons of 90%+ signals.
    """
    # Combine a few normalized signals conservatively
    rvol = float(np.clip(features.get("rvol", 1.0), 0.0, 5.0))
    ema_9_21 = float(np.clip(features.get("ema_9_21", 0.0), -0.2, 0.2))
    ema_21_55 = float(np.clip(features.get("ema_21_55", 0.0), -0.2, 0.2))
    ema21_slope = float(np.clip(features.get("ema21_slope", 0.0), -0.2, 0.2))
    ema_strength = 10.0*(ema_9_21 + 0.7*ema_21_55) + 5.0*ema21_slope
    adx = float(np.clip(features.get("adx14", 10.0), 0.0, 60.0))
    vwap_loc = float(np.clip(features.get("vwap_loc", 0.0), -0.05, 0.05))
    dist_to_target_atr = float(np.clip(features.get("dist_to_target_atr", 5.0), 0.0, 10.0))

    # Simple score in [-, +]
    score = (
        0.35 * np.tanh((rvol - 1.2) / 1.0)
        + 0.25 * np.tanh(ema_strength)
        + 0.20 * np.tanh((adx - 18) / 10)
        + 0.20 * np.tanh(-vwap_loc / 0.02)
        + 0.25 * np.tanh(-(dist_to_target_atr - 2.0) / 2.0)
    )

    # Larger targets are harder
    pct_i = int(round(float(pct)))
    hardness = {2: 0.0, 5: 0.6, 10: 1.1}.get(pct_i, 0.0)
    score = score - hardness
    # Map to probability (conservative)
    p = 1 / (1 + np.exp(-2.2 * score))
    return float(np.clip(p, 0.01, 0.85))




async def compute_scores(settings: Settings, cb: CoinbaseClient, alpaca: AlpacaCryptoClient, product_ids: List[str]) -> Tuple[pd.DataFrame, dict]:
    """Compute probabilities for each Coinbase-tradeable product using Alpaca batched crypto bars.

    Uses 5-minute bars for features; horizon is fixed in hours (HORIZON_HOURS).
    This implementation is designed for reliability at scale: it avoids per-symbol candle calls
    and instead batches via Alpaca's multi-symbol bars endpoint.
    """
    now = datetime.now(timezone.utc)
    horizon_end = now + timedelta(hours=int(settings.horizon_hours))
    # Pull enough history for indicators + stability (12h is plenty for 5m indicators and RVOL)
    lookback_start = now - timedelta(hours=12)

    # thresholds
    thresholds = [int(round(x)) for x in settings.thresholds()]
    thresholds = sorted(set([t for t in thresholds if t > 0]))

    meta: dict = {
        "now_utc": now.isoformat().replace("+00:00","Z"),
        "horizon_end_utc": horizon_end.isoformat().replace("+00:00","Z"),
        "thresholds": thresholds,
        "skipped": {"no_bars": 0, "too_few_bars": 0, "illiquid": 0},
    }

    # Liquidity gates (not a cap): score everything that has sufficient notional volume and sane ATR
    min_notional = float(getattr(settings, "min_notional_volume", 50000.0))
    min_bars = int(getattr(settings, "min_bars_5m", 60))  # 5 hours=60 bars; need more for indicators
    min_bars = max(min_bars, 80)

    # Fetch BTC regime bars (for btc_ret_30m feature)
    btc_pid = f"BTC-{settings.quote_currency.upper()}"
    try:
        btc_map = await alpaca.get_bars_batch([btc_pid], start=lookback_start, end=now, timeframe="5Min")
        btc_df = None
        if btc_pid in btc_map:
            btc_df = btc_map[btc_pid].copy()
            btc_df["time"] = pd.to_datetime(btc_df["time"], utc=True)
            btc_df = btc_df.set_index("time").sort_index()
    except Exception:
        btc_df = None

    # Batch fetch bars for products in chunks to avoid long URLs
    all_bars: Dict[str, pd.DataFrame] = {}
    chunk_size = int(getattr(settings, "alpaca_batch_symbols", 200))
    chunk_size = max(25, min(chunk_size, 400))
    for i in range(0, len(product_ids), chunk_size):
        chunk = product_ids[i:i+chunk_size]
        try:
            bars_map = await alpaca.get_bars_batch(chunk, start=lookback_start, end=now, timeframe="5Min")
            all_bars.update(bars_map)
        except Exception as e:
            # don't fail the whole cycle
            meta.setdefault("batch_errors", 0)
            meta["batch_errors"] += 1
            meta.setdefault("batch_error_last", str(e))
            continue

    models = await load_models(settings)
    rows=[]
    for pid in product_ids:
        df = all_bars.get(pid)
        if df is None or df.empty:
            meta["skipped"]["no_bars"] += 1
            continue
        # normalize
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        # basic sufficiency
        if len(df) < min_bars:
            meta["skipped"]["too_few_bars"] += 1
            continue

        # liquidity: last 6 hours notional volume
        tail = df.tail(72)  # 72*5m = 6h
        notional = float((tail["volume"] * tail["close"]).sum())
        if notional < min_notional:
            meta["skipped"]["illiquid"] += 1
            continue

        base_price = float(df["close"].iloc[-1])

        # compute features per threshold; some features depend on target_pct
        probs={}
        feats_cache={}
        for pct in thresholds:
            target_pct = float(pct) / 100.0
            feats = compute_features(df5=df, df1=None, btc5=btc_df, now=now, horizon_end=horizon_end, target_pct=target_pct)
            feats_cache[pct]=feats
            art = models.get(pct)
            if isinstance(art, ModelArtifacts) and art.trained:
                # model returns calibrated probability
                try:
                    p = float(art.predict_proba(pd.DataFrame([feats]))[0])
                except Exception:
                    p = _fallback_prob(feats, pct)
            else:
                p = _fallback_prob(feats, pct)
            probs[pct]=float(np.clip(p, 0.0, 1.0))

        # enforce monotonicity: smaller threshold must be >= larger threshold
        for a,b in zip(thresholds, thresholds[1:]):
            probs[b] = min(probs[b], probs[a])

        # expose key diagnostics from the primary threshold (smallest)
        primary = thresholds[0] if thresholds else 2
        feats0 = feats_cache.get(primary, {})
        rows.append({
            "product_id": pid,
            "price": base_price,
            "notional_6h": notional,
            "atr_pct": float(feats0.get("atr_pct", 0.0)),
            "rvol": float(feats0.get("rvol", 0.0)),
            "ret_30m": float(feats0.get("ret_30m", 0.0)),
            "vwap_loc": float(feats0.get("vwap_loc", 0.0)),
            "dist_to_target_atr": float(feats0.get("dist_to_target_atr", 0.0)),
            **{f"prob_{pct}": probs[pct] for pct in thresholds},
            "updated_utc": meta["now_utc"],
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        # sort by smallest threshold probability desc
        df_out.sort_values(by=f"prob_{thresholds[0]}", ascending=False, inplace=True)
    meta["scored"] = int(len(df_out))
    meta["total_universe"] = int(len(product_ids))
    return df_out, meta

