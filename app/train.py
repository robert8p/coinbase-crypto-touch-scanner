from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from .config import Settings
from .coinbase import CoinbaseClient
from .alpaca_crypto import AlpacaCryptoClient
from .features import compute_features, FEATURES
from .modeling import ModelArtifacts, save_artifacts, default_artifacts


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_pcts(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    # preserve order but unique-ish
    uniq = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    return uniq


def _pct_dir(pct: float) -> str:
    # folder-safe
    if abs(pct - round(pct)) < 1e-9:
        return f"pt{int(round(pct))}"
    return "pt" + str(pct).replace(".", "_")


def _df_from_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca bars df (time/open/high/low/close/volume) to the expected schema for features."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time")
    out = out.set_index("time", drop=False)
    return out[["time","open","high","low","close","volume"]]


def _pick_p0(df1: pd.DataFrame, t0: pd.Timestamp) -> float:
    # midquote not available in training; use last 1m close at/ before t0
    if df1 is None or df1.empty:
        return np.nan
    sub = df1[df1["time"] <= t0]
    if sub.empty:
        return np.nan
    return float(sub["close"].iloc[-1])


def _future_max_high(df1: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    sub = df1[(df1["time"] > t0) & (df1["time"] <= t1)]
    if sub.empty:
        return np.nan
    return float(sub["high"].max())


async def _fetch_training_bars(
    alp: AlpacaCryptoClient,
    product_ids: List[str],
    start: datetime,
    end: datetime,
    max_syms_per_req: int,
) -> Tuple[Dict[str,pd.DataFrame], Dict[str,pd.DataFrame]]:
    # Fetch 5m and 1m bars in batches
    bars5: Dict[str,pd.DataFrame] = {}
    bars1: Dict[str,pd.DataFrame] = {}
    for i in range(0, len(product_ids), max_syms_per_req):
        chunk = product_ids[i:i+max_syms_per_req]
        b5 = await alp.get_bars_batch(chunk, start=start, end=end, timeframe="5Min", limit=10000)
        b1 = await alp.get_bars_batch(chunk, start=start, end=end, timeframe="1Min", limit=10000)
        bars5.update(b5)
        bars1.update(b1)
    return bars5, bars1


def train_for_threshold(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[ModelArtifacts, Dict[str,Any]]:
    # Basic logistic + isotonic calibration
    lr = LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
    lr.fit(X, y)
    p_raw = lr.predict_proba(X)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y)
    p_cal = iso.transform(p_raw)
    auc = float(roc_auc_score(y, p_cal)) if len(np.unique(y)) > 1 else 0.5
    brier = float(brier_score_loss(y, p_cal))
    art = ModelArtifacts(model=lr, calibrator=iso, features=list(X.columns))
    meta = {"auc_cal": auc, "brier_cal": brier, "n": int(len(y)), "pos_rate": float(np.mean(y))}
    return art, meta


async def main_async() -> None:
    s = Settings()
    # Coinbase universe (tradeable)
    cb = CoinbaseClient(base_url=s.coinbase_base_url, timeout=s.coinbase_timeout_seconds, max_concurrency=s.coinbase_max_concurrency)
    products = await cb.get_products_cached()
    # Filter to USD quote, online
    pids = [p["id"] for p in products if p.get("quote_currency")==s.quote_currency and p.get("status")=="online"]
    if s.max_products and int(s.max_products) > 0:
        pids = pids[: int(s.max_products)]
    # For training, cap to manageable size unless user overrides
    pids = pids[: int(getattr(s, "train_max_products", 50))]

    if not pids:
        raise RuntimeError("No products to train on after filtering")

    alp = AlpacaCryptoClient(
        api_key=s.alpaca_api_key,
        api_secret=s.alpaca_api_secret,
        base_url=s.alpaca_data_base_url,
        location=s.alpaca_crypto_location,
        timeout=s.alpaca_timeout_seconds,
        max_concurrency=s.alpaca_max_concurrency,
    )

    now = _utcnow()
    days = int(getattr(s, "train_lookback_days", 30))
    start = now - timedelta(days=days)
    # Need extra horizon to label futures
    end = now

    max_syms_req = int(getattr(s, "alpaca_max_symbols_per_request", 200))
    bars5_map, bars1_map = await _fetch_training_bars(alp, pids + ["BTC-USD"], start, end, max_syms_req)

    # BTC regime series
    btc5 = _df_from_bars(bars5_map.get("BTC-USD", pd.DataFrame()))
    if btc5.empty:
        btc5 = None

    scan_step = int(getattr(s, "train_scan_step_minutes", 15))
    horizon = timedelta(hours=float(s.horizon_hours))
    X_rows: List[Dict[str,float]] = []
    y_rows: List[int] = []
    sym_rows: List[str] = []
    t_rows: List[pd.Timestamp] = []

    thresholds = _parse_pcts(s.target_move_pcts)
    if not thresholds:
        thresholds = [2.0, 5.0, 10.0]

    # Build dataset across symbols; for efficiency, compute per-threshold labels later
    # We'll store features once per scan, and then compute y per threshold from df1 futures.
    feats_by_key: List[Dict[str,float]] = []
    futures_by_key: List[Tuple[float, List[float]]] = []  # p0 and future_max_high for each threshold? we'll compute per threshold on fly
    keys: List[Tuple[str,pd.Timestamp]] = []

    for pid in pids:
        df5 = _df_from_bars(bars5_map.get(pid, pd.DataFrame()))
        df1 = _df_from_bars(bars1_map.get(pid, pd.DataFrame()))
        if df5.empty or df1.empty or len(df5) < 60 or len(df1) < 300:
            continue

        # Choose scan times on 5m grid, stepping by scan_step
        times = df5["time"].iloc[::max(1, scan_step//5)]
        for t0 in times:
            t0 = pd.Timestamp(t0)
            t1 = t0 + horizon
            if t1 > df1["time"].iloc[-1]:
                break
            # slice to t0
            df5_now = df5[df5["time"] <= t0].copy()
            if len(df5_now) < 60:
                continue
            p0 = _pick_p0(df1, t0)
            if not np.isfinite(p0) or p0 <= 0:
                continue
            # features computed per threshold because target_pct matters (dist_to_target_atr)
            # We'll store per-threshold later.
            keys.append((pid, t0))
            feats_by_key.append({"_p0": float(p0), "_t0": t0})
            futures_by_key.append((float(p0), float(_future_max_high(df1, t0, t1))))
            # store df refs in loop? no.

        # end pid loop

    if not keys:
        raise RuntimeError("No training samples were generated; loosen requirements or increase lookback.")

    # Prepare output dir
    base_out = Path(s.model_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Train per threshold
    for pct in thresholds:
        Xmat = []
        y = []
        for (pid, t0), meta0, (p0, hmax) in zip(keys, feats_by_key, futures_by_key):
            # need df slices again? we didn't store df5_now; recompute minimal by using bars map
            df5 = _df_from_bars(bars5_map.get(pid, pd.DataFrame()))
            if df5.empty:
                continue
            df5_now = df5[df5["time"] <= t0].copy()
            if len(df5_now) < 60:
                continue
            target_pct = float(pct) / 100.0
            f = compute_features(df5_now, None, btc5, now=t0.to_pydatetime(), horizon_end=(t0+horizon).to_pydatetime(), target_pct=target_pct)
            # drop any keys not in FEATURES (protect)
            row = {k: float(f.get(k, 0.0)) for k in FEATURES if k in f}
            Xmat.append(row)
            y.append(1 if (np.isfinite(hmax) and hmax >= (1.0+target_pct)*p0) else 0)
        if not Xmat:
            continue
        Xdf = pd.DataFrame(Xmat).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        yarr = np.array(y, dtype=int)

        art, meta = train_for_threshold(Xdf, yarr)

        out_dir = base_out / _pct_dir(pct)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_artifacts(out_dir, art, meta)

    # Also write a default artifacts set for fallback usage
    save_artifacts(base_out, default_artifacts(FEATURES), {"note":"default fallback"})

    print(json.dumps({"ok": True, "trained_thresholds": thresholds, "out_dir": str(base_out)}, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
