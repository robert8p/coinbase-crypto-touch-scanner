from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.clients.alpaca import AlpacaClient
from app.config import Settings
from app.utils.features_fast import compute_features_5m_fast


log = logging.getLogger("training")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_1m_arrays(bars: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bars:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float), np.array([], dtype=float)
    # bars should be dicts with t,h,c
    t = np.array([np.datetime64(b.get("t")) for b in bars], dtype="datetime64[ns]")
    high = np.array([float(b.get("h", np.nan)) for b in bars], dtype=float)
    close = np.array([float(b.get("c", np.nan)) for b in bars], dtype=float)
    idx = np.argsort(t)
    return t[idx], high[idx], close[idx]


def _find_idx_leq(times: np.ndarray, t: np.datetime64) -> int:
    if times.size == 0:
        return -1
    return int(np.searchsorted(times, t, side="right") - 1)


def _label_touch(high_arr: np.ndarray, i0: int, window_mins: int, target: float) -> int:
    j1 = i0 + 1
    j2 = min(len(high_arr), j1 + window_mins)
    if j1 >= len(high_arr) or j1 >= j2:
        return 0
    mx = float(np.nanmax(high_arr[j1:j2]))
    return 1 if np.isfinite(mx) and mx >= target else 0


def _slice_5m(bars: List[Dict[str, Any]], times5: np.ndarray, t0: np.datetime64, lookback_bars: int) -> List[Dict[str, Any]]:
    if not bars or times5.size == 0:
        return []
    i0 = _find_idx_leq(times5, t0)
    if i0 < 0:
        return []
    j0 = max(0, i0 - lookback_bars + 1)
    return bars[j0 : i0 + 1]


def _regime_features(btc: Dict[str, float], eth: Dict[str, float]) -> Dict[str, float]:
    keys = {"ret_30m", "ret_2h", "rv_1h", "rv_5h"}
    out: Dict[str, float] = {}
    for k in keys:
        out[f"btc_{k}"] = float(btc.get(k, 0.0))
        out[f"eth_{k}"] = float(eth.get(k, 0.0))
    return out


def train_all(settings: Settings, alpaca_symbols: List[str]) -> Dict[str, Any]:
    """Train per-threshold hazard models and Platt-calibrate aggregated 5h probabilities.

    - Hazard label: touch within next 5 minutes.
    - Calibration label: touch within next 5 hours.
    """

    if not settings.alpaca_api_key or not settings.alpaca_api_secret:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_API_SECRET must be set for training")

    target_pcts = settings.target_pcts()
    horizon_steps = settings.horizon_steps()

    end_dt = _utcnow() - timedelta(hours=settings.horizon_hours)
    start_dt = end_dt - timedelta(days=settings.train_lookback_days)
    cutoff_dt = start_dt + (end_dt - start_dt) * 0.8

    # Ensure regime symbols exist
    for s in ["BTC/USD", "ETH/USD"]:
        if s not in alpaca_symbols:
            alpaca_symbols = [s] + alpaca_symbols

    alpaca = AlpacaClient(
        key=settings.alpaca_api_key,
        secret=settings.alpaca_api_secret,
        loc=settings.alpaca_crypto_location,
        timeout_s=settings.alpaca_timeout_seconds,
        max_concurrency=settings.alpaca_max_concurrency,
    )

    lookback_bars_5m = int((settings.feature_lookback_hours * 60) // 5)

    log.info("Fetching 5m bars for %d symbols", len(alpaca_symbols))
    bars5_raw = asyncio.run(
        alpaca.fetch_bars_batched(
            alpaca_symbols,
            timeframe="5Min",
            start=start_dt - timedelta(hours=settings.feature_lookback_hours),
            end=end_dt,
            max_symbols_per_request=settings.alpaca_max_symbols_per_request,
        )
    )

    # Sort 5m bars and create time arrays
    bars5: Dict[str, List[Dict[str, Any]]] = {}
    times5: Dict[str, np.ndarray] = {}
    for sym in alpaca_symbols:
        rows = bars5_raw.get(sym, [])
        rows = sorted(rows, key=lambda b: b.get("t", ""))
        bars5[sym] = rows
        times5[sym] = np.array([np.datetime64(r.get("t")) for r in rows], dtype="datetime64[ns]") if rows else np.array([], dtype="datetime64[ns]")

    # Feature schema
    base_features = [
        "atr_pct",
        "ret_5m",
        "ret_30m",
        "ret_2h",
        "rv_1h",
        "rv_5h",
        "vol_of_vol",
        "ema12_ema26",
        "ema12_ema50",
        "ema_slope_12",
        "vwap_loc",
        "rvol_30m",
        "close_pos",
    ]
    regime = [
        "btc_ret_30m",
        "btc_ret_2h",
        "btc_rv_1h",
        "btc_rv_5h",
        "eth_ret_30m",
        "eth_ret_2h",
        "eth_rv_1h",
        "eth_rv_5h",
    ]

    feature_names_by_pct: Dict[int, List[str]] = {pct: [f"dist_to_target_atr_{pct}"] + base_features + regime for pct in target_pcts}

    scalers: Dict[int, StandardScaler] = {pct: StandardScaler() for pct in target_pcts}
    clfs: Dict[int, SGDClassifier] = {
        pct: SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            fit_intercept=True,
            max_iter=1,
            tol=None,
            learning_rate="optimal",
            random_state=42,
        )
        for pct in target_pcts
    }

    hold_X: Dict[int, List[List[float]]] = {pct: [] for pct in target_pcts}
    hold_y5h: Dict[int, List[int]] = {pct: [] for pct in target_pcts}

    step = timedelta(minutes=max(5, settings.train_scan_step_minutes))
    day = timedelta(days=1)
    max_samples_soft = int(os.environ.get("TRAIN_MAX_SAMPLES", "250000"))
    seen = 0

    log.info("Training window: %s -> %s (cutoff=%s)", start_dt.isoformat(), end_dt.isoformat(), cutoff_dt.isoformat())

    cursor = start_dt
    while cursor < end_dt:
        chunk_start = cursor
        chunk_scan_end = min(end_dt, chunk_start + day)
        chunk_label_end = min(end_dt + timedelta(hours=settings.horizon_hours), chunk_start + day + timedelta(hours=settings.horizon_hours))

        log.info("Fetching 1m bars chunk: %s -> %s", chunk_start.isoformat(), chunk_label_end.isoformat())
        bars1_raw = asyncio.run(
            alpaca.fetch_bars_batched(
                alpaca_symbols,
                timeframe="1Min",
                start=chunk_start,
                end=chunk_label_end,
                max_symbols_per_request=max(50, settings.alpaca_max_symbols_per_request // 2),
            )
        )

        arrays1: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for sym in alpaca_symbols:
            arrays1[sym] = _parse_1m_arrays(sorted(bars1_raw.get(sym, []), key=lambda b: b.get("t", "")))

        # Iterate scan times within the day chunk
        t_dt = chunk_start
        while t_dt < chunk_scan_end:
            t64 = np.datetime64(t_dt.replace(tzinfo=timezone.utc))

            # regime features for this time
            btc_slice = _slice_5m(bars5.get("BTC/USD", []), times5.get("BTC/USD", np.array([])), t64, lookback_bars_5m)
            eth_slice = _slice_5m(bars5.get("ETH/USD", []), times5.get("ETH/USD", np.array([])), t64, lookback_bars_5m)
            btc_price = float(btc_slice[-1].get("c", np.nan)) if btc_slice else np.nan
            eth_price = float(eth_slice[-1].get("c", np.nan)) if eth_slice else np.nan
            btc_feat = compute_features_5m_fast(btc_slice, btc_price, target_pcts)
            eth_feat = compute_features_5m_fast(eth_slice, eth_price, target_pcts)
            reg = _regime_features(btc_feat, eth_feat)

            for sym in alpaca_symbols:
                if sym in {"BTC/USD", "ETH/USD"}:
                    continue
                ts, high, close = arrays1.get(sym, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float), np.array([], dtype=float)))
                i0 = _find_idx_leq(ts, t64)
                if i0 < 0:
                    continue
                p0 = float(close[i0])
                if not np.isfinite(p0) or p0 <= 0:
                    continue

                slice5 = _slice_5m(bars5.get(sym, []), times5.get(sym, np.array([])), t64, lookback_bars_5m)
                feats = compute_features_5m_fast(slice5, p0, target_pcts)
                if not feats:
                    continue
                feats.update(reg)

                # Liquidity gate
                notional = float(feats.get("notional_6h", 0.0))
                if np.isfinite(notional) and notional < settings.min_notional_volume_6h:
                    continue

                in_holdout = t_dt >= cutoff_dt

                for pct in target_pcts:
                    m = float(pct) / 100.0
                    target_price = (1.0 + m) * p0

                    y5m = _label_touch(high, i0, window_mins=5, target=target_price)
                    y5h = _label_touch(high, i0, window_mins=int(settings.horizon_hours * 60), target=target_price)

                    fn = feature_names_by_pct[pct]
                    x = [float(feats.get(f, 0.0)) if np.isfinite(feats.get(f, 0.0)) else 0.0 for f in fn]

                    if in_holdout:
                        hold_X[pct].append(x)
                        hold_y5h[pct].append(int(y5h))
                    else:
                        Xb = np.array(x, dtype=float).reshape(1, -1)
                        scalers[pct].partial_fit(Xb)
                        Xs = scalers[pct].transform(Xb)
                        if not hasattr(clfs[pct], "classes_"):
                            clfs[pct].partial_fit(Xs, np.array([y5m], dtype=int), classes=np.array([0, 1], dtype=int))
                        else:
                            clfs[pct].partial_fit(Xs, np.array([y5m], dtype=int))

                seen += 1
                if seen >= max_samples_soft:
                    log.warning("Reached TRAIN_MAX_SAMPLES=%d; stopping early", max_samples_soft)
                    cursor = end_dt
                    break

            if seen >= max_samples_soft:
                break

            t_dt += step

        cursor += day

    # Save models + calibrators
    os.makedirs(settings.model_dir, exist_ok=True)
    summary: Dict[str, Any] = {
        "window": {
            "start_utc": start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "cutoff_utc": cutoff_dt.isoformat(),
            "train_samples_seen": int(seen),
        },
        "trained": {},
    }

    for pct in target_pcts:
        Xh = np.array(hold_X[pct], dtype=float)
        yh = np.array(hold_y5h[pct], dtype=int)

        cal: Optional[Dict[str, float]] = None
        auc: Optional[float] = None

        if Xh.size and yh.size and 0 < yh.sum() < len(yh):
            Xhs = scalers[pct].transform(Xh)
            hazard = clfs[pct].predict_proba(Xhs)[:, 1]
            p5h_raw = 1.0 - np.power(1.0 - hazard, float(horizon_steps))
            z = np.log(np.clip(p5h_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p5h_raw, 1e-9, 1 - 1e-9)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(z, yh)
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])
            cal = {"a": a, "b": b}
            p5h_cal = 1.0 / (1.0 + np.exp(-(a * z[:, 0] + b)))
            auc = float(roc_auc_score(yh, p5h_cal))
        else:
            log.warning("Holdout for pt%d insufficient for calibration (n=%d, pos=%d)", pct, int(len(yh)), int(yh.sum()))

        obj = {
            "model": clfs[pct],
            "scaler": scalers[pct],
            "calibrator": cal,
            "meta": {
                "feature_names": feature_names_by_pct[pct],
                "horizon_steps": horizon_steps,
                "pct": pct,
                "lookback_hours": settings.feature_lookback_hours,
            },
        }

        path = os.path.join(settings.model_dir, f"model_pt{pct}.joblib")
        joblib.dump(obj, path)

        summary["trained"][str(pct)] = {
            "saved": True,
            "path": path,
            "holdout_n": int(len(yh)),
            "holdout_pos": int(yh.sum()),
            "auc_cal": auc,
            "calibrator": cal,
        }

    meta_path = os.path.join(settings.model_dir, "training_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)
    summary["training_summary_path"] = meta_path

    return summary
