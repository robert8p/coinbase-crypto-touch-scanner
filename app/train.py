from __future__ import annotations
import asyncio
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
from .features import candles_to_df, compute_features, FEATURES
from .modeling import ModelArtifacts, save_artifacts, default_artifacts

def parse_pcts(s: str) -> List[float]:
    out=[]
    for part in (s or "").split(","):
        part=part.strip()
        if not part: 
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return sorted(set(out))

def pct_dir(p: float) -> str:
    if abs(p-round(p)) < 1e-9:
        return f"pt{int(round(p))}"
    return "pt"+str(p).replace(".","_")

async def _fetch_5m_history(cb: CoinbaseClient, product_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    # Coinbase candles limit ~300 per call; chunk by 24h to be safe (24h=288 5m candles)
    dfs=[]
    cur=start
    while cur < end:
        nxt=min(cur+timedelta(hours=24), end)
        candles = await cb.get_candles(product_id, cur, nxt, 300)
        if candles:
            dfs.append(candles_to_df(candles))
        cur=nxt
        await asyncio.sleep(0.05)
    if not dfs:
        return pd.DataFrame()
    df=pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _make_samples(df5: pd.DataFrame, btc5: pd.DataFrame, pcts: List[float], scan_step_min: int, horizon_hours: int) -> Tuple[pd.DataFrame, Dict[float, np.ndarray]]:
    # Build samples at scan times aligned to df5 index.
    if df5.empty:
        return pd.DataFrame(), {}
    df5=df5.sort_index()
    idx=df5.index
    step=max(1, int(scan_step_min/5))
    scan_points=list(range(100, len(idx)-int(horizon_hours*12)-1, step))  # ensure enough past + future
    feats=[]
    labels={pct:[] for pct in pcts}
    for i in scan_points:
        t = idx[i]
        past5 = df5.loc[:t].tail(96)  # 8h
        if len(past5)<60:
            continue
        fut = df5.loc[t: t+timedelta(hours=horizon_hours)].head(int(horizon_hours*12)+1)
        if len(fut)<int(horizon_hours*12):
            continue
        # build pseudo 1m df using 5m (acceptable approximation)
        df1 = past5.rename(columns={"volume":"volume"})  # reuse; vwap uses volume
        now = t.to_pydatetime()
        horizon_end = (t+timedelta(hours=horizon_hours)).to_pydatetime()
        base = compute_features(past5, df1, btc5, now.replace(tzinfo=timezone.utc), horizon_end.replace(tzinfo=timezone.utc), pcts[0])
        row = {k: float(base.get(k,0.0)) for k in FEATURES}
        row["_t"]=t
        feats.append(row)
        p0=float(base["price"])
        max_future_high=float(fut["high"].max())
        for pct in pcts:
            labels[pct].append(1 if max_future_high >= (1.0+pct/100.0)*p0 else 0)
    X=pd.DataFrame(feats)
    Y={pct: np.asarray(labels[pct], dtype=int) for pct in pcts}
    return X, Y

def _fit_one(X: pd.DataFrame, y: np.ndarray) -> ModelArtifacts:
    Xv=X.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # split last 20% as calibration
    n=len(Xv)
    if n<500 or len(np.unique(y))<2:
        return default_artifacts(FEATURES)
    cut=int(n*0.8)
    Xtr=Xv.iloc[:cut]; ytr=y[:cut]
    Xcal=Xv.iloc[cut:]; ycal=y[cut:]
    lr=LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
    lr.fit(Xtr[FEATURES].to_numpy(), ytr)
    raw=lr.predict_proba(Xcal[FEATURES].to_numpy())[:,1]
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, ycal)
    pcal=iso.transform(raw)
    auc=float(roc_auc_score(ycal, pcal)) if len(np.unique(ycal))>1 else None
    brier=float(brier_score_loss(ycal, pcal))
    meta={"trained": True, "auc_cal": auc, "brier_cal": brier, "n": int(n), "pos_rate": float(y.mean())}
    return ModelArtifacts(features=FEATURES, coef=lr.coef_.reshape(-1), intercept=float(lr.intercept_[0]), calibrator={"x": raw.tolist(), "y": pcal.tolist()}, meta=meta)

async def train_all(settings: Settings, cb: CoinbaseClient, products: List[str]) -> Dict[str, Any]:
    pcts=parse_pcts(settings.target_move_pcts)
    primary=float(settings.target_move_pct)
    # train set
    maxp=min(settings.train_max_products, len(products))
    products=products[:maxp]
    end=datetime.now(timezone.utc)
    start=end-timedelta(days=int(settings.train_lookback_days))
    # BTC regime
    btc5=await _fetch_5m_history(cb,"BTC-USD", start, end)
    results=[]
    for pid in products:
        df5=await _fetch_5m_history(cb, pid, start, end)
        if df5.empty:
            continue
        X, Y = _make_samples(df5, btc5 if not btc5.empty else None, pcts, settings.train_scan_step_minutes, settings.horizon_hours)
        if X.empty:
            continue
        for pct in pcts:
            art=_fit_one(X, Y[pct])
            out_dir=Path(settings.model_dir) / pct_dir(pct)
            if abs(pct-primary)<1e-9:
                out_dir=Path(settings.model_dir)
            save_artifacts(art, out_dir)
            results.append({"product": pid, "pct": pct, "meta": art.meta})
    return {"trained": True, "results": results, "pcts": pcts, "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
