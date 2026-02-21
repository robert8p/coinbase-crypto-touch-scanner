from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import asyncio

from .config import Settings
from .coinbase import CoinbaseClient
from .features import candles_to_df, compute_features, FEATURES
from .modeling import ModelArtifacts, load_artifacts, default_artifacts

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

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0/(1.0+np.exp(-x))

def apply_probability_hygiene(df: pd.DataFrame, col: str, settings: Settings) -> None:
    if col not in df.columns: 
        return
    p = df[col].astype(float).clip(0,1).to_numpy()
    ttc = pd.to_numeric(df.get("time_to_horizon_frac", 1.0), errors="coerce").fillna(1.0).clip(0,1).to_numpy()
    dist = pd.to_numeric(df.get("dist_to_target_atr", 1.0), errors="coerce").fillna(1.0).clip(0,50).to_numpy()
    global_cap = float(getattr(settings, "prob_global_cap", 0.95))
    k = float(getattr(settings, "dist_atr_k", 1.2))
    cap_dist = np.exp(-k*dist)
    cap_time = 0.05 + 0.95*ttc
    cap = np.minimum(global_cap, np.maximum(cap_time, cap_dist))

    rvol = pd.to_numeric(df.get("rvol", 1.0), errors="coerce").fillna(1.0).clip(0,50).to_numpy()
    rsoft=float(getattr(settings,"rvol_soft",1.0)); rhard=float(getattr(settings,"rvol_hard",0.5))
    rscale=np.where(rvol>=rsoft,1.0,np.where(rvol<=rhard,0.35,0.35+0.65*(rvol-rhard)/(rsoft-rhard+1e-9)))
    cap*=rscale

    # hard rule: >0.9 only if very close in ATR
    dist_hi=float(getattr(settings,"dist_atr_high_prob_max",0.25))
    cap=np.where(dist<=dist_hi, cap, np.minimum(cap, 0.85))

    df[col]=np.minimum(p, cap)

def enforce_probability_mass(df: pd.DataFrame, prob_cols: List[str], primary_col: str, target_sum: float) -> float:
    if df.empty or primary_col not in df.columns: 
        return 1.0
    p0 = df[primary_col].astype(float).clip(0,1).to_numpy()
    s0=float(np.nansum(p0))
    if s0 <= target_sum or target_sum <= 0:
        return 1.0
    l0=_logit(p0)
    def sum_for_temp(temp: float) -> float:
        return float(np.nansum(_sigmoid(l0/temp)))
    lo,hi=1.0,20.0
    for _ in range(20):
        if sum_for_temp(hi) <= target_sum:
            break
        hi*=1.5
        if hi>200: 
            break
    for _ in range(30):
        mid=0.5*(lo+hi)
        if sum_for_temp(mid) > target_sum: 
            lo=mid
        else:
            hi=mid
    temp=hi
    for c in prob_cols:
        if c in df.columns:
            p=df[c].astype(float).clip(0,1).to_numpy()
            df[c]=_sigmoid(_logit(p)/temp)
    return temp

async def load_models(settings: Settings) -> Dict[float, ModelArtifacts]:
    base=Path(settings.model_dir)
    models={}
    for pct in parse_pcts(settings.target_move_pcts):
        d=base/pct_dir(pct)
        art=load_artifacts(d) or (load_artifacts(base) if abs(pct-float(settings.target_move_pct))<1e-9 else None)
        if art is None:
            art=default_artifacts(FEATURES)
        models[pct]=art
    return models

async def compute_scores(settings: Settings, cb: CoinbaseClient, products: List[str]) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    now = datetime.now(timezone.utc)
    horizon_end = now + timedelta(hours=int(settings.horizon_hours))
    # get BTC candles as regime
    btc5=None
    try:
        btc = await cb.get_candles("BTC-USD", now-timedelta(hours=8), now, 300)
        btc5=candles_to_df(btc)
    except Exception:
        btc5=None

    models = await load_models(settings)
    pcts = parse_pcts(settings.target_move_pcts)
    primary = float(settings.target_move_pct)

    # pull candles with limited concurrency
    sem=asyncio.Semaphore(10)
    async def fetch_one(pid: str):
        async with sem:
            end=now
            start=now-timedelta(hours=8)
            c5 = await cb.get_candles(pid, start, end, 300)
            c1 = await cb.get_candles(pid, now-timedelta(hours=2), now, 60)
            return pid, c5, c1

    rows=[]
    meta={}
    tasks=[fetch_one(pid) for pid in products]
    for fut in asyncio.as_completed(tasks):
        try:
            pid,c5,c1 = await fut
            df5=candles_to_df(c5)
            df1=candles_to_df(c1)
            if len(df5)<60 or len(df1)<60:
                continue
            base_feats = compute_features(df5, df1, btc5, now, horizon_end, primary)
            feat_row = {k: float(base_feats.get(k,0.0)) for k in FEATURES}
            feat_row["product_id"]=pid
            feat_row["price"]=float(base_feats["price"])
            feat_row["vwap"]=float(base_feats["vwap"])
            # compute probs for each pct by recomputing dist_to_target_atr
            for pct in pcts:
                fr=feat_row.copy()
                fr["dist_to_target_atr"]=float((pct/100.0)/(fr["atr_pct"]+1e-9))
                X=np.asarray([fr[f] for f in FEATURES], dtype=float)
                p=float(models[pct].predict_proba(X)[0])
                feat_row[f"prob_{pct}"]=p
            # map to fixed columns 2/5/10 for UI
            rows.append(feat_row)
        except Exception as e:
            continue

    df=pd.DataFrame(rows)
    if df.empty:
        return df, {}
    # hygiene + mass enforcement on primary and each column
    for pct in pcts:
        col=f"prob_{pct}"
        apply_probability_hygiene(df, col, settings)
    # mass enforcement on primary column (closest match)
    primary_col=f"prob_{primary}" if f"prob_{primary}" in df.columns else f"prob_{pcts[-1]}"
    target_sum = min(float(getattr(settings,"prob_mass_target_abs_max",12)), float(getattr(settings,"prob_mass_target_mult",2.0))*5.0)
    temp=enforce_probability_mass(df, [c for c in df.columns if c.startswith("prob_")], primary_col, target_sum)
    meta["prob_mass_temp"]=temp
    # produce standardized columns
    def getprob(p):
        col=f"prob_{p}"
        return df[col] if col in df.columns else 0.0
    df["prob_2"]=getprob(2.0)
    df["prob_5"]=getprob(5.0)
    df["prob_10"]=getprob(10.0)
    df["prob_primary"]=df[primary_col]
    return df, meta
