from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _inv_logit(x: float) -> float:
    return _sigmoid(x)


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def apply(self, p: float) -> float:
        return _inv_logit(self.a * _logit(p) + self.b)


@dataclass
class ModelBundle:
    pct: int
    model: Any  # sklearn classifier
    scaler: Any  # sklearn scaler
    calibrator: Optional[PlattCalibrator] = None
    meta: Dict[str, Any] = None


def load_models(model_dir: str, target_pcts: List[int]) -> Dict[int, ModelBundle]:
    bundles: Dict[int, ModelBundle] = {}
    for pct in target_pcts:
        path = os.path.join(model_dir, f"model_pt{pct}.joblib")
        if not os.path.exists(path):
            continue
        obj = joblib.load(path)
        model = obj.get("model")
        scaler = obj.get("scaler")
        cal = obj.get("calibrator")
        meta = obj.get("meta") or {}
        calibrator = None
        if isinstance(cal, dict) and "a" in cal and "b" in cal:
            calibrator = PlattCalibrator(a=float(cal["a"]), b=float(cal["b"]))
        if model is None or scaler is None:
            continue
        bundles[pct] = ModelBundle(pct=pct, model=model, scaler=scaler, calibrator=calibrator, meta=meta)
    return bundles


def _fallback_hazard(features: Dict[str, float], pct: int) -> float:
    dist = features.get(f"dist_to_target_atr_{pct}", float("nan"))
    ret2h = features.get("ret_2h", 0.0)
    ret30 = features.get("ret_30m", 0.0)
    rv1h = features.get("rv_1h", 0.0)
    rvol = features.get("rvol_30m", 1.0)
    vwap_loc = features.get("vwap_loc", 0.0)

    if not np.isfinite(dist):
        dist = 10.0

    x = (
        -2.2
        + (-0.55) * float(dist)
        + 3.0 * float(ret30)
        + 1.2 * float(ret2h)
        + 0.6 * float(math.log(max(float(rvol), 1e-3)))
        + 0.15 * float(vwap_loc)
        + 0.2 * float(rv1h)
    )

    if pct >= 10:
        x -= 0.6
    elif pct >= 5:
        x -= 0.25

    return _sigmoid(x)


def _plausibility_cap(p: float, dist_to_target_atr: float, spread_bps: float, quote_age_s: float) -> float:
    if np.isfinite(dist_to_target_atr):
        cap_dist = min(0.98, math.exp(-0.35 * max(0.0, float(dist_to_target_atr) - 0.5)))
    else:
        cap_dist = 0.15

    cap = cap_dist

    if np.isfinite(spread_bps) and float(spread_bps) > 0:
        if float(spread_bps) > 50:
            cap *= min(1.0, 50.0 / float(spread_bps))

    if np.isfinite(quote_age_s) and float(quote_age_s) > 30:
        cap *= min(1.0, 30.0 / float(quote_age_s))

    return min(float(p), float(cap))


def enforce_monotone(probs_by_pct: Dict[int, float], target_pcts: List[int]) -> Dict[int, float]:
    ordered = sorted(target_pcts)
    out = dict(probs_by_pct)
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        out[cur] = min(out.get(cur, 0.0), out.get(prev, 0.0))
    return out


def temperature_soften_probs(rows: List[Dict[str, Any]], target_pcts: List[int], frac_cap: float, temp_max: float) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    if not rows:
        return rows, None
    pct0 = sorted(target_pcts)[0]
    probs = [r.get(f"p_touch_{pct0}") for r in rows]
    probs = [p for p in probs if isinstance(p, (int, float)) and np.isfinite(p)]
    if not probs:
        return rows, None

    exp_touches = float(np.sum(probs))
    n = len(probs)
    cap = float(frac_cap) * float(n)
    if cap <= 0 or exp_touches <= cap:
        return rows, None

    temp = min(float(temp_max), max(1.0, exp_touches / cap))

    def soften(p: float) -> float:
        if not np.isfinite(p):
            return p
        return _inv_logit(_logit(float(p)) / temp)

    for r in rows:
        for pct in target_pcts:
            k = f"p_touch_{pct}"
            if k in r and isinstance(r[k], (int, float)):
                r[k] = float(soften(float(r[k])))

    return rows, float(temp)


def score_symbol(
    bundles: Dict[int, ModelBundle],
    features: Dict[str, float],
    spread_bps: float,
    quote_age_s: float,
    horizon_steps: int,
    target_pcts: List[int],
) -> Dict[int, float]:
    probs: Dict[int, float] = {}

    for pct in target_pcts:
        bundle = bundles.get(pct)

        if bundle and bundle.model is not None and bundle.scaler is not None:
            fnames = (bundle.meta or {}).get("feature_names") or []
            x = np.array([features.get(f, float("nan")) for f in fnames], dtype=float).reshape(1, -1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            xs = bundle.scaler.transform(x)
            hazard = float(bundle.model.predict_proba(xs)[0, 1])
        else:
            hazard = float(_fallback_hazard(features, pct))

        hazard = min(max(hazard, 1e-6), 1 - 1e-6)
        p5h = 1.0 - (1.0 - hazard) ** float(horizon_steps)

        if bundle and bundle.calibrator:
            p5h = float(bundle.calibrator.apply(p5h))

        dist = float(features.get(f"dist_to_target_atr_{pct}", float("nan")))
        p5h = float(_plausibility_cap(p5h, dist, float(spread_bps), float(quote_age_s)))

        probs[pct] = float(min(max(p5h, 0.0), 1.0))

    return enforce_monotone(probs, target_pcts)
