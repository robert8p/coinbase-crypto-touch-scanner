from __future__ import annotations

import asyncio
import logging
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.clients.alpaca import AlpacaClient
from app.clients.alpaca_trading import fetch_crypto_assets
from app.config import Settings
from app.state import APP_STATE
from app.utils.features_fast import compute_features_5m_fast
from app.utils.modeling import load_models, score_symbol, temperature_soften_probs


def _clean_float(x):
    """Return a JSON-safe float (None if NaN/inf/invalid)."""
    try:
        xf = float(x)
    except Exception:
        return None
    if xf != xf or xf == float("inf") or xf == float("-inf"):
        return None
    return float(xf)


def _clean_prob(x):
    """Return probability in [0,1] and JSON-safe."""
    v = _clean_float(x)
    if v is None:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

log = logging.getLogger("scanner")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)



def _parse_quote(q: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
    if not isinstance(q, dict):
        return None, None, None
    # Alpaca latest quotes commonly: ap, bp, t
    ap = q.get("ap") or q.get("ask_price") or q.get("ask")
    bp = q.get("bp") or q.get("bid_price") or q.get("bid")
    t = q.get("t") or q.get("timestamp")

    try:
        ap_f = float(ap) if ap is not None else None
    except Exception:
        ap_f = None

    try:
        bp_f = float(bp) if bp is not None else None
    except Exception:
        bp_f = None

    t_dt: Optional[datetime] = None
    if t:
        try:
            t_dt = datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            t_dt = None

    return bp_f, ap_f, t_dt


async def refresh_universe(settings: Settings) -> None:
    """Refresh universe from Alpaca Trading API /v2/assets (asset_class=crypto, tradable=true).

    Filters to pairs with QUOTE_CURRENCY (default USD) and requires `tradable` flag.
    """
    try:
        APP_STATE.set_alpaca_trading_request()
        assets = await fetch_crypto_assets(
            base_url=settings.alpaca_trading_base_url,
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            status="active",
            timeout_s=settings.alpaca_trading_timeout_seconds,
            backoff_base_s=settings.alpaca_trading_backoff_base_seconds,
        )

        quote = (settings.quote_currency or "USD").upper()
        products = []
        for a in assets:
            if not isinstance(a, dict):
                continue
            if str(a.get("class") or a.get("asset_class") or "").lower() != "crypto":
                continue
            if not bool(a.get("tradable", False)):
                continue
            sym = str(a.get("symbol") or "")
            if "/" not in sym:
                continue
            base, q = sym.split("/", 1)
            if q.upper() != quote:
                continue
            products.append({"id": sym, "symbol": sym, "base": base, "quote": q, "raw": a})

        with APP_STATE.lock:
            APP_STATE.universe = products
            APP_STATE.universe_last_refresh_utc = _utcnow().isoformat()
            APP_STATE.alpaca_trading.last_error = None
    except Exception as e:
        s = str(e)
        with APP_STATE.lock:
            APP_STATE.alpaca_trading.last_error = s
            if "429" in s:
                APP_STATE.alpaca_trading.rate_limit_warn = s
        log.exception("Universe refresh failed")


def _need_universe_refresh() -> bool:
    with APP_STATE.lock:
        ts = APP_STATE.universe_last_refresh_utc
        n = len(APP_STATE.universe)
    if not ts or n == 0:
        return True
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return True
    return (_utcnow() - t) > timedelta(minutes=30)


async def scan_once(settings: Settings) -> None:
    if settings.demo_mode:
        _scan_demo(settings)
        return

    if _need_universe_refresh():
        await refresh_universe(settings)

    with APP_STATE.lock:
        products = list(APP_STATE.universe)

    if not products:
        with APP_STATE.lock:
            APP_STATE.last_scan_error = (
                "Universe is empty. Alpaca Trading API /v2/assets likely failed (check ALPACA_API_KEY/SECRET and ALPACA_TRADING_BASE_URL: paper vs live)."
            )
        return

    with APP_STATE.lock:
        bad_syms = set(APP_STATE.alpaca_bad_symbols.keys())

    target_pcts = settings.target_pcts()
    horizon_steps = settings.horizon_steps()

    # Build alpaca symbols list
    symbols = []
    for p in products:
        sym = p.get("symbol") or p.get("id")
        if sym:
            symbols.append(str(sym))


    # Safety: always include BTC and ETH for regime
    for s in ["BTC/USD", "ETH/USD"]:
        if s not in symbols:
            symbols.append(s)

    symbols = sorted(set(symbols))
    # Avoid repeatedly requesting symbols Alpaca has rejected
    symbols = [s for s in symbols if (s in {"BTC/USD", "ETH/USD"} or s not in bad_syms)]

    alpaca = AlpacaClient(
        key=settings.alpaca_api_key,
        secret=settings.alpaca_api_secret,
        loc=settings.alpaca_crypto_location,
        timeout_s=settings.alpaca_timeout_seconds,
        max_concurrency=settings.alpaca_max_concurrency,
    )

    now = _utcnow()

    # Mark scan started (helps diagnostics even if scan takes time)
    with APP_STATE.lock:
        APP_STATE.last_scan_utc = now.isoformat()
        APP_STATE.last_scan_error = None

    # Use a small rolling cache for 5m bars to reduce API load and 429 risk
    with APP_STATE.lock:
        cache_last = APP_STATE.bars5_cache_last_utc
        cache = dict(APP_STATE.bars5_cache)

    full_fetch = True
    if cache_last:
        try:
            t_last = datetime.fromisoformat(cache_last)
            if (now - t_last) < timedelta(minutes=12) and cache:
                full_fetch = False
        except Exception:
            full_fetch = True

    if full_fetch:
        start_5m = now - timedelta(hours=settings.feature_lookback_hours)
    else:
        # incremental refresh window
        start_5m = now - timedelta(minutes=25)
    end_5m = now


    try:
        APP_STATE.set_alpaca_request()

        # Step 1: use latest quotes to discover which symbols Alpaca actually supports.
        quotes = await alpaca.fetch_latest_quotes_batched(
            symbols,
            max_symbols_per_request=max(50, settings.alpaca_max_symbols_per_request // 2),
        )

        # Record any Alpaca symbol rejections discovered during quotes fetch
        if getattr(alpaca, "bad_symbols_last", None):
            with APP_STATE.lock:
                for s in alpaca.bad_symbols_last:
                    APP_STATE.alpaca_bad_symbols[s] = now.isoformat()
                if len(APP_STATE.alpaca_bad_symbols) > 2000:
                    for k in list(APP_STATE.alpaca_bad_symbols.keys())[:200]:
                        APP_STATE.alpaca_bad_symbols.pop(k, None)


        symbols_supported = sorted(set(quotes.keys()) | {"BTC/USD", "ETH/USD"})
        missing = [s for s in symbols if s not in symbols_supported and s not in {"BTC/USD", "ETH/USD"}]

        with APP_STATE.lock:
            APP_STATE.alpaca_supported_symbols_count = len(symbols_supported)
            APP_STATE.alpaca_missing_symbols_count = len(missing)

        if len(symbols_supported) <= 2:
            raise RuntimeError("Alpaca quotes returned no supported symbols beyond BTC/USD & ETH/USD. Check universe source and Alpaca crypto location.")

        # Step 2: fetch 5m bars only for supported symbols (reduces 400s / invalid symbol issues).
        bars5_new = await alpaca.fetch_bars_batched(
            symbols_supported,
            timeframe="5Min",
            start=start_5m,
            end=end_5m,
            max_symbols_per_request=settings.alpaca_max_symbols_per_request,
        )

        # Merge 5m cache
        if full_fetch:
            merged: Dict[str, List[Dict[str, Any]]] = {}
            for sym in symbols_supported:
                rows = sorted(bars5_new.get(sym, []), key=lambda b: b.get("t", ""))
                merged[sym] = rows
        else:
            merged = cache
            for sym, rows_new in bars5_new.items():
                existing = merged.get(sym, [])
                comb = existing + rows_new
                seen_t = set()
                out_rows = []
                for b in sorted(comb, key=lambda b: b.get("t", "")):
                    tt = b.get("t")
                    if not tt or tt in seen_t:
                        continue
                    seen_t.add(tt)
                    out_rows.append(b)
                keep_n = int((settings.feature_lookback_hours * 60) // 5) + 10
                if len(out_rows) > keep_n:
                    out_rows = out_rows[-keep_n:]
                merged[sym] = out_rows

        with APP_STATE.lock:
            APP_STATE.bars5_cache = merged
            APP_STATE.bars5_cache_last_utc = now.isoformat()

        bars5 = merged


        # Record any Alpaca symbol rejections / partial batch failures (if we did hit 400s internally)
        if getattr(alpaca, "bad_symbols_last", None):
            with APP_STATE.lock:
                for s in alpaca.bad_symbols_last:
                    APP_STATE.alpaca_bad_symbols[s] = now.isoformat()
                if len(APP_STATE.alpaca_bad_symbols) > 2000:
                    for k in list(APP_STATE.alpaca_bad_symbols.keys())[:200]:
                        APP_STATE.alpaca_bad_symbols.pop(k, None)

        # If we got partial errors but some data, surface as a non-fatal warning
        if getattr(alpaca, "batch_errors_last", None) and bars5_new:
            warn = f"Alpaca batch partial failures: {len(alpaca.batch_errors_last)} (continuing)."
            with APP_STATE.lock:
                APP_STATE.alpaca.last_error = warn

        # Only request 1m bars for symbols that actually returned 5m bars
        symbols_data = sorted(set(bars5.keys()) | {"BTC/USD", "ETH/USD"})

        # Fetch last 3 minutes of 1m bars for fallback close
        bars1 = await alpaca.fetch_bars_batched(
            symbols_data,
            timeframe="1Min",
            start=now - timedelta(minutes=3),
            end=now,
            max_symbols_per_request=max(50, settings.alpaca_max_symbols_per_request // 2),
        )

        with APP_STATE.lock:
            if not APP_STATE.alpaca.last_error:
                APP_STATE.alpaca.last_error = None

    except Exception as e:
        s = str(e)
        with APP_STATE.lock:
            APP_STATE.alpaca.last_error = s
            if "429" in s:
                APP_STATE.alpaca.rate_limit_warn = s
        raise

    # Regime features
    def price_from_last_bar(sym: str) -> float:
        rows = bars5.get(sym, [])
        if not rows:
            return float("nan")
        rows = sorted(rows, key=lambda b: b.get("t", ""))
        return float(rows[-1].get("c", np.nan))

    btc_slice = sorted(bars5.get("BTC/USD", []), key=lambda b: b.get("t", ""))
    eth_slice = sorted(bars5.get("ETH/USD", []), key=lambda b: b.get("t", ""))
    btc_price = price_from_last_bar("BTC/USD")
    eth_price = price_from_last_bar("ETH/USD")
    btc_feat = compute_features_5m_fast(btc_slice, btc_price, target_pcts)
    eth_feat = compute_features_5m_fast(eth_slice, eth_price, target_pcts)
    reg = {}
    for k in ["ret_30m", "ret_2h", "rv_1h", "rv_5h"]:
        reg[f"btc_{k}"] = float(btc_feat.get(k, 0.0))
        reg[f"eth_{k}"] = float(eth_feat.get(k, 0.0))

    # Load models (cached by caller, but safe)
    bundles = load_models(settings.model_dir, target_pcts)

    rows_out: List[Dict[str, Any]] = []
    rows_out_nogate: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "symbols_total": len(symbols_data),
        "symbols_with_bars5_ge_20": 0,
        "skip_short_bars5": 0,
        "skip_p0_missing": 0,
        "skip_features_empty": 0,
        "skip_notional_gate": 0,
        "scored": 0,
    }
    for _s in symbols_data:
        if len(sorted(bars5.get(_s, []), key=lambda b: b.get("t", ""))) >= 20:
            diag["symbols_with_bars5_ge_20"] += 1

    for sym in symbols_data:
        bars5_sym = sorted(bars5.get(sym, []), key=lambda b: b.get("t", ""))
        if len(bars5_sym) < 20:
            diag["skip_short_bars5"] += 1
            continue

        # Determine P0
        bp, ap, qt = _parse_quote(quotes.get(sym, {}))
        quote_age = None
        spread_bps = float("nan")
        p0 = float("nan")

        if bp is not None and ap is not None and bp > 0 and ap > 0:
            p0_mid = (bp + ap) / 2.0
            if qt:
                quote_age = (now - qt).total_seconds()
            else:
                quote_age = None

            if quote_age is not None and quote_age <= settings.quote_max_age_seconds:
                p0 = float(p0_mid)
            spread_bps = float(((ap - bp) / p0_mid) * 10000.0) if p0_mid > 0 else float("nan")

        if not np.isfinite(p0):
            bars1_sym = sorted(bars1.get(sym, []), key=lambda b: b.get("t", ""))
            if bars1_sym:
                try:
                    p0 = float(bars1_sym[-1].get("c", np.nan))
                except Exception:
                    p0 = float("nan")

        if not np.isfinite(p0) or p0 <= 0:
            diag["skip_p0_missing"] += 1
            continue

        feats = compute_features_5m_fast(bars5_sym, p0, target_pcts)
        if not feats:
            diag["skip_features_empty"] += 1
            continue

        feats.update(reg)

        notional_6h = float(feats.get("notional_6h", 0.0))

        probs = score_symbol(
            bundles=bundles,
            features=feats,
            spread_bps=float(spread_bps) if np.isfinite(spread_bps) else float("nan"),
            quote_age_s=float(quote_age) if quote_age is not None else float("nan"),
            horizon_steps=horizon_steps,
            target_pcts=target_pcts,
        )

        row = {
            "symbol": sym,
            "price": float(p0),
            "atr_pct": _clean_float(feats.get("atr_pct", float("nan"))),
            "spread_bps": float(spread_bps) if np.isfinite(spread_bps) else None,
            "quote_age_s": float(quote_age) if quote_age is not None else None,
            "notional_6h": _clean_float(notional_6h),
            "updated_utc": now.isoformat(),
        }

        for pct in target_pcts:
            row[f"p_touch_{pct}"] = _clean_prob(probs.get(pct, 0.0))
        # Convenience fields
        if 2 in target_pcts:
            row["dist_to_target_atr_2"] = _clean_float(feats.get("dist_to_target_atr_2", float("nan")))

        # Always keep a candidate (in case the liquidity gate excludes everything)
        rows_out_nogate.append(row)

        if np.isfinite(notional_6h) and notional_6h < settings.min_notional_volume_6h:
            diag["skip_notional_gate"] += 1
            continue

        rows_out.append(row)
        diag["scored"] += 1

    # If liquidity gate excluded everything, show candidates anyway (flagging it)
    if (not rows_out) and rows_out_nogate:
        rows_out = rows_out_nogate
        diag["liquidity_gate_bypassed"] = True
        with APP_STATE.lock:
            APP_STATE.last_scan_error = (
                f"Liquidity gate MIN_NOTIONAL_VOLUME_6H={settings.min_notional_volume_6h:g} excluded all; showing results anyway. Tune MIN_NOTIONAL_VOLUME_6H."
            )

    # Universe-level sanity soften
    rows_out, temp = temperature_soften_probs(
        rows_out,
        target_pcts=target_pcts,
        frac_cap=settings.universe_expected_touch_frac_cap,
        temp_max=settings.universe_temperature_max,
    )

    # Sort by p2 desc
    p0 = sorted(target_pcts)[0]
    rows_out.sort(key=lambda r: float(r.get(f"p_touch_{p0}", 0.0)), reverse=True)

    with APP_STATE.lock:
        APP_STATE.last_scan_rows = rows_out
        APP_STATE.last_scan_utc = now.isoformat()
        APP_STATE.last_scan_diag = diag
        # Preserve any non-fatal warning already set in last_scan_error
        if APP_STATE.last_scan_error is None:
            APP_STATE.last_scan_error = None


def _scan_demo(settings: Settings) -> None:
    target_pcts = settings.target_pcts()
    now = _utcnow()
    syms = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "ADA/USD", "XRP/USD", "DOGE/USD"]

    rows = []
    for s in syms:
        price = random.uniform(0.1, 50000)
        atr_pct = random.uniform(0.002, 0.08)
        spread_bps = random.uniform(2, 50)
        quote_age = random.uniform(1, 10)
        notional_6h = random.uniform(5e4, 5e7)
        base = 0.35 / (1 + (0.05 / (atr_pct + 1e-9)))
        probs = {}
        for pct in target_pcts:
            m = pct / 100.0
            dist = m / atr_pct
            p = min(0.95, math.exp(-0.35 * max(0.0, dist - 0.5))) * (0.3 + base)
            probs[pct] = max(0.0, min(1.0, p))
        # enforce monotone
        ordered = sorted(target_pcts)
        for i in range(1, len(ordered)):
            probs[ordered[i]] = min(probs[ordered[i]], probs[ordered[i - 1]])

        row = {
            "symbol": s,
            "price": float(price),
            "atr_pct": float(atr_pct),
            "spread_bps": float(spread_bps),
            "quote_age_s": float(quote_age),
            "notional_6h": float(notional_6h),
            "dist_to_target_atr_2": float((0.02 / atr_pct) if atr_pct > 0 else float("nan")),
            "updated_utc": now.isoformat(),
        }
        for pct in target_pcts:
            row[f"p_touch_{pct}"] = float(probs[pct])
        rows.append(row)

    rows.sort(key=lambda r: r.get("p_touch_2", 0.0), reverse=True)

    with APP_STATE.lock:
        APP_STATE.universe_last_refresh_utc = now.isoformat()
        APP_STATE.last_scan_rows = rows
        APP_STATE.last_scan_utc = now.isoformat()
        APP_STATE.last_scan_error = None
