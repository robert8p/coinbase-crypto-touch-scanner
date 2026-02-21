from __future__ import annotations

import asyncio
import logging
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.clients.alpaca import AlpacaClient
from app.clients.coinbase import fetch_products
from app.config import Settings
from app.state import APP_STATE
from app.utils.features_fast import compute_features_5m_fast
from app.utils.modeling import load_models, score_symbol, temperature_soften_probs


log = logging.getLogger("scanner")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _alpaca_symbol_from_coinbase(product_id: str) -> Optional[str]:
    # Coinbase Exchange product id is typically BASE-QUOTE (e.g. BTC-USD)
    if not product_id or "-" not in product_id:
        return None
    base, quote = product_id.split("-", 1)
    return f"{base}/{quote}"


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
    try:
        APP_STATE.set_coinbase_request()
        products = await fetch_products(
            base_url=settings.coinbase_base_url,
            quote_currency=settings.quote_currency,
            timeout_s=settings.coinbase_timeout_seconds,
            backoff_base_s=settings.coinbase_backoff_base_seconds,
        )
        with APP_STATE.lock:
            APP_STATE.universe = products
            APP_STATE.universe_last_refresh_utc = _utcnow().isoformat()
            APP_STATE.coinbase.last_error = None
    except Exception as e:
        s = str(e)
        with APP_STATE.lock:
            APP_STATE.coinbase.last_error = s
            if '429' in s:
                APP_STATE.coinbase.rate_limit_warn = s
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

    with APP_STATE.lock:
        bad_syms = set(APP_STATE.alpaca_bad_symbols.keys())

    target_pcts = settings.target_pcts()
    horizon_steps = settings.horizon_steps()

    # Build alpaca symbols list
    symbols = []
    for p in products:
        sym = _alpaca_symbol_from_coinbase(p.get("id"))
        if sym:
            symbols.append(sym)

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

        symbols_supported = sorted(set(quotes.keys()) | {"BTC/USD", "ETH/USD"})
        missing = [s for s in symbols if s not in symbols_supported and s not in {"BTC/USD", "ETH/USD"}]

        with APP_STATE.lock:
            APP_STATE.alpaca_supported_symbols_count = len(symbols_supported)
            APP_STATE.alpaca_missing_symbols_count = len(missing)

        if len(symbols_supported) <= 2:
            raise RuntimeError("Alpaca returned quotes for 0 symbols (check Alpaca key/plan/crypto location).")

        # Step 2: fetch 5m bars only for supported symbols (reduces 400s / invalid symbol issues).
        bars5_new = await alpaca.fetch_bars_batched(
            symbols_supported,
            timeframe="5Min",
            start=start_5m,
            end=end_5m,
            max_symbols_per_request=settings.alpaca_max_symbols_per_request,
        )

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
        symbols_data = sorted(set(bars5_new.keys()) | {"BTC/USD", "ETH/USD"})

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

    for sym in symbols_data:
        if sym in {"BTC/USD", "ETH/USD"}:
            continue

        bars5_sym = sorted(bars5.get(sym, []), key=lambda b: b.get("t", ""))
        if len(bars5_sym) < 20:
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
            continue

        feats = compute_features_5m_fast(bars5_sym, p0, target_pcts)
        if not feats:
            continue

        feats.update(reg)

        notional_6h = float(feats.get("notional_6h", 0.0))
        if np.isfinite(notional_6h) and notional_6h < settings.min_notional_volume_6h:
            continue

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
            "atr_pct": float(feats.get("atr_pct", float("nan"))),
            "spread_bps": float(spread_bps) if np.isfinite(spread_bps) else None,
            "quote_age_s": float(quote_age) if quote_age is not None else None,
            "notional_6h": float(notional_6h) if np.isfinite(notional_6h) else None,
            "updated_utc": now.isoformat(),
        }

        for pct in target_pcts:
            row[f"p_touch_{pct}"] = float(probs.get(pct, 0.0))
        # Convenience fields
        if 2 in target_pcts:
            row["dist_to_target_atr_2"] = float(feats.get("dist_to_target_atr_2", float("nan")))

        rows_out.append(row)

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
