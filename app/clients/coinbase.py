from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from app.utils.http import request_json_with_backoff


log = logging.getLogger("coinbase")


async def fetch_products(
    base_url: str,
    quote_currency: str,
    timeout_s: int,
    backoff_base_s: float,
) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/products"
    headers = {"User-Agent": "coinbase-crypto-touch-scanner/1.0"}

    async with httpx.AsyncClient(http2=True) as client:
        data = await request_json_with_backoff(
            client,
            "GET",
            url,
            headers=headers,
            params=None,
            timeout=timeout_s,
            max_tries=6,
            backoff_base=backoff_base_s,
        )

    if not isinstance(data, list):
        raise RuntimeError(f"unexpected /products response type: {type(data)}")

    out: List[Dict[str, Any]] = []
    for p in data:
        try:
            qc = (p.get("quote_currency") or "").upper()
            status = (p.get("status") or "").lower()
            trading_disabled = bool(p.get("trading_disabled", False))
            cancel_only = bool(p.get("cancel_only", False))
            # We keep post_only pairs; they are still tradable, but may behave differently.
            if qc != quote_currency.upper():
                continue
            if status and status != "online":
                continue
            if trading_disabled:
                continue
            if cancel_only:
                continue

            pid = p.get("id")
            if not pid:
                continue
            out.append(p)
        except Exception:
            continue

    log.info("Coinbase products: %s (quote=%s)", len(out), quote_currency)
    return out
