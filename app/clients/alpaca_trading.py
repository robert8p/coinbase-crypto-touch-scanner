from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from app.utils.http import request_json_with_backoff


log = logging.getLogger("alpaca_trading")


def _alt_trading_base_url(base_url: str) -> str:
    b = (base_url or "").rstrip("/")
    if "paper-api.alpaca.markets" in b:
        return "https://api.alpaca.markets"
    if "api.alpaca.markets" in b:
        return "https://paper-api.alpaca.markets"
    # Default alternate
    return "https://paper-api.alpaca.markets"


async def fetch_crypto_assets(
    base_url: str,
    api_key: str,
    api_secret: str,
    status: str = "active",
    timeout_s: int = 10,
    backoff_base_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Fetch Alpaca tradable crypto assets/pairs via Trading API /v2/assets.

    GET /v2/assets?asset_class=crypto
    - Use live base URL for live keys: https://api.alpaca.markets
    - Use paper base URL for paper keys: https://paper-api.alpaca.markets
    This helper auto-retries the alternate base URL on 401 to reduce config friction.
    """
    def _headers() -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": api_key or "",
            "APCA-API-SECRET-KEY": api_secret or "",
            "User-Agent": "alpaca-crypto-touch-scanner/1.1",
        }

    async def _call(url_base: str) -> List[Dict[str, Any]]:
        url = url_base.rstrip("/") + "/v2/assets"
        params: Dict[str, Any] = {"asset_class": "crypto"}
        if status:
            params["status"] = status
        async with httpx.AsyncClient(http2=False) as client:
            data = await request_json_with_backoff(
                client,
                "GET",
                url,
                headers=_headers(),
                params=params,
                timeout=timeout_s,
                max_tries=6,
                backoff_base=backoff_base_s,
            )
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected /v2/assets response type: {type(data)}")
        return data

    try:
        return await _call(base_url)
    except httpx.HTTPStatusError as e:
        status_code = int(getattr(e.response, "status_code", 0) or 0)
        if status_code == 401:
            alt = _alt_trading_base_url(base_url)
            log.warning("Trading API 401 from %s; retrying assets with %s", base_url, alt)
            return await _call(alt)
        raise
