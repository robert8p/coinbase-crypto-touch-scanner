from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.utils.http import request_json_with_backoff


log = logging.getLogger("alpaca_trading")


async def fetch_crypto_assets(
    base_url: str,
    api_key: str,
    api_secret: str,
    status: str = "active",
    timeout_s: int = 10,
    backoff_base_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Fetch Alpaca tradable crypto assets/pairs via Trading API /v2/assets.

    Docs: GET /v2/assets?asset_class=crypto (filter tradable=true client-side).
    """
    url = base_url.rstrip("/") + "/v2/assets"
    params: Dict[str, Any] = {"asset_class": "crypto"}
    if status:
        params["status"] = status

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "User-Agent": "alpaca-crypto-touch-scanner/1.0",
    }

    async with httpx.AsyncClient(http2=False) as client:
        data = await request_json_with_backoff(
            client,
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=timeout_s,
            max_tries=6,
            backoff_base=backoff_base_s,
        )

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /v2/assets response type: {type(data)}")
    return data
