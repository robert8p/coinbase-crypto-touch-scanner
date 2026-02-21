from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CoinbaseResponse:
    ok: bool
    status_code: int
    json: Optional[Any] = None
    error: Optional[str] = None


class CoinbaseClient:
    """Minimal Coinbase Exchange REST client with sensible retry & rate-limit handling.

    This app uses the public Exchange API endpoints:
      - /products
      - /products/{product_id}/candles

    The Exchange API enforces fairly strict request limits. To avoid 429s we:
      - limit concurrency with a semaphore
      - retry 429/5xx with backoff (respecting Retry-After when provided)
      - keep the scan universe size capped (MAX_PRODUCTS)
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        max_concurrency: int,
        max_retries: int,
        backoff_base_seconds: float,
        requests_per_second: float = 2.5,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = max(0.1, float(backoff_base_seconds))
        self._rate_lock = asyncio.Lock()
        self._min_interval = 1.0 / max(0.1, float(requests_per_second))
        self._last_req_monotonic: float = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    def _retry_after_seconds(self, headers: Dict[str, str]) -> Optional[float]:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    async def _request_json(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> CoinbaseResponse:
        last_err: Optional[str] = None
        status_code = 0
        for attempt in range(self._max_retries + 1):
            async with self._sem:
                try:
                    # Global per-process pacing to reduce Coinbase HTTP 429.
                    async with self._rate_lock:
                        now_m = time.monotonic()
                        wait = self._min_interval - (now_m - self._last_req_monotonic)
                        if wait > 0:
                            await asyncio.sleep(wait)
                        self._last_req_monotonic = time.monotonic()

                    resp = await self._client.request(method, path, params=params)
                    status_code = resp.status_code

                    if resp.status_code == 200:
                        return CoinbaseResponse(ok=True, status_code=resp.status_code, json=resp.json())

                    # 429 / 5xx retryable
                    if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                        ra = self._retry_after_seconds(dict(resp.headers))
                        wait = ra if ra is not None else (self._backoff_base * (2 ** attempt))
                        # add small jitter
                        wait = wait + (0.05 * attempt)
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        logger.warning("Coinbase request %s %s failed (attempt %s/%s): %s; sleeping %.2fs", method, path, attempt + 1, self._max_retries + 1, last_err, wait)
                        if attempt < self._max_retries:
                            await asyncio.sleep(wait)
                            continue
                        return CoinbaseResponse(ok=False, status_code=resp.status_code, error=last_err)

                    # Non-retryable
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return CoinbaseResponse(ok=False, status_code=resp.status_code, error=last_err)

                except httpx.TimeoutException as e:
                    last_err = f"Timeout: {e}"
                except Exception as e:
                    last_err = f"Exception: {e}"

            # retry path for exceptions
            if attempt < self._max_retries:
                wait = (self._backoff_base * (2 ** attempt)) + (0.05 * attempt)
                logger.warning("Coinbase request %s %s errored (attempt %s/%s): %s; sleeping %.2fs", method, path, attempt + 1, self._max_retries + 1, last_err, wait)
                await asyncio.sleep(wait)

        return CoinbaseResponse(ok=False, status_code=status_code or 0, error=last_err or "unknown")

    async def list_products(self) -> List[Dict[str, Any]]:
        r = await self._request_json("GET", "/products")
        if not r.ok or not isinstance(r.json, list):
            raise RuntimeError(r.error or "Failed to list products")
        return r.json

    async def get_candles(self, product_id: str, start: datetime, end: datetime, granularity: int) -> List[List[float]]:
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "granularity": int(granularity),
        }
        r = await self._request_json("GET", f"/products/{product_id}/candles", params=params)
        if not r.ok or not isinstance(r.json, list):
            raise RuntimeError(r.error or f"Failed to get candles for {product_id}")
        return r.json
