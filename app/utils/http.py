from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, Optional

import httpx


class HttpError(RuntimeError):
    pass


async def request_json_with_backoff(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    max_tries: int = 6,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
) -> Any:
    """Resilient JSON request. Retries on 429/5xx/timeouts with jittered exponential backoff."""

    last_exc: Optional[Exception] = None
    for i in range(max_tries):
        try:
            resp = await client.request(method, url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = None
                else:
                    delay = None
                if delay is None:
                    delay = min(backoff_cap, backoff_base * (2**i))
                    delay = delay * (0.8 + 0.4 * random.random())
                await asyncio.sleep(delay)
                last_exc = HttpError(f"HTTP {resp.status_code} {resp.text[:200]}")
                continue

            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
            last_exc = e
            delay = min(backoff_cap, backoff_base * (2**i))
            delay = delay * (0.8 + 0.4 * random.random())
            await asyncio.sleep(delay)

    raise HttpError(str(last_exc) if last_exc else "request failed")