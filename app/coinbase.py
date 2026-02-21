from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx

@dataclass
class Product:
    product_id: str
    base_currency: str
    quote_currency: str
    status: str

class CoinbaseClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0, max_concurrency: int = 8):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        import asyncio
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=20)


    async def _get_json(self, path: str, params: dict | None = None) -> Any:
        import asyncio
        url = f"{self.base_url}{path}"
        last_err = None
        for attempt in range(4):
            try:
                async with self._sem:
                    r = await self._client.get(url, params=params, headers={"Accept":"application/json"})
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise last_err

    async def close(self):
        await self._client.aclose()

    async def get_products(self) -> List[Product]:
        url = f"{self.base_url}/products"
        data = await self._get_json("/products")
        out=[]
        for p in data:
            out.append(Product(
                product_id=p.get("id"),
                base_currency=p.get("base_currency",""),
                quote_currency=p.get("quote_currency",""),
                status=p.get("status","")
            ))
        return out

    async def get_product_stats(self, product_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/products/{product_id}/stats"
        r = await self._client.get(url, headers={"Accept":"application/json"})
        r.raise_for_status()
        return r.json()

    async def get_candles(self, product_id: str, start: datetime, end: datetime, granularity: int) -> List[List[float]]:
        # returns list of [time, low, high, open, close, volume]
        url = f"{self.base_url}/products/{product_id}/candles"
        params = {
            "start": start.replace(tzinfo=timezone.utc).isoformat(),
            "end": end.replace(tzinfo=timezone.utc).isoformat(),
            "granularity": str(int(granularity)),
        }
        r = await self._client.get(url, params=params, headers={"Accept":"application/json"})
        r.raise_for_status()
        data = r.json()
        # Coinbase returns newest-first; reverse
        data.sort(key=lambda x: x[0])
        return data