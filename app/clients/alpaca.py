from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from app.utils.http import request_json_with_backoff


log = logging.getLogger("alpaca")


def _iso(dt: datetime) -> str:
    # Alpaca accepts RFC3339/ISO8601
    return dt.isoformat()


class AlpacaClient:
    def __init__(
        self,
        key: str,
        secret: str,
        loc: str = "us",
        timeout_s: int = 10,
        max_concurrency: int = 4,
    ):
        self.key = key
        self.secret = secret
        self.loc = loc
        self.timeout_s = timeout_s
        self.sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.base = f"https://data.alpaca.markets/v1beta3/crypto/{loc}"

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "User-Agent": "coinbase-crypto-touch-scanner/1.0",
        }

    async def fetch_bars(
        self,
        client: httpx.AsyncClient,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        url = f"{self.base}/bars"
        params: Dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": _iso(start),
            "end": _iso(end),
            "limit": limit,
        }
        out: Dict[str, List[Dict[str, Any]]] = {}

        page_token: Optional[str] = None
        tries = 0
        while True:
            tries += 1
            if page_token:
                params["page_token"] = page_token
            else:
                params.pop("page_token", None)

            data = await request_json_with_backoff(
                client,
                "GET",
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout_s,
                max_tries=6,
                backoff_base=0.5,
            )

            bars_obj = data.get("bars") if isinstance(data, dict) else None
            if isinstance(bars_obj, dict):
                for sym, rows in bars_obj.items():
                    if not isinstance(rows, list):
                        continue
                    out.setdefault(sym, []).extend(rows)

            page_token = data.get("next_page_token") if isinstance(data, dict) else None
            if not page_token:
                break
            # Safety valve
            if tries > 50:
                log.warning("bars pagination exceeded 50 pages; stopping")
                break

        return out

    async def fetch_latest_quotes(
        self,
        client: httpx.AsyncClient,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        url = f"{self.base}/latest/quotes"
        params = {"symbols": ",".join(symbols)}
        data = await request_json_with_backoff(
            client,
            "GET",
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout_s,
            max_tries=6,
            backoff_base=0.5,
        )
        qobj = data.get("quotes") if isinstance(data, dict) else None
        return qobj if isinstance(qobj, dict) else {}

    async def fetch_bars_batched(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        max_symbols_per_request: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        batches: List[List[str]] = []
        for i in range(0, len(symbols), max_symbols_per_request):
            batches.append(symbols[i : i + max_symbols_per_request])

        out: Dict[str, List[Dict[str, Any]]] = {}
        async with httpx.AsyncClient(http2=False) as client:
            async def run_batch(batch: List[str]) -> None:
                async with self.sem:
                    data = await self.fetch_bars(client, batch, timeframe, start, end)
                    for k, v in data.items():
                        out.setdefault(k, []).extend(v)

            await asyncio.gather(*(run_batch(b) for b in batches))

        return out

    async def fetch_latest_quotes_batched(
        self,
        symbols: List[str],
        max_symbols_per_request: int,
    ) -> Dict[str, Dict[str, Any]]:
        batches: List[List[str]] = []
        for i in range(0, len(symbols), max_symbols_per_request):
            batches.append(symbols[i : i + max_symbols_per_request])

        out: Dict[str, Dict[str, Any]] = {}
        async with httpx.AsyncClient(http2=False) as client:
            async def run_batch(batch: List[str]) -> None:
                async with self.sem:
                    data = await self.fetch_latest_quotes(client, batch)
                    out.update(data)

            await asyncio.gather(*(run_batch(b) for b in batches))

        return out
