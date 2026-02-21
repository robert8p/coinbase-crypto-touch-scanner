from __future__ import annotations

import asyncio
import logging
import urllib.parse
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
        # Diagnostics from last batched call
        self.bad_symbols_last: List[str] = []
        self.batch_errors_last: List[str] = []

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "User-Agent": "coinbase-crypto-touch-scanner/1.0",
        }

    def _make_url_bounded_batches(
        self,
        endpoint_url: str,
        base_params: Dict[str, Any],
        symbols: List[str],
        max_symbols_per_request: int,
        url_len_cap: int = 7000,
    ) -> List[List[str]]:
        """Split symbols so that encoded URL length stays under a safe cap."""
        batches: List[List[str]] = []
        cur: List[str] = []
        for sym in symbols:
            candidate = cur + [sym]
            if len(candidate) > max(1, int(max_symbols_per_request)):
                if cur:
                    batches.append(cur)
                cur = [sym]
                continue
            params = dict(base_params)
            params["symbols"] = ",".join(candidate)
            q = urllib.parse.urlencode(params, safe=":+,")
            full_len = len(endpoint_url) + 1 + len(q)
            if full_len > url_len_cap and cur:
                batches.append(cur)
                cur = [sym]
            else:
                cur = candidate
        if cur:
            batches.append(cur)
        return batches

    async def _fetch_bars_resilient(
        self,
        client: httpx.AsyncClient,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        """Fetch bars; if Alpaca returns 400 for a mixed batch, bisect to isolate bad symbols."""
        try:
            data = await self.fetch_bars(client, symbols, timeframe, start, end, limit=limit)
            return data, []
        except Exception as e:
            msg = str(e)
            if ("400" in msg or "Bad Request" in msg) and len(symbols) > 1:
                mid = max(1, len(symbols) // 2)
                left, bad_l = await self._fetch_bars_resilient(client, symbols[:mid], timeframe, start, end, limit=limit)
                right, bad_r = await self._fetch_bars_resilient(client, symbols[mid:], timeframe, start, end, limit=limit)
                merged: Dict[str, List[Dict[str, Any]]] = {}
                for k, v in left.items():
                    merged.setdefault(k, []).extend(v)
                for k, v in right.items():
                    merged.setdefault(k, []).extend(v)
                return merged, bad_l + bad_r
            if ("400" in msg or "Bad Request" in msg) and len(symbols) == 1:
                return {}, [symbols[0]]
            raise

    async def _fetch_quotes_resilient(
        self,
        client: httpx.AsyncClient,
        symbols: List[str],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Fetch latest quotes; bisect on 400 to isolate bad symbols."""
        try:
            data = await self.fetch_latest_quotes(client, symbols)
            return data, []
        except Exception as e:
            msg = str(e)
            if ("400" in msg or "Bad Request" in msg) and len(symbols) > 1:
                mid = max(1, len(symbols) // 2)
                left, bad_l = await self._fetch_quotes_resilient(client, symbols[:mid])
                right, bad_r = await self._fetch_quotes_resilient(client, symbols[mid:])
                merged = dict(left)
                merged.update(right)
                return merged, bad_l + bad_r
            if ("400" in msg or "Bad Request" in msg) and len(symbols) == 1:
                return {}, [symbols[0]]
            raise

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
        url = f"{self.base}/bars"
        base_params: Dict[str, Any] = {
            "timeframe": timeframe,
            "start": _iso(start),
            "end": _iso(end),
            "limit": 10000,
        }
        batches = self._make_url_bounded_batches(
            endpoint_url=url,
            base_params=base_params,
            symbols=symbols,
            max_symbols_per_request=max_symbols_per_request,
            url_len_cap=7000,
        )

        out: Dict[str, List[Dict[str, Any]]] = {}
        bad: List[str] = []
        errs: List[str] = []

        async with httpx.AsyncClient(http2=False) as client:
            async def run_batch(batch: List[str]) -> None:
                nonlocal out, bad, errs
                async with self.sem:
                    try:
                        data, bad_syms = await self._fetch_bars_resilient(
                            client, batch, timeframe, start, end, limit=10000
                        )
                        bad.extend(bad_syms)
                        for k, v in data.items():
                            out.setdefault(k, []).extend(v)
                    except Exception as e:
                        errs.append(str(e))
                        log.warning("Alpaca bars batch failed (%d syms): %s", len(batch), str(e)[:200])

            await asyncio.gather(*(run_batch(b) for b in batches))

        self.bad_symbols_last = sorted(set(bad))
        self.batch_errors_last = errs

        if not out and errs:
            raise RuntimeError(errs[0])

        return out

    async def fetch_latest_quotes_batched(
        self,
        symbols: List[str],
        max_symbols_per_request: int,
    ) -> Dict[str, Dict[str, Any]]:
        url = f"{self.base}/latest/quotes"
        base_params: Dict[str, Any] = {}
        batches = self._make_url_bounded_batches(
            endpoint_url=url,
            base_params=base_params,
            symbols=symbols,
            max_symbols_per_request=max_symbols_per_request,
            url_len_cap=7000,
        )

        out: Dict[str, Dict[str, Any]] = {}
        bad: List[str] = []
        errs: List[str] = []

        async with httpx.AsyncClient(http2=False) as client:
            async def run_batch(batch: List[str]) -> None:
                nonlocal out, bad, errs
                async with self.sem:
                    try:
                        data, bad_syms = await self._fetch_quotes_resilient(client, batch)
                        bad.extend(bad_syms)
                        out.update(data)
                    except Exception as e:
                        errs.append(str(e))
                        log.warning("Alpaca quotes batch failed (%d syms): %s", len(batch), str(e)[:200])

            await asyncio.gather(*(run_batch(b) for b in batches))

        self.bad_symbols_last = sorted(set(self.bad_symbols_last + bad))
        self.batch_errors_last = list(self.batch_errors_last) + errs

        if not out and errs:
            raise RuntimeError(errs[0])

        return out

