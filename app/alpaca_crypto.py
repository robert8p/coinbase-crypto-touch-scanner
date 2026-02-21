from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class AlpacaStatus:
    ok: bool
    message: str
    last_request_utc: Optional[str]
    base_url: str
    location: str

class AlpacaCryptoClient:
    def __init__(self, api_key: str | None, api_secret: str | None, base_url: str = "https://data.alpaca.markets", location: str = "us", timeout: float = 10.0, max_concurrency: int = 4):
        self.base_url = base_url.rstrip("/")
        self.location = location
        self._headers = {}
        if api_key and api_secret:
            # Alpaca data API auth headers
            self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        self._timeout = timeout
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._last_request_utc: Optional[str] = None

    @property
    def last_request_utc(self) -> Optional[str]:
        return self._last_request_utc

    async def _get_json(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        async with self._sem:
            async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                r = await client.get(url, params=params)
                self._last_request_utc = datetime.utcnow().isoformat() + "Z"
                r.raise_for_status()
                return r.json()

    @staticmethod
    def _to_alpaca_symbol(product_id: str) -> str:
        # Coinbase product_id: BTC-USD -> Alpaca: BTC/USD
        if "-" in product_id:
            a,b = product_id.split("-",1)
            return f"{a}/{b}"
        return product_id

    async def get_bars_batch(self, product_ids: List[str], start: datetime, end: datetime, timeframe: str = "5Min", limit: int = 10000) -> Dict[str, pd.DataFrame]:
        if not product_ids:
            return {}
        symbols = ",".join([self._to_alpaca_symbol(pid) for pid in product_ids])
        path = f"/v1beta3/crypto/{self.location}/bars"
        params = {
            "symbols": symbols,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": limit,
        }
        data = await self._get_json(path, params=params)
        bars = data.get("bars", {}) or {}
        out: Dict[str, pd.DataFrame] = {}
        # bars keys are alpaca symbols; map back to product_id
        inv = {self._to_alpaca_symbol(pid): pid for pid in product_ids}
        for sym, rows in bars.items():
            pid = inv.get(sym, sym.replace("/","-"))
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Alpaca fields: t,o,h,l,c,v
            # Normalize to columns expected by features: time, open, high, low, close, volume
            df.rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.sort_values("time", inplace=True)
            out[pid] = df[["time","open","high","low","close","volume"]].reset_index(drop=True)
        return out
