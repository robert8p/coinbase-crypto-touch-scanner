from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ApiDiag:
    last_request_utc: Optional[str] = None
    last_error: Optional[str] = None
    rate_limit_warn: Optional[str] = None


@dataclass
class TrainingDiag:
    running: bool = False
    last_started_utc: Optional[str] = None
    last_finished_utc: Optional[str] = None
    last_error: Optional[str] = None
    last_summary: Optional[Dict[str, Any]] = None


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Universe
    universe: List[Dict[str, Any]] = field(default_factory=list)
    universe_last_refresh_utc: Optional[str] = None

    # Scan
    last_scan_utc: Optional[str] = None
    last_scan_error: Optional[str] = None
    last_scan_rows: List[Dict[str, Any]] = field(default_factory=list)
    last_scan_diag: Dict[str, Any] = field(default_factory=dict)

    # Alpaca symbol rejections (invalid / unsupported)
    alpaca_bad_symbols: Dict[str, str] = field(default_factory=dict)

    # Alpaca scan selection diagnostics
    alpaca_supported_symbols_count: int = 0
    alpaca_missing_symbols_count: int = 0

    # Cached bars (to reduce API load)
    bars5_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    bars5_cache_last_utc: Optional[str] = None
    # Diagnostics
    alpaca: ApiDiag = field(default_factory=ApiDiag)
    alpaca_trading: ApiDiag = field(default_factory=ApiDiag)
    training: TrainingDiag = field(default_factory=TrainingDiag)

    # Model
    model_loaded: Dict[str, Any] = field(default_factory=dict)

    def set_alpaca_trading_request(self):
        self.alpaca_trading.last_request_utc = utcnow().isoformat()
        self.alpaca_trading.rate_limit_warn = None

    def set_alpaca_request(self):
        self.alpaca.last_request_utc = utcnow().isoformat()
        self.alpaca.rate_limit_warn = None


APP_STATE = AppState()
