from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, List

@dataclass
class SymbolScore:
    product_id: str
    prob_2: float
    prob_5: float
    prob_10: float
    updated_utc: datetime
    price: float
    vwap: float
    reasons: Dict[str, Any] = field(default_factory=dict)
    contrib: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AppState:
    lock: Lock = field(default_factory=Lock)
    last_scores: Dict[str, SymbolScore] = field(default_factory=dict)
    last_run_utc: Optional[datetime] = None
    last_error: Optional[str] = None
    training_running: bool = False
    training_last_error: Optional[str] = None
    training_last_result: Optional[str] = None
