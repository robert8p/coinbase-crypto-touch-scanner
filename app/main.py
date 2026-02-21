from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.scanner import refresh_universe, scan_once
from app.state import APP_STATE
from app.training import train_all
from app.utils.logging import setup_logging
from app.utils.modeling import load_models


log = logging.getLogger("main")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Alpaca Crypto Touch Probability Scanner", version="1.0.0")

# Ensure dirs exist (avoid startup crashes if missing)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

scan_lock = asyncio.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_admin(password: str, x_admin_password: Optional[str]) -> None:
    if not password:
        raise HTTPException(status_code=500, detail="ADMIN_PASSWORD is not set")
    if not x_admin_password or x_admin_password != password:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _model_status(settings: Settings) -> Dict[str, Any]:
    pcts = settings.target_pcts()
    bundles = load_models(settings.model_dir, pcts)
    out: Dict[str, Any] = {}
    for pct in pcts:
        b = bundles.get(pct)
        out[str(pct)] = {
            "present": b is not None,
            "has_calibrator": bool(b and b.calibrator),
            "meta": (b.meta if b else None),
        }
    return out


async def _scan_loop(settings: Settings):
    # initial small delay for cold starts
    await asyncio.sleep(0.5)
    while True:
        async with scan_lock:
            try:
                if (not settings.demo_mode) and (not settings.alpaca_api_key or not settings.alpaca_api_secret):
                    with APP_STATE.lock:
                        APP_STATE.last_scan_error = "ALPACA_API_KEY/ALPACA_API_SECRET missing (set DEMO_MODE=true to run without external APIs)"
                        APP_STATE.last_scan_utc = _utcnow_iso()
                else:
                    await scan_once(settings)
            except Exception as e:
                with APP_STATE.lock:
                    APP_STATE.last_scan_error = str(e)
                    APP_STATE.last_scan_utc = _utcnow_iso()
                log.exception("scan failed")
        await asyncio.sleep(max(30, int(settings.scan_interval_minutes) * 60))


@app.on_event("startup")
async def on_startup():
    settings = get_settings()
    setup_logging(settings.log_level)

    # Ensure dirs
    os.makedirs(settings.model_dir, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    # Start scanner loop
    asyncio.create_task(_scan_loop(settings))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.head("/")
async def index_head():
    return HTMLResponse(status_code=200)


@app.get("/api/scan")
async def api_scan(settings: Settings = Depends(get_settings)):
    with APP_STATE.lock:
        rows = list(APP_STATE.last_scan_rows)
        last_scan_utc = APP_STATE.last_scan_utc
        err = APP_STATE.last_scan_error

    return {
        "last_scan_utc": last_scan_utc,
        "error": err,
        "rows": rows,
    }


@app.get("/api/status")
async def api_status(settings: Settings = Depends(get_settings)):
    with APP_STATE.lock:
        uni_count = len(APP_STATE.universe)
        scored_count = len(APP_STATE.last_scan_rows)
        state = {
            "demo_mode": settings.demo_mode,            "alpaca": APP_STATE.alpaca.__dict__,
            "alpaca_trading": APP_STATE.alpaca_trading.__dict__,
            "alpaca_bad_symbols_count": len(APP_STATE.alpaca_bad_symbols),
            "alpaca_bad_symbols_sample": list(APP_STATE.alpaca_bad_symbols.keys())[:10],
            "alpaca_supported_symbols_count": APP_STATE.alpaca_supported_symbols_count,
            "alpaca_missing_symbols_count": APP_STATE.alpaca_missing_symbols_count,
            "universe_count": uni_count,
            "scored_count": scored_count,
            "universe_last_refresh_utc": APP_STATE.universe_last_refresh_utc,
            "last_scan_utc": APP_STATE.last_scan_utc,
            "last_scan_error": APP_STATE.last_scan_error,
            "training": APP_STATE.training.__dict__,
        }

    state["models"] = _model_status(settings)
    state["config"] = {
        "target_move_pcts": settings.target_pcts(),
        "horizon_hours": settings.horizon_hours,
        "scan_interval_minutes": settings.scan_interval_minutes,
        "min_notional_volume_6h": settings.min_notional_volume_6h,
        "alpaca_crypto_location": settings.alpaca_crypto_location,    }

    return state


@app.post("/admin/universe/refresh")
async def admin_refresh_universe(
    settings: Settings = Depends(get_settings),
    x_admin_password: Optional[str] = Header(default=None, alias="X-Admin-Password"),
):
    _require_admin(settings.admin_password, x_admin_password)
    await refresh_universe(settings)
    return {"ok": True, "universe_count": len(APP_STATE.universe)}


@app.post("/admin/train")
async def admin_train(
    settings: Settings = Depends(get_settings),
    x_admin_password: Optional[str] = Header(default=None, alias="X-Admin-Password"),
):
    _require_admin(settings.admin_password, x_admin_password)

    with APP_STATE.lock:
        if APP_STATE.training.running:
            raise HTTPException(status_code=409, detail="Training already running")
        APP_STATE.training.running = True
        APP_STATE.training.last_started_utc = _utcnow_iso()
        APP_STATE.training.last_error = None
        APP_STATE.training.last_summary = None

    # Build training universe from current universe / last scan
    with APP_STATE.lock:
        rows = list(APP_STATE.last_scan_rows)
        products = list(APP_STATE.universe)

    # Prefer last_scan notional ordering
    symbols: List[str] = []
    if rows:
        rows = sorted(rows, key=lambda r: float(r.get("notional_6h") or 0.0), reverse=True)
        for r in rows:
            sym = r.get("symbol")
            if sym:
                symbols.append(sym)
    else:
        for p in products:
            pid = p.get("id")
            if pid and "-" in pid:
                base, quote = pid.split("-", 1)
                symbols.append(f"{base}/{quote}")

    # unique + cap
    seen = set()
    ordered = []
    for s in symbols:
        if s in seen:
            continue
        seen.add(s)
        ordered.append(s)
        if len(ordered) >= settings.train_max_products:
            break

    async def run_training() -> Dict[str, Any]:
        return await anyio.to_thread.run_sync(train_all, settings, ordered)

    try:
        summary = await run_training()
        with APP_STATE.lock:
            APP_STATE.training.last_finished_utc = _utcnow_iso()
            APP_STATE.training.last_summary = summary
            APP_STATE.training.running = False
        return {"ok": True, "summary": summary}
    except Exception as e:
        with APP_STATE.lock:
            APP_STATE.training.last_finished_utc = _utcnow_iso()
            APP_STATE.training.last_error = str(e)
            APP_STATE.training.running = False
        log.exception("training failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/storage")
async def admin_storage(
    settings: Settings = Depends(get_settings),
    x_admin_password: Optional[str] = Header(default=None, alias="X-Admin-Password"),
):
    _require_admin(settings.admin_password, x_admin_password)

    path = settings.model_dir
    try:
        os.makedirs(path, exist_ok=True)
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        used = total - free
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "path": path,
        "total_bytes": int(total),
        "used_bytes": int(used),
        "free_bytes": int(free),
        "files": sorted(os.listdir(path)) if os.path.exists(path) else [],
    }
