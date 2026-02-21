from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings, Settings
from .logging_utils import setup_logging
from .coinbase import CoinbaseClient
from .scoring import compute_scores, parse_pcts
from .state import AppState, SymbolScore

logger = setup_logging()
app = FastAPI(title="Coinbase Crypto Touch Scanner")
BASE_DIR = Path(__file__).resolve().parent
static_path = BASE_DIR / "static"
templates_path = BASE_DIR / "templates"
# Ensure directories exist even if repo has no assets/templates (prevents Render startup crashes)
try:
    templates_path.mkdir(parents=True, exist_ok=True)
    static_path.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
templates = Jinja2Templates(directory=str(templates_path))

STATE = AppState()
SETTINGS: Optional[Settings] = None
CB: Optional[CoinbaseClient] = None
PRODUCTS: List[str] = []

try:
    app.mount("/static", StaticFiles(directory=str(static_path), check_dir=False), name="static")
except Exception as e:
    # Never hard-fail on static mount (Render requires app import to succeed)
    print(f"WARN: failed to mount /static: {e}")

def _utcnow():
    return datetime.now(timezone.utc)

async def refresh_products():
    global PRODUCTS
    assert CB and SETTINGS
    prods = await CB.get_products()
    # filter to quote currency and online status
    quote = SETTINGS.quote_currency.upper()
    candidates = [p.product_id for p in prods if p.quote_currency.upper()==quote and p.status=="online"]
    # rank by 24h volume (quote volume proxy via stats volume*price is heavy; use stats "volume" base)
    # To keep rate limits sane, just take first MAX_PRODUCTS if stats ranking fails.
    top = candidates[: SETTINGS.max_products]
    PRODUCTS = top
    return PRODUCTS

async def score_cycle():
    global PRODUCTS
    try:
        if not PRODUCTS:
            try:
                await refresh_products()
            except Exception as e:
                logger.warning(f"Product refresh failed: {type(e).__name__}: {e}")
                # proceed with empty universe; API will return 0 rows + last_error in status.
        df, meta = await compute_scores(SETTINGS, CB, PRODUCTS)
        updated=_utcnow()
        with STATE.lock:
            STATE.last_run_utc = updated
            STATE.last_error = None
            STATE.last_scores.clear()
            for _, r in df.iterrows():
                pid = r["product_id"]
                STATE.last_scores[pid] = SymbolScore(
                    product_id=pid,
                    prob_2=float(r.get("prob_2",0.0)),
                    prob_5=float(r.get("prob_5",0.0)),
                    prob_10=float(r.get("prob_10",0.0)),
                    updated_utc=updated,
                    price=float(r.get("price",0.0)),
                    vwap=float(r.get("vwap",0.0)),
                    reasons={"meta": meta},
                    contrib=[]
                )
    except Exception as e:
        logger.exception("score cycle failed")
        with STATE.lock:
            STATE.last_error = f"{type(e).__name__}: {e}"

async def scheduler_loop():
    assert SETTINGS
    interval = int(SETTINGS.scan_interval_minutes)
    while True:
        start = _utcnow()
        await score_cycle()
        # align to interval
        elapsed = (_utcnow()-start).total_seconds()
        sleep_for = max(1.0, interval*60 - elapsed)
        await asyncio.sleep(sleep_for)

@app.on_event("startup")
async def startup():
    global SETTINGS, CB
    SETTINGS = get_settings()
    (Path(SETTINGS.model_dir)).mkdir(parents=True, exist_ok=True)
    CB = CoinbaseClient(SETTINGS.coinbase_base_url, timeout_seconds=SETTINGS.coinbase_timeout_seconds, max_concurrency=SETTINGS.coinbase_max_concurrency)
    # Do not let startup fail if Coinbase is temporarily unreachable
    try:
        await refresh_products()
    except Exception as e:
        logger.warning(f"Product refresh failed on startup: {type(e).__name__}: {e}")
    if SETTINGS.scheduler_enabled:
        asyncio.create_task(scheduler_loop())
        logger.info("Started scheduler")

@app.on_event("shutdown")
async def shutdown():
    if CB:
        await CB.close()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/symbol/{product_id}", response_class=HTMLResponse)
def symbol_page(request: Request, product_id: str):
    return templates.TemplateResponse("symbol.html", {"request": request, "product_id": product_id})

@app.get("/api/status")
def api_status():
    with STATE.lock:
        return {
            "demo_mode": bool(getattr(SETTINGS,"demo_mode",False)) if SETTINGS else False,
            "now_utc": _utcnow().isoformat().replace("+00:00","Z"),
            "alpaca": None,
            "coinbase": {
                "ok": True if CB else False,
                "base_url": SETTINGS.coinbase_base_url if SETTINGS else None,
                "last_request_utc": None,
            },
            "universe_count": len(PRODUCTS),
            "model": {"thresholds": parse_pcts(SETTINGS.target_move_pcts) if SETTINGS else []},
            "training": {
                "running": STATE.training_running,
                "last_result": STATE.training_last_result,
                "last_error": STATE.training_last_error,
            },
            "last_run_utc": STATE.last_run_utc.isoformat().replace("+00:00","Z") if STATE.last_run_utc else None,
            "scores_count": len(STATE.last_scores),
            "last_error": STATE.last_error,
        }

@app.get("/api/scores")
def api_scores(min_prob: float = Query(0.0), limit: int = Query(200)):
    with STATE.lock:
        items = list(STATE.last_scores.values())
    # sort by prob_2 desc
    items.sort(key=lambda x: x.prob_2, reverse=True)
    out=[]
    for it in items:
        if it.prob_2 < min_prob:
            continue
        out.append({
            "product_id": it.product_id,
            "prob_2": round(it.prob_2,4),
            "prob_5": round(it.prob_5,4),
            "prob_10": round(it.prob_10,4),
            "price": it.price,
            "vwap": it.vwap,
            "updated_utc": it.updated_utc.isoformat().replace("+00:00","Z"),
        })
        if len(out)>=limit:
            break
    return {"rows": out, "last_run_utc": STATE.last_run_utc.isoformat().replace("+00:00","Z") if STATE.last_run_utc else None, "last_error": STATE.last_error}

@app.get("/api/symbol/{product_id}")
def api_symbol(product_id: str):
    with STATE.lock:
        it = STATE.last_scores.get(product_id)
    if not it:
        return JSONResponse({"detail":"Not Found"}, status_code=404)
    return {
        "product_id": it.product_id,
        "prob_2": it.prob_2,
        "prob_5": it.prob_5,
        "prob_10": it.prob_10,
        "price": it.price,
        "vwap": it.vwap,
        "updated_utc": it.updated_utc.isoformat().replace("+00:00","Z"),
    }

@app.get("/api/symbol/{product_id}/series")
async def api_series(product_id: str):
    assert CB
    now = datetime.now(timezone.utc)
    c5 = await CB.get_candles(product_id, now-datetime.timedelta(hours=8), now, 300)
    # fallback if no data
    return {"candles_5m": c5}

@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request, "error": None, "result": None})

@app.post("/train", response_class=HTMLResponse)
def train_start(request: Request, password: str = Form(...)):
    if not SETTINGS.admin_password or password != SETTINGS.admin_password:
        return templates.TemplateResponse("train.html", {"request": request, "error": "Invalid admin password.", "result": None})
    # Minimal: mark training running; actual training requires heavy data pulls; we run in background task.
    if STATE.training_running:
        return templates.TemplateResponse("train.html", {"request": request, "error": None, "result": "Training already running."})
    STATE.training_running = True
    async def _run():
        try:
            from .train import train_all
            res = await train_all(SETTINGS, CB, PRODUCTS)
            with STATE.lock:
                STATE.training_last_result = json.dumps(res)
                STATE.training_last_error = None
        except Exception as e:
            with STATE.lock:
                STATE.training_last_error = f"{type(e).__name__}: {e}"
        finally:
            with STATE.lock:
                STATE.training_running = False
    asyncio.create_task(_run())
    return templates.TemplateResponse("train.html", {"request": request, "error": None, "result": "Training started (lightweight placeholder)."})
