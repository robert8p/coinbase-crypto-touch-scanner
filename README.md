# Coinbase Tradeable Crypto Touch Probability Scanner

Production-ready FastAPI web app that scans **Coinbase-tradeable** crypto pairs (via Coinbase Exchange `/products`) and uses **Alpaca Crypto Data API** for bars/quotes to estimate the probability that price will **touch +2%, +5%, +10% within the next 5 hours**.

## Key points
- Coinbase API is used **only** for the tradable universe.
- No Coinbase candle calls.
- Alpaca bars/quotes are **batched** and concurrency-limited.
- Hazard-style aggregation over 5-minute steps: `P = 1 - (1-h)^60`.
- Calibration + monotonicity across thresholds.
- `/api/status` provides rich diagnostics.

## Run locally
```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ADMIN_PASSWORD=change-me

# optional
export ALPACA_CRYPTO_LOCATION=us
export COINBASE_BASE_URL=https://api.exchange.coinbase.com
export TARGET_MOVE_PCTS=2,5,10
export HORIZON_HOURS=5
export SCAN_INTERVAL_MINUTES=5
export QUOTE_CURRENCY=USD

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Deploy to Render
- Push this repo to GitHub.
- Create a Render **Web Service** (Docker) from the repo.
- Add the required env vars.
- Attach a persistent disk mounted at `/var/data` (or set `MODEL_DIR` to your mount path).

## Admin endpoints
- `POST /admin/train` (Header `X-Admin-Password: <ADMIN_PASSWORD>`)
- `GET /admin/storage`
- `POST /admin/universe/refresh`

## API
- `GET /api/scan` latest scan results
- `GET /api/status` diagnostics (API health, counts, errors, rate limits, training state)

## Demo mode
Set `DEMO_MODE=true` to run without external API calls.
