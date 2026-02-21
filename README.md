# Coinbase Crypto Touch Scanner

Real-time scanner for Coinbase tradeable spot products (default: USD-quoted) that estimates the probability of touching +2%, +5%, +10% **at any point within the next 5 hours**.

## Deploy on Render (one service)

1. Push this repo to GitHub.
2. Create a Render **Web Service** from the repo (or use `render.yaml`).
3. Add environment variables (Render → Service → Environment):

Required (works without auth using public endpoints):
- `TIMEZONE` (default `UTC`)
- `SCAN_INTERVAL_MINUTES` (default `5`)
- `HORIZON_HOURS` (default `5`)
- `TARGET_MOVE_PCTS` (default `2,5,10`)
- `MAX_PRODUCTS` (default `200`)
- `ADMIN_PASSWORD` (recommended to protect training)

Optional (Advanced Trade auth not required for public data):
- `COINBASE_API_KEY`
- `COINBASE_API_SECRET`

Persistence (recommended):
- Attach a persistent disk at `/var/data`
- Set `MODEL_DIR=/var/data/models`

## Pages
- `/` dashboard
- `/symbol/{product_id}` detail
- `/train` training UI (requires `ADMIN_PASSWORD`)
- `/admin/storage` disk check

## Training
Training uses historical candles from Coinbase public endpoints. To keep memory/rate limits sane:
- `TRAIN_LOOKBACK_DAYS` (default 30)
- `TRAIN_MAX_PRODUCTS` (default 50)
- `TRAIN_SCAN_STEP_MINUTES` (default 15)

Run training in UI: `/train`

## API
- `/api/status`
- `/api/scores`
- `/api/symbol/{product_id}`
- `/api/symbol/{product_id}/series`

