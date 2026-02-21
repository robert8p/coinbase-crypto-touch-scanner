from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App configuration.

    Notes
    -----
    * We intentionally avoid defining `target_move_pct` because this app is
      multi-threshold (2%, 5%, 10%). Previous iterations introduced it as a
      `@property`, which Pydantic tried to validate as a field and caused startup
      failures.
    """

    # Coinbase
    coinbase_base_url: str = Field(
        default="https://api.exchange.coinbase.com", alias="COINBASE_BASE_URL"
    )
    coinbase_api_key: str | None = Field(default=None, alias="COINBASE_API_KEY")
    coinbase_api_secret: str | None = Field(default=None, alias="COINBASE_API_SECRET")
    coinbase_passphrase: str | None = Field(default=None, alias="COINBASE_PASSPHRASE")

    # Alpaca Crypto Data (Algo Trader Plus)
    alpaca_api_key: str | None = Field(default=None, alias="ALPACA_API_KEY")
    alpaca_api_secret: str | None = Field(default=None, alias="ALPACA_API_SECRET")
    alpaca_data_base_url: str = Field(default="https://data.alpaca.markets", alias="ALPACA_DATA_BASE_URL")
    alpaca_crypto_location: str = Field(default="us", alias="ALPACA_CRYPTO_LOCATION")
    alpaca_timeout_seconds: float = Field(default=10.0, alias="ALPACA_TIMEOUT_SECONDS")
    alpaca_max_concurrency: int = Field(default=4, alias="ALPACA_MAX_CONCURRENCY")

    # Universe filter (Coinbase products are used for tradeability, Alpaca used for bars)
    quote_currency: str = Field(default="USD", alias="QUOTE_CURRENCY")


    # Scanner
    horizon_hours: int = Field(default=5, alias="HORIZON_HOURS")
    target_move_pcts: str = Field(default="2,5,10", alias="TARGET_MOVE_PCTS")

    # MAX_PRODUCTS=0 means no cap; liquidity gates and Alpaca batching keep this scalable.
    max_products: int = Field(default=0, alias="MAX_PRODUCTS")

    # Rate limiting / concurrency
    # Coinbase Exchange API is strict. Keep this conservative to avoid 429s.
    coinbase_max_concurrency: int = Field(default=2, alias="COINBASE_MAX_CONCURRENCY")
    coinbase_requests_per_second: float = Field(default=3.0, alias="COINBASE_RPS")
    coinbase_max_retries: int = Field(default=5, alias="COINBASE_MAX_RETRIES")

    # Liquidity gates (not a cap)
    min_notional_volume: float = Field(default=50000.0, alias="MIN_NOTIONAL_VOLUME_6H")
    min_bars_5m: int = Field(default=80, alias="MIN_BARS_5M")
    alpaca_batch_symbols: int = Field(default=200, alias="ALPACA_MAX_SYMBOLS_PER_REQUEST")

    # Storage
    model_dir: str = Field(default="/var/data/models", alias="MODEL_DIR")
    artifacts_dir: str = Field(default="models", alias="ARTIFACTS_DIR")
    train_max_products: int = Field(default=50, alias="TRAIN_MAX_PRODUCTS")
    train_scan_step_minutes: int = Field(default=15, alias="TRAIN_SCAN_STEP_MINUTES")
    train_lookback_days: int = Field(default=30, alias="TRAIN_LOOKBACK_DAYS")

    # Pydantic v2 configuration.
    # * `protected_namespaces` removes the default protection of `model_` which
    #   would otherwise warn on our `model_dir` field.
    # * `env_file` allows local `.env` development; Render uses env vars.
    model_config = {
        "protected_namespaces": ("settings_",),
        "env_file": ".env",
        "extra": "ignore",
    }

    def thresholds(self) -> List[float]:
        out: List[float] = []
        for part in (self.target_move_pcts or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except Exception:
                continue
        return out or [2.0, 5.0, 10.0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
