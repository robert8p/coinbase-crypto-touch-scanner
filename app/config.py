from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Coinbase
    coinbase_base_url: str = Field(default="https://api.exchange.coinbase.com", alias="COINBASE_BASE_URL")
    coinbase_api_key: str | None = Field(default=None, alias="COINBASE_API_KEY")
    coinbase_api_secret: str | None = Field(default=None, alias="COINBASE_API_SECRET")

    # Scanner
    timezone: str = Field(default="UTC", alias="TIMEZONE")
    scan_interval_minutes: int = Field(default=5, alias="SCAN_INTERVAL_MINUTES")
    horizon_hours: int = Field(default=5, alias="HORIZON_HOURS")
    target_move_pcts: str = Field(default="2,5,10", alias="TARGET_MOVE_PCTS")
    max_products: int = Field(default=200, alias="MAX_PRODUCTS")
    quote_currency: str = Field(default="USD", alias="QUOTE_CURRENCY")

    # Model/persistence
    model_dir: str = Field(default="data/models", alias="MODEL_DIR")

    # Training
    train_lookback_days: int = Field(default=30, alias="TRAIN_LOOKBACK_DAYS")
    train_max_products: int = Field(default=50, alias="TRAIN_MAX_PRODUCTS")
    train_scan_step_minutes: int = Field(default=15, alias="TRAIN_SCAN_STEP_MINUTES")

    # Admin
    admin_password: str | None = Field(default=None, alias="ADMIN_PASSWORD")

    # Demo
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")

    # Runtime
    scheduler_enabled: bool = Field(default=True, alias="SCHEDULER_ENABLED")
    coinbase_timeout_seconds: float = Field(default=10.0, alias="COINBASE_TIMEOUT_SECONDS")
    coinbase_max_concurrency: int = Field(default=8, alias="COINBASE_MAX_CONCURRENCY")

def get_settings() -> Settings:
    return Settings()
