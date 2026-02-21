from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", protected_namespaces=("settings_",))

    # Required (no defaults)
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    admin_password: str = Field(default="", alias="ADMIN_PASSWORD")

    # Core
    alpaca_crypto_location: str = Field(default="us", alias="ALPACA_CRYPTO_LOCATION")
    alpaca_trading_base_url: str = Field(default="https://api.alpaca.markets", alias="ALPACA_TRADING_BASE_URL")
    alpaca_trading_timeout_seconds: int = Field(default=10, alias="ALPACA_TRADING_TIMEOUT_SECONDS")
    alpaca_trading_backoff_base_seconds: float = Field(default=0.5, alias="ALPACA_TRADING_BACKOFF_BASE_SECONDS")
    target_move_pcts: str = Field(default="2,5,10", alias="TARGET_MOVE_PCTS")
    horizon_hours: int = Field(default=5, alias="HORIZON_HOURS")
    scan_interval_minutes: int = Field(default=5, alias="SCAN_INTERVAL_MINUTES")
    quote_currency: str = Field(default="USD", alias="QUOTE_CURRENCY")

    # Performance
    alpaca_max_symbols_per_request: int = Field(default=200, alias="ALPACA_MAX_SYMBOLS_PER_REQUEST")
    alpaca_max_concurrency: int = Field(default=4, alias="ALPACA_MAX_CONCURRENCY")
    alpaca_timeout_seconds: int = Field(default=10, alias="ALPACA_TIMEOUT_SECONDS")


    # Reliability gates
    min_notional_volume_6h: float = Field(default=50000.0, alias="MIN_NOTIONAL_VOLUME_6H")

    # Training
    model_dir: str = Field(default="./models", alias="MODEL_DIR")
    train_lookback_days: int = Field(default=30, alias="TRAIN_LOOKBACK_DAYS")
    train_scan_step_minutes: int = Field(default=15, alias="TRAIN_SCAN_STEP_MINUTES")
    train_max_products: int = Field(default=200, alias="TRAIN_MAX_PRODUCTS")

    # Optional / quality
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")
    quote_max_age_seconds: int = Field(default=15, alias="QUOTE_MAX_AGE_SECONDS")

    # Feature windows
    feature_lookback_hours: int = Field(default=6, alias="FEATURE_LOOKBACK_HOURS")

    # Sanity controls
    universe_expected_touch_frac_cap: float = Field(default=0.30, alias="UNIVERSE_EXPECTED_TOUCH_FRAC_CAP")
    universe_temperature_max: float = Field(default=4.0, alias="UNIVERSE_TEMPERATURE_MAX")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def target_pcts(self) -> List[int]:
        out: List[int] = []
        for p in self.target_move_pcts.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(float(p)))
            except ValueError:
                continue
        out = sorted(set(out))
        return out if out else [2, 5, 10]

    def horizon_steps(self) -> int:
        # 5m steps
        return int((self.horizon_hours * 60) // 5)


def get_settings() -> Settings:
    return Settings()
