import logging, os

def setup_logging():
    level = os.getenv("LOG_LEVEL","INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("crypto_scanner")
