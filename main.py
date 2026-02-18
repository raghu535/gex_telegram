"""GEX 0DTE Trading Engine — Entry Point."""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gex_engine.log", encoding="utf-8"),
    ],
)

log = logging.getLogger("gex_engine")


def main():
    log.info("=" * 60)
    log.info("GEX 0DTE Trading Engine starting...")
    log.info("=" * 60)

    # Import here to ensure logging is configured first
    from telegram_listener import start_listener

    try:
        asyncio.run(start_listener())
    except KeyboardInterrupt:
        log.info("Shutting down gracefully...")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
