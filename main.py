"""GEX 0DTE Trading Engine — Entry Point."""

import asyncio
import logging
import os
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

_DIR = os.path.dirname(os.path.abspath(__file__))
_PID_FILE = os.path.join(_DIR, "engine.pid")
_SESSION_BASE = os.path.join(_DIR, "gex_telegram")


def _cleanup_stale_session_locks():
    """Remove stale SQLite journal/lock files from Telethon session."""
    for suffix in ("-journal", "-wal", "-shm"):
        lock = _SESSION_BASE + ".session" + suffix
        if os.path.exists(lock):
            try:
                os.remove(lock)
                log.info(f"Removed stale session lock: {lock}")
            except OSError:
                pass


def _check_pid_lock():
    """Ensure only one engine instance runs. Kill stale if needed."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as f:
                old_pid = int(f.read().strip())
            # Check if that process is still alive
            import signal
            try:
                os.kill(old_pid, 0)  # doesn't kill, just checks
                log.error(f"Engine already running (PID {old_pid}). Exiting.")
                sys.exit(0)
            except (OSError, ProcessLookupError):
                log.info(f"Stale PID file (PID {old_pid} dead). Taking over.")
        except (ValueError, IOError):
            pass
    # Write our PID
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _cleanup_pid():
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                if int(f.read().strip()) == os.getpid():
                    os.remove(_PID_FILE)
    except Exception:
        pass


def main():
    _check_pid_lock()
    _cleanup_stale_session_locks()

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
        _cleanup_pid()
        sys.exit(1)
    finally:
        _cleanup_pid()


if __name__ == "__main__":
    main()
