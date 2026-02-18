"""Backfill historical messages from Telegram channel into the database."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

import pytz
import yaml
from dotenv import load_dotenv
from telethon import TelegramClient

from gex_parser import parse_message
from gex_db import init_db, save_snapshot, get_snapshot_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("backfill")

PT_TZ = pytz.timezone("US/Pacific")

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)


async def backfill(days: int = 7, limit: int = 5000):
    """Read historical messages from the Telegram channel and store them.

    Args:
        days: How many days back to fetch.
        limit: Max messages to fetch (Telegram caps at ~3000 per request).
    """
    env_path = CFG["telegram"].get("env_path", "")
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_id = int(os.getenv("TG_API_ID", "0"))
    api_hash = os.getenv("TG_API_HASH", "")
    channel = CFG["telegram"]["channel"]

    if not api_id or not api_hash:
        log.error("TG_API_ID and TG_API_HASH must be set.")
        return

    # Use a separate session file so backfill can run while the live listener is active.
    session_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gex_telegram_backfill")
    client = TelegramClient(session_path, api_id, api_hash)

    await client.start()
    log.info(f"Connected. Fetching up to {limit} messages from '{channel}' (last {days} days)...")

    init_db()
    before_count = get_snapshot_count()

    cutoff = datetime.now(pytz.utc) - timedelta(days=days)
    parsed = 0
    skipped = 0
    total = 0

    async for message in client.iter_messages(channel, limit=limit, offset_date=None):
        total += 1
        if message.date < cutoff:
            break

        text = message.text or ""
        if not text or len(text) < 50:
            skipped += 1
            continue

        snapshot = parse_message(text, envelope_dt=message.date)
        if snapshot and snapshot.session_tag != "EXPIRED_ECHO":
            save_snapshot(snapshot)
            parsed += 1

            if parsed % 50 == 0:
                log.info(f"  ...processed {parsed} snapshots ({total} messages scanned)")
        else:
            skipped += 1

    after_count = get_snapshot_count()
    new = after_count - before_count

    log.info(f"Backfill complete:")
    log.info(f"  Messages scanned: {total}")
    log.info(f"  Snapshots parsed: {parsed}")
    log.info(f"  Skipped (non-GEX): {skipped}")
    log.info(f"  New rows in DB: {new}")
    log.info(f"  Total DB snapshots: {after_count}")

    await client.disconnect()


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    log.info(f"Backfilling {days} days (max {limit} messages)...")
    asyncio.run(backfill(days=days, limit=limit))


if __name__ == "__main__":
    main()
