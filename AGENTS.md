# Project Scope (gex_telegram)

This repository is a Python trading-signal pipeline that ingests SpotGamma GEX updates from a Telegram channel, parses them into structured snapshots, stores them in SQLite, computes intraday/overnight metrics and regimes, and posts formatted alerts to Discord webhooks. It also provides utilities to backfill historical data, replay a day for analysis, and generate a pre-market context report.

## Primary Flow

1. `main.py` starts the Telethon listener.
1. `telegram_listener.py` receives Telegram messages, parses into `GEXSnapshot`, persists to SQLite, computes metrics, evaluates signals, and posts Discord alerts. It also schedules daily alerts (morning brief, final 15, EOD summary).
1. `gex_parser.py` cleans raw messages, extracts price/gamma/positions, derives call wall / put floor, classifies session tags, and filters expired echoes.
1. `gex_db.py` stores snapshots and daily summaries in SQLite (`gex_data.db`).
1. `trend_engine.py` computes 15‑min and 1‑hour metrics, regime classification, overnight drift, and advanced metrics (GCI, squeeze/pin probabilities, pivot strikes).
1. `signal_interpreter.py` turns metrics into signals and formatted Discord messages, with stateful debounce and dedup logic.
1. `discord_alerts.py` sends messages via Discord webhooks and enforces cooldowns.
1. `scheduler.py` triggers daily scheduled alerts at configured times.

## Key Entry Points

- `main.py`: production listener (Telegram -> DB -> signals -> Discord).
- `backfill.py`: fetch historical Telegram messages into DB.
- `replay.py`: re-run a day’s snapshots through the signal engine with cooldown simulation.
- `context.py`: print a pre-market context summary from DB.
- `test_parser.py`: lightweight test suite covering parser, DB, metrics, and signals.

## Configuration and Environment

- `config.yaml`: thresholds, windows, cooldowns, schedules, session windows, and DB retention.
- `.env` (or the path in `config.yaml` under `telegram.env_path`) must provide:
  - `TG_API_ID`, `TG_API_HASH` (Telethon)
  - `DISCORD_GEX_OPS_WEBHOOK`, `DISCORD_MARKET_PULSE_WEBHOOK`, `DISCORD_DAILY_INTEL_WEBHOOK`
- `discord_alerts.py` currently loads webhooks from environment variables (not from `config.yaml`). A default webhook is embedded in code; override via env to avoid using it.

## Data and Files

- `gex_data.db`: SQLite DB for snapshots and daily summaries (live data).
- `gex_telegram.session`: Telethon session file (auth token). Treat as sensitive.
- `cooldown_state.json`: persisted cooldown state for Discord alerts.
- `gex_engine.log`: runtime logs.

## Operational Notes

- All time logic is in `US/Pacific`. Session windows are defined in `config.yaml`.
- Regime changes are debounced; adjacent regime flips require longer hold times.
- Cooldowns are halved during the final hour (12:00–13:00 PT).
- Parsing is robust to emoji / non‑ASCII, but some files contain mojibake text (e.g., “â€””); avoid reintroducing encoding issues.

## Running Locally

```powershell
python .\main.py
```

Backfill, replay, and context:

```powershell
python .\backfill.py 7 5000
python .\replay.py 2026-02-10
python .\context.py
```

Tests:

```powershell
python .\test_parser.py
```

## Implementation Guidance

- Keep `config.yaml` as the source of truth for thresholds and schedules.
- Avoid breaking schema compatibility in `gex_db.py`; existing data is live.
- Changes to parsing logic should preserve compatibility with existing Telegram formats.
- When adding signals, update `SignalType`, routing, cooldowns, and formatting together.
- If you adjust cooldown/schedule behavior, update both runtime logic and `config.yaml`.
