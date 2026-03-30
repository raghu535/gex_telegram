# GEX Telegram Engine — Claude Agent Guide

> **Type:** Long-running Python process (Telegram listener via Telethon)
> **Entry:** `main.py` → `telegram_listener.py` → `gex_parser.py`
> **DB:** `gex_data.db` (SQLite) | **Log:** `gex_engine.log`
> **Start:** `python main.py`

## What This Does

Listens to a SpotGamma Telegram channel, parses GEX (Gamma Exposure) snapshots every ~90 seconds during RTH, stores them in SQLite, computes regime/trend signals, and sends alerts to Discord + Telegram channels.

---

## Architecture

```
main.py                 — Entry point, PID lock, session cleanup
telegram_listener.py    — Telethon async listener, GEXPipeline processing class
gex_parser.py           — Message parsing → GEXSnapshot dataclass
gex_db.py               — SQLite read/write operations
trend_engine.py         — 15min/1hr metrics, regime classification, gamma concentration
signal_interpreter.py   — Signal generation from snapshots
discord_alerts.py       — Discord webhook alerts
telegram_alerts.py      — Telegram channel alerts (@codexchannels, @codex_gex)
recap.py                — Hourly recap builder
scheduler.py            — Scheduled tasks (EOD summary, etc.)
consensus_tracker.py    — Cross-signal consensus
day_classifier.py       — Day type classification
delivery_watchdog.py    — Stale data detection
conviction_engine.py    — Conviction scoring
config.yaml             — All configuration
```

---

## CRITICAL: Gamma Interpretation

### Net Gamma Sign Is Everything
- **Positive net gamma** → Dealers are LONG gamma → They sell rallies, buy dips → **STABLE, mean-reverting, range-bound**
- **Negative net gamma** → Dealers are SHORT gamma → They sell dips, buy rallies → **VOLATILE, trend-amplifying**

### GEX Levels
- **Call Wall** = Strike with most positive call gamma → Acts as **RESISTANCE** (dealers sell into rallies toward this level)
- **Put Floor** = Strike with most negative put gamma → Acts as **SUPPORT** (dealers buy into dips toward this level)
- **Gamma Flip** = Where net gamma crosses zero → Key inflection point

### Do NOT invert:
- Call wall is RESISTANCE, not support
- Put floor is SUPPORT, not resistance
- Positive gamma = STABILITY, not "bullish momentum"
- Negative gamma = VOLATILITY, not "bearish momentum"

---

## Message Parsing (`gex_parser.py`)

### SpotGamma Message Format
```
📊 SPX Gamma Exposure Update (20260210)
🕒 2026-02-11 03:30:08
💵 Current Price: $6,941.81

Top 5 Call Positions (Positive Gamma):
- Strike: $6,945 | Gamma: 2,925,349,624
...
Top 5 Put Positions (Negative Gamma):
- Strike: $6,940 | Gamma: -5,373,756,151
...
 Total Call Gamma: 5,578,057,074
 Total Put Gamma: -15,162,168,237
⚖️ Net Gamma: -9,584,111,163
```

### Unicode Minus Handling
SpotGamma uses unicode minus signs (−, –, —) not ASCII hyphen (-). The parser translates these:
```python
UNICODE_MINUSES = {'\u2212': '-', '\u2013': '-', '\u2014': '-', '\u00AD': '-'}
```
**If parsing breaks on negative numbers, check for new unicode minus characters.**

### Key Regex Patterns
```python
RE_PRICE = r"Current Price:\s*\$?([\d,]+\.?\d*)"
RE_NET_GAMMA = r"Net Gamma:\s*([-\d,]+)"
RE_TOTAL_CALL = r"Total Call Gamma:\s*([-\d,]+)"
RE_TOTAL_PUT = r"Total Put Gamma:\s*([-\d,]+)"
RE_STRIKE_GAMMA = r"Strike:\s*\$?([\d,]+)\s*\|\s*Gamma:\s*([-\d,]+)"
```
Numbers use commas (e.g., `2,925,349,624`). The parser strips commas before converting to int.

### Call Wall / Put Floor Derivation
- **Call Wall** = Strike with the HIGHEST gamma from the Top 5 Call Positions
- **Put Floor** = Strike with the MOST NEGATIVE gamma from the Top 5 Put Positions
- These are the max-gamma strikes, not just any strike

---

## Session Tagging (`classify_session()`)

| Tag | Time Range (PT) | Meaning |
|-----|-----------------|---------|
| `PRE_MARKET` | Before 6:30 AM | Pre-market data |
| `RTH` | 6:30 AM - 1:00 PM | Regular Trading Hours |
| `TRANSITION` | 1:00 PM - 1:30 PM | Market close transition |
| `NEXT_DAY_PREVIEW` | After 1:30 PM | After-hours preview |

**Only RTH snapshots should be used for intraday analysis.** Pre-market and after-hours data has different gamma dynamics.

---

## Database Schema (`gex_data.db`)

### `gex_snapshots` (37K+ rows)
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Auto-increment PK |
| timestamp_pt | TEXT | Pacific time ISO string |
| date_pt | TEXT | 'YYYY-MM-DD' |
| time_pt | TEXT | 'HH:MM:SS' |
| session_tag | TEXT | RTH, PRE_MARKET, TRANSITION, NEXT_DAY_PREVIEW |
| curr_price | REAL | SPX price |
| net_gamma | INTEGER | Raw net gamma (billions scale, e.g., -9584111163) |
| total_call_gamma | INTEGER | Total positive gamma |
| total_put_gamma | INTEGER | Total negative gamma (stored as negative) |
| call_wall | INTEGER | Max call gamma strike (e.g., 6945) |
| put_floor | INTEGER | Max put gamma strike (e.g., 6940) |
| top5_calls | TEXT | JSON list of [strike, gamma] pairs |
| top5_puts | TEXT | JSON list of [strike, gamma] pairs |
| raw_text | TEXT | Original message text |

### `daily_summaries`
EOD summary rows computed at market close.

### Querying Notes
- `sqlite3` CLI is NOT installed — use Python: `python -c "import sqlite3; ..."`
- `net_gamma` is raw integer (e.g., -9584111163). Divide by 1e9 for billions.
- `call_wall` and `put_floor` are strike prices (integers), NOT gamma values
- `top5_calls` / `top5_puts` are JSON strings — parse with `json.loads()`

---

## Crash-Prone Areas

### Telegram Connection
- Telethon reconnects automatically but can lose messages during reconnection
- PID lock (`engine.pid`) prevents duplicate instances
- Stale session locks are cleaned on startup

### Parse Failures
- If regex doesn't match, fields default to 0/empty
- Malformed messages (format changes from SpotGamma) will silently produce zero values
- **Check `gex_engine.log` for parse warnings**

### Delivery Watchdog
Built-in stale data detection — alerts if no snapshot received for >10 minutes during RTH.

### Rolling Price Buffer
Detects "stale-stream oscillation" (repeated identical prices) which indicates the data feed is stuck.

---

## Do NOT
- Invert gamma interpretation (positive = stable, negative = volatile)
- Treat call wall as support (it's resistance)
- Treat put floor as resistance (it's support)
- Use non-RTH snapshots for intraday analysis
- Forget to divide net_gamma by 1e9 for human-readable billions
- Run multiple instances (PID lock exists for a reason)
- Modify the Telegram session file while engine is running
