"""Debug: inspect raw message text and parser behavior."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gex_db import get_conn
from gex_parser import _clean_text, RE_PRICE, RE_NET_GAMMA, RE_TOTAL_CALL, RE_TOTAL_PUT, RE_SECTION_SPLIT, RE_STRIKE_GAMMA

conn = get_conn()
row = conn.execute("SELECT raw_text FROM gex_snapshots ORDER BY timestamp_pt DESC LIMIT 1").fetchone()
conn.close()

if not row:
    print("No data")
    sys.exit()

raw = row[0]
print(f"RAW LENGTH: {len(raw)}")
print()

# Show non-ASCII characters
print("NON-ASCII CHARS:")
for i, ch in enumerate(raw[:500]):
    if ord(ch) > 127:
        print(f"  pos {i}: U+{ord(ch):04X}")

print()
cleaned = _clean_text(raw)
print("CLEANED TEXT:")
print(cleaned)
print()

# Test each regex
print("REGEX MATCHES:")
m = RE_PRICE.search(cleaned)
print(f"  Price: {m.group(1) if m else 'NO MATCH'}")

m = RE_NET_GAMMA.search(cleaned)
print(f"  Net Gamma: {m.group(1) if m else 'NO MATCH'}")

m = RE_TOTAL_CALL.search(cleaned)
print(f"  Total Call: {m.group(1) if m else 'NO MATCH'}")

m = RE_TOTAL_PUT.search(cleaned)
print(f"  Total Put: {m.group(1) if m else 'NO MATCH'}")

parts = RE_SECTION_SPLIT.split(cleaned)
print(f"  Section split parts: {len(parts)}")
for i, p in enumerate(parts):
    print(f"    [{i}] {p[:80]}...")

strikes = RE_STRIKE_GAMMA.findall(cleaned)
print(f"  Strike/Gamma matches: {len(strikes)}")
for s, g in strikes[:3]:
    print(f"    Strike {s}, Gamma {g}")
