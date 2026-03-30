# FINAL BUILD INSTRUCTIONS — SIMPLE
# ===================================
# Drop this file in C:\Code\gex_telegram\
# Give to Claude Code: "Read final_build.py and build day_classifier.py"
# That's it.


# ===================================
# THE SYSTEM: 4 ALERTS, 2 DISCORD CHANNELS
# ===================================
#
# #gex-trades channel — you watch this
# #gex-engine channel — you ignore this
#
# Every day, max 4 alerts hit #gex-trades:
#   9:30 AM  → Morning Read (context, every day)
#   10:00 AM → Morning Bet (only if first 30 min moved enough)
#   2:00 PM  → Pin Butterfly (only if gamma positive + near wall)
#   Anytime  → EM Fade (only if price hits extreme level)
#
# You check your phone. Place the trade. Go back to work.
# Engine goes quiet after trades are done.


# ===================================
# ALERT 1: MORNING READ — 9:30 AM ET (every day)
# ===================================
#
# Purpose: context, not a trade
#
# Compute at 9:30:
#   anchor = prior day SPX close (yfinance or last RTH snapshot)
#   em_pts = (prior day VIX close / 16) / 100 * anchor
#   open_gamma = first RTH snapshot net_gamma
#   prior_range = prior day high - low from snapshots
#   prior_em = prior day's em_pts
#   compression = prior_range < prior_em (True/False)
#   coil = abs(prior day eod_gamma - open_gamma) > 5e9
#
# Message:
#   📊 MORNING READ
#   Anchor: {anchor} | EM: ±{em_pts}pts
#   Range: {anchor - 0.75*em_pts} to {anchor + 0.75*em_pts}
#   Gamma: {open_gamma in Bn, plain english}
#   Prior day: {compressed or normal}
#   Coil: {yes/no}


# ===================================
# ALERT 2: MORNING BET — 10:00 AM ET (conditional)
# ===================================
#
# ONLY fires when first_30_ratio > 0.3
#
# Compute at 10:00 AM ET (7:00 AM PT):
#   open_price = first RTH snapshot curr_price
#   price_at_10am = snapshot nearest 10:00 AM ET
#   first_30_high = max(curr_price) between 9:30-10:00 ET
#   first_30_low = min(curr_price) between 9:30-10:00 ET
#   first_30_range = first_30_high - first_30_low
#   first_30_ratio = first_30_range / em_pts
#   first_30_direction = price_at_10am - open_price
#
# If first_30_ratio <= 0.3 → send quiet day message, no trade
# If first_30_ratio > 0.3 → fire morning bet:
#
#   direction = same as first 30 min
#   if first_30_direction < 0: buy PUT
#   if first_30_direction > 0: buy CALL
#
#   entry = price_at_10am
#   stop = entry + 15 (if put) or entry - 15 (if call)
#   target = em_lower (if put) or em_upper (if call)
#   time_exit = 2:00 PM ET
#
# Big move score (count TRUE out of 5):
#   coil: abs(prior_eod_gamma - open_gamma) > 5e9
#   expansion: first_30_ratio > 0.3 (always true here)
#   gamma_accel: net_gamma < -3e9 AND gamma_at_10am < open_gamma - 1e9
#   compression: prior_day_range < prior_day_em_pts
#   sweep: first_30_low < prior_day_low OR first_30_high > prior_day_high
#
# If big_move_score >= 2:
#   label = "🔥 BIG MOVE — hold for runner, trail 10pts"
# Else:
#   label = "📎 Normal — take profit at target"
#
# Message:
#   🟢 MORNING BET — CALL (or 🔴 PUT)
#   SPX {price} | 10:00 AM
#   First 30m: {up/down} {pts}pts | Ratio {ratio}
#   Buy ATM {call/put}
#   Stop: {stop} | Target: {target}
#   Exit by 2:00 PM
#   Big Move: {score}/5 {label}
#
# ⚠️ DO NOT add gamma filter. It drops win rate from 54% to 22%.
# ⚠️ DO NOT fire if first_30_ratio > 0.3 is not met.


# ===================================
# ALERT 3: PIN BUTTERFLY — 2:00 PM ET (conditional)
# ===================================
#
# Check at 2:00 PM ET (11:00 AM PT):
#   gamma_2pm = net_gamma at snapshot nearest 2:00 PM ET
#   call_wall_2pm = call_wall at snapshot nearest 2:00 PM ET
#   price_2pm = curr_price at snapshot nearest 2:00 PM ET
#   distance = abs(price_2pm - call_wall_2pm)
#
# ONLY fires when:
#   gamma_2pm > 3e9 (positive gamma > 3 billion)
#   AND distance < 15 (price within 15pts of call wall)
#
# If conditions NOT met → no message, silence
# If conditions met:
#
#   Message:
#   🦋 PIN TRADE
#   SPX {price} | 2:00 PM
#   Gamma: {gamma in Bn} | Call wall: {call_wall}
#   Distance: {distance}pts
#   Butterfly at {call_wall} strike
#   Risk ~$500 | Reward ~$2,000
#
# Can fire same day as morning bet (morning exits by 2pm).
# Max 1 pin per day.
#
# ⚠️ Use 10pt win zone, NOT 5pt
# ⚠️ Use call_wall at 2pm for strike, NOT closing wall


# ===================================
# ALERT 4: EM FADE — anytime (conditional)
# ===================================
#
# Monitor every RTH snapshot:
#   em_position = (curr_price - anchor) / em_pts
#
# LONG trigger: price crosses BELOW anchor - (0.75 * em_pts)
#   → first snapshot where curr_price < fade_long_level
#   → buy CALL
# SHORT trigger: price crosses ABOVE anchor + (0.75 * em_pts)
#   → first snapshot where curr_price > fade_short_level
#   → buy PUT
#
# One trigger per direction per day.
#
# SKIP if:
#   first_30_ratio > 0.5 (big open = trend day, fades blow up)
#   OR morning bet is currently active
#
# entry = curr_price at trigger
# stop = em_lower (for long) or em_upper (for short)
# target = anchor - 0.25*em_pts (for long) or anchor + 0.25*em_pts (for short)
# time_exit = 60 minutes after entry
#
# Message:
#   🟢 EM FADE LONG (or 🔴 SHORT)
#   SPX {price} | {time}
#   Hit {level} (EM position: {em_position})
#   Entry: {price} | Stop: {stop} | Target: {target}
#   Exit in 60 min


# ===================================
# TRADE EXIT — when any trade ends
# ===================================
#
# Monitor active trades each snapshot.
# Check: did price hit stop? target? time exit?
#
# Stop hit:
#   ❌ STOPPED — {trade type}
#   SPX {price} | {time}
#
# Target hit:
#   ✅ TARGET — {trade type}
#   SPX {price} | {time}
#
# Time exit:
#   ⏰ TIME EXIT — {trade type}
#   SPX {price} | {time}
#
# After last trade completes:
#   🔇 DONE FOR TODAY


# ===================================
# CHANNEL RELAY — external signals
# ===================================
#
# When Telegram channels post SPX trades,
# relay to #gex-trades with GEX context:
#
#   📡 {channel name}: {their call}
#   GEX: {gamma english} | EM position: {position}
#   Big Move: {score}/5
#   {verdict: ALIGNS / CONFLICTS / NEUTRAL}
#
# Keep it short. One line verdict.


# ===================================
# ROUTING
# ===================================
#
# #gex-trades (clean):
#   MORNING_READ, MORNING_BET, PIN_BUTTERFLY,
#   EM_FADE, TRADE_EXIT, ENGINE_QUIET,
#   CHANNEL_RELAY, REGIME_SHIFT
#
# #gex-engine (raw — ignore this):
#   RTH_PULSE, MICRO_PULSE, TREND_DASHBOARD,
#   HOURLY_RECAP, QUIET_SUMMARY, OVERNIGHT_DRIFT,
#   GAMMA_COMPRESSION, STRIKE_FLIP, TRADING_SUMMARY
#
# SUPPRESS entirely (dead signals):
#   RUBBER_BAND, GAMMA_SNAP, WALL_REJECTION,
#   LIQUIDITY_SWEEP


# ===================================
# DAILY RULES
# ===================================
#
# Max 1 directional trade (morning bet OR em fade, not both)
# Pin butterfly is separate (can coexist)
# Max 2 trades total per day
# After done → ENGINE_QUIET → silence until tomorrow


# ===================================
# DO NOT
# ===================================
#
# ❌ Gamma filter on morning bet (kills it)
# ❌ 5pt pin zone (too tight, use 10pt)
# ❌ Fade on Group A days (blows up 22-72pts)
# ❌ Coil/swing alerts (not proven yet)
# ❌ Gamma snap / wall rejection alerts (dead)
# ❌ Raw numbers in messages (say +5.2Bn not 5200000000)
# ❌ Telegram alerts (use Discord)
# ❌ More than 2 trades per day
# ❌ Purging gex_data.db
