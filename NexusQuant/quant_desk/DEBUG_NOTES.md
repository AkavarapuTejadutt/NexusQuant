# Paper trading debugging

This supplied project contains a Python trading engine and Rich terminal dashboard.
It contains no website, web server, or Paper One / Paper Two frontend.
Both available modes execute orders through the virtual PaperBroker:

```powershell
python quant_desk/main.py --mode live
```

`live` means live FYERS market data with virtual orders in this source tree. It
requires a valid access token and the existing 09:15–15:30 IST execution gate.
`--mode sim` replays historical FYERS one-minute closes and also requires a token.
Restart the engine after updating source or configuration.

## Fixed defects

- FYERS SymbolUpdate's `vol_traded_today` was ignored. Live volume often stayed
  zero, blocking the momentum liquidity gate. Cumulative volume now becomes
  per-update volume, with daily reset handling. The first snapshot establishes
  a baseline rather than adding the whole day's volume to a single candle.
- Historical 15-minute candles were copies of 5-minute candles. They now
  resample OHLCV correctly. History uses floating-point columns for tick updates.
- VWAP used inconsistent definitions between history and live candles. Momentum
  now calculates session VWAP from price-volume sums and resets each day.
- Dynamic scanner selections could omit pair legs. All configured pair symbols
  now receive history and live subscriptions.
- Strategy callback exceptions were swallowed. They are logged and counted;
  the dashboard shows the last error, tick count, last tick, and entry reasons.
- Slippage was charged in both fill prices and cash fees. Cash fees now exclude
  slippage, which remains in execution prices. Partial closes allocate original
  entry fees and release proportional position risk. Realized plus unrealized
  P&L reconciles to equity less initial capital.
- The supplied `.env` had `STT_PCT=0.025` (2.5%). It now matches the intended
  code setting `0.00025` (0.025%).
- Paper orders now reject duplicate symbols and insufficient cash. Pair entries
  check both legs' capital and fees, use price-adjusted log-beta hedge quantities,
  and avoid overwriting positions belonging to another strategy.
- Dashboard snapshots now capture ledger totals and copied positions together
  under the strategy lock. Rendering no longer triggers the risk kill switch.
- Historical playback no longer seeds indicator buffers with later intraday
  candles before replay. Daily pair calibration still uses downloaded history;
  simulation is not a validated backtest.

## Expected behavior

Initial capital remains ₹10 lakh. Cash changes on entry/exit; equity and P&L
change with market prices while positions are open. A connected API alone does
not create trades: indicator warmup, trend, volume, RL confidence, and risk gates
must pass. No open trades means no market-driven P&L fluctuation. Read the
dashboard's entry/waiting reason instead of assuming a connection implies entry.

Paper positions belong to this in-memory ledger, not FYERS account positions.
Restarting currently resets virtual cash and positions. Website wiring and live
FYERS-session verification remain unverified because no website source was supplied.

## Validation

```powershell
python -m unittest discover -s quant_desk/tests -v
```

Regression tests cover FYERS volume deltas, exchange timestamps, daily reset,
callback failure reporting, real 15-minute resampling, session VWAP, momentum
entry, long/short mark-to-market, partial-close reconciliation, cash rejection,
and tick-to-position-to-dashboard execution using mocked entry decisions.
