# Research and implementation audit — 6 October 2026

## Decision

**No tested candidate qualifies for deployment.** The web dashboard defaults to
observe-only. Experimental candidates are selectable for paper testing; no real
orders are implemented. The original trend and pair engines remain in the code.
The web session runs one selected strategy without the unvalidated pair overlay
so results can be attributed to that strategy. The CLI retains its original
pair configuration.

## Actual historical experiment

FYERS returned 6,227 five-minute bars each for IDEA, RELIANCE, SBIN, SUNPHARMA,
TATASTEEL and TCS: 83 common sessions, 8 June–5 October 2026. The first 60% is
indicator history. Validation starts 17 August; the final test starts 9 September.
Five fixed candidates were specified before opening the held-out results.

| Candidate | Validation net P&L | Validation trades | Held-out net P&L | Doubled-slippage test |
| --- | ---: | ---: | ---: | ---: |
| Original EMA trend | -12,727.89 | 60 | Not tested | Not tested |
| 20-bar breakout | -7,160.35 | 36 | Not tested | Not tested |
| 15-minute opening range | -4,682.35 | 14 | Not tested | Not tested |
| Trend pullback | -526.64 | 26 | +1,005.90 | -1,009.85 |
| Range reversion | -7,266.72 | 40 | Not tested | Not tested |

All amounts are rupees summed across **six separate ₹10 lakh symbol accounts**;
they are not results for a single ₹10 lakh portfolio. Pullback was frozen as the
validation winner before testing. We did not switch to another candidate after
seeing its test outcome. IDEA is excluded by the new candidates' price filter,
so zero trades there also fails the deliberately strict per-symbol acceptance
rule. Validation is negative even without that acceptance issue.

Costs include configured taxes, slippage and an additional **estimated** ₹20 per
order plus 18% GST on brokerage/exchange/SEBI fees. This is a conservative research
assumption, not a verified FYERS tariff. Held-out stress doubles slippage only.
Signals use completed candles and fill at the next bar open; stop precedes target
when both are touched; adverse gaps fill at open. No parameter was tuned using
the test period. Machine-readable results are in `strategy_report.json`.

This is a small fixed cross-section, not an exchange-wide historical top-volume
portfolio. Historical universe reconstruction, spread/impact, corporate actions,
survivorship, multiple rolling windows and forward paper validation remain open.
An untouched hold-out does not eliminate every form of overfitting.

## Research rationale

- [Zarattini, Barbon and Aziz: Opening Range Breakout research](https://alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content)
  examines US equities and elevated trading activity. Our 15-minute NSE candidate
  adds trend/VWAP confirmation and shared exits; it is not a replication and does
  not inherit the paper's reported returns.
- [Bailey et al.: The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
  explains why searching millions of combinations on finite data creates false
  discoveries. This release tests five explicit hypotheses and reports failures.
  It does not implement the paper's full CSCV/PBO procedure.
- The pullback and range-reversion rules are independent engineering hypotheses,
  not journal-validated recipes. Earlier primary research and its horizon limits
  are recorded in `NSE_AND_STRATEGY_RESEARCH.md`.

## Price discrepancy and data fixes

The user's loss log was historical replay of earlier sessions. Its prices should
not match today's live screen. The website distinguishes live FYERS quotes from
historical replay, shows timestamps, and labels quotes older than 60 seconds stale.
Quote snapshots retain FYERS' timestamp even when it is coarse or old. No clock
adjustment invents freshness. Snapshot requests mark prices but do not generate
trades or candle volume. New live entries require fresh ticks. Entry/exit fills
include simulated slippage and are displayed separately from quote LTP.

The FYERS authentication validation failure was caused by a missing log directory;
that directory is now created. The session and quote endpoint both validated.
Bulk history requests are paced and rate-limit responses get one bounded retry.
Out-of-order ticks and regressing cumulative volume no longer corrupt candles.
Invalid prices and fractional quantities cannot corrupt the virtual ledger.

## Run and verify

From `C:\Users\akava\OneDrive\Desktop\NexusQuant`:

```powershell
python -m quant_desk.web_app
python -m unittest discover -s quant_desk/tests -q
python -m quant_desk.smoke_test
python -m quant_desk.evaluate_strategies
```

The first command opens the browser at http://127.0.0.1:8765. Alternatively,
double-click `START_DASHBOARD.bat`. Keep the server terminal open. Live prices
require FYERS authentication; historical replay deliberately uses old prices.
Stop freezes the paper session; starting again creates a fresh account. Account
state is in memory and is not restored across server restarts.

The unit suite includes 2,000 deterministic randomized long/short/partial-exit
ledger scenarios. `smoke_test` replays five sessions of actual cached prices
through all five strategies and observe-only, checking errors and end-of-day
positions. Its close-only replay is an integration check, not return evidence.
