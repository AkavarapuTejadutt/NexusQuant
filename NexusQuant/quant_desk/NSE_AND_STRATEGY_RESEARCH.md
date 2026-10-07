# NSE selection and strategy research — 6 October 2026

Subsequent work added a local website, validated FYERS authentication, and completed
historical candidate research. See [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) for
the current results and limitations; the verification section below describes
the earlier implementation stage.

## Implemented market coverage

NSE supplies stock rankings; FYERS supplies ticks and candles, as requested.
Momentum now selects 50 stocks by today's **shares traded**, rather than a fixed
Nifty list, percentage price change, or traded rupee value. An independent
relative-volume filter still controls momentum entries.

The official 6 October 2026 daily report produced 50 stocks from 2,328 traded
EQ equities present in NSE's current equity security master. The saved JSON and
CSV record source, session date, coverage, ranking and sectors. This is an exact
end-of-day ranking within that domain; ETFs, SME and other trading series are
excluded. It is not a list of every exchange instrument.

Sources: [NSE daily report](https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_06102026.csv),
[NSE equity master](https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv),
[NSE most-active equities](https://www.nseindia.com/market-data/most-active-equities).

During trading hours the adapter combines NSE's public broad-market, sector and
most-active snapshots. Their coverage is incomplete: the public most-active
endpoint returned only 20 stocks, and broader snapshots do not contain every EQ
stock. Intraday results are therefore explicitly labeled **partial coverage**,
never an exchange-wide top 50. A complete intraday ranking requires a complete
volume dataset; NSE's [market-data subscriptions](https://www.nseindia.com/static/market-data/real-time-data-subscription)
describe exchange data services. The integration does not claim such a feed.

The selection refreshes every five minutes. Invalid or stale snapshots fail
visibly rather than substituting a static Nifty list. Refresh failures block new
entries while existing positions retain protective exits. Existing subscriptions
and open positions survive selection changes. Sector classifications come from
NSE lists where available; unknown stocks share a conservative risk bucket.

Bank, metal, pharma and other sectors are eligible when their stocks rank in the
selection. NSE-listed Sensex constituents are also eligible. This does not add
direct trading of the Sensex index or BSE instruments; Sensex is a BSE index.
No sector is guaranteed a place in a volume-ranked list.

## Your retained strategies

| Strategy | Current behavior |
| --- | --- |
| Momentum | Original EMA 6/30, relative volume, ADX and VWAP logic; completed 5-minute decisions with 15-minute direction confirmation; supports long and short signals |
| Pairs / statistical arbitrage | Original configured pairs retained and enabled by default; additional same-sector candidates at startup; require both cointegration tests, valid history and two-leg risk controls |
| RL filter | Original module retained; optional and disabled by default; untrained random weights no longer veto otherwise valid trades |
| Breakout candidate | Optional previous-20-bar breakout using the same regime and risk controls; default remains the original trend candidate |

Added rejection of unusually extended entries and sudden volatility shocks
(three times the preceding true-range baseline). These numerical settings,
cooldown and risk limits are engineering candidates, not optimized discoveries.
Downward movement can produce short signals; a declining stock alone is not a
short-entry condition. Broker/instrument short-sale eligibility still requires
operational validation. In weak or conflicting regimes, holding cash is allowed.

## Primary research and what it actually supports

1. Moskowitz, Ooi and Pedersen (2012), *Time series momentum*, Journal of
   Financial Economics. [Paper](https://doi.org/10.1016/j.jfineco.2011.11.003).
   Documents trend effects across futures with horizons measured in months.
   It motivates examining direction-based approaches across markets; it does
   not validate EMA 6/30 on five-minute Indian cash equities.
2. Gatev, Goetzmann and Rouwenhorst (2006), *Pairs Trading: Performance of a
   Relative-Value Arbitrage Rule*, Review of Financial Studies.
   [Author paper](https://depot.som.yale.edu/icf/papers/fileuploads/2573/original/08-03.pdf).
   Studies historical daily US stock pairs using normalized-distance selection.
   It motivates relative-value research, not a claim that this cointegration
   implementation reproduces their results intraday.
3. Daniel and Moskowitz (2016), *Momentum crashes*, Journal of Financial
   Economics. [Paper](https://kentdaniel.net/papers/published/jfe_16.pdf).
   Momentum can suffer severe losses around rebounds following declines and
   high volatility. This supports investigating regime and exposure controls;
   it does not specify our three-times-ATR threshold.
4. Moreira and Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance.
   [Working paper](https://www.nber.org/papers/w22208).
   Historical factor evidence motivates testing lower exposure in high
   volatility. It is not evidence of universal intraday profitability.
5. Heston, Korajczyk and Sadka (2010), *Intraday Patterns in the Cross-section
   of Stock Returns*, Journal of Finance.
   [Paper](https://doi.org/10.1111/j.1540-6261.2010.01573.x).
   Intraday seasonality and liquidity effects motivate evaluating signals by
   time of day and modeling trading friction rather than treating every bar
   identically.

## Verification and remaining evidence

31 unit tests pass, including ledger reconciliation, open positions, protective
stops, bearish signals, shock rejection, NSE filtering, stale-data rejection,
partial-coverage disclosure and watchlist refresh. The official daily ranking
was downloaded and generated successfully. No real FYERS session or historical
profitability test was completed because no access token was available.

The chronological evaluator compares trend and breakout candidates with
validation and held-out windows and next-bar execution. It does not currently
evaluate the combined portfolio, pairs or RL strategy. Selecting today's volume
leaders for historical evaluation creates hindsight bias; robust universe tests
must reconstruct each historical day's available ranking. Full actual costs,
slippage, liquidity, survivorship and forward paper results remain necessary.
No candidate is approved as optimal or profitable.

Commands from the project root:

```powershell
python -m unittest discover -s quant_desk/tests -q
python -m quant_desk.data.nse_market --date 2026-10-06
python quant_desk/main.py --auth
python -m quant_desk.evaluate_strategies --download --days 90 --symbols SBIN RELIANCE TCS
```

The workspace contains a Python engine and terminal dashboard. The separate
website/Paper One/Paper Two frontend was not present, so its account binding and
UI updates could not be inspected or verified.
