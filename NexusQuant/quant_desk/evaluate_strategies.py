"""Chronological candidate comparison using completed 5m OHLCV and next-bar fills.

Run: python -m quant_desk.evaluate_strategies --download --days 90
Or:  python -m quant_desk.evaluate_strategies --data-dir path/to/csvs
CSV schema: datetime,open,high,low,close,volume (timestamps in IST).
"""
import argparse
import contextlib
import io
import json
from dataclasses import replace
from datetime import datetime, timedelta, time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from quant_desk.core.config import FYERS_CONFIG, MOMENTUM_CONFIG, RISK_CONFIG, FRICTION_CONFIG, get_access_token
from quant_desk.engines.research_strategies import STYLES
from quant_desk.data.data_feed import CandleAggregator
from quant_desk.engines.momentum import MomentumEngine
from quant_desk.execution.paper_broker import PaperBroker
from quant_desk.risk.portfolio_risk import PortfolioRiskManager
from quant_desk.data.universe import UniverseManager


def read_candles(path):
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    required = ["open", "high", "low", "close", "volume"]
    df = df[required].astype(float)
    if df.index.has_duplicates or not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{path.name}: duplicate timestamps or invalid numeric data")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError(f"{path.name}: nonpositive prices or negative volume")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
        (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{path.name}: invalid OHLC ordering")
    if any(t.second or t.minute % 5 for t in df.index):
        raise ValueError(f"{path.name}: expected 5-minute candle starts")
    return df.between_time("09:15", "15:25")


class ResearchBroker(PaperBroker):
    """Conservative estimated Rs20/order plus 18% GST; not a broker tariff quote."""
    def _execution_fees(self, price, quantity, side):
        return super()._execution_fees(price, quantity, side) + 20 * 1.18 + price * quantity * (self.friction.exchange_txn_pct + self.friction.sebi_fee_pct) * .18

    def affordable_quantity(self, price, side):
        fill = price * (1 + self.friction.slippage_pct if side == "BUY" else 1 - self.friction.slippage_pct)
        low, high = 0, max(0, int(self.cash / fill))
        while low < high:
            mid = (low + high + 1) // 2
            if fill * mid + self._execution_fees(fill, mid, side) <= self.cash:
                low = mid
            else:
                high = mid - 1
        return low

    def calculate_roundtrip_friction(self, price, quantity):
        values = list(super().calculate_roundtrip_friction(price, quantity))
        values[0] += 40 * 1.18 + price * quantity * 2 * (self.friction.exchange_txn_pct + self.friction.sebi_fee_pct) * .18
        return tuple(values)


def run_candidate(df, symbol, style, start, end, slippage_multiplier=1):
    """Independent single-symbol account; no cross-symbol portfolio claim."""
    engine = MomentumEngine(replace(MOMENTUM_CONFIG, entry_style=style))
    broker = ResearchBroker(friction_cfg=replace(FRICTION_CONFIG, slippage_pct=FRICTION_CONFIG.slippage_pct * slippage_multiplier))
    risk = PortfolioRiskManager()
    history = df.loc[df.index < end].copy()
    agg = CandleAggregator(symbol, buffer_size=len(history))
    agg.load_history(history)
    five, fifteen = agg.get_5m_dataframe(), agg.get_15m_dataframe()
    five["vwap"] = engine.calculate_vwap(five)
    five["ema_fast"] = engine.calculate_ema(five["close"], 20)
    five["ema_slow"] = engine.calculate_ema(five["close"], 50)
    fifteen["adx_14"] = engine.calculate_adx(fifteen)
    fifteen["higher_ema"] = engine.calculate_ema(fifteen["close"], 30)
    marks = [broker.initial_cash]
    last_entry = None
    pending = None
    pending_exit = False
    day = None
    locked = False
    daily_baseline = broker.initial_cash
    for ts, bar in history.loc[history.index >= start].iterrows():
        if day != ts.date():
            day, locked, daily_baseline = ts.date(), False, marks[-1]
        if pending_exit and symbol in broker.open_positions:
            broker.close_position(symbol, float(bar.open), "SIGNAL_REVERSAL")
            last_entry = ts
        pending_exit = False
        pos = broker.open_positions.get(symbol)
        if pos:
            # Gap through stop fills at open; if stop and target share a bar,
            # stop wins. Trailing updates become effective on the next bar.
            stop_hit = bar.low <= pos.stop_loss_price if pos.side == "LONG" else bar.high >= pos.stop_loss_price
            if stop_hit:
                fill = min(bar.open, pos.stop_loss_price) if pos.side == "LONG" else max(bar.open, pos.stop_loss_price)
                broker.close_position(symbol, float(fill), "STOP_LOSS")
                last_entry = ts
        pos = broker.open_positions.get(symbol)
        if pending and pos is None and not locked and time(9, 30) <= ts.time() < time(15, 0):
            action, atr, signal_ts = pending
            # A signal cannot be carried across a data gap or a session boundary.
            if ts.date() == signal_ts.date() and ts - signal_ts == pd.Timedelta(minutes=5):
                price = float(bar.open)
                stop_dist = max(atr * engine.config.atr_stop_mult, price * .01)
                unit_risk = stop_dist + price * broker.friction.slippage_pct
                equity = marks[-1]
                qty = min(int(equity * risk.config.max_risk_per_trade_pct / unit_risk),
                          int(equity * risk.config.max_position_size_pct / price),
                          broker.affordable_quantity(price, action))
                target_dist = max(atr * engine.config.atr_target_mult, stop_dist * 1.5)
                if qty > 0 and broker.passes_friction_payoff_gate(target_dist * qty, price, qty):
                    stop = price - stop_dist if action == "BUY" else price + stop_dist
                    broker.execute_market_order(symbol, action, qty, price, atr_14=atr, stop_loss_price=stop)
                    broker.open_positions[symbol].entry_time = ts.to_pydatetime()
                    last_entry = ts
                    # Also test the newly entered position's stop in its fill bar.
                    pos = broker.open_positions.get(symbol)
                    if (action == "BUY" and bar.low <= stop) or (action == "SELL" and bar.high >= stop):
                        broker.close_position(symbol, stop, "STOP_LOSS")
                        last_entry = ts
        pending = None
        pos = broker.open_positions.get(symbol)
        if pos:
            target_dist = max(pos.atr_14 * engine.config.atr_target_mult,
                              max(pos.atr_14 * engine.config.atr_stop_mult, pos.entry_price * .01) * 1.5)
            target = pos.entry_price + target_dist if pos.side == "LONG" else pos.entry_price - target_dist
            target_hit = bar.high >= target if pos.side == "LONG" else bar.low <= target
            if not pos.stage1_taken and target_hit:
                if pos.quantity > 1:
                    qty = min(pos.quantity - 1, max(1, int(pos.quantity * engine.config.partial_tp_pct)))
                    broker.close_partial_position(symbol, qty, target, "STAGE1_TARGET")
                    pos.stop_loss_price = pos.entry_price
                else:
                    broker.close_position(symbol, target, "TARGET")
                    last_entry = ts
            pos = broker.open_positions.get(symbol)
            if pos and pos.stage1_taken:
                trail_dist = max(pos.atr_14 * engine.config.atr_stop_mult, pos.entry_price * .01)
                if pos.side == "LONG":
                    pos.highest_price = max(pos.highest_price, float(bar.high))
                    pos.stop_loss_price = max(pos.stop_loss_price, pos.highest_price - trail_dist)
                else:
                    pos.lowest_price = min(pos.lowest_price, float(bar.low))
                    pos.stop_loss_price = min(pos.stop_loss_price, pos.lowest_price + trail_dist)
        equity = broker.get_portfolio_summary({symbol: float(bar.close)})["total_equity"]
        if equity - daily_baseline <= daily_baseline * risk.config.daily_kill_switch_pct:
            broker.square_off_all_positions({symbol: float(bar.close)}, "DAILY_LIMIT_BAR_CLOSE")
            locked = True
        is_last = ts == history.index[-1] or not (history.index > ts).any()
        next_rows = history.index[history.index > ts]
        day_end = not len(next_rows) or next_rows[0].date() != ts.date()
        if ts.time() >= time(15, 20) or day_end or is_last:
            broker.square_off_all_positions({symbol: float(bar.close)}, "END_OF_DAY")
        marks.append(broker.get_portfolio_summary({symbol: float(bar.close)})["total_equity"])
        if not locked and time(9, 25) <= ts.time() < time(15, 20):
            if symbol not in broker.open_positions and last_entry is not None and ts - last_entry < pd.Timedelta(minutes=5 * engine.config.cooldown_bars):
                continue
            idx_15m = fifteen.index.searchsorted(ts - pd.Timedelta(minutes=10), side="right")
            idx_5m = five.index.searchsorted(ts, side="right")
            complete15 = fifteen.iloc[max(0, idx_15m - 500) : idx_15m]
            candles = five.iloc[max(0, idx_5m - 500) : idx_5m]
            signal = engine.evaluate_signal(symbol, candles, complete15.iloc[-500:])
            pos = broker.open_positions.get(symbol)
            if pos and ((pos.side == "LONG" and signal.action == "SELL") or (pos.side == "SHORT" and signal.action == "BUY")):
                pending_exit = True
            elif pos is None and ts.time() < time(14, 55) and signal.action in ("BUY", "SELL"):
                pending = (signal.action, risk.calculate_atr(candles), ts)
    # Count complete trades rather than partial exit records.
    pnl_by_entry = {}
    for rec in broker.trade_history:
        pnl_by_entry[rec.entry_time] = pnl_by_entry.get(rec.entry_time, 0) + rec.net_pnl
    pnls = list(pnl_by_entry.values())
    wins, losses = sum(max(0, p) for p in pnls), -sum(min(0, p) for p in pnls)
    curve = np.array(marks)
    peaks = np.maximum.accumulate(curve)
    drawdown = float(np.max((peaks - curve) / peaks) * 100)
    return {"net_pnl": float(curve[-1] - curve[0]), "return_pct": float((curve[-1] / curve[0] - 1) * 100),
            "max_drawdown_pct": drawdown, "trades": len(pnls),
            "win_rate_pct": sum(p > 0 for p in pnls) / len(pnls) * 100 if pnls else 0,
            "profit_factor": wins / losses if losses else None,
            "cost_model": "configured taxes + estimated Rs20/order + GST18% + slippage; no measured spread/impact/latency",
            "slippage_multiplier": slippage_multiplier}


def download(directory, symbols, days):
    from fyers_apiv3 import fyersModel
    token = get_access_token()
    if not token:
        raise RuntimeError("No FYERS access token available")
    client = fyersModel.FyersModel(client_id=FYERS_CONFIG.client_id, token=token, is_async=False, log_path="")
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    directory.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        rows = []
        begin = today - timedelta(days=days)
        while begin < today:
            until = min(today - timedelta(days=1), begin + timedelta(days=29))
            response = client.history({"symbol": UniverseManager.to_fyers_symbol(symbol), "resolution": "5",
                                       "date_format": "1", "range_from": str(begin), "range_to": str(until), "cont_flag": "1"})
            if response.get("s") != "ok":
                raise RuntimeError(f"FYERS history rejected for {symbol} (code {response.get('code')}); authenticate and retry")
            rows.extend(response.get("candles", []))
            begin = until + timedelta(days=1)
        df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
        if df.empty:
            raise RuntimeError(f"No FYERS historical candles for {symbol}")
        df["datetime"] = pd.to_datetime(df.datetime, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df.drop_duplicates("datetime").sort_values("datetime").to_csv(directory / f"{symbol}.csv", index=False)
        print(f"Saved {symbol}: {len(df)} bars")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", nargs="+", default=["SBIN", "RELIANCE", "TCS"])
    parser.add_argument("--data-dir", type=Path, default=Path("quant_desk/research_data"))
    parser.add_argument("--report", type=Path, default=Path("quant_desk/strategy_report.json"))
    args = parser.parse_args()
    try:
        if args.download:
            download(args.data_dir, args.symbols, args.days)
        files = sorted(args.data_dir.glob("*.csv"))
        if not files:
            raise RuntimeError("No historical CSV files available; use --download or --data-dir")
        datasets = {p.stem: read_candles(p) for p in files}
        sessions = sorted(set.intersection(*(set(df.index.date) for df in datasets.values())))
        if len(sessions) < 30:
            raise RuntimeError("Need at least 30 common trading sessions for chronological evaluation")
        validation_start = pd.Timestamp(sessions[int(len(sessions) * .6)])
        test_start = pd.Timestamp(sessions[int(len(sessions) * .8)])
        end = pd.Timestamp(sessions[-1]) + pd.Timedelta(days=1)
        report = {"status": "evaluated", "common_sessions": len(sessions),
                  "validation_start": str(validation_start), "test_start": str(test_start),
                  "method": "First 60% warmup, next 20% candidate selection, final 20% untouched evaluation; independent symbol accounts",
                  "candidates": {}, "selected_candidate": None, "approved_for_use": False}
        report["candidate_count"] = len(STYLES)
        report["data_start"] = str(sessions[0])
        report["data_end"] = str(sessions[-1])
        for style in STYLES:
            report["candidates"][style] = {}
            for symbol, df in datasets.items():
                print(f"Validating {style}: {symbol}", flush=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_candidate(df, symbol, style, validation_start, test_start)
                report["candidates"][style][symbol] = {"validation": result}
        scores = {style: sum(r["validation"]["net_pnl"] for r in results.values())
                  for style, results in report["candidates"].items()}
        selected = max(scores, key=scores.get)
        report["best_validation_candidate"] = selected
        # Freeze selection before opening the final test period. Rejected winners
        # are reported, never replaced by another strategy based on test results.
        for symbol, df in datasets.items():
            print(f"Held-out evaluation {selected}: {symbol}", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                report["candidates"][selected][symbol]["test"] = run_candidate(df, symbol, selected, test_start, end)
                report["candidates"][selected][symbol]["stress_test"] = run_candidate(df, symbol, selected, test_start, end, 2)
        checks = []
        for result in report["candidates"][selected].values():
            for period in ("validation", "test", "stress_test"):
                r = result[period]
                checks.append(r["trades"] >= 10 and r["net_pnl"] > 0 and r["profit_factor"] is not None
                              and r["profit_factor"] > 1 and r["max_drawdown_pct"] <= 2)
        report["passes_exploratory_checks"] = all(checks)
        report["selected_candidate"] = selected if all(checks) else "observe"
        report["deployment_decision"] = "Forward paper validation needed" if all(checks) else "No candidate passed; observe only"
        report["limitations"] = "Selection is exploratory, not proof of optimality. Requires broader history, full broker costs and forward paper validation."
    except (RuntimeError, ValueError) as exc:
        report = {"status": "unavailable", "approved_for_use": False, "reason": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
