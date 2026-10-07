import os
import sys
import time
import argparse
import threading
import pytz
import pandas as pd
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import replace

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box

from quant_desk.core.config import RISK_CONFIG, MOMENTUM_CONFIG, get_access_token
from quant_desk.core.auth import FyersAuthenticator
from quant_desk.data.universe import UniverseManager, NIFTY_PAIRS, NIFTY_50_SYMBOLS
from quant_desk.data.data_feed import DataFeedManager
from quant_desk.engines.stat_arb import StatArbEngine, StatArbSignal
from quant_desk.engines.momentum import MomentumEngine, MomentumSignal
from quant_desk.engines.rl_agent import RLMetaFilter, RLActionDecision
from quant_desk.risk.portfolio_risk import PortfolioRiskManager
from quant_desk.execution.paper_broker import PaperBroker
from quant_desk.data.nse_market import NSEDataError


def enforce_market_hours() -> None:
    """Strict execution gate for LIVE mode: Halts program outside 09:15 - 15:30 IST."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now.weekday() >= 5 or not (market_open <= now <= market_close):
        print(f"\n[SYSTEM FATAL] Current Time: {now.strftime('%I:%M %p')} IST.")
        print("[SYSTEM FATAL] Market is CLOSED. Live execution requires active trading hours (09:15 AM - 03:30 PM).")
        print("[SYSTEM NOTE] To test offline or after-hours, run in simulation mode:")
        print("              python quant_desk/main.py --mode sim\n")
        sys.exit(0)


class MasterQuantDesk:
    """Master Orchestrator for the Hybrid Orthogonal Algorithmic Trading Framework + RL Meta-Agent."""

    def __init__(self, mode: str = "sim", style: Optional[str] = None):
        self.mode = mode
        self.console = Console()
        
        # 1. Universe & Data Infrastructure
        self.universe = UniverseManager()
        self.symbols = self.universe.get_nse_top_volume(top_n=50)
        self.momentum_symbols = set(self.symbols)
        self.pairs = self.universe.get_eligible_pairs(NIFTY_PAIRS)
        self.symbols = list(dict.fromkeys(self.symbols + [s for pair in self.pairs for s in pair]))
        self.data_feed = DataFeedManager(self.symbols)
        
        # 2. Strategy & RL Engines
        self.stat_arb = StatArbEngine()
        self.momentum = MomentumEngine(replace(MOMENTUM_CONFIG, entry_style=style)) if style else MomentumEngine()
        self.rl_filter = RLMetaFilter()

        # 3. Risk & Money Management
        self.risk_mgr = PortfolioRiskManager(initial_capital=RISK_CONFIG.initial_capital)

        # 4. Execution Ledger
        self.broker = PaperBroker(initial_cash=RISK_CONFIG.initial_capital)

        # Tracking state
        self.latest_prices: Dict[str, float] = {}
        self.latest_stat_signals: Dict[str, StatArbSignal] = {}
        self.latest_mom_signals: Dict[str, MomentumSignal] = {}
        self.latest_rl_decisions: Dict[str, RLActionDecision] = {}
        self.execution_status: Dict[str, str] = {}
        self.last_pair_holding_bars: Dict[str, int] = {}
        self.last_evaluated_bar = {}
        self.last_entry_bar = {}
        self.latest_price_times = {}
        self.session_date = None
        self.entries_allowed = True
        self.lock = threading.Lock()
        self.is_running = False
        self.universe_error = ""
        self.universe_refresh_thread = None
        self._universe_stop = threading.Event()
        self.now_ist = lambda: pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

    def initialize_engines(self) -> None:
        """Calibrate Cointegration pairs and backfill historical candles."""
        self.console.print("[bold cyan][INIT] Initializing Institutional Quant Desk Framework + RL Meta-Agent...[/bold cyan]")
        if self.mode == "live":
            # Quote snapshots mark the ledger only. Never feed a closing snapshot
            # into candle construction or trigger orders from it.
            for symbol, price, stamp in self.data_feed.fetch_quote_snapshot():
                with self.lock:
                    if symbol not in self.latest_prices:
                        self.latest_prices[symbol] = price
                        self.latest_price_times[symbol] = stamp
        
        df_daily = self.data_feed.fetch_historical_daily(days=180) if self.stat_arb.config.enabled else pd.DataFrame()
        if not self.stat_arb.config.enabled:
            self.console.print("[yellow]Pair entries disabled pending intraday validation (ENABLE_STAT_ARB=false).[/yellow]")
        else:
            discovered = self.universe.discover_pairs(df_daily, sorted(self.momentum_symbols))
            self.pairs = list(dict.fromkeys(self.pairs + discovered))
        
        for asset_a, asset_b in self.pairs:
            if self.stat_arb.config.enabled and asset_a in df_daily.columns and asset_b in df_daily.columns:
                try:
                    coint_model = self.stat_arb.calibrate_pair(
                        asset_a, asset_b, df_daily[asset_a], df_daily[asset_b]
                    )
                except (ValueError, ArithmeticError) as exc:
                    self.console.print(f"Pair {asset_a}/{asset_b} calibration failed: {exc}")
                    continue
                status = "COINTEGRATED (p<0.05)" if coint_model.is_cointegrated else "NON-STATIONARY"
                self.console.print(
                    f"  - Pair [yellow]{asset_a}/{asset_b}[/yellow]: {status} | "
                    f"Beta: [green]{coint_model.beta:.4f}[/green] | Half-life: {coint_model.half_life:.1f} days"
                )

        if self.mode != "sim":
            self.data_feed.backfill_intraday_candles(days=10)
        self.data_feed.register_tick_callback(self.on_tick_update)

    def on_tick_update(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """Core event loop handler executed on every tick update."""
        with self.lock:
            symbol = UniverseManager.to_clean_symbol(symbol)
            ts = pd.Timestamp(ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
            previous_time = self.latest_price_times.get(symbol)
            if previous_time is not None and ts < previous_time:
                return
            if self.session_date is not None and ts.date() < self.session_date:
                return
            self.latest_prices[symbol] = price
            self.latest_price_times[symbol] = ts
            if self.session_date != ts.date():
                if self.session_date is not None and self.broker.open_positions:
                    self.broker.square_off_all_positions(self.latest_prices, reason="SESSION_ROLLOVER")
                equity = self.broker.get_portfolio_summary(self.latest_prices)["total_equity"]
                self.risk_mgr.reset_daily_session(equity)
                self.session_date = ts.date()
            # Strict EOD entry cutoff at 2:45 PM IST (14:45) to avoid late intraday market noise
            self.entries_allowed = ts.weekday() < 5 and clock_time(9, 30) <= ts.time() < clock_time(14, 45)
            if self.mode == "live":
                quote_age = (self.now_ist() - ts).total_seconds()
                self.entries_allowed = self.entries_allowed and -5 <= quote_age <= 60
            # Strict NSE intraday square-off at 3:15 PM IST (15:15) to prevent cash short auction penalties
            if ts.time() >= clock_time(15, 15):
                self.broker.square_off_all_positions(self.latest_prices, reason="END_OF_DAY_315PM")
                self.execution_status[symbol] = "End-of-day: new entries blocked"
                return
            
            # -------------------------------------------------------------
            # Central Risk Gate: Check Daily Kill Switch (-2.0% Stop)
            # -------------------------------------------------------------
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            self.risk_mgr.current_capital = summary["total_equity"]
            total_pnl = summary["total_equity"] - self.risk_mgr.starting_daily_capital
            kill_triggered, kill_reason = self.risk_mgr.evaluate_kill_switch(total_pnl, now=ts.to_pydatetime())

            if kill_triggered:
                if self.broker.open_positions:
                    closed = self.broker.square_off_all_positions(self.latest_prices, reason="KILL_SWITCH_-2pct")
                    for rec in closed:
                        self.rl_filter.process_reward_step(rec.symbol, rec.net_pnl, rec.total_friction, is_closed=True)
                return  # Block execution

            # Existing protective stops must run even if indicator evaluation fails.
            protected = self.broker.open_positions.get(symbol)
            if protected and protected.strategy.startswith("MOMENTUM"):
                stop_hit = ((protected.side == "LONG" and price <= protected.stop_loss_price) or
                            (protected.side == "SHORT" and price >= protected.stop_loss_price))
                if stop_hit:
                    rec = self.broker.close_position(symbol, price, exit_reason="ATR_TRAILING_STOP" if protected.stage1_taken else "STOP_LOSS")
                    self.last_entry_bar[symbol] = ts.floor("5min")
                    self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)

            # -------------------------------------------------------------
            # Strategy 1: Stat-Arb Signal Evaluation
            # -------------------------------------------------------------
            for asset_a, asset_b in self.pairs:
                pair_key = (asset_a, asset_b)
                if symbol not in pair_key:
                    continue
                if asset_a in self.latest_prices and asset_b in self.latest_prices:
                    px_a = self.latest_prices[asset_a]
                    px_b = self.latest_prices[asset_b]
                    
                    pos_a = self.broker.open_positions.get(asset_a)
                    pos_b = self.broker.open_positions.get(asset_b)
                    
                    curr_pos = 0
                    if pos_a and pos_b and pos_a.pair_ref == pair_key and pos_b.pair_ref == pair_key:
                        curr_pos = 1 if pos_a.side == "LONG" else -1

                    entry_ts = self.stat_arb.trade_entry_time.get(pair_key)
                    open_bars = max(0.0, (ts - entry_ts).total_seconds() / (375 * 60)) if entry_ts is not None else 0
                    sig = self.stat_arb.evaluate_pair_signal(pair_key, px_a, px_b, curr_pos, open_bars)
                    if curr_pos:
                        hit = any((p.side == "LONG" and px <= p.stop_loss_price) or
                                  (p.side == "SHORT" and px >= p.stop_loss_price)
                                  for p, px in ((pos_a, px_a), (pos_b, px_b)))
                        if hit:
                            sig = replace(sig, action="HARD_STOP_DIVERGENCE", reason="Pair leg stop hit; closing both legs")
                    self.latest_stat_signals[f"{asset_a}_{asset_b}"] = sig

                    self._execute_stat_arb_signal(sig, asset_a, asset_b, px_a, px_b)

            # -------------------------------------------------------------
            # Strategy 2: Volatility-Gated Momentum + RL Meta-Filter
            # -------------------------------------------------------------
            agg = self.data_feed.aggregators.get(symbol)
            if agg:
                df_5m = agg.get_5m_dataframe()
                df_15m = agg.get_15m_dataframe()
                # Use strictly-less-than so the bar starting at ts is excluded
                # (it is still forming). Only fully completed bars are used.
                df_5m = df_5m[df_5m.index < ts]
                df_15m = df_15m[df_15m.index < ts]
                if not df_5m.empty:
                    bar_ts = df_5m.index[-1]
                    new_bar = self.last_evaluated_bar.get(symbol) != bar_ts
                    if new_bar:
                        mom_sig = self.momentum.evaluate_signal(symbol, df_5m, df_15m)
                        self.latest_mom_signals[symbol] = mom_sig

                        # Evaluate RL Meta-Filter Decision
                        rl_dec = self.rl_filter.evaluate_decision(symbol, df_5m, df_15m, proposed_action=mom_sig.action)
                        self.latest_rl_decisions[symbol] = rl_dec
                        self.last_evaluated_bar[symbol] = bar_ts
                    else:
                        mom_sig = self.latest_mom_signals.get(symbol)
                        rl_dec = self.latest_rl_decisions.get(symbol)
                        if mom_sig is None or rl_dec is None:
                            self.execution_status[symbol] = "Waiting for first signal evaluation"
                            return

                    # A completed bar is fresh if it belongs to today and the
                    # tick arrived within 30 minutes of the bar's open time.
                    # This tolerates thin-market / slow-feed gaps without
                    # permanently locking out entries for a whole session.
                    bar_is_today = bar_ts.date() == ts.date()
                    bar_is_fresh = ts - bar_ts <= pd.Timedelta(minutes=30)
                    allow = bar_is_today and bar_is_fresh and self.entries_allowed and symbol in self.momentum_symbols
                    self._execute_momentum_signal(mom_sig, symbol, price, df_5m, rl_dec, allow_entry=allow)
                else:
                    self.execution_status[symbol] = "Waiting for candle history"
                    pos = self.broker.open_positions.get(symbol)
                    if pos and pos.strategy.startswith("MOMENTUM"):
                        if (pos.side == "LONG" and price <= pos.stop_loss_price) or (pos.side == "SHORT" and price >= pos.stop_loss_price):
                            self.broker.close_position(symbol, price, exit_reason="STOP_LOSS")

    def _execute_stat_arb_signal(self, sig: StatArbSignal, asset_a: str, asset_b: str, px_a: float, px_b: float) -> None:
        """Route Stat-Arb orders through Half-Kelly sizer & paper broker."""
        agg_a = self.data_feed.aggregators.get(asset_a)
        df_a = agg_a.get_5m_dataframe() if agg_a else pd.DataFrame()

        if sig.action.startswith("ENTER_"):
            if not self.stat_arb.config.enabled or not self.entries_allowed:
                return
            if abs(sig.z_score) >= self.stat_arb.config.hard_stop_zscore:
                return
            if len(self.broker.open_positions) + 2 > self.risk_mgr.config.max_open_positions:
                return
            equity = self.broker.get_portfolio_summary(self.latest_prices)["total_equity"]
            existing_risk = sum(p.stop_loss_risk for p in self.broker.open_positions.values())
            if existing_risk + equity * self.risk_mgr.config.max_risk_per_trade_pct * 2 > equity * self.risk_mgr.config.max_portfolio_open_risk_pct:
                return
            if any(not self.risk_mgr.can_accept_new_trade(s, self.broker.open_positions)[0] for s in (asset_a, asset_b)):
                return
            times = [self.latest_price_times.get(s) for s in (asset_a, asset_b)]
            if any(t is None for t in times) or abs(times[0] - times[1]) > pd.Timedelta(seconds=30):
                return

        if sig.action.startswith("ENTER_") and (asset_a in self.broker.open_positions or asset_b in self.broker.open_positions):
            return
        if sig.action.startswith("ENTER_") and (sig.beta <= 0 or not pd.notna(sig.beta) or sig.beta == float("inf")):
            return

        if sig.action.startswith("ENTER_"):
            # Size the whole two-leg basket, rather than risk-sizing only leg A.
            size_res = self.risk_mgr.calculate_half_kelly_size(asset_a, px_a, df_a)
            agg_b = self.data_feed.aggregators.get(asset_b)
            df_b = agg_b.get_5m_dataframe() if agg_b else pd.DataFrame()
            atr_b = self.risk_mgr.calculate_atr(df_b)
            stop_a = max(size_res.atr_14 * 1.5, px_a * .01)
            stop_b = max(atr_b * 1.5, px_b * .01)
            ratio = sig.beta * px_a / px_b
            basket_unit_risk = stop_a + px_a * self.broker.friction.slippage_pct + ratio * (stop_b + px_b * self.broker.friction.slippage_pct)
            basket_budget = equity * self.risk_mgr.config.max_risk_per_trade_pct
            qty_a = min(size_res.share_quantity, int(basket_budget / basket_unit_risk),
                        int(equity * self.risk_mgr.config.max_position_size_pct / (ratio * px_b)))
            qty_b = int(qty_a * ratio)
            if qty_a <= 0 or qty_b <= 0:
                return
            model = self.stat_arb.models.get((asset_a, asset_b))
            if not model:
                return
            estimated_payoff = qty_a * px_a * abs(sig.z_score) * model.spread_std
            estimated_friction = sum(self.broker.calculate_roundtrip_friction(p, q)[0] for p, q in ((px_a, qty_a), (px_b, qty_b)))
            if estimated_payoff < self.broker.friction.friction_payoff_mult * estimated_friction:
                return

        if sig.action == "ENTER_LONG_SPREAD" and asset_a not in self.broker.open_positions:
            
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            cost_a = px_a * (1 + self.broker.friction.slippage_pct) * qty_a
            cost_b = px_b * (1 - self.broker.friction.slippage_pct) * qty_b
            if qty_a > 0 and summary["cash"] >= cost_a + cost_b + self.broker._execution_fees(cost_a, 1, "BUY") + self.broker._execution_fees(cost_b, 1, "SELL"):
                self.broker.execute_market_order(asset_a, "BUY", qty_a, px_a, strategy="STAT_ARB", pair_ref=(asset_a, asset_b), atr_14=size_res.atr_14, stop_loss_price=px_a - stop_a)
                self.broker.execute_market_order(asset_b, "SELL", qty_b, px_b, strategy="STAT_ARB", pair_ref=(asset_a, asset_b), atr_14=atr_b, stop_loss_price=px_b + stop_b)
                self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0
                self.stat_arb.trade_entry_time[(asset_a, asset_b)] = max(self.latest_price_times[asset_a], self.latest_price_times[asset_b])

        elif sig.action == "ENTER_SHORT_SPREAD" and asset_a not in self.broker.open_positions:

            summary = self.broker.get_portfolio_summary(self.latest_prices)
            cost_a = px_a * (1 - self.broker.friction.slippage_pct) * qty_a
            cost_b = px_b * (1 + self.broker.friction.slippage_pct) * qty_b
            if qty_a > 0 and summary["cash"] >= cost_a + cost_b + self.broker._execution_fees(cost_a, 1, "SELL") + self.broker._execution_fees(cost_b, 1, "BUY"):
                self.broker.execute_market_order(asset_a, "SELL", qty_a, px_a, strategy="STAT_ARB", pair_ref=(asset_a, asset_b), atr_14=size_res.atr_14, stop_loss_price=px_a + stop_a)
                self.broker.execute_market_order(asset_b, "BUY", qty_b, px_b, strategy="STAT_ARB", pair_ref=(asset_a, asset_b), atr_14=atr_b, stop_loss_price=px_b - stop_b)
                self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0
                self.stat_arb.trade_entry_time[(asset_a, asset_b)] = max(self.latest_price_times[asset_a], self.latest_price_times[asset_b])

        elif sig.action in ["EXIT_TARGET", "HARD_STOP_DIVERGENCE", "TIME_STOP_EXPIRED"]:
            pos_a = self.broker.open_positions.get(asset_a)
            pos_b = self.broker.open_positions.get(asset_b)
            rec_a = self.broker.close_position(asset_a, px_a, exit_reason=sig.action) if pos_a and pos_a.pair_ref == (asset_a, asset_b) else None
            rec_b = self.broker.close_position(asset_b, px_b, exit_reason=sig.action) if pos_b and pos_b.pair_ref == (asset_a, asset_b) else None
            if rec_a:
                self.rl_filter.process_reward_step(asset_a, rec_a.net_pnl, rec_a.total_friction, is_closed=True)
            if rec_b:
                self.rl_filter.process_reward_step(asset_b, rec_b.net_pnl, rec_b.total_friction, is_closed=True)
            self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0

    def _execute_momentum_signal(self, sig: MomentumSignal, symbol: str, price: float, df_5m: pd.DataFrame, rl_dec: RLActionDecision, allow_entry: bool = True) -> None:
        """Route Momentum orders with sector limits, fixed risk sizing, Stage-1 partial TP, and ATR trailing stops."""
        pos = self.broker.open_positions.get(symbol)
        self.execution_status[symbol] = sig.reason
        
        if sig.action in ["BUY", "SELL"] and not pos:
            if not allow_entry:
                self.execution_status[symbol] = "Waiting for a fresh completed candle within entry hours"
                return
            last_entry = self.last_entry_bar.get(symbol)
            if last_entry is not None and df_5m.index[-1] - last_entry < pd.Timedelta(minutes=5 * self.momentum.config.cooldown_bars):
                self.execution_status[symbol] = "Entry cooldown active"
                return
            # 1. RL Filter Floor Check (w_RL >= 0.45)
            if rl_dec.confidence_weight < self.rl_filter.agent.config.confidence_floor:
                self.execution_status[symbol] = f"RL confidence {rl_dec.confidence_weight:.1%} below floor"
                return

            # 2. Portfolio & Sector Constraint Check (Max 3 open positions, Max 1 per sector)
            can_accept, reject_reason = self.risk_mgr.can_accept_new_trade(symbol, self.broker.open_positions)
            if not can_accept:
                self.execution_status[symbol] = reject_reason
                return

            # 3. Dynamic Fixed-Risk Sizing (0.75% max risk per trade)
            size_res = self.risk_mgr.calculate_half_kelly_size(symbol, price, df_5m)
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            
            actual_qty = min(size_res.share_quantity, self.broker.affordable_quantity(price, sig.action))
            stop_dist = max(size_res.atr_14 * self.momentum.config.atr_stop_mult, price * 0.010)
            fill = price * (1 + self.broker.friction.slippage_pct if sig.action == "BUY" else 1 - self.broker.friction.slippage_pct)
            unit_risk = stop_dist + abs(fill - price)
            open_risk = sum(p.stop_loss_risk for p in self.broker.open_positions.values())
            risk_budget = max(0, min(summary["total_equity"] * self.risk_mgr.config.max_risk_per_trade_pct,
                                     summary["total_equity"] * self.risk_mgr.config.max_portfolio_open_risk_pct - open_risk))
            actual_qty = min(actual_qty, int(risk_budget / unit_risk))
            if actual_qty > 0:
                target_dist = max(size_res.atr_14 * self.momentum.config.atr_target_mult, stop_dist * 1.5)
                if not self.broker.passes_friction_payoff_gate(target_dist * actual_qty, price, actual_qty):
                    self.execution_status[symbol] = "Target distance insufficient after estimated costs"
                    return
                stop_price = (price - stop_dist) if sig.action == "BUY" else (price + stop_dist)
                
                opened = self.broker.execute_market_order(
                    symbol=symbol,
                    side=sig.action,
                    quantity=actual_qty,
                    current_market_price=price,
                    strategy="MOMENTUM_RL",
                    atr_14=size_res.atr_14,
                    stop_loss_price=stop_price
                )
                self.execution_status[symbol] = "Position opened" if opened else "Order rejected by paper broker"
                if opened:
                    self.last_entry_bar[symbol] = df_5m.index[-1]
                    self.rl_filter.record_entry(symbol)
            else:
                self.execution_status[symbol] = "Insufficient paper cash for entry"

        elif pos and pos.strategy.startswith("MOMENTUM"):
            # Update position peak/trough prices
            if pos.side == "LONG":
                pos.highest_price = max(pos.highest_price, price)
                gain = price - pos.entry_price
                
                # Stage 1: Partial Take Profit (50% @ 1.5x ATR, min 1.0% gain) -> Move Stop Loss to Break-Even
                stage1_dist = max(pos.atr_14 * self.momentum.config.atr_target_mult, max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010) * 1.5)
                if pos.quantity == 1 and gain >= stage1_dist:
                    rec = self.broker.close_position(symbol, price, exit_reason="TARGET")
                    self.last_entry_bar[symbol] = df_5m.index[-1]
                    self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)
                    return
                if pos.quantity > 1 and not pos.stage1_taken and gain >= stage1_dist:
                    close_qty = min(pos.quantity - 1, max(1, int(pos.quantity * self.momentum.config.partial_tp_pct)))
                    rec = self.broker.close_partial_position(symbol, close_qty, price, exit_reason="STAGE1_TARGET")
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=False)
                    pos.stop_loss_price = max(pos.stop_loss_price, pos.entry_price) # Break-even stop

                # Stage 2: ATR Trailing Stop
                atr_stop_dist = max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010)
                trailing_stop = pos.highest_price - atr_stop_dist
                effective_stop = max(pos.stop_loss_price, trailing_stop) if pos.stage1_taken else pos.stop_loss_price
                pos.stop_loss_price = effective_stop
                pos.stop_loss_risk = max(0, pos.entry_price - effective_stop) * pos.quantity
                
                signal_reversal = (sig.action == "SELL")
                stop_hit = price <= effective_stop

                if stop_hit or signal_reversal:
                    reason = "ATR_TRAILING_STOP" if (stop_hit and pos.stage1_taken) else ("STOP_LOSS" if stop_hit else "SIGNAL_REVERSAL")
                    rec = self.broker.close_position(symbol, price, exit_reason=reason)
                    if rec:
                        self.last_entry_bar[symbol] = df_5m.index[-1]
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)

            elif pos.side == "SHORT":
                pos.lowest_price = min(pos.lowest_price, price)
                gain = pos.entry_price - price

                # Stage 1: Partial Take Profit (50% @ 1.5x ATR, min 1.0% gain) -> Move Stop Loss to Break-Even
                stage1_dist = max(pos.atr_14 * self.momentum.config.atr_target_mult, max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010) * 1.5)
                if pos.quantity == 1 and gain >= stage1_dist:
                    rec = self.broker.close_position(symbol, price, exit_reason="TARGET")
                    self.last_entry_bar[symbol] = df_5m.index[-1]
                    self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)
                    return
                if pos.quantity > 1 and not pos.stage1_taken and gain >= stage1_dist:
                    close_qty = min(pos.quantity - 1, max(1, int(pos.quantity * self.momentum.config.partial_tp_pct)))
                    rec = self.broker.close_partial_position(symbol, close_qty, price, exit_reason="STAGE1_TARGET")
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=False)
                    pos.stop_loss_price = min(pos.stop_loss_price, pos.entry_price) # Break-even stop

                # Stage 2: ATR Trailing Stop
                atr_stop_dist = max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010)
                trailing_stop = pos.lowest_price + atr_stop_dist
                effective_stop = min(pos.stop_loss_price, trailing_stop) if pos.stage1_taken else pos.stop_loss_price
                pos.stop_loss_price = effective_stop
                pos.stop_loss_risk = max(0, effective_stop - pos.entry_price) * pos.quantity

                signal_reversal = (sig.action == "BUY")
                stop_hit = price >= effective_stop

                if stop_hit or signal_reversal:
                    reason = "ATR_TRAILING_STOP" if (stop_hit and pos.stage1_taken) else ("STOP_LOSS" if stop_hit else "SIGNAL_REVERSAL")
                    rec = self.broker.close_position(symbol, price, exit_reason=reason)
                    if rec:
                        self.last_entry_bar[symbol] = df_5m.index[-1]
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)



    def render_terminal_dashboard(self) -> Table:
        with self.lock:
            prices = self.latest_prices.copy()
            stat_signals = self.latest_stat_signals.copy()
            mom_signals = self.latest_mom_signals.copy()
            rl_decisions = self.latest_rl_decisions.copy()
            open_positions = {sym: replace(pos) for sym, pos in self.broker.open_positions.items()}
            summary = self.broker.get_portfolio_summary(prices)
            kill_triggered = self.risk_mgr.kill_switch_triggered
            execution_status = self.execution_status.copy()
            daily_pnl = summary["total_equity"] - self.risk_mgr.starting_daily_capital
            universe_status = self.universe.status
            universe_error = self.universe_error
            snapshot = self.universe.latest_snapshot

        grid = Table.grid(expand=True)
        grid.add_column()
        
        kill_status = "[bold white on red] [ALERT] KILL SWITCH ACTIVE [/bold white on red]" if kill_triggered else "[bold black on green] [OK] SYSTEM ACTIVE [/bold black on green]"
        if self.mode == "sim" and not self.data_feed.is_running:
            kill_status = self.data_feed.replay_status
        pnl_color = "green" if summary["total_pnl"] >= 0 else "red"
        eps_val = self.rl_filter.agent.epsilon
        
        header_pnl = (
            f"[bold]Equity:[/bold] Rs. {summary['total_equity']:,.2f} | "
            f"[bold]Cash:[/bold] Rs. {summary['cash']:,.2f} | "
            f"[bold]Total PnL:[/bold] [{pnl_color}]Rs. {summary['total_pnl']:,.2f} ({summary['total_pnl']/summary['initial_cash']*100:+.2f}%)[/{pnl_color}] | "
            f"[bold]Today PnL:[/bold] Rs. {daily_pnl:,.2f} | "
            f"[bold]RL Epsilon:[/bold] {eps_val:.3f} | Mode: [bold cyan]{self.mode.upper()}[/bold cyan] | Status: {kill_status}"
        )
        rl_status = "enabled" if self.rl_filter.agent.config.enabled else "disabled"
        grid.add_row(Panel(header_pnl, title=f"Quant Desk - Paper execution | RL filter {rl_status}", border_style="cyan"))
        last_tick = self.data_feed.last_tick_time
        feed_info = f"Ticks: {self.data_feed.tick_count} | Last tick: {last_tick or 'Waiting'} | Strategy errors: {self.data_feed.callback_errors} | Open positions: {summary['open_positions_count']} | Closed trades: {summary['trades_count']}"
        if self.mode == "sim":
            feed_info += f"\nReplay: {self.data_feed.replay_completed}/{self.data_feed.replay_total} minute timestamps | {self.data_feed.replay_status}"
        if self.data_feed.last_error:
            feed_info += f" | Last error: {self.data_feed.last_error}"
        grid.add_row(Panel(feed_info, title="Data feed and execution health"))
        universe_info = universe_status
        if universe_error:
            universe_info += f" | New entries blocked: {universe_error}"
        if snapshot and not snapshot.global_ranking:
            universe_info += " | Public intraday coverage is partial; this is not an exchange-wide top 50"
        grid.add_row(Panel(universe_info, title="NSE volume universe"))

        # Stat Arb Table
        stat_table = Table(title="Strategy 1: Stat-Arb Pairs (Johansen & OU Time Stop)", box=box.SIMPLE_HEAD)
        stat_table.add_column("Pair")
        stat_table.add_column("Z-Score")
        stat_table.add_column("Beta")
        stat_table.add_column("Action Signal")
        stat_table.add_column("Reason")
        
        for p_key, sig in stat_signals.items():
            z_color = "red" if abs(sig.z_score) >= 2.0 else "green"
            stat_table.add_row(p_key, f"[{z_color}]{sig.z_score:+.2f}[/{z_color}]", f"{sig.beta:.3f}", sig.action, sig.reason)

        # Momentum + RL Meta-Agent Table
        mom_table = Table(title="Strategy 2: Volatility-Gated Momentum + Adaptive RL Meta-Filter", box=box.SIMPLE_HEAD)
        mom_table.add_column("Ticker")
        mom_table.add_column("LTP (Rs.)")
        mom_table.add_column("ADX 14")
        mom_table.add_column("Vol Spike")
        mom_table.add_column("Momentum Gate")
        mom_table.add_column("RL Decision")
        mom_table.add_column("RL Conf. %")
        mom_table.add_column("Entry / waiting reason")

        for sym, sig in mom_signals.items():
            adx_col = "red" if sig.adx_14 <= 20 else ("green" if sig.adx_14 > 25 else "yellow")
            rl_dec = rl_decisions.get(sym)
            rl_act = rl_dec.action_name if rl_dec else "HOLD"
            rl_conf = f"{rl_dec.confidence_weight*100:.1f}%" if rl_dec else "50.0%"
            conf_col = "green" if (rl_dec and rl_dec.confidence_weight >= 0.5) else "yellow"
            
            mom_table.add_row(
                sym, f"{prices.get(sym, sig.price):.2f}", f"[{adx_col}]{sig.adx_14:.1f}[/{adx_col}]",
                f"{sig.volume_spike_ratio:.1f}x", sig.action, rl_act, f"[{conf_col}]{rl_conf}[/{conf_col}]", execution_status.get(sym, sig.reason)
            )

        grid.add_row(stat_table)
        grid.add_row(mom_table)

        # Positions Table
        pos_table = Table(title="Active Virtual Ledger Positions", box=box.SIMPLE_HEAD)
        pos_table.add_column("Symbol")
        pos_table.add_column("Side")
        pos_table.add_column("Qty")
        pos_table.add_column("Entry (Rs.)")
        pos_table.add_column("LTP (Rs.)")
        pos_table.add_column("Unrealized PnL")

        for sym, pos in open_positions.items():
            ltp = prices.get(sym, pos.entry_price)
            unrealized = (ltp - pos.entry_price) * pos.quantity if pos.side == "LONG" else (pos.entry_price - ltp) * pos.quantity
            col = "green" if unrealized >= 0 else "red"
            pos_table.add_row(sym, pos.side, str(pos.quantity), f"{pos.entry_price:.2f}", f"{ltp:.2f}", f"[{col}]Rs. {unrealized:,.2f}[/{col}]")
        if not open_positions:
            pos_table.add_row("No open positions", "", "", "", "", "Check signal / entry reasons above")

        grid.add_row(pos_table)
        return grid

    def refresh_universe(self) -> None:
        """Refresh selection without discarding the ledger or open-position feeds."""
        selected = self.universe.get_nse_top_volume(top_n=50)
        self.data_feed.add_symbols(selected)
        with self.lock:
            self.momentum_symbols = set(selected)
            self.symbols = self.data_feed.symbols.copy()
            self.universe_error = ""

    def _start_universe_refresh(self) -> None:
        self._universe_stop.clear()
        def loop():
            while not self._universe_stop.wait(300):
                if not self.is_running:
                    break
                try:
                    self.refresh_universe()
                except Exception as exc:
                    with self.lock:
                        self.universe_error = f"{type(exc).__name__}: {exc}"
                    self.console.print(f"NSE universe refresh failed; existing stops remain active: {type(exc).__name__}")
        self.universe_refresh_thread = threading.Thread(target=loop, daemon=True)
        self.universe_refresh_thread.start()

    def start(self, headless: bool = False, stop_event=None) -> None:
        """Start data feed and live terminal dashboard loop."""
        self.initialize_engines()
        if stop_event is not None and stop_event.is_set():
            return
        self.is_running = True
        
        # Start Feed
        if self.mode == "sim":
            self.data_feed.start_simulation_feed(tick_interval_sec=0.2)
        else:
            token = get_access_token()
            if not token:
                self.console.print("[bold red]Error: No FYERS Access Token found. Run with --auth first.[/bold red]")
                sys.exit(1)
            self.console.print("[bold green]Connecting to Live FYERS Data Stream...[/bold green]")
            self.data_feed.start_live_websocket()
            self._start_universe_refresh()

        # Terminal Live Dashboard Loop
        if headless:
            try:
                while self.is_running:
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(0.5)
                    if self.mode == "sim" and not self.data_feed.is_running:
                        break
                    if self.mode != "sim" and datetime.now(pytz.timezone("Asia/Kolkata")).time() >= clock_time(15, 20):
                        with self.lock:
                            self.entries_allowed = False
                            self.broker.square_off_all_positions(self.latest_prices, reason="END_OF_DAY_LAST_QUOTE")
            finally:
                self.is_running = False
                self._universe_stop.set()
                self.data_feed.stop()
            return
        with Live(self.render_terminal_dashboard(), refresh_per_second=2, console=self.console) as live:
            try:
                while self.is_running:
                    time.sleep(0.5)
                    if self.mode != "sim" and datetime.now(pytz.timezone("Asia/Kolkata")).time() >= clock_time(15, 20):
                        with self.lock:
                            self.entries_allowed = False
                            self.broker.square_off_all_positions(self.latest_prices, reason="END_OF_DAY_LAST_QUOTE")
                    live.update(self.render_terminal_dashboard())
                    if self.mode == "sim" and not self.data_feed.is_running:
                        break
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Shutting down Quant Desk Engine...[/bold yellow]")
            finally:
                self.is_running = False
                self._universe_stop.set()
                self.data_feed.stop()
        if self.mode == "sim":
            self.console.print(f"Replay {self.data_feed.replay_status}: {self.data_feed.replay_completed}/{self.data_feed.replay_total} timestamps, {self.data_feed.tick_count} ticks.")


def main():
    parser = argparse.ArgumentParser(description="Institutional Quantitative Trading Framework (FYERS API v3)")
    parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Execution mode: sim (tick playback simulator) or live (FYERS API v3)")
    parser.add_argument("--style", choices=["trend", "breakout", "opening_range", "pullback", "range_reversion", "adaptive_alpha"], default=None, help="Strategy style: e.g. adaptive_alpha, pullback, trend")
    parser.add_argument("--auth", action="store_true", help="Launch FYERS OAuth2 token generator")
    args = parser.parse_args()

    if args.auth:
        auth = FyersAuthenticator()
        print("=== FYERS OAuth2 Token Generator ===")
        url = auth.open_browser_for_auth()
        print("1. Opening FYERS Login URL in browser:")
        print(url)
        raw_input_str = input("\n2. Paste the full redirect URL or the 'auth_code' parameter here: ").strip()
        if raw_input_str:
            try:
                token = auth.generate_access_token(raw_input_str)
                print(f"\n[SUCCESS] Access token generated & saved to .env: {token[:20]}...")
                if auth.validate_session(token):
                    print("[SUCCESS] Live FYERS API Session validated successfully!")
                else:
                    print("[WARNING] Access token saved, but profile check returned non-ok status.")
            except Exception as e:
                print(f"\n[ERROR] Error generating access token: {e}")
        sys.exit(0)

    if args.mode == "live":
        enforce_market_hours()

    try:
        desk = MasterQuantDesk(mode=args.mode, style=args.style)
    except NSEDataError as exc:
        print(f"NSE volume universe unavailable: {exc}")
        sys.exit(1)
    desk.start()


if __name__ == "__main__":
    main()
