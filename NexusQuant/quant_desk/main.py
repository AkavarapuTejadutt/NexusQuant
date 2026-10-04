import os
import sys
import time
import argparse
import threading
import pytz
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box

from quant_desk.core.config import RISK_CONFIG, get_access_token
from quant_desk.core.auth import FyersAuthenticator
from quant_desk.data.universe import UniverseManager, NIFTY_PAIRS, NIFTY_50_SYMBOLS
from quant_desk.data.data_feed import DataFeedManager
from quant_desk.engines.stat_arb import StatArbEngine, StatArbSignal
from quant_desk.engines.momentum import MomentumEngine, MomentumSignal
from quant_desk.engines.rl_agent import RLMetaFilter, RLActionDecision
from quant_desk.risk.portfolio_risk import PortfolioRiskManager
from quant_desk.execution.paper_broker import PaperBroker


def enforce_market_hours() -> None:
    """Strict execution gate for LIVE mode: Halts program outside 09:15 - 15:30 IST."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if not (market_open <= now <= market_close):
        print(f"\n[SYSTEM FATAL] Current Time: {now.strftime('%I:%M %p')} IST.")
        print("[SYSTEM FATAL] Market is CLOSED. Live execution requires active trading hours (09:15 AM - 03:30 PM).")
        print("[SYSTEM NOTE] To test offline or after-hours, run in simulation mode:")
        print("              python quant_desk/main.py --mode sim\n")
        sys.exit(0)


class MasterQuantDesk:
    """Master Orchestrator for the Hybrid Orthogonal Algorithmic Trading Framework + RL Meta-Agent."""

    def __init__(self, mode: str = "sim"):
        self.mode = mode
        self.console = Console()
        
        # 1. Universe & Data Infrastructure
        self.universe = UniverseManager()
        self.symbols = self.universe.get_dynamic_volume_shockers(top_n=30)
        self.pairs = self.universe.get_eligible_pairs(NIFTY_PAIRS)
        self.data_feed = DataFeedManager(self.symbols)
        
        # 2. Strategy & RL Engines
        self.stat_arb = StatArbEngine()
        self.momentum = MomentumEngine()
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
        self.last_pair_holding_bars: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.is_running = False

    def initialize_engines(self) -> None:
        """Calibrate Cointegration pairs and backfill historical candles."""
        self.console.print("[bold cyan][INIT] Initializing Institutional Quant Desk Framework + RL Meta-Agent...[/bold cyan]")
        
        df_daily = self.data_feed.fetch_historical_daily(days=180)
        
        for asset_a, asset_b in self.pairs:
            if asset_a in df_daily.columns and asset_b in df_daily.columns:
                coint_model = self.stat_arb.calibrate_pair(
                    asset_a, asset_b, df_daily[asset_a], df_daily[asset_b]
                )
                status = "COINTEGRATED (p<0.05)" if coint_model.is_cointegrated else "NON-STATIONARY"
                self.console.print(
                    f"  - Pair [yellow]{asset_a}/{asset_b}[/yellow]: {status} | "
                    f"Beta: [green]{coint_model.beta:.4f}[/green] | Half-life: {coint_model.half_life:.1f} days"
                )

        self.data_feed.backfill_intraday_candles(days=3)
        self.data_feed.register_tick_callback(self.on_tick_update)

    def on_tick_update(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """Core event loop handler executed on every tick update."""
        with self.lock:
            self.latest_prices[symbol] = price
            
            # -------------------------------------------------------------
            # Central Risk Gate: Check Daily Kill Switch (-2.0% Stop)
            # -------------------------------------------------------------
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            total_pnl = summary["total_pnl"]
            kill_triggered, kill_reason = self.risk_mgr.evaluate_kill_switch(total_pnl)

            if kill_triggered:
                if self.broker.open_positions:
                    closed = self.broker.square_off_all_positions(self.latest_prices, reason="KILL_SWITCH_-2pct")
                    for rec in closed:
                        self.rl_filter.process_reward_step(rec.symbol, rec.net_pnl, rec.total_friction, is_closed=True)
                return  # Block execution

            # -------------------------------------------------------------
            # Strategy 1: Stat-Arb Signal Evaluation
            # -------------------------------------------------------------
            for asset_a, asset_b in self.pairs:
                pair_key = (asset_a, asset_b)
                if asset_a in self.latest_prices and asset_b in self.latest_prices:
                    px_a = self.latest_prices[asset_a]
                    px_b = self.latest_prices[asset_b]
                    
                    pos_a = self.broker.open_positions.get(asset_a)
                    pos_b = self.broker.open_positions.get(asset_b)
                    
                    curr_pos = 0
                    if pos_a and pos_b:
                        curr_pos = 1 if pos_a.side == "LONG" else -1

                    open_bars = self.last_pair_holding_bars.get(f"{asset_a}_{asset_b}", 0)
                    sig = self.stat_arb.evaluate_pair_signal(pair_key, px_a, px_b, curr_pos, open_bars)
                    self.latest_stat_signals[f"{asset_a}_{asset_b}"] = sig

                    self._execute_stat_arb_signal(sig, asset_a, asset_b, px_a, px_b)

            # -------------------------------------------------------------
            # Strategy 2: Volatility-Gated Momentum + RL Meta-Filter
            # -------------------------------------------------------------
            agg = self.data_feed.aggregators.get(symbol)
            if agg:
                df_5m = agg.get_5m_dataframe()
                df_15m = agg.get_15m_dataframe()
                if len(df_5m) >= 30 and len(df_15m) >= 15:
                    mom_sig = self.momentum.evaluate_signal(symbol, df_5m, df_15m)
                    self.latest_mom_signals[symbol] = mom_sig
                    
                    # Evaluate RL Meta-Filter Decision
                    rl_dec = self.rl_filter.evaluate_decision(symbol, df_5m, df_15m, proposed_action=mom_sig.action)
                    self.latest_rl_decisions[symbol] = rl_dec

                    self._execute_momentum_signal(mom_sig, symbol, price, df_5m, rl_dec)

    def _execute_stat_arb_signal(self, sig: StatArbSignal, asset_a: str, asset_b: str, px_a: float, px_b: float) -> None:
        """Route Stat-Arb orders through Half-Kelly sizer & paper broker."""
        agg_a = self.data_feed.aggregators.get(asset_a)
        df_a = agg_a.get_5m_dataframe() if agg_a else pd.DataFrame()

        if sig.action == "ENTER_LONG_SPREAD" and asset_a not in self.broker.open_positions:
            size_res = self.risk_mgr.calculate_half_kelly_size(asset_a, px_a, df_a)
            qty_a = size_res.share_quantity
            qty_b = max(1, int(qty_a * sig.beta))
            
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            if summary["cash"] >= (qty_a * px_a):
                self.broker.execute_market_order(asset_a, "BUY", qty_a, px_a, strategy="STAT_ARB", pair_ref=(asset_a, asset_b))
                self.broker.execute_market_order(asset_b, "SELL", qty_b, px_b, strategy="STAT_ARB", pair_ref=(asset_a, asset_b))
                self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0

        elif sig.action == "ENTER_SHORT_SPREAD" and asset_a not in self.broker.open_positions:
            size_res = self.risk_mgr.calculate_half_kelly_size(asset_a, px_a, df_a)
            qty_a = size_res.share_quantity
            qty_b = max(1, int(qty_a * sig.beta))

            summary = self.broker.get_portfolio_summary(self.latest_prices)
            if summary["cash"] >= (qty_b * px_b):
                self.broker.execute_market_order(asset_a, "SELL", qty_a, px_a, strategy="STAT_ARB", pair_ref=(asset_a, asset_b))
                self.broker.execute_market_order(asset_b, "BUY", qty_b, px_b, strategy="STAT_ARB", pair_ref=(asset_a, asset_b))
                self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0

        elif sig.action in ["EXIT_TARGET", "HARD_STOP_DIVERGENCE", "TIME_STOP_EXPIRED"]:
            rec_a = self.broker.close_position(asset_a, px_a, exit_reason=sig.action) if asset_a in self.broker.open_positions else None
            rec_b = self.broker.close_position(asset_b, px_b, exit_reason=sig.action) if asset_b in self.broker.open_positions else None
            if rec_a:
                self.rl_filter.process_reward_step(asset_a, rec_a.net_pnl, rec_a.total_friction, is_closed=True)
            if rec_b:
                self.rl_filter.process_reward_step(asset_b, rec_b.net_pnl, rec_b.total_friction, is_closed=True)
            self.last_pair_holding_bars[f"{asset_a}_{asset_b}"] = 0

    def _execute_momentum_signal(self, sig: MomentumSignal, symbol: str, price: float, df_5m: pd.DataFrame, rl_dec: RLActionDecision) -> None:
        """Route Momentum orders with sector limits, fixed risk sizing, Stage-1 partial TP, and ATR trailing stops."""
        pos = self.broker.open_positions.get(symbol)
        
        if sig.action in ["BUY", "SELL"] and not pos:
            # 1. RL Filter Floor Check (w_RL >= 0.45)
            if rl_dec.confidence_weight < self.rl_filter.agent.config.confidence_floor:
                return

            # 2. Portfolio & Sector Constraint Check (Max 3 open positions, Max 1 per sector)
            can_accept, reject_reason = self.risk_mgr.can_accept_new_trade(symbol, self.broker.open_positions)
            if not can_accept:
                return

            # 3. Dynamic Fixed-Risk Sizing (0.75% max risk per trade)
            size_res = self.risk_mgr.calculate_half_kelly_size(symbol, price, df_5m)
            summary = self.broker.get_portfolio_summary(self.latest_prices)
            
            actual_qty = min(size_res.share_quantity, int(summary["cash"] / price))
            if actual_qty > 0:
                stop_dist = size_res.atr_14 * self.momentum.config.atr_stop_mult
                stop_price = (price - stop_dist) if sig.action == "BUY" else (price + stop_dist)
                
                self.broker.execute_market_order(
                    symbol=symbol,
                    side=sig.action,
                    quantity=actual_qty,
                    current_market_price=price,
                    strategy="MOMENTUM_RL",
                    atr_14=size_res.atr_14,
                    stop_loss_price=stop_price
                )

        elif pos and pos.strategy.startswith("MOMENTUM"):
            # Update position peak/trough prices
            if pos.side == "LONG":
                pos.highest_price = max(pos.highest_price, price)
                gain = price - pos.entry_price
                
                # Stage 1: Partial Take Profit (50% @ 1.5x ATR, min 1.0% gain) -> Move Stop Loss to Break-Even
                stage1_dist = max(pos.atr_14 * self.momentum.config.atr_target_mult, pos.entry_price * 0.010)
                if not pos.stage1_taken and gain >= stage1_dist:
                    close_qty = max(1, pos.quantity // 2)
                    rec = self.broker.close_partial_position(symbol, close_qty, price, exit_reason="STAGE1_TP_1.5xATR")
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=False)
                    pos.stop_loss_price = max(pos.stop_loss_price, pos.entry_price) # Break-even stop

                # Stage 2: ATR Trailing Stop
                atr_stop_dist = max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010)
                trailing_stop = pos.highest_price - atr_stop_dist
                effective_stop = max(pos.stop_loss_price, trailing_stop)
                
                signal_reversal = (sig.action == "SELL")
                stop_hit = price <= effective_stop

                if stop_hit or signal_reversal:
                    reason = "ATR_TRAILING_STOP" if (stop_hit and pos.stage1_taken) else ("STOP_LOSS" if stop_hit else "SIGNAL_REVERSAL")
                    rec = self.broker.close_position(symbol, price, exit_reason=reason)
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)

            elif pos.side == "SHORT":
                pos.lowest_price = min(pos.lowest_price, price)
                gain = pos.entry_price - price

                # Stage 1: Partial Take Profit (50% @ 1.5x ATR, min 1.0% gain) -> Move Stop Loss to Break-Even
                stage1_dist = max(pos.atr_14 * self.momentum.config.atr_target_mult, pos.entry_price * 0.010)
                if not pos.stage1_taken and gain >= stage1_dist:
                    close_qty = max(1, pos.quantity // 2)
                    rec = self.broker.close_partial_position(symbol, close_qty, price, exit_reason="STAGE1_TP_1.5xATR")
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=False)
                    pos.stop_loss_price = min(pos.stop_loss_price, pos.entry_price) # Break-even stop

                # Stage 2: ATR Trailing Stop
                atr_stop_dist = max(pos.atr_14 * self.momentum.config.atr_stop_mult, pos.entry_price * 0.010)
                trailing_stop = pos.lowest_price + atr_stop_dist
                effective_stop = min(pos.stop_loss_price, trailing_stop)

                signal_reversal = (sig.action == "BUY")
                stop_hit = price >= effective_stop

                if stop_hit or signal_reversal:
                    reason = "ATR_TRAILING_STOP" if (stop_hit and pos.stage1_taken) else ("STOP_LOSS" if stop_hit else "SIGNAL_REVERSAL")
                    rec = self.broker.close_position(symbol, price, exit_reason=reason)
                    if rec:
                        self.rl_filter.process_reward_step(symbol, rec.net_pnl, rec.total_friction, is_closed=True)



    def render_terminal_dashboard(self) -> Table:
        with self.lock:
            prices = self.latest_prices.copy()
            stat_signals = self.latest_stat_signals.copy()
            mom_signals = self.latest_mom_signals.copy()
            rl_decisions = self.latest_rl_decisions.copy()
            open_positions = self.broker.open_positions.copy()

        summary = self.broker.get_portfolio_summary(prices)
        kill_triggered, kill_reason = self.risk_mgr.evaluate_kill_switch(summary["total_pnl"])

        grid = Table.grid(expand=True)
        grid.add_column()
        
        kill_status = "[bold white on red] [ALERT] KILL SWITCH ACTIVE [/bold white on red]" if kill_triggered else "[bold black on green] [OK] SYSTEM ACTIVE [/bold black on green]"
        pnl_color = "green" if summary["total_pnl"] >= 0 else "red"
        eps_val = self.rl_filter.agent.epsilon
        
        header_pnl = (
            f"[bold]Equity:[/bold] Rs. {summary['total_equity']:,.2f} | "
            f"[bold]Cash:[/bold] Rs. {summary['cash']:,.2f} | "
            f"[bold]Intraday PnL:[/bold] [{pnl_color}]Rs. {summary['total_pnl']:,.2f} ({summary['total_pnl']/summary['initial_cash']*100:+.2f}%)[/{pnl_color}] | "
            f"[bold]RL Epsilon:[/bold] {eps_val:.3f} | Mode: [bold cyan]{self.mode.upper()}[/bold cyan] | Status: {kill_status}"
        )
        grid.add_row(Panel(header_pnl, title="Institutional Quant Desk - Master Control (+ Adaptive RL Meta-Agent)", border_style="cyan"))

        # Stat Arb Table
        stat_table = Table(title="Strategy 1: Stat-Arb Pairs (Johansen & OU Time Stop)", box=box.SIMPLE_HEAD)
        stat_table.add_column("Pair")
        stat_table.add_column("Z-Score")
        stat_table.add_column("Beta")
        stat_table.add_column("Action Signal")
        
        for p_key, sig in stat_signals.items():
            z_color = "red" if abs(sig.z_score) >= 2.0 else "green"
            stat_table.add_row(p_key, f"[{z_color}]{sig.z_score:+.2f}[/{z_color}]", f"{sig.beta:.3f}", sig.action)

        # Momentum + RL Meta-Agent Table
        mom_table = Table(title="Strategy 2: Volatility-Gated Momentum + Adaptive RL Meta-Filter", box=box.SIMPLE_HEAD)
        mom_table.add_column("Ticker")
        mom_table.add_column("LTP (Rs.)")
        mom_table.add_column("ADX 14")
        mom_table.add_column("Vol Spike")
        mom_table.add_column("Momentum Gate")
        mom_table.add_column("RL Decision")
        mom_table.add_column("RL Conf. %")

        for sym, sig in mom_signals.items():
            adx_col = "red" if sig.adx_14 <= 20 else ("green" if sig.adx_14 > 25 else "yellow")
            rl_dec = rl_decisions.get(sym)
            rl_act = rl_dec.action_name if rl_dec else "HOLD"
            rl_conf = f"{rl_dec.confidence_weight*100:.1f}%" if rl_dec else "50.0%"
            conf_col = "green" if (rl_dec and rl_dec.confidence_weight >= 0.5) else "yellow"
            
            mom_table.add_row(
                sym, f"{sig.price:.2f}", f"[{adx_col}]{sig.adx_14:.1f}[/{adx_col}]",
                f"{sig.volume_spike_ratio:.1f}x", sig.action, rl_act, f"[{conf_col}]{rl_conf}[/{conf_col}]"
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

        grid.add_row(pos_table)
        return grid

    def start(self) -> None:
        """Start data feed and live terminal dashboard loop."""
        self.initialize_engines()
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

        # Terminal Live Dashboard Loop
        with Live(self.render_terminal_dashboard(), refresh_per_second=2, console=self.console) as live:
            try:
                while self.is_running:
                    time.sleep(0.5)
                    live.update(self.render_terminal_dashboard())
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Shutting down Quant Desk Engine...[/bold yellow]")
                self.data_feed.stop()


def main():
    parser = argparse.ArgumentParser(description="Institutional Quantitative Trading Framework (FYERS API v3)")
    parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Execution mode: sim (tick playback simulator) or live (FYERS API v3)")
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

    desk = MasterQuantDesk(mode=args.mode)
    desk.start()


if __name__ == "__main__":
    main()