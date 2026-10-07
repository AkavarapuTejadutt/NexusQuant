import time
import math
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from quant_desk.core.config import FRICTION_CONFIG, FrictionConfig, RISK_CONFIG
from quant_desk.data.universe import UniverseManager


@dataclass
class PaperPosition:
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_friction: float
    strategy: str  # "STAT_ARB" or "MOMENTUM"
    pair_ref: Optional[Tuple[str, str]] = None
    atr_14: float = 1.0
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    stop_loss_price: float = 0.0
    stop_loss_risk: float = 0.0
    stage1_taken: bool = False


@dataclass
class TradeExecutionRecord:
    trade_id: str
    symbol: str
    side: str
    quantity: int
    fill_price: float
    gross_pnl: float
    net_pnl: float
    total_friction: float
    slippage: float
    stt: float
    exchange_fees: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    exit_reason: str


class PaperBroker:
    """Virtual Order Ledger with 0.03% Slippage, Indian STT/tax calculator, & 3x Friction Filter."""

    def __init__(self, initial_cash: float = RISK_CONFIG.initial_capital, friction_cfg: FrictionConfig = FRICTION_CONFIG):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.friction = friction_cfg
        
        self.open_positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[TradeExecutionRecord] = []
        self.trade_counter = 0

    def get_dynamic_slippage_pct(self, quantity: int, adv: float = 100000.0) -> float:
        """Almgren-Chriss Square Root Law Slippage Model: Base 0.03% + 0.05 * sqrt(Order_Qty / ADV), strictly capped at 0.2% (0.002)."""
        ratio = max(0.0, float(quantity) / max(adv, 1000.0))
        impact = 0.0003 + 0.05 * math.sqrt(ratio)
        return float(min(0.002, max(0.0003, impact)))

    def _execution_fees(self, price: float, quantity: int, side: str) -> float:
        """Cash fees including GST on Exchange & SEBI charges; slippage is reflected in fill price."""
        turnover = price * quantity
        ex_sebi_charges = turnover * (self.friction.exchange_txn_pct + self.friction.sebi_fee_pct)
        gst = ex_sebi_charges * self.friction.gst_pct
        stt_or_stamp = (turnover * self.friction.stt_pct) if side == "SELL" else (turnover * self.friction.stamp_duty_pct)
        return float(ex_sebi_charges + gst + stt_or_stamp)

    def affordable_quantity(self, price: float, side: str) -> int:
        base_slip = self.get_dynamic_slippage_pct(1)
        fill = price * (1 + base_slip if side == "BUY" else 1 - base_slip)
        cost = fill + self._execution_fees(fill, 1, side)
        return max(0, int(self.cash / cost)) if cost > 0 else 0

    def calculate_roundtrip_friction(self, price: float, quantity: int) -> Tuple[float, float, float, float, float]:
        """Calculate exact Indian market frictions (STT, Exchange, SEBI, GST, Dynamic Slippage) for round-trip trade."""
        turnover = price * quantity * 2.0  # Round-trip buy + sell
        slip_pct = self.get_dynamic_slippage_pct(quantity)
        
        # 1. Dynamic Slippage
        slippage = turnover * slip_pct
        
        # 2. STT (0.025% on intraday sell)
        sell_turnover = price * quantity
        stt = sell_turnover * self.friction.stt_pct
        
        # 3. Exchange Transaction Charges (0.00297%)
        exchange_fees = turnover * self.friction.exchange_txn_pct
        
        # 4. Stamp Duty (0.003% on buy)
        stamp_duty = (price * quantity) * self.friction.stamp_duty_pct
        
        # 5. SEBI Fees
        sebi_fees = turnover * self.friction.sebi_fee_pct

        # 6. GST (18% on broker/exchange/sebi fees)
        gst_fees = (exchange_fees + sebi_fees) * self.friction.gst_pct

        total_friction = slippage + stt + exchange_fees + stamp_duty + sebi_fees + gst_fees
        return total_friction, slippage, stt, exchange_fees, sebi_fees

    def passes_friction_payoff_gate(self, expected_payoff: float, price: float, quantity: int) -> bool:
        """Check if expected trade payoff exceeds 3x total friction costs."""
        total_friction, _, _, _, _ = self.calculate_roundtrip_friction(price, quantity)
        min_required_payoff = self.friction.friction_payoff_mult * total_friction
        return expected_payoff >= min_required_payoff

    def execute_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        current_market_price: float,
        strategy: str = "GENERIC",
        pair_ref: Optional[Tuple[str, str]] = None,
        expected_payoff: Optional[float] = None,
        atr_14: float = 1.0,
        stop_loss_price: float = 0.0
    ) -> Optional[PaperPosition]:
        """Execute virtual market order applying dynamic slippage penalty & full Indian tax structure."""
        sym = UniverseManager.to_clean_symbol(symbol)
        
        if (side not in ("BUY", "SELL") or not math.isfinite(quantity) or quantity <= 0
                or int(quantity) != quantity or sym in self.open_positions
                or not math.isfinite(current_market_price) or current_market_price <= 0
                or not math.isfinite(atr_14) or atr_14 <= 0
                or not math.isfinite(stop_loss_price) or stop_loss_price < 0):
            return None

        # Calculate round-trip friction and check 3x friction filter
        total_friction, slippage_fee, stt_fee, ex_fee, sebi_fee = self.calculate_roundtrip_friction(current_market_price, quantity)
        
        if expected_payoff is not None:
            if not self.passes_friction_payoff_gate(expected_payoff, current_market_price, quantity):
                print(f"[PaperBroker] REJECTED {sym} setup: Expected payoff Rs. {expected_payoff:.2f} < 3x friction (Rs. {total_friction*3:.2f})")
                return None

        # Apply dynamic slippage on entry execution price
        slip_pct = self.get_dynamic_slippage_pct(quantity)
        slippage_mult = (1.0 + slip_pct) if side == "BUY" else (1.0 - slip_pct)
        execution_price = current_market_price * slippage_mult

        entry_friction_half = self._execution_fees(execution_price, quantity, side)
        required_capital = execution_price * quantity
        if required_capital + entry_friction_half > self.cash:
            return None

        # Calculate dollar risk
        stop_dist = abs(execution_price - stop_loss_price) if stop_loss_price > 0 else (atr_14 * 1.5)
        stop_risk = stop_dist * quantity

        if side == "BUY":
            pos = PaperPosition(
                symbol=sym,
                side="LONG",
                quantity=quantity,
                entry_price=execution_price,
                entry_time=datetime.now(),
                entry_friction=entry_friction_half,
                strategy=strategy,
                pair_ref=pair_ref,
                atr_14=atr_14,
                highest_price=execution_price,
                lowest_price=execution_price,
                stop_loss_price=stop_loss_price or (execution_price - stop_dist),
                stop_loss_risk=stop_risk
            )
            self.cash -= (required_capital + entry_friction_half)
            self.open_positions[sym] = pos
            print(f"[PaperBroker] ENTRY LONG: {quantity} {sym} @ Rs. {execution_price:.2f} (Slippage: Rs. {slippage_fee/2:.2f}, Risk: Rs. {stop_risk:.2f})")
            return pos

        elif side == "SELL":
            pos = PaperPosition(
                symbol=sym,
                side="SHORT",
                quantity=quantity,
                entry_price=execution_price,
                entry_time=datetime.now(),
                entry_friction=entry_friction_half,
                strategy=strategy,
                pair_ref=pair_ref,
                atr_14=atr_14,
                highest_price=execution_price,
                lowest_price=execution_price,
                stop_loss_price=stop_loss_price or (execution_price + stop_dist),
                stop_loss_risk=stop_risk
            )
            self.cash -= (required_capital + entry_friction_half)  # Reserve required capital
            self.open_positions[sym] = pos
            print(f"[PaperBroker] ENTRY SHORT: {quantity} {sym} @ Rs. {execution_price:.2f} (Slippage: Rs. {slippage_fee/2:.2f}, Risk: Rs. {stop_risk:.2f})")
            return pos

        return None

    def close_partial_position(
        self,
        symbol: str,
        close_qty: int,
        current_market_price: float,
        exit_reason: str = "PARTIAL_TP"
    ) -> Optional[TradeExecutionRecord]:
        """Close a fraction of an open position for partial profit booking."""
        sym = UniverseManager.to_clean_symbol(symbol)
        pos = self.open_positions.get(sym)
        if (not pos or not math.isfinite(close_qty) or close_qty <= 0 or close_qty >= pos.quantity
                or int(close_qty) != close_qty or not math.isfinite(current_market_price) or current_market_price <= 0):
            return None

        slip_pct = self.get_dynamic_slippage_pct(close_qty)
        slippage_mult = (1.0 - slip_pct) if pos.side == "LONG" else (1.0 + slip_pct)
        exit_price = current_market_price * slippage_mult

        entry_fees = pos.entry_friction * close_qty / pos.quantity
        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        exit_friction_half = self._execution_fees(exit_price, close_qty, exit_side)
        total_friction = entry_fees + exit_friction_half
        entry_market = pos.entry_price / (1 + slip_pct if pos.side == "LONG" else 1 - slip_pct)
        slippage = (abs(pos.entry_price - entry_market) + abs(exit_price - current_market_price)) * close_qty
        stt = (exit_price if pos.side == "LONG" else pos.entry_price) * close_qty * self.friction.stt_pct
        ex_fees = (pos.entry_price + exit_price) * close_qty * self.friction.exchange_txn_pct

        if pos.side == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * close_qty
        else:
            gross_pnl = (pos.entry_price - exit_price) * close_qty

        net_pnl = gross_pnl - total_friction
        self.cash += (pos.entry_price * close_qty + gross_pnl - exit_friction_half)

        # Update position remaining quantity
        pos.entry_friction -= entry_fees
        pos.stop_loss_risk *= (pos.quantity - close_qty) / pos.quantity
        pos.quantity -= close_qty
        pos.stage1_taken = True

        self.trade_counter += 1
        record = TradeExecutionRecord(
            trade_id=f"TRD_{self.trade_counter:04d}",
            symbol=sym,
            side=pos.side,
            quantity=close_qty,
            fill_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_friction=total_friction,
            slippage=slippage,
            stt=stt,
            exchange_fees=ex_fees,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            strategy=pos.strategy,
            exit_reason=exit_reason
        )

        self.trade_history.append(record)
        pnl_str = f"+Rs. {net_pnl:,.2f}" if net_pnl >= 0 else f"-Rs. {abs(net_pnl):,.2f}"
        print(f"[PaperBroker] PARTIAL CLOSE {close_qty} {pos.side} {sym}: Net PnL {pnl_str} (Reason: {exit_reason}, Left: {pos.quantity})")
        return record

    def close_position(self, symbol: str, current_market_price: float, exit_reason: str = "SIGNAL") -> Optional[TradeExecutionRecord]:
        """Close an open position, deducting exit slippage/taxes, updating ledger cash and PnL."""
        sym = UniverseManager.to_clean_symbol(symbol)
        pos = self.open_positions.get(sym)
        if not pos or not math.isfinite(current_market_price) or current_market_price <= 0:
            return None

        # Apply dynamic slippage on exit price
        slip_pct = self.get_dynamic_slippage_pct(pos.quantity)
        slippage_mult = (1.0 - slip_pct) if pos.side == "LONG" else (1.0 + slip_pct)
        exit_price = current_market_price * slippage_mult

        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        exit_friction_half = self._execution_fees(exit_price, pos.quantity, exit_side)
        total_friction = pos.entry_friction + exit_friction_half
        entry_market = pos.entry_price / (1 + self.friction.slippage_pct if pos.side == "LONG" else 1 - self.friction.slippage_pct)
        slippage = (abs(pos.entry_price - entry_market) + abs(exit_price - current_market_price)) * pos.quantity
        stt = (exit_price if pos.side == "LONG" else pos.entry_price) * pos.quantity * self.friction.stt_pct
        ex_fees = (pos.entry_price + exit_price) * pos.quantity * self.friction.exchange_txn_pct

        if pos.side == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:  # SHORT
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_friction
        self.cash += (pos.entry_price * pos.quantity + gross_pnl - exit_friction_half)

        self.trade_counter += 1
        record = TradeExecutionRecord(
            trade_id=f"TRD_{self.trade_counter:04d}",
            symbol=sym,
            side=pos.side,
            quantity=pos.quantity,
            fill_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_friction=total_friction,
            slippage=slippage,
            stt=stt,
            exchange_fees=ex_fees,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            strategy=pos.strategy,
            exit_reason=exit_reason
        )

        del self.open_positions[sym]
        self.trade_history.append(record)
        
        pnl_str = f"+Rs. {net_pnl:,.2f}" if net_pnl >= 0 else f"-Rs. {abs(net_pnl):,.2f}"
        print(f"[PaperBroker] CLOSED {pos.side} {sym}: Net PnL {pnl_str} (Reason: {exit_reason})")
        return record

    def square_off_all_positions(self, current_prices: Dict[str, float], reason: str = "KILL_SWITCH") -> List[TradeExecutionRecord]:
        """Liquidate all open positions immediately at market."""
        closed_records = []
        open_syms = list(self.open_positions.keys())
        for sym in open_syms:
            px = current_prices.get(sym, self.open_positions[sym].entry_price)
            rec = self.close_position(sym, px, exit_reason=reason)
            if rec:
                closed_records.append(rec)
        return closed_records

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Return total portfolio equity, cash, Bid-Ask MTM unrealized PnL, and realized PnL."""
        current_prices = {UniverseManager.to_clean_symbol(s): p for s, p in current_prices.items()}
        unrealized_pnl = 0.0
        for sym, pos in self.open_positions.items():
            px = current_prices.get(sym, pos.entry_price)
            slip_pct = self.get_dynamic_slippage_pct(pos.quantity)
            bid_px = px * (1.0 - slip_pct)
            ask_px = px * (1.0 + slip_pct)
            if pos.side == "LONG":
                unrealized_pnl += (bid_px - pos.entry_price) * pos.quantity
            else:
                unrealized_pnl += (pos.entry_price - ask_px) * pos.quantity

        realized_pnl = sum(r.net_pnl for r in self.trade_history)
        total_equity = self.cash + sum(pos.entry_price * pos.quantity for pos in self.open_positions.values()) + unrealized_pnl

        open_entry_fees = sum(pos.entry_friction for pos in self.open_positions.values())
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "total_equity": total_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl - open_entry_fees,
            "total_pnl": total_equity - self.initial_cash,
            "open_positions_count": len(self.open_positions),
            "trades_count": len(self.trade_history)
        }
