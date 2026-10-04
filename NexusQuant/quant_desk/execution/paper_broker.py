import time
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

    def calculate_roundtrip_friction(self, price: float, quantity: int) -> Tuple[float, float, float, float, float]:
        """Calculate exact Indian market frictions for round-trip trade."""
        turnover = price * quantity * 2.0  # Round-trip buy + sell
        
        # 1. Slippage (0.03% on entry + exit)
        slippage = turnover * self.friction.slippage_pct
        
        # 2. STT (0.025% on intraday sell)
        sell_turnover = price * quantity
        stt = sell_turnover * self.friction.stt_pct
        
        # 3. Exchange Transaction Charges (0.003%)
        exchange_fees = turnover * self.friction.exchange_txn_pct
        
        # 4. Stamp Duty (0.003% on buy)
        stamp_duty = (price * quantity) * self.friction.stamp_duty_pct
        
        # 5. SEBI Fees
        sebi_fees = turnover * self.friction.sebi_fee_pct

        total_friction = slippage + stt + exchange_fees + stamp_duty + sebi_fees
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
        """Execute virtual market order applying 0.03% slippage penalty & Indian taxes."""
        sym = UniverseManager.to_clean_symbol(symbol)
        
        if quantity <= 0:
            return None

        # Calculate round-trip friction and check 3x friction filter
        total_friction, slippage_fee, stt_fee, ex_fee, sebi_fee = self.calculate_roundtrip_friction(current_market_price, quantity)
        
        if expected_payoff is not None:
            if not self.passes_friction_payoff_gate(expected_payoff, current_market_price, quantity):
                print(f"[PaperBroker] REJECTED {sym} setup: Expected payoff Rs. {expected_payoff:.2f} < 3x friction (Rs. {total_friction*3:.2f})")
                return None

        # Apply slippage on entry execution price
        slippage_mult = (1.0 + self.friction.slippage_pct) if side == "BUY" else (1.0 - self.friction.slippage_pct)
        execution_price = current_market_price * slippage_mult

        entry_friction_half = total_friction / 2.0
        required_capital = execution_price * quantity

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
        if not pos or close_qty <= 0 or close_qty >= pos.quantity:
            return None

        slippage_mult = (1.0 - self.friction.slippage_pct) if pos.side == "LONG" else (1.0 + self.friction.slippage_pct)
        exit_price = current_market_price * slippage_mult

        total_friction, slippage, stt, ex_fees, sebi = self.calculate_roundtrip_friction(exit_price, close_qty)
        exit_friction_half = total_friction / 2.0

        if pos.side == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * close_qty
        else:
            gross_pnl = (pos.entry_price - exit_price) * close_qty

        net_pnl = gross_pnl - total_friction
        self.cash += (pos.entry_price * close_qty + gross_pnl - exit_friction_half)

        # Update position remaining quantity
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
        if not pos:
            return None

        # Apply slippage on exit price
        # Closing LONG -> SELL execution price (lower)
        # Closing SHORT -> BUY execution price (higher)
        slippage_mult = (1.0 - self.friction.slippage_pct) if pos.side == "LONG" else (1.0 + self.friction.slippage_pct)
        exit_price = current_market_price * slippage_mult

        total_friction, slippage, stt, ex_fees, sebi = self.calculate_roundtrip_friction(exit_price, pos.quantity)
        exit_friction_half = total_friction / 2.0

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
        """Return total portfolio equity, cash, unrealized PnL, and realized PnL."""
        unrealized_pnl = 0.0
        for sym, pos in self.open_positions.items():
            px = current_prices.get(sym, pos.entry_price)
            if pos.side == "LONG":
                unrealized_pnl += (px - pos.entry_price) * pos.quantity
            else:
                unrealized_pnl += (pos.entry_price - px) * pos.quantity

        realized_pnl = sum(r.net_pnl for r in self.trade_history)
        total_equity = self.cash + sum(pos.entry_price * pos.quantity for pos in self.open_positions.values()) + unrealized_pnl

        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "total_equity": total_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_equity - self.initial_cash,
            "open_positions_count": len(self.open_positions),
            "trades_count": len(self.trade_history)
        }
