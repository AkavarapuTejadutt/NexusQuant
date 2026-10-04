import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime, time
from quant_desk.core.config import RISK_CONFIG, RiskConfig


@dataclass
class PositionSizingResult:
    symbol: str
    target_capital: float
    kelly_fraction: float
    share_quantity: int
    atr_14: float
    reason: str


class PortfolioRiskManager:
    """Central Risk Engine: Half-Kelly position sizing, ATR volatility parity, & Daily -2% Kill Switch."""

    def __init__(self, initial_capital: float = RISK_CONFIG.initial_capital, config: RiskConfig = RISK_CONFIG):
        self.config = config
        self.starting_daily_capital = initial_capital
        self.current_capital = initial_capital
        self.kill_switch_triggered = False
        self.kill_switch_timestamp: Optional[datetime] = None
        self.lock_until: Optional[datetime] = None

    def reset_daily_session(self, current_capital: float) -> None:
        """Reset session baseline at start of trading day (09:00 AM)."""
        self.starting_daily_capital = current_capital
        self.current_capital = current_capital
        self.kill_switch_triggered = False
        self.kill_switch_timestamp = None
        self.lock_until = None
        print(f"[RiskEngine] New Session Initialized. Baseline Capital: Rs. {current_capital:,.2f}")

    def evaluate_kill_switch(self, total_pnl: float, open_positions_pnl: float = 0.0) -> Tuple[bool, str]:
        """Evaluate if cumulative intraday PnL violates the -2.0% Daily Kill-Switch circuit breaker."""
        now = datetime.now()
        
        # Check if already locked from previous trigger
        if self.kill_switch_triggered:
            if self.lock_until and now < self.lock_until:
                return True, f"SYSTEMIC KILL SWITCH ACTIVE. Locked until {self.lock_until.strftime('%Y-%m-%d %H:%M:%S')}."
            elif self.lock_until and now >= self.lock_until:
                # Expired lock
                self.kill_switch_triggered = False

        net_pnl = total_pnl + open_positions_pnl
        pnl_pct = net_pnl / self.starting_daily_capital if self.starting_daily_capital > 0 else 0.0

        if pnl_pct <= self.config.daily_kill_switch_pct:
            self.kill_switch_triggered = True
            self.kill_switch_timestamp = now
            # Lock until next day 09:00 AM
            next_day = now + pd.Timedelta(days=1)
            self.lock_until = datetime.combine(next_day.date(), time(9, 0, 0))
            
            reason = (
                f"[ALERT] SYSTEMIC KILL SWITCH TRIGGERED! "
                f"Intraday PnL: Rs. {net_pnl:,.2f} ({pnl_pct*100:.2f}%) <= Limit ({self.config.daily_kill_switch_pct*100:.1f}%). "
                f"Liquidating all positions & halting engine until 09:00 AM tomorrow."
            )
            return True, reason

        return False, f"Risk OK. PnL: Rs. {net_pnl:,.2f} ({pnl_pct*100:.2f}%)"

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate 14-period Average True Range (ATR)."""
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1]) if not df.empty else 1.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_series = tr.rolling(window=period).mean()
        atr_val = float(atr_series.iloc[-1])
        return atr_val if not np.isnan(atr_val) and atr_val > 0 else 1.0

    def can_accept_new_trade(
        self,
        symbol: str,
        open_positions: Dict,
        portfolio_equity: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Check portfolio constraints: Max open positions (3), Max sector positions (1), and Max open risk (1.5%)."""
        from quant_desk.data.universe import UniverseManager

        capital = portfolio_equity or self.current_capital
        
        # 1. Total Open Positions Cap
        if len(open_positions) >= self.config.max_open_positions:
            return False, f"Portfolio Cap Reached: Max {self.config.max_open_positions} concurrent positions allowed."

        # 2. Sector Exposure Cap
        target_sector = UniverseManager.get_sector(symbol)
        sector_count = sum(
            1 for sym in open_positions.keys()
            if UniverseManager.get_sector(sym) == target_sector
        )
        if sector_count >= self.config.max_positions_per_sector:
            return False, f"Sector Exposure Cap Reached: Sector '{target_sector}' already has {sector_count} open trade(s)."

        # 3. Total Open Portfolio Risk Cap
        total_open_risk = sum(
            pos.stop_loss_risk if hasattr(pos, 'stop_loss_risk') else (pos.entry_price * pos.quantity * 0.015)
            for pos in open_positions.values()
        )
        max_allowed_risk = capital * self.config.max_portfolio_open_risk_pct
        if total_open_risk >= max_allowed_risk:
            return False, f"Portfolio Open Risk Cap Reached: Total Risk (Rs. {total_open_risk:,.2f}) >= Max (Rs. {max_allowed_risk:,.2f})."

        return True, "Risk constraints satisfied."

    def calculate_half_kelly_size(
        self,
        symbol: str,
        price: float,
        df_candles: pd.DataFrame,
        expected_return: float = 0.12,  # 12% annualized expected return
        variance: float = 0.04,          # 20% annual volatility -> 0.04 variance
        portfolio_equity: Optional[float] = None
    ) -> PositionSizingResult:
        """Calculate dynamic share quantity using Fixed Risk per Trade (0.75% max risk) & ATR volatility parity."""
        capital = portfolio_equity or self.current_capital
        
        # 1. Calculate ATR_14 & Stop Loss Distance (1.5 x ATR or min 1.0% price cushion)
        atr_14 = self.calculate_atr(df_candles, period=14)
        stop_distance = max(atr_14 * 1.5, price * 0.010) # At least 1.0% price cushion


        # 2. Max Dollar Risk per Trade (0.75% of Equity)
        max_dollar_risk = capital * self.config.max_risk_per_trade_pct

        # 3. Dynamic Share Quantity derived from Risk Limit
        risk_based_qty = int(max_dollar_risk / stop_distance) if stop_distance > 0 else 0

        # 4. Cap Position Size at max allowed fraction of capital (e.g. 15%)
        max_position_capital = capital * self.config.max_position_size_pct
        max_capital_qty = int(max_position_capital / price) if price > 0 else 0

        final_quantity = max(1, min(risk_based_qty, max_capital_qty))
        target_capital = final_quantity * price

        reason = (
            f"Risk Sizer ({self.config.max_risk_per_trade_pct*100:.2f}% risk = Rs. {max_dollar_risk:,.2f}) "
            f"allocated {final_quantity} shares @ Rs. {price:.2f} (ATR_14=Rs. {atr_14:.2f}, StopDistance=Rs. {stop_distance:.2f})"
        )

        return PositionSizingResult(
            symbol=symbol,
            target_capital=target_capital,
            kelly_fraction=self.config.max_risk_per_trade_pct,
            share_quantity=final_quantity,
            atr_14=atr_14,
            reason=reason
        )

