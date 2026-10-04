import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
from quant_desk.core.config import MOMENTUM_CONFIG, MomentumConfig
from quant_desk.data.universe import UniverseManager


@dataclass
class MomentumSignal:
    symbol: str
    action: str  # "BUY", "SELL", "MUTE_CHOP", "HOLD"
    price: float
    ema_fast: float
    ema_slow: float
    adx_14: float
    volume_spike_ratio: float
    vwap: float
    reason: str


class MomentumEngine:
    """Volatility-Gated Momentum Engine (Upgraded EMA 6/30 + ADX 14 + Volume Spike + Daily VWAP)."""

    def __init__(self, config: MomentumConfig = MOMENTUM_CONFIG):
        self.config = config

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) 14 without external library dependencies."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed TR and DM
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)

        # Directional Index (DX) & ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)
        return adx

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        if "vwap" in df.columns:
            return df["vwap"]
        cum_vol = df["volume"].cumsum()
        cum_pv = (df["close"] * df["volume"]).cumsum()
        return cum_pv / cum_vol.replace(0, np.nan)

    def evaluate_signal(self, symbol: str, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> MomentumSignal:
        """Evaluate Volatility-Gated Momentum Signal on 5-min and 15-min candle buffers."""
        sym = UniverseManager.to_clean_symbol(symbol)
        
        if len(df_5m) < self.config.ema_slow + 5 or len(df_15m) < self.config.adx_period + 5:
            return MomentumSignal(
                symbol=sym, action="HOLD", price=0.0, ema_fast=0.0,
                ema_slow=0.0, adx_14=0.0, volume_spike_ratio=0.0, vwap=0.0,
                reason="Insufficient candle data to calculate indicators."
            )

        # 1. Calculate Indicators on 5m
        df_5m_calc = df_5m.copy()
        df_5m_calc["ema_fast"] = self.calculate_ema(df_5m_calc["close"], self.config.ema_fast)
        df_5m_calc["ema_slow"] = self.calculate_ema(df_5m_calc["close"], self.config.ema_slow)
        df_5m_calc["vol_sma"] = df_5m_calc["volume"].rolling(window=self.config.volume_sma_period).mean()
        df_5m_calc["vwap"] = self.calculate_vwap(df_5m_calc)

        latest_5m = df_5m_calc.iloc[-1]
        prev_5m = df_5m_calc.iloc[-2]

        current_price = float(latest_5m["close"])
        ema_fast_curr = float(latest_5m["ema_fast"])
        ema_slow_curr = float(latest_5m["ema_slow"])
        ema_fast_prev = float(prev_5m["ema_fast"])
        ema_slow_prev = float(prev_5m["ema_slow"])
        
        vol_curr = float(latest_5m["volume"])
        vol_sma = float(latest_5m["vol_sma"]) if not np.isnan(latest_5m["vol_sma"]) else 1.0
        volume_spike = vol_curr / vol_sma if vol_sma > 0 else 1.0
        vwap_curr = float(latest_5m["vwap"]) if not np.isnan(latest_5m["vwap"]) else current_price

        # 2. Calculate ADX 14 on 15m Chart
        df_15m_calc = df_15m.copy()
        df_15m_calc["adx_14"] = self.calculate_adx(df_15m_calc, period=self.config.adx_period)
        adx_curr = float(df_15m_calc["adx_14"].iloc[-1])

        # -------------------------------------------------------------
        # Defense 1: The Mute Button (ADX <= 20)
        # -------------------------------------------------------------
        if adx_curr <= self.config.adx_mute:
            return MomentumSignal(
                symbol=sym,
                action="MUTE_CHOP",
                price=current_price,
                ema_fast=ema_fast_curr,
                ema_slow=ema_slow_curr,
                adx_14=adx_curr,
                volume_spike_ratio=volume_spike,
                vwap=vwap_curr,
                reason=f"Mute Gate Active: 15m ADX ({adx_curr:.1f}) <= {self.config.adx_mute}. Range-bound chop protection."
            )

        # Crossover Detection OR Expanding Momentum Trend
        ema_diff_curr = ema_fast_curr - ema_slow_curr
        ema_diff_prev = ema_fast_prev - ema_slow_prev

        bullish_signal = (
            (ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr) or
            (ema_fast_curr > ema_slow_curr and ema_diff_curr > ema_diff_prev)
        )

        bearish_signal = (
            (ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr) or
            (ema_fast_curr < ema_slow_curr and ema_diff_curr < ema_diff_prev)
        )

        # Gates Verification
        regime_gate_passed = adx_curr > self.config.adx_trigger
        liquidity_gate_passed = volume_spike >= self.config.volume_spike_mult

        # -------------------------------------------------------------
        # Long Execution Trigger
        # -------------------------------------------------------------
        if bullish_signal and regime_gate_passed and liquidity_gate_passed and (current_price > vwap_curr):
            return MomentumSignal(
                symbol=sym,
                action="BUY",
                price=current_price,
                ema_fast=ema_fast_curr,
                ema_slow=ema_slow_curr,
                adx_14=adx_curr,
                volume_spike_ratio=volume_spike,
                vwap=vwap_curr,
                reason=f"LONG TRIGGER: EMA 6/30 Trend AND ADX={adx_curr:.1f}>{self.config.adx_trigger} AND Vol={volume_spike:.1f}x AND Price>VWAP."
            )

        # -------------------------------------------------------------
        # Short Execution Trigger
        # -------------------------------------------------------------
        if bearish_signal and regime_gate_passed and liquidity_gate_passed and (current_price < vwap_curr):
            return MomentumSignal(
                symbol=sym,
                action="SELL",
                price=current_price,
                ema_fast=ema_fast_curr,
                ema_slow=ema_slow_curr,
                adx_14=adx_curr,
                volume_spike_ratio=volume_spike,
                vwap=vwap_curr,
                reason=f"SHORT TRIGGER: EMA 6/30 Trend AND ADX={adx_curr:.1f}>{self.config.adx_trigger} AND Vol={volume_spike:.1f}x AND Price<VWAP."
            )

        return MomentumSignal(
            symbol=sym,
            action="HOLD",
            price=current_price,
            ema_fast=ema_fast_curr,
            ema_slow=ema_slow_curr,
            adx_14=adx_curr,
            volume_spike_ratio=volume_spike,
            vwap=vwap_curr,
            reason="No crossover or conditions not fully satisfied."
        )
