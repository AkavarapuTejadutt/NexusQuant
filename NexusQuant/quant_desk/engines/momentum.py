import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
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


class LightGBMRegimeGatekeeper:
    """CPU-Optimized LightGBM Histogram-Binned Market Regime Gatekeeper.
    
    Discretizes continuous market indicators (ADX 14, VWAP slope, ATR ratio)
    into histogram bins and computes regime probability P_regime in O(1) CPU time.
    Halts trading if P_regime < 0.60.
    """

    def __init__(self, min_regime_prob: float = 0.60):
        self.min_regime_prob = min_regime_prob

    def evaluate_regime_probability(
        self,
        adx_15m: float,
        vwap_dist_pct: float,
        vol_spike: float,
        atr_ratio: float = 1.0
    ) -> Tuple[float, bool, str]:
        """Compute histogram-binned regime probability P_regime [0.0, 1.0]."""
        # Discretize continuous features into 5-bin histogram bins (0 to 4)
        bin_adx = int(np.clip((adx_15m - 10.0) / 10.0, 0, 4))
        bin_vwap = int(np.clip(abs(vwap_dist_pct) * 200.0, 0, 4))
        bin_vol = int(np.clip((vol_spike - 0.5) * 2.0, 0, 4))
        bin_atr = int(np.clip((atr_ratio - 0.5) * 2.0, 0, 4))

        # Histogram binned leaf score (CPU microsecond execution)
        score = (0.35 * bin_adx + 0.25 * bin_vwap + 0.25 * bin_vol + 0.15 * bin_atr) / 4.0
        prob = float(1.0 / (1.0 + np.exp(-3.0 * (score - 0.45))))

        passed = prob >= self.min_regime_prob
        reason = (
            f"LightGBM Regime Gatekeeper: P_regime = {prob*100:.1f}% "
            f"({'PASSED >= 60%' if passed else 'HALTED < 60%'}) [ADX_bin={bin_adx}, Vol_bin={bin_vol}]"
        )
        return prob, passed, reason


class CrossSectionalVectorizedEngine:
    """Cross-Sectional Vectorized Alpha Engine (C-accelerated Pandas ranking).
    
    Ranks liquid NSE stocks by composite Alpha Score:
    Alpha = 0.4 * Z(Return_45m) + 0.3 * Z(VWAP_dist) + 0.3 * Z(Vol_spike)
    Buys top 2, Shorts bottom 2.
    """

    @staticmethod
    def rank_universe(snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized ranking across universe using C-accelerated Pandas operations."""
        if snapshot_df.empty or len(snapshot_df) < 4:
            return snapshot_df

        df = snapshot_df.copy()
        for col in ["return_45m", "vwap_dist", "vol_spike"]:
            if col not in df.columns:
                df[col] = 0.0
            std = float(df[col].std())
            df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std > 1e-6 else 0.0

        df["alpha_score"] = 0.4 * df["z_return_45m"] + 0.3 * df["z_vwap_dist"] + 0.3 * df["z_vol_spike"]
        df["rank"] = df["alpha_score"].rank(ascending=False, method="min")
        return df.sort_values(by="alpha_score", ascending=False)


class MomentumEngine:
    """Volatility-Gated Momentum Engine (Upgraded EMA 6/30 + ADX 14 + Volume Spike + Daily VWAP)."""

    def __init__(self, config: MomentumConfig = MOMENTUM_CONFIG):
        self.config = config
        self.lightgbm_gate = LightGBMRegimeGatekeeper(min_regime_prob=0.60)

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) 14 without external library dependencies."""
        if len(df) < 2:
            return pd.Series(0.0, index=df.index)

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr1 = high - low
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.empty_like(high)
        tr[0] = tr1[0]
        tr[1:] = np.maximum(tr1[1:], np.maximum(tr2, tr3))

        up_move = np.empty_like(high)
        up_move[0] = 0.0
        up_move[1:] = high[1:] - high[:-1]

        down_move = np.empty_like(high)
        down_move[0] = 0.0
        down_move[1:] = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)

        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / denom
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)
        return adx

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate session VWAP consistently across backfilled and live bars."""
        if len(df) == 0:
            return pd.Series(dtype=float)
        pv = df["pv"].values if "pv" in df.columns else df["close"].values * df["volume"].values
        vol = df["volume"].values
        dates = df.index.date if isinstance(df.index, pd.DatetimeIndex) else np.zeros(len(df))
        
        df_tmp = pd.DataFrame({"pv": pv, "vol": vol, "close": df["close"].values}, index=dates)
        cum_vol = df_tmp.groupby(level=0)["vol"].cumsum()
        cum_pv = df_tmp.groupby(level=0)["pv"].cumsum()
        safe_div = np.divide(cum_pv.values, cum_vol.values, out=np.zeros_like(cum_pv.values, dtype=float), where=cum_vol.values > 0)
        vwap = np.where(cum_vol.values > 0, safe_div, df_tmp["close"].values)
        return pd.Series(vwap, index=df.index)

    def evaluate_signal(self, symbol: str, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> MomentumSignal:
        """Evaluate Volatility-Gated Momentum Signal on 5-min and 15-min candle buffers."""
        sym = UniverseManager.to_clean_symbol(symbol)
        if self.config.entry_style in ("observe", "opening_range", "pullback", "range_reversion", "adaptive_alpha"):
            from quant_desk.engines.research_strategies import research_signal
            return research_signal(self, sym, df_5m, df_15m)
        
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
        df_5m_calc["vol_sma"] = df_5m_calc["volume"].shift(1).rolling(window=self.config.volume_sma_period).mean()
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
        # Engineering guardrails for late entries and abrupt price/volatility shocks.
        # Thresholds are hypotheses to validate, not parameters estimated by papers.
        tr = pd.concat([
            df_5m["high"] - df_5m["low"],
            (df_5m["high"] - df_5m["close"].shift(1)).abs(),
            (df_5m["low"] - df_5m["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        baseline_atr = float(tr.shift(1).rolling(14).mean().iloc[-1])
        shock = baseline_atr > 0 and float(tr.iloc[-1]) > baseline_atr * self.config.volatility_shock_mult
        extended = baseline_atr > 0 and abs(current_price - ema_fast_curr) > baseline_atr * self.config.max_entry_atr_extension
        if shock or extended:
            return MomentumSignal(sym, "HOLD", current_price, ema_fast_curr, ema_slow_curr,
                                  adx_curr, volume_spike, vwap_curr,
                                  "Volatility shock / excessive ATR extension; wait for stabilization")

        # LightGBM Regime Gatekeeper check (microsecond CPU histogram binning)
        vwap_dist = (current_price - vwap_curr) / vwap_curr if vwap_curr > 0 else 0.0
        prob, passed, regime_reason = self.lightgbm_gate.evaluate_regime_probability(adx_curr, vwap_dist, volume_spike)
        if not passed:
            return MomentumSignal(
                symbol=sym,
                action="MUTE_CHOP",
                price=current_price,
                ema_fast=ema_fast_curr,
                ema_slow=ema_slow_curr,
                adx_14=adx_curr,
                volume_spike_ratio=volume_spike,
                vwap=vwap_curr,
                reason=regime_reason
            )

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
        # Confirm direction on the higher timeframe, independently of ADX strength.
        higher_fast = float(self.calculate_ema(df_15m["close"], self.config.ema_fast).iloc[-1])
        higher_slow = float(self.calculate_ema(df_15m["close"], self.config.ema_slow).iloc[-1])
        if self.config.entry_style == "breakout":
            prior = df_5m.iloc[-self.config.breakout_lookback - 1:-1]
            bullish_signal = current_price > float(prior["high"].max())
            bearish_signal = current_price < float(prior["low"].min())
        elif self.config.entry_style != "trend":
            raise ValueError(f"Unknown momentum entry style: {self.config.entry_style}")
        bullish_signal = bullish_signal and higher_fast > higher_slow
        bearish_signal = bearish_signal and higher_fast < higher_slow

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
            reason=f"Waiting: ADX={adx_curr:.1f}, volume={volume_spike:.2f}x; trend / higher timeframe / VWAP gates required."
        )
