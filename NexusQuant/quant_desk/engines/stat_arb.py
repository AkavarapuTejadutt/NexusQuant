import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from quant_desk.core.config import STAT_ARB_CONFIG, StatArbConfig


@dataclass
class CointegrationModel:
    pair: Tuple[str, str]
    is_cointegrated: bool
    p_value: float
    beta: float
    spread_mean: float
    spread_std: float
    half_life: float  # In bars/days
    max_holding_bars: float
    last_updated: pd.Timestamp


@dataclass
class StatArbSignal:
    pair: Tuple[str, str]
    action: str  # "ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD", "EXIT_TARGET", "HARD_STOP_DIVERGENCE", "TIME_STOP_EXPIRED", "HOLD"
    z_score: float
    current_spread: float
    beta: float
    reason: str


class StatArbEngine:
    """Statistical Arbitrage (Pairs Trading) Engine with Johansen coint, OU half-life & divergence stops."""

    def __init__(self, config: StatArbConfig = STAT_ARB_CONFIG):
        self.config = config
        self.models: Dict[Tuple[str, str], CointegrationModel] = {}
        self.trade_entry_time: Dict[Tuple[str, str], pd.Timestamp] = {}
        self.trade_entry_bars: Dict[Tuple[str, str], int] = {}

    def calibrate_pair(self, asset_a: str, asset_b: str, prices_a: pd.Series, prices_b: pd.Series) -> CointegrationModel:
        """Calibrate cointegration relationship, hedge ratio (beta), and OU half-life for asset pair."""
        pair = (asset_a, asset_b)
        
        # Align series & log transform
        df = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
        log_a = np.log(df["a"])
        log_b = np.log(df["b"])

        # 1. Johansen & Engle-Granger Cointegration Test
        score, p_value, _ = coint(log_a, log_b)
        
        # Determine Johansen trace statistic as secondary validation
        joh_res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        joh_coint = joh_res.lr1[0] > joh_res.cvt[0, 1]  # 95% confidence level

        is_coint = (p_value < self.config.p_value_threshold) or joh_coint

        # 2. OLS Regression for Hedge Ratio (Beta)
        X = sm.add_constant(log_b)
        model = sm.OLS(log_a, X).fit()
        beta = float(model.params.iloc[1])

        # 3. Calculate Residual Spread
        spread = log_a - beta * log_b
        mean = float(spread.mean())
        std = float(spread.std())

        # 4. Ornstein-Uhlenbeck Half-Life Calculation
        # Delta e_t = -theta * (e_{t-1} - mean) + error
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        df_ou = pd.DataFrame({"diff": spread_diff, "lag": spread_lag - mean}).dropna()
        
        if len(df_ou) > 10:
            ou_model = sm.OLS(df_ou["diff"], df_ou["lag"]).fit()
            theta = -float(ou_model.params.iloc[0])
            if theta > 0:
                half_life = np.log(2) / theta
            else:
                half_life = 30.0  # Default fallback
        else:
            half_life = 30.0

        max_holding_bars = half_life * self.config.half_life_time_mult

        coint_model = CointegrationModel(
            pair=pair,
            is_cointegrated=is_coint,
            p_value=float(p_value),
            beta=beta,
            spread_mean=mean,
            spread_std=std if std > 1e-8 else 1.0,
            half_life=float(half_life),
            max_holding_bars=float(max_holding_bars),
            last_updated=pd.Timestamp.now()
        )
        self.models[pair] = coint_model
        return coint_model

    def calculate_zscore(self, pair: Tuple[str, str], price_a: float, price_b: float) -> Tuple[float, float]:
        """Calculate real-time residual spread and Z-score for pair."""
        if pair not in self.models:
            raise KeyError(f"Pair {pair} is not calibrated. Run calibrate_pair first.")
        
        model = self.models[pair]
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        spread = log_a - model.beta * log_b
        z_score = (spread - model.spread_mean) / model.spread_std
        return float(spread), float(z_score)

    def evaluate_pair_signal(
        self,
        pair: Tuple[str, str],
        price_a: float,
        price_b: float,
        current_position: int = 0,  # +1: Long Spread, -1: Short Spread, 0: Flat
        open_bars: int = 0
    ) -> StatArbSignal:
        """Evaluate real-time signal, incorporating $3.5\sigma$ hard stop and OU time stop."""
        model = self.models.get(pair)
        if not model or not model.is_cointegrated:
            return StatArbSignal(
                pair=pair, action="HOLD", z_score=0.0, current_spread=0.0,
                beta=1.0, reason="Pair not cointegrated (p >= 0.05)"
            )

        spread, z = self.calculate_zscore(pair, price_a, price_b)

        # -------------------------------------------------------------
        # Defense 1: The 3.5 Sigma Hard Stop (Divergence Trap Defense)
        # -------------------------------------------------------------
        if abs(z) >= self.config.hard_stop_zscore and current_position != 0:
            return StatArbSignal(
                pair=pair,
                action="HARD_STOP_DIVERGENCE",
                z_score=z,
                current_spread=spread,
                beta=model.beta,
                reason=f"Hard Stop Fired: |Z|={abs(z):.2f} >= {self.config.hard_stop_zscore}. Fundamental relationship broken."
            )

        # -------------------------------------------------------------
        # Defense 2: Ornstein-Uhlenbeck Time Stop
        # -------------------------------------------------------------
        if current_position != 0 and open_bars >= model.max_holding_bars:
            return StatArbSignal(
                pair=pair,
                action="TIME_STOP_EXPIRED",
                z_score=z,
                current_spread=spread,
                beta=model.beta,
                reason=f"OU Time Stop Fired: Trade open for {open_bars} bars >= max limit ({model.max_holding_bars:.1f} bars)."
            )

        # -------------------------------------------------------------
        # Execution Signals
        # -------------------------------------------------------------
        if current_position == 0:
            # Long Spread: Buy Asset A, Short Asset B
            if z <= -self.config.entry_zscore:
                return StatArbSignal(
                    pair=pair,
                    action="ENTER_LONG_SPREAD",
                    z_score=z,
                    current_spread=spread,
                    beta=model.beta,
                    reason=f"Z-Score {z:.2f} <= -{self.config.entry_zscore}. Spread oversold."
                )
            # Short Spread: Short Asset A, Buy Asset B
            elif z >= self.config.entry_zscore:
                return StatArbSignal(
                    pair=pair,
                    action="ENTER_SHORT_SPREAD",
                    z_score=z,
                    current_spread=spread,
                    beta=model.beta,
                    reason=f"Z-Score {z:.2f} >= +{self.config.entry_zscore}. Spread overbought."
                )
        elif current_position == 1: # Currently Long Spread
            # Exit Target: Mean Reversion to Z = 0
            if z >= self.config.exit_zscore:
                return StatArbSignal(
                    pair=pair,
                    action="EXIT_TARGET",
                    z_score=z,
                    current_spread=spread,
                    beta=model.beta,
                    reason=f"Target Reached: Z-Score {z:.2f} >= {self.config.exit_zscore}."
                )
        elif current_position == -1: # Currently Short Spread
            # Exit Target: Mean Reversion to Z = 0
            if z <= self.config.exit_zscore:
                return StatArbSignal(
                    pair=pair,
                    action="EXIT_TARGET",
                    z_score=z,
                    current_spread=spread,
                    beta=model.beta,
                    reason=f"Target Reached: Z-Score {z:.2f} <= {self.config.exit_zscore}."
                )

        return StatArbSignal(
            pair=pair,
            action="HOLD",
            z_score=z,
            current_spread=spread,
            beta=model.beta,
            reason=f"Z-Score {z:.2f} inside normal range."
        )
