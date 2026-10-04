import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_desk.core.config import StatArbConfig, MomentumConfig, RiskConfig, FrictionConfig, RLConfig
from quant_desk.data.universe import UniverseManager
from quant_desk.engines.stat_arb import StatArbEngine, StatArbSignal
from quant_desk.engines.momentum import MomentumEngine, MomentumSignal
from quant_desk.engines.rl_agent import RLState, QLearningAgent, RLMetaFilter
from quant_desk.risk.portfolio_risk import PortfolioRiskManager
from quant_desk.execution.paper_broker import PaperBroker


class TestInstitutionalQuantFramework(unittest.TestCase):

    def setUp(self):
        self.stat_arb = StatArbEngine()
        self.momentum = MomentumEngine()
        self.risk_mgr = PortfolioRiskManager(initial_capital=1000000.0)
        self.broker = PaperBroker(initial_cash=1000000.0)
        self.rl_meta = RLMetaFilter()

    def test_universe_symbol_conversions(self):
        self.assertEqual(UniverseManager.to_yfinance_symbol("SBIN"), "SBIN.NS")
        self.assertEqual(UniverseManager.to_fyers_symbol("SBIN.NS"), "NSE:SBIN-EQ")
        self.assertEqual(UniverseManager.to_clean_symbol("NSE:SBIN-EQ"), "SBIN")

    def test_stat_arb_cointegration_and_zscore(self):
        # Generate synthetic cointegrated series
        np.random.seed(42)
        n = 200
        asset_b = np.cumsum(np.random.normal(0, 1, n)) + 500
        # asset_a = 1.2 * asset_b + mean-reverting noise
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 0.5)
        asset_a = 1.2 * asset_b + noise

        series_a = pd.Series(asset_a)
        series_b = pd.Series(asset_b)

        model = self.stat_arb.calibrate_pair("A", "B", series_a, series_b)
        self.assertTrue(model.is_cointegrated, "Pair should be cointegrated")
        self.assertGreater(model.half_life, 0, "Half life should be positive")

        # Test Z-score calculation
        spread, z = self.stat_arb.calculate_zscore(("A", "B"), asset_a[-1], asset_b[-1])
        self.assertIsInstance(z, float)

        # Test 3.5 Sigma Hard Stop Defense
        # Artificially set model mean & std to test high z-score
        self.stat_arb.models[("A", "B")].spread_mean = 0.0
        self.stat_arb.models[("A", "B")].spread_std = 0.1
        
        # Large divergence
        sig = self.stat_arb.evaluate_pair_signal(("A", "B"), price_a=1000.0, price_b=500.0, current_position=1)
        self.assertEqual(sig.action, "HARD_STOP_DIVERGENCE")
        self.assertIn("Hard Stop Fired", sig.reason)

    def test_momentum_adx_and_mute_gate(self):
        # Create synthetic 5m and 15m OHLCV DataFrames
        dates_5m = pd.date_range("2026-01-01 09:15", periods=50, freq="5min")
        dates_15m = pd.date_range("2026-01-01 09:15", periods=30, freq="15min")

        # Flat sideways data -> low ADX (Chop)
        df_5m = pd.DataFrame({
            "open": [500.0]*50,
            "high": [501.0]*50,
            "low": [499.0]*50,
            "close": [500.0]*50,
            "volume": [1000]*50,
            "vwap": [500.0]*50
        }, index=dates_5m)

        df_15m = pd.DataFrame({
            "open": [500.0]*30,
            "high": [501.0]*30,
            "low": [499.0]*30,
            "close": [500.0]*30,
            "volume": [3000]*30,
            "vwap": [500.0]*30
        }, index=dates_15m)

        sig = self.momentum.evaluate_signal("SBIN", df_5m, df_15m)
        self.assertEqual(sig.action, "MUTE_CHOP")
        self.assertIn("Mute Gate Active", sig.reason)

    def test_half_kelly_and_kill_switch(self):
        # 1. Test Half Kelly
        dates = pd.date_range("2026-01-01", periods=30, freq="5min")
        df_candles = pd.DataFrame({
            "high": [105]*30, "low": [95]*30, "close": [100]*30, "volume": [1000]*30
        }, index=dates)

        res = self.risk_mgr.calculate_half_kelly_size("SBIN", 100.0, df_candles)
        self.assertGreater(res.share_quantity, 0)
        self.assertGreater(res.atr_14, 0)

        # 2. Test Daily Kill Switch (-2.0% PnL)
        kill_triggered, reason = self.risk_mgr.evaluate_kill_switch(-25000.0)  # -2.5% of 1M
        self.assertTrue(kill_triggered)
        self.assertIn("SYSTEMIC KILL SWITCH TRIGGERED", reason)

    def test_paper_broker_slippage_and_taxes(self):
        # Execute BUY market order of 100 shares @ ₹500
        pos = self.broker.execute_market_order("SBIN", "BUY", 100, 500.0, strategy="MOMENTUM")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.quantity, 100)
        # Entry price should include 0.03% buy slippage: 500 * 1.0003 = 500.15
        self.assertAlmostEqual(pos.entry_price, 500.15, places=2)

        # Close position @ ₹520
        record = self.broker.close_position("SBIN", 520.0, exit_reason="TARGET")
        self.assertIsNotNone(record)
        self.assertGreater(record.total_friction, 0)
        self.assertGreater(record.net_pnl, 0)
        self.assertEqual(len(self.broker.open_positions), 0)

    def test_rl_agent_state_and_q_learning(self):
        # 1. Test state vector normalization
        state = RLState(
            z_score=1.75,
            adx_14=30.0,
            ema_spread_ratio=0.005,
            vwap_distance_pct=0.002,
            volume_spike_ratio=1.5,
            return_5m_pct=0.001
        )
        vec = state.to_vector()
        self.assertEqual(len(vec), 6)
        self.assertTrue(np.all(vec >= -1.0) and np.all(vec <= 1.0))

        # 2. Test Q-learning decision evaluation
        dates_5m = pd.date_range("2026-01-01 09:15", periods=50, freq="5min")
        dates_15m = pd.date_range("2026-01-01 09:15", periods=30, freq="15min")
        df_5m = pd.DataFrame({
            "open": [500.0 + i*0.5 for i in range(50)],
            "high": [501.0 + i*0.5 for i in range(50)],
            "low": [499.0 + i*0.5 for i in range(50)],
            "close": [500.5 + i*0.5 for i in range(50)],
            "volume": [2000]*50,
            "vwap": [500.0 + i*0.5 for i in range(50)]
        }, index=dates_5m)
        df_15m = pd.DataFrame({
            "open": [500.0]*30, "high": [502.0]*30, "low": [498.0]*30, "close": [501.0]*30,
            "volume": [5000]*30, "adx_14": [28.0]*30
        }, index=dates_15m)

        decision = self.rl_meta.evaluate_decision("SBIN", df_5m, df_15m, z_score=0.5, proposed_action="BUY")
        self.assertIn(decision.action_name, ["HOLD", "BUY", "SELL", "EXIT"])
        self.assertGreaterEqual(decision.confidence_weight, 0.0)
        self.assertLessEqual(decision.confidence_weight, 1.0)

    def test_sector_caps_and_fixed_risk_sizing(self):
        # 1. Test Fixed Risk Sizing
        dates = pd.date_range("2026-01-01", periods=30, freq="5min")
        df_candles = pd.DataFrame({
            "high": [105]*30, "low": [95]*30, "close": [100]*30, "volume": [1000]*30
        }, index=dates)

        res = self.risk_mgr.calculate_half_kelly_size("TCS", 2000.0, df_candles)
        self.assertGreater(res.share_quantity, 0)
        # Verify single trade risk does not exceed 0.75% of capital (₹7,500)
        max_risk = self.risk_mgr.config.max_risk_per_trade_pct * self.risk_mgr.current_capital
        expected_risk = (res.atr_14 * 1.5) * res.share_quantity
        self.assertLessEqual(expected_risk, max_risk + 100.0)

        # 2. Test Sector Limit Enforcement (Max 1 per sector)
        self.broker.execute_market_order("TCS", "BUY", 10, 2000.0, strategy="MOMENTUM")
        can_accept_infy, reason_infy = self.risk_mgr.can_accept_new_trade("INFY", self.broker.open_positions)
        self.assertFalse(can_accept_infy, "Should reject INFY because TCS is already open in IT sector")

        # Non-IT stock (e.g. SBIN in BANK sector) should be accepted
        can_accept_sbin, reason_sbin = self.risk_mgr.can_accept_new_trade("SBIN", self.broker.open_positions)
        self.assertTrue(can_accept_sbin, "Should accept SBIN as BANK sector is open")


if __name__ == "__main__":
    unittest.main()


