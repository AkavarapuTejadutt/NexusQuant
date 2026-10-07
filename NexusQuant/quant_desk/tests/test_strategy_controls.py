import contextlib
import io
import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from quant_desk.core.config import RLConfig
from quant_desk.engines.momentum import MomentumSignal
from quant_desk.engines.rl_agent import RLMetaFilter, RLActionDecision
from quant_desk.main import MasterQuantDesk
from quant_desk.evaluate_strategies import run_candidate


def candles():
    index = pd.date_range(end="2026-10-06 09:55", periods=90, freq="5min")
    return pd.DataFrame({"open": 500., "high": 502., "low": 498., "close": 500., "volume": 1000.}, index=index)


class TestStrategyControls(unittest.TestCase):
    def desk(self):
        with patch("quant_desk.main.UniverseManager.get_nse_top_volume", return_value=["SBIN"]):
            desk = MasterQuantDesk("live")
        desk.data_feed.aggregators["SBIN"].load_history(candles())
        desk.now_ist = lambda: pd.Timestamp("2026-10-06 10:00")
        return desk

    def signal(self, action="BUY"):
        return MomentumSignal("SBIN", action, 500, 501, 499, 30, 2, 499, "test")

    def decision(self):
        return RLActionDecision(1, "BUY", 1, [0, 0, 0, 0], False, "bypass")

    def test_closed_candles_once_per_bar_and_stop_on_each_tick(self):
        desk = self.desk()
        with patch.object(desk.momentum, "evaluate_signal", return_value=self.signal()) as evaluate:
            desk.on_tick_update("SBIN", 500, 10, datetime(2026, 10, 6, 10))
            self.assertIn("SBIN", desk.broker.open_positions)
            self.assertEqual(evaluate.call_args.args[1].index[-1], pd.Timestamp("2026-10-06 09:55"))
            self.assertLessEqual(evaluate.call_args.args[2].index[-1], pd.Timestamp("2026-10-06 09:45"))
            desk.on_tick_update("SBIN", 490, 10, datetime(2026, 10, 6, 10, 0, 30))
            self.assertNotIn("SBIN", desk.broker.open_positions)
            desk.on_tick_update("SBIN", 500, 10, datetime(2026, 10, 6, 10, 1))
            self.assertNotIn("SBIN", desk.broker.open_positions)
            self.assertEqual(evaluate.call_count, 1)

    def test_proposed_trade_does_not_overrun_portfolio_risk(self):
        desk = self.desk()
        existing = desk.broker.execute_market_order("TCS", "BUY", 10, 2000)
        existing.stop_loss_risk = 7400
        desk._execute_momentum_signal(self.signal(), "SBIN", 500, candles(), self.decision())
        self.assertLessEqual(sum(p.stop_loss_risk for p in desk.broker.open_positions.values()),
                             1000000 * desk.risk_mgr.config.max_portfolio_open_risk_pct)
        self.assertLessEqual(desk.broker.open_positions["SBIN"].quantity, 20)

    def test_eod_exit_blocks_new_entries(self):
        desk = self.desk()
        desk.broker.execute_market_order("SBIN", "BUY", 10, 500, strategy="MOMENTUM_RL")
        desk.on_tick_update("SBIN", 501, 10, datetime(2026, 10, 6, 15, 20))
        self.assertFalse(desk.broker.open_positions)
        self.assertFalse(desk.entries_allowed)
        self.assertEqual(desk.broker.trade_history[-1].exit_reason, "END_OF_DAY")

    def test_protective_stop_executes_before_indicator_failure(self):
        desk = self.desk()
        desk.broker.execute_market_order("SBIN", "BUY", 10, 500,
                                         strategy="MOMENTUM_RL", stop_loss_price=495)
        desk.data_feed.register_tick_callback(desk.on_tick_update)
        with patch.object(desk.momentum, "evaluate_signal", side_effect=ValueError("bad indicator input")), \
             self.assertLogs("quant_desk.data.data_feed", level="ERROR"):
            desk.data_feed.process_tick("SBIN", 490, 10, datetime(2026, 10, 6, 10))
        self.assertNotIn("SBIN", desk.broker.open_positions)
        self.assertEqual(desk.broker.trade_history[-1].exit_reason, "STOP_LOSS")
        self.assertEqual(desk.data_feed.callback_errors, 1)

    def test_daily_baseline_resets_after_previous_day_loss(self):
        desk = self.desk()
        desk.on_tick_update("SBIN", 500, 10, datetime(2026, 10, 6, 10))
        desk.broker.cash -= 30000
        desk.on_tick_update("SBIN", 500, 10, datetime(2026, 10, 6, 10, 1))
        self.assertTrue(desk.risk_mgr.kill_switch_triggered)
        desk.on_tick_update("SBIN", 500, 10, datetime(2026, 10, 7, 10))
        self.assertFalse(desk.risk_mgr.kill_switch_triggered)
        self.assertEqual(desk.risk_mgr.starting_daily_capital, 970000)

    def test_rl_bypass_deterministic_and_entry_state_not_overwritten(self):
        filter_ = RLMetaFilter(replace(RLConfig(), enabled=True, min_training_steps=100))
        df = candles()
        first = filter_.evaluate_decision("SBIN", df, df, proposed_action="BUY")
        self.assertEqual(first.action_name, "BUY")
        self.assertEqual(first.confidence_weight, 1)
        self.assertFalse(first.is_exploration)
        filter_.record_entry("SBIN")
        filter_.evaluate_decision("SBIN", df, df, proposed_action="SELL")
        self.assertEqual(filter_.last_actions["SBIN"], 1)

    def test_historical_evaluator_does_not_read_future_bars(self):
        df = candles()
        df.loc[pd.Timestamp("2026-10-06 10:00")] = [500, 502, 498, 500, 1000]
        df.loc[pd.Timestamp("2026-10-06 10:05")] = [500, 502, 498, 500, 1000]
        end = pd.Timestamp("2026-10-06 10:10")
        with contextlib.redirect_stdout(io.StringIO()), patch("quant_desk.evaluate_strategies.MomentumEngine.evaluate_signal", return_value=self.signal()):
            before = run_candidate(df, "SBIN", "trend", pd.Timestamp("2026-10-06 09:55"), end)
            df.loc[pd.Timestamp("2026-10-06 10:10")] = [900, 990, 890, 980, 999999]
            after = run_candidate(df, "SBIN", "trend", pd.Timestamp("2026-10-06 09:55"), end)
        self.assertEqual(before, after)
        self.assertEqual(before["trades"], 1)
        self.assertLess(before["net_pnl"], 0)  # Flat market loses the configured execution costs.


if __name__ == "__main__":
    unittest.main()
