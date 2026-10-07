import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from quant_desk.core.config import FrictionConfig
from quant_desk.data.data_feed import CandleAggregator, DataFeedManager
from quant_desk.engines.momentum import MomentumEngine, MomentumSignal
from quant_desk.engines.rl_agent import RLActionDecision
from quant_desk.execution.paper_broker import PaperBroker
from quant_desk.main import MasterQuantDesk


class TestPaperPipeline(unittest.TestCase):
    def test_fyers_volume_delta_and_daily_reset(self):
        feed = DataFeedManager(["SBIN"])
        ts = int(pd.Timestamp("2026-10-06 10:00", tz="Asia/Kolkata").timestamp())
        for cumulative in [10000, 10040, 10040, 10100]:
            feed.process_live_message({"symbol": "NSE:SBIN-EQ", "ltp": 500,
                                       "vol_traded_today": cumulative, "exch_feed_time": ts})
        self.assertEqual(feed.tick_count, 4)
        candles = feed.aggregators["SBIN"].get_5m_dataframe()
        self.assertEqual(candles.iloc[-1]["volume"], 100)
        self.assertEqual(candles.index[-1], pd.Timestamp("2026-10-06 10:00"))
        feed.process_live_message({"symbol": "NSE:SBIN-EQ", "ltp": 501,
                                   "vol_traded_today": 50, "exch_feed_time": ts + 86400})
        self.assertEqual(feed.aggregators["SBIN"].get_5m_dataframe().iloc[-1]["volume"], 0)

    def test_callback_errors_are_visible_and_other_callbacks_continue(self):
        feed = DataFeedManager(["SBIN"])
        calls = []
        def broken(*args):
            raise RuntimeError("indicator failed")
        feed.register_tick_callback(broken)
        feed.register_tick_callback(lambda *args: calls.append(args))
        with self.assertLogs("quant_desk.data.data_feed", level="ERROR"):
            feed.process_tick("SBIN", 500, 10)
        self.assertEqual(feed.callback_errors, 1)
        self.assertIn("indicator failed", feed.last_error)
        self.assertEqual(len(calls), 1)

    def test_history_resamples_real_fifteen_minute_candles(self):
        df = pd.DataFrame({"open": [100, 101, 102, 103], "high": [102, 103, 104, 105],
                           "low": [99, 100, 101, 102], "close": [101, 102, 103, 104],
                           "volume": [10, 20, 30, 40]},
                          index=pd.date_range("2026-10-06 09:15", periods=4, freq="5min"))
        agg = CandleAggregator("SBIN")
        agg.load_history(df)
        fifteen = agg.get_15m_dataframe()
        self.assertEqual(len(fifteen), 2)
        self.assertEqual(list(fifteen.iloc[0][["open", "high", "low", "close", "volume"]]),
                         [100, 104, 99, 103, 60])
        agg.add_tick(106, 5, datetime(2026, 10, 6, 9, 32))
        self.assertEqual(agg.get_15m_dataframe().iloc[-1]["volume"], 45)

    def test_session_vwap_resets_each_day(self):
        df = pd.DataFrame({"close": [100, 200, 300], "volume": [10, 10, 10]},
                          index=pd.to_datetime(["2026-10-05 15:25", "2026-10-06 09:15", "2026-10-06 09:20"]))
        self.assertEqual(list(MomentumEngine.calculate_vwap(df)), [100, 200, 250])

    def test_actual_momentum_entry_with_volume_spike(self):
        closes = [500 + i for i in range(90)]
        df = pd.DataFrame({"open": closes, "high": [p + 2 for p in closes],
                           "low": [p - 2 for p in closes], "close": closes,
                           "volume": [1000]*89 + [8000]},
                          index=pd.date_range("2026-10-06 09:15", periods=90, freq="5min"))
        agg = CandleAggregator("SBIN")
        agg.load_history(df)
        signal = MomentumEngine().evaluate_signal("SBIN", agg.get_5m_dataframe(), agg.get_15m_dataframe())
        self.assertEqual(signal.action, "BUY")
        self.assertGreater(signal.volume_spike_ratio, 1.5)

    def test_actual_momentum_short_in_downtrend(self):
        closes = [600 - i for i in range(90)]
        df = pd.DataFrame({"open": closes, "high": [p + 2 for p in closes],
                           "low": [p - 2 for p in closes], "close": closes,
                           "volume": [1000]*89 + [8000]},
                          index=pd.date_range("2026-10-06 09:15", periods=90, freq="5min"))
        agg = CandleAggregator("NONNIFTY")
        agg.load_history(df)
        signal = MomentumEngine().evaluate_signal("NONNIFTY", agg.get_5m_dataframe(), agg.get_15m_dataframe())
        self.assertEqual(signal.action, "SELL")

    def test_momentum_does_not_chase_a_price_shock(self):
        closes = [500 + i for i in range(90)]
        closes[-1] += 50
        df = pd.DataFrame({"open": closes, "high": [p + 2 for p in closes],
                           "low": [p - 2 for p in closes], "close": closes,
                           "volume": [1000]*89 + [8000]},
                          index=pd.date_range("2026-10-06 09:15", periods=90, freq="5min"))
        agg = CandleAggregator("SBIN")
        agg.load_history(df)
        signal = MomentumEngine().evaluate_signal("SBIN", agg.get_5m_dataframe(), agg.get_15m_dataframe())
        self.assertEqual(signal.action, "HOLD")
        self.assertIn("Volatility shock", signal.reason)

    def test_long_and_short_equity_tracks_prices_and_reconciles_closures(self):
        for side, profitable_price in [("BUY", 520), ("SELL", 480)]:
            with self.subTest(side=side):
                broker = PaperBroker(1000000, FrictionConfig(stt_pct=0.00025))
                broker.execute_market_order("NSE:SBIN-EQ", side, 100, 500)
                entry = broker.get_portfolio_summary({"NSE:SBIN-EQ": 500})
                gain = broker.get_portfolio_summary({"NSE:SBIN-EQ": profitable_price})
                self.assertEqual(gain["open_positions_count"], 1)
                self.assertGreater(gain["total_equity"], entry["total_equity"])
                self.assertAlmostEqual(gain["realized_pnl"] + gain["unrealized_pnl"], gain["total_pnl"])
                original_fees = broker.open_positions["SBIN"].entry_friction
                broker.close_partial_position("SBIN", 40, profitable_price)
                self.assertAlmostEqual(broker.open_positions["SBIN"].entry_friction, original_fees * 0.6)
                broker.close_position("SBIN", profitable_price)
                final = broker.get_portfolio_summary({})
                self.assertAlmostEqual(final["cash"] - 1000000, final["realized_pnl"])
                self.assertAlmostEqual(final["total_pnl"], final["realized_pnl"])

    def test_broker_rejects_overwrite_and_unaffordable_orders(self):
        broker = PaperBroker(1000)
        self.assertIsNone(broker.execute_market_order("SBIN", "BUY", 2, 500))
        self.assertEqual(broker.cash, 1000)
        first = broker.execute_market_order("SBIN", "BUY", 1, 500)
        cash = broker.cash
        self.assertIsNone(broker.execute_market_order("SBIN", "SELL", 1, 500))
        self.assertIs(broker.open_positions["SBIN"], first)
        self.assertEqual(broker.cash, cash)

    def test_tick_to_position_to_dashboard_without_network(self):
        with patch("quant_desk.main.UniverseManager.get_nse_top_volume", return_value=["SBIN"]):
            desk = MasterQuantDesk("live")
        desk.now_ist = lambda: pd.Timestamp("2026-10-06 10:00")
        self.assertIn("INFY", desk.data_feed.aggregators)
        df = pd.DataFrame({"open": [500]*90, "high": [502]*90, "low": [498]*90,
                           "close": [500]*90, "volume": [1000]*90},
                          index=pd.date_range(end="2026-10-06 09:55", periods=90, freq="5min"))
        desk.data_feed.aggregators["SBIN"].load_history(df)
        desk.data_feed.register_tick_callback(desk.on_tick_update)
        signal = MomentumSignal("SBIN", "BUY", 500, 501, 499, 30, 2, 499, "Test valid entry")
        decision = RLActionDecision(1, "BUY", 0.6, [0, 1, 0, 0], False, "Test")
        with patch.object(desk.momentum, "evaluate_signal", return_value=signal), \
             patch.object(desk.rl_filter, "evaluate_decision", return_value=decision):
            desk.data_feed.process_tick("NSE:SBIN-EQ", 500, 100, datetime(2026, 10, 6, 10))
            self.assertEqual(desk.data_feed.callback_errors, 0)
            self.assertIn("SBIN", desk.broker.open_positions)
            before = desk.broker.get_portfolio_summary(desk.latest_prices)["total_equity"]
            desk.data_feed.process_tick("NSE:SBIN-EQ", 501, 100, datetime(2026, 10, 6, 10, 1))
            after = desk.broker.get_portfolio_summary(desk.latest_prices)["total_equity"]
            self.assertGreater(after, before)
            desk.render_terminal_dashboard()
        self.assertEqual(desk.data_feed.callback_errors, 0)


if __name__ == "__main__":
    unittest.main()
