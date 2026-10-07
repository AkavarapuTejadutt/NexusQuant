import unittest
from datetime import date, datetime
from unittest.mock import Mock, patch

import pandas as pd

from quant_desk.data.nse_market import NSEMarketClient, NSEDataError, NSESnapshot
from quant_desk.data.universe import UniverseManager
from quant_desk.data.data_feed import DataFeedManager
from quant_desk.main import MasterQuantDesk


class TestNSEUniverse(unittest.TestCase):
    def test_ranking_is_volume_not_price_return_or_old_index_list(self):
        rows = [{"symbol": "NEWSTOCK", "totalTradedVolume": 900, "lastPrice": 50, "pChange": -8},
                {"symbol": "SBIN", "totalTradedVolume": 500, "lastPrice": 600, "pChange": 7},
                {"symbol": "ETF", "totalTradedVolume": 9999, "lastPrice": 10},
                {"symbol": "BAD", "totalTradedVolume": float("inf"), "lastPrice": 10},
                {"symbol": "SBIN", "totalTradedVolume": 400, "lastPrice": 600},
                {"symbol": "BE_STOCK", "series": "BE", "totalTradedVolume": 99999, "lastPrice": 50}]
        ranked, count = NSEMarketClient.rank_rows(rows, {"NEWSTOCK", "SBIN", "BAD", "BE_STOCK"}, 50)
        self.assertEqual([r["symbol"] for r in ranked], ["NEWSTOCK", "SBIN"])
        self.assertEqual(count, 2)
        self.assertEqual(ranked[0]["change_pct"], -8)

    def test_full_bhavcopy_exact_date_and_equity_filters(self):
        text = ("SYMBOL,SERIES,DATE1,CLOSE_PRICE,PREV_CLOSE,TTL_TRD_QNTY\n"
                "NEWSTOCK,EQ,06-Oct-2026,50,55,900\n"
                "SBIN,EQ,06-Oct-2026,600,590,500\n"
                "ETF,EQ,06-Oct-2026,10,10,9999\n")
        client = NSEMarketClient()
        with patch.object(client, "_get", return_value=Mock(text=text)), \
             patch.object(client, "equity_master", return_value={"NEWSTOCK", "SBIN"}):
            snapshot = client.daily_snapshot(date(2026, 10, 6), top_n=2)
            self.assertTrue(snapshot.global_ranking)
            self.assertEqual(snapshot.stocks[0]["symbol"], "NEWSTOCK")
            self.assertLess(snapshot.stocks[0]["change_pct"], 0)
            with self.assertRaises(NSEDataError):
                client.daily_snapshot(date(2026, 10, 7), top_n=2)

    def test_stale_live_snapshot_is_rejected(self):
        client = NSEMarketClient()
        row = {"symbol": "SBIN", "series": "EQ", "totalTradedVolume": 500,
               "lastPrice": 600, "lastUpdateTime": "2026-10-05 10:00:00"}
        with patch.object(client, "index_rows", return_value=[row]), \
             patch.object(client, "_json", return_value={"data": [row]}):
            with self.assertRaisesRegex(NSEDataError, "not today's"):
                client.live_snapshot(1, datetime(2026, 10, 6, 10))

    def test_intraday_coverage_is_never_labeled_global(self):
        client = NSEMarketClient()
        row = {"symbol": "NEWSTOCK", "series": "EQ", "totalTradedVolume": 900,
               "lastPrice": 50, "lastUpdateTime": "2026-10-06 10:00:00"}
        with patch.object(client, "index_rows", return_value=[row]), \
             patch.object(client, "_json", return_value={"data": [row]}), \
             patch.object(client, "equity_master", return_value={"NEWSTOCK"}):
            snapshot = client.live_snapshot(1, datetime(2026, 10, 6, 10))
            self.assertFalse(snapshot.global_ranking)
            self.assertTrue(snapshot.warnings)

    def test_nse_failure_does_not_fall_back_to_static_nifty_list(self):
        client = Mock()
        client.select_snapshot.side_effect = NSEDataError("NSE unavailable")
        universe = UniverseManager(nse_client=client)
        with self.assertRaises(NSEDataError):
            universe.get_nse_top_volume()

    def test_refresh_preserves_open_stock_feed_but_changes_entry_selection(self):
        with patch("quant_desk.main.UniverseManager.get_nse_top_volume", return_value=["SBIN"]):
            desk = MasterQuantDesk("live")
        desk.broker.execute_market_order("SBIN", "BUY", 10, 500, strategy="MOMENTUM_RL")
        with patch.object(desk.universe, "get_nse_top_volume", return_value=["NEWSTOCK"]), \
             patch.object(desk.data_feed, "backfill_intraday_candles"):
            desk.refresh_universe()
        self.assertEqual(desk.momentum_symbols, {"NEWSTOCK"})
        self.assertIn("SBIN", desk.data_feed.aggregators)
        self.assertIn("SBIN", desk.broker.open_positions)
        self.assertIn("NEWSTOCK", desk.data_feed.aggregators)

    def test_dynamic_same_sector_pairs_keep_original_pair_candidates(self):
        universe = UniverseManager()
        prices = pd.DataFrame({"A": [100 + i + i % 3 for i in range(90)],
                               "B": [200 + 2 * i + 2 * (i % 3) for i in range(90)],
                               "C": [300 + i for i in range(90)]})
        with patch.object(universe, "get_sector", side_effect=lambda s: "METALS" if s in ("A", "B") else "BANK"):
            pairs = universe.discover_pairs(prices, ["A", "B", "C"])
        self.assertEqual(pairs, [("A", "B")])
        self.assertIn(("TCS", "INFY"), universe.get_eligible_pairs())


if __name__ == "__main__":
    unittest.main()
