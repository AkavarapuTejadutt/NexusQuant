import unittest
from unittest.mock import patch

from quant_desk.data.data_feed import DataFeedManager


class TestReplayStatus(unittest.TestCase):
    def run_replay(self, candles, broken=False):
        feed = DataFeedManager(["SBIN"])
        with patch("quant_desk.data.data_feed.get_access_token", return_value="test"), \
             patch("quant_desk.data.data_feed.fyersModel.FyersModel") as model:
            model.return_value.history.return_value = {"s": "ok", "candles": candles}
            if broken:
                feed.process_tick = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad replay row"))
            feed.start_simulation_feed(tick_interval_sec=0)
            feed.socket_thread.join(timeout=5)
            self.assertFalse(feed.socket_thread.is_alive())
        return feed

    def test_completion_counts_timestamps_and_ticks(self):
        feed = self.run_replay([[1791258300, 100, 102, 99, 101, 1000],
                                [1791258360, 101, 103, 100, 102, 1200]])
        self.assertEqual(feed.replay_status, "Completed")
        self.assertEqual((feed.replay_completed, feed.replay_total, feed.tick_count), (2, 2, 2))
        self.assertFalse(feed.is_running)

    def test_missing_history_has_terminal_failure(self):
        feed = self.run_replay([])
        self.assertTrue(feed.replay_status.startswith("Failed:"))
        self.assertFalse(feed.is_running)

    def test_worker_exception_is_visible_and_stops_feed(self):
        with self.assertLogs("quant_desk.data.data_feed", level="ERROR"):
            feed = self.run_replay([[1791258300, 100, 102, 99, 101, 1000]], broken=True)
        self.assertEqual(feed.replay_status, "Failed")
        self.assertIn("bad replay row", feed.last_error)
        self.assertFalse(feed.is_running)
