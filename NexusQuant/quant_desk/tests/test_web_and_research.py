import json
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import urllib.request
import urllib.error
import pandas as pd

from quant_desk.web_app import Dashboard, make_handler
from quant_desk.main import MasterQuantDesk
from quant_desk.core.config import MOMENTUM_CONFIG
from quant_desk.engines.momentum import MomentumEngine, MomentumSignal
from quant_desk.evaluate_strategies import ResearchBroker
from quant_desk.data.data_feed import CandleAggregator, DataFeedManager


class WebResearchTests(unittest.TestCase):
    def opening_candles(self, short=False):
        index = pd.date_range("2026-10-01 09:15", periods=75, freq="5min").append(
            pd.date_range("2026-10-05 09:15", periods=75, freq="5min")).append(
            pd.date_range("2026-10-06 09:15", periods=4, freq="5min"))
        prices = [100+i*9/149 for i in range(150)] + [110,111,111,113]
        if short:
            prices = [250-p for p in prices]
        frame = pd.DataFrame({"close":prices,"volume":[20000]*153+[40000]},index=index)
        frame["open"] = frame.close
        frame["high"],frame["low"] = frame.close+1,frame.close-1
        agg = CandleAggregator("SBIN")
        agg.load_history(frame)
        return frame,agg.get_15m_dataframe().iloc[:-1]

    def test_opening_range_long_and_short(self):
        engine = MomentumEngine(replace(MOMENTUM_CONFIG,entry_style="opening_range"))
        for short,action in ((False,"BUY"),(True,"SELL")):
            five,fifteen=self.opening_candles(short)
            self.assertEqual(engine.evaluate_signal("SBIN",five,fifteen).action,action)
            missing = five.drop(pd.Timestamp("2026-10-06 09:20"))
            self.assertEqual(engine.evaluate_signal("SBIN",missing,fifteen).action,"HOLD")

    def test_quote_snapshot_preserves_exchange_price_and_old_timestamp(self):
        feed = DataFeedManager(["SBIN"])
        with patch("quant_desk.data.data_feed.fyersModel.FyersModel") as model:
            model.return_value.quotes.return_value={"s":"ok","d":[{"n":"NSE:SBIN-EQ","v":{"lp":958.75,"tt":"1791244800"}}]}
            quotes = feed.fetch_quote_snapshot()
        self.assertEqual(quotes[0][1],958.75)
        self.assertEqual(quotes[0][2],pd.Timestamp(1791244800,unit="s",tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None))
        self.assertEqual(feed.tick_count,0)

    def test_observe_never_enters_even_without_history(self):
        engine = MomentumEngine(replace(MOMENTUM_CONFIG, entry_style="observe"))
        self.assertEqual(engine.evaluate_signal("SBIN", pd.DataFrame(), pd.DataFrame()).action, "HOLD")

    def test_research_costs_reduce_cash_and_affordability(self):
        broker = ResearchBroker(initial_cash=1000)
        qty = broker.affordable_quantity(100, "BUY")
        self.assertEqual(qty, 9)
        broker.execute_market_order("SBIN", "BUY", qty, 100)
        broker.close_position("SBIN", 100)
        self.assertLess(broker.cash, 953)
        self.assertAlmostEqual(broker.cash-1000, sum(t.net_pnl for t in broker.trade_history))

    def test_stale_live_quote_blocks_entry_and_dashboard_marks_stale(self):
        with patch("quant_desk.main.UniverseManager.get_nse_top_volume", return_value=["SBIN"]):
            desk = MasterQuantDesk("live")
        desk.now_ist = lambda: pd.Timestamp("2026-10-06 11:00")
        df = pd.DataFrame({"open":500., "high":502., "low":498., "close":500., "volume":1000.},
                          index=pd.date_range(end="2026-10-06 09:55", periods=90, freq="5min"))
        desk.data_feed.aggregators["SBIN"].load_history(df)
        with patch.object(desk.momentum, "evaluate_signal", return_value=MomentumSignal("SBIN","BUY",500,501,499,30,2,499,"test")):
            desk.on_tick_update("SBIN",500,10,pd.Timestamp("2026-10-06 10:00"))
        self.assertFalse(desk.entries_allowed)
        self.assertEqual(desk.broker.open_positions, {})
        app = Dashboard()
        app.desk = desk
        data = app.snapshot()
        self.assertEqual(data["quotes"][0]["price"],500)
        self.assertEqual(data["quotes"][0]["timestamp"],"2026-10-06 10:00:00")
        json.dumps(data, default=str, allow_nan=False)

    def test_http_state_and_cross_origin_rejection(self):
        app = Dashboard()
        server = ThreadingHTTPServer(("127.0.0.1",0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(url+"/api/state") as response:
                self.assertEqual(json.load(response)["status"],"Ready")
            req = urllib.request.Request(url+"/api/start", data=b'{}',headers={"Content-Type":"application/json","Origin":"https://foreign.example"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,403)
            req = urllib.request.Request(url+"/api/start", data=b'{"mode":"real","strategy":"trend"}',headers={"Content-Type":"application/json","Origin":url})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
