"""Deterministic adversarial scenarios; synthetic data is never performance evidence."""
import contextlib
import io
import random
import unittest
import pandas as pd
from quant_desk.execution.paper_broker import PaperBroker
from quant_desk.data.data_feed import DataFeedManager


class StressInvariants(unittest.TestCase):
    def test_delayed_and_decreasing_cumulative_volume_do_not_double_count(self):
        feed=DataFeedManager(["SBIN"])
        epoch=int(pd.Timestamp("2026-10-06 10:00",tz="Asia/Kolkata").timestamp())
        for seconds,cumulative in ((0,100),(2,120),(1,110),(3,110),(4,140)):
            feed.process_live_message({"symbol":"NSE:SBIN-EQ","ltp":500,"vol_traded_today":cumulative,"exch_feed_time":epoch+seconds})
        self.assertEqual(feed.aggregators["SBIN"].get_5m_dataframe().iloc[-1].volume,40)

    def test_two_thousand_random_long_short_partial_roundtrips(self):
        rng = random.Random(5719)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2000):
                broker = PaperBroker(initial_cash=1_000_000)
                price = rng.uniform(2, 20000)
                qty = rng.randint(2,max(2,int(150000/price)))
                pos = broker.execute_market_order("TEST",rng.choice(["BUY","SELL"]),qty,price)
                self.assertIsNotNone(pos)
                marked = rng.uniform(.7,1.3)*price
                before = broker.get_portfolio_summary({"TEST":marked})
                self.assertAlmostEqual(before["total_equity"]-1_000_000,before["realized_pnl"]+before["unrealized_pnl"],places=6)
                broker.close_partial_position("TEST",rng.randint(1,qty-1),marked)
                broker.close_position("TEST",rng.uniform(.7,1.3)*price)
                self.assertAlmostEqual(broker.cash-1_000_000,sum(r.net_pnl for r in broker.trade_history),places=6)
                self.assertEqual(broker.open_positions,{})
                self.assertGreater(broker.cash,0)

    def test_bad_exit_prices_cannot_corrupt_ledger(self):
        with contextlib.redirect_stdout(io.StringIO()):
            broker=PaperBroker()
            broker.execute_market_order("SBIN","BUY",10,500)
            cash=broker.cash
            for price in (float('nan'),float('inf'),0,-1):
                self.assertIsNone(broker.close_position("SBIN",price))
                self.assertIsNone(broker.close_partial_position("SBIN",5,price))
                self.assertEqual(broker.cash,cash)
                self.assertEqual(broker.open_positions["SBIN"].quantity,10)

    def test_bad_and_out_of_order_ticks_do_not_change_candles(self):
        feed=DataFeedManager(["SBIN"])
        ts=pd.Timestamp("2026-10-06 10:00")
        feed.process_tick("SBIN",500,100,ts)
        for price,vol,stamp in ((float('nan'),10,ts),(0,10,ts),(500,-1,ts),(999,500,ts-pd.Timedelta(minutes=5))):
            feed.process_tick("SBIN",price,vol,stamp)
        self.assertEqual(feed.tick_count,1)
        frame=feed.aggregators["SBIN"].get_5m_dataframe()
        self.assertEqual(len(frame),1)
        self.assertEqual(frame.iloc[-1].close,500)
        self.assertEqual(frame.iloc[-1].volume,100)
