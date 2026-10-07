import gc
import math
import queue
import threading
import time
import logging
import collections
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional

import pandas as pd
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from quant_desk.data.universe import UniverseManager
from quant_desk.core.config import FYERS_CONFIG, get_access_token

logger = logging.getLogger(__name__)

# Worker queue max depth — oldest ticks dropped when strategy falls behind
_QUEUE_MAX = 4096


class CandleAggregator:
    """Per-symbol OHLCV candle builder backed by a deque ring buffer (O(1) append)."""

    def __init__(self, symbol: str, buffer_size: int = 500):
        self.symbol = UniverseManager.to_clean_symbol(symbol)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

        # Deque ring buffer: O(1) append, automatic eviction, no unbounded growth
        self.ticks: collections.deque = collections.deque(maxlen=2000)
        self._df_5m = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "pv"])
        self._df_15m = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "pv"])

    def add_tick(self, ltp: float, volume: float, timestamp: Optional[datetime] = None) -> None:
        # Validate before touching any data structure
        if not math.isfinite(ltp) or ltp <= 0:
            return
        if not math.isfinite(volume) or volume < 0:
            return
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime()
        with self.lock:
            self.ticks.append({"timestamp": ts, "price": ltp, "volume": volume})
            self._update_candles(ts, ltp, volume)

    def _update_candles(self, ts: datetime, price: float, volume: float) -> None:
        ts_5m = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        ts_15m = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
        
        self._df_5m = self._fast_update_df(self._df_5m, ts_5m, price, volume)
        self._df_15m = self._fast_update_df(self._df_15m, ts_15m, price, volume)

    def _fast_update_df(self, df: pd.DataFrame, ts: datetime, price: float, volume: float) -> pd.DataFrame:
        if df.empty:
            pv = price * volume
            return pd.DataFrame(
                [{"open": price, "high": price, "low": price, "close": price, "volume": volume, "vwap": price, "pv": pv}],
                index=pd.DatetimeIndex([ts], name="datetime")
            )
        
        if df.index[-1] == ts:
            last_idx = df.index[-1]
            h = max(float(df.at[last_idx, "high"]), price)
            l = min(float(df.at[last_idx, "low"]), price)
            vol = float(df.at[last_idx, "volume"]) + volume
            prev_pv = float(df.at[last_idx, "pv"])
            new_pv = prev_pv + (price * volume)
            vwap = new_pv / vol if vol > 0 else price
            df.loc[last_idx, ["high", "low", "close", "volume", "vwap", "pv"]] = [h, l, price, vol, vwap, new_pv]
            return df
        elif ts in df.index:
            h = max(float(df.at[ts, "high"]), price)
            l = min(float(df.at[ts, "low"]), price)
            vol = float(df.at[ts, "volume"]) + volume
            prev_pv = float(df.at[ts, "pv"])
            new_pv = prev_pv + (price * volume)
            vwap = new_pv / vol if vol > 0 else price
            df.loc[ts, ["high", "low", "close", "volume", "vwap", "pv"]] = [h, l, price, vol, vwap, new_pv]
            return df
        else:
            pv = price * volume
            new_row = pd.DataFrame(
                [{"open": price, "high": price, "low": price, "close": price, "volume": volume, "vwap": price, "pv": pv}],
                index=pd.DatetimeIndex([ts], name="datetime")
            )
            updated = pd.concat([df, new_row])
            if len(updated) > self.buffer_size:
                updated = updated.iloc[-self.buffer_size:]
            return updated

    def get_5m_dataframe(self) -> pd.DataFrame:
        with self.lock:
            return self._df_5m.copy()

    def get_15m_dataframe(self) -> pd.DataFrame:
        with self.lock:
            return self._df_15m.copy()

    def load_history(self, df: pd.DataFrame) -> None:
        """Seed distinct OHLCV timeframes using exchange timestamps in IST."""
        with self.lock:
            df = df.sort_index().copy().astype(float)
            df = df[~df.index.duplicated(keep="last")]
            df["pv"] = df["close"] * df["volume"]
            df["vwap"] = df["close"]
            df_5m = df.iloc[-self.buffer_size:].copy()
            fifteen = df.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum", "pv": "sum",
            }).dropna(subset=["close"])
            fifteen["vwap"] = (fifteen["pv"] / fifteen["volume"].replace(0, float("nan"))).fillna(fifteen["close"])
            df_15m = fifteen.iloc[-self.buffer_size:]
            self._df_5m = df_5m
            self._df_15m = df_15m


class DataFeedManager:
    """Manages live WebSocket streams and FYERS historical replay.

    Strategy callbacks are dispatched from a dedicated worker thread so the
    websocket receive loop is never blocked by indicator computation.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = [UniverseManager.to_clean_symbol(s) for s in symbols]
        self.aggregators: Dict[str, CandleAggregator] = {s: CandleAggregator(s) for s in self.symbols}
        self.on_tick_callbacks: List[Callable[[str, float, float, datetime], None]] = []
        self.is_running = False
        self.socket_thread: Optional[threading.Thread] = None
        self.ws = None
        self.tick_count = 0
        self.callback_errors = 0
        self.last_error = ""
        self.last_tick_time = None
        self._cumulative_volumes: Dict[str, tuple] = {}
        self._last_symbol_time: Dict[str, datetime] = {}
        self.replay_status = "Not started"
        self.replay_total = 0
        self.replay_completed = 0
        self.history_status = "Not started"
        self._last_history_request = 0.0
        self._history_lock = threading.Lock()

        # Worker queue: websocket pushes ticks; worker drains and calls strategies
        self._tick_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

    # ------------------------------------------------------------------
    # Worker thread — strategy callbacks run here, not on the socket thread
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="tick-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                item = self._tick_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            sym, price, volume, ts = item
            for cb in self.on_tick_callbacks:
                try:
                    cb(sym, price, volume, ts)
                except Exception as exc:
                    self.callback_errors += 1
                    self.last_error = f"{sym}: {type(exc).__name__}: {exc}"
                    logger.exception("Strategy callback failed for %s", sym)
            self._tick_queue.task_done()

    def _stop_worker(self) -> None:
        self._worker_stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _history_request(self, client, payload):
        """Pace bulk downloads and retry a rate-limited request once."""
        with self._history_lock:
            time.sleep(max(0, 1.1 - (time.monotonic() - self._last_history_request)))
            self._last_history_request = time.monotonic()
            response = client.history(data=payload)
            if response.get("code") == 429 or "limit" in str(response.get("message", "")).lower():
                time.sleep(3)
                self._last_history_request = time.monotonic()
                response = client.history(data=payload)
            return response

    @staticmethod
    def _parse_fyers_candles(candles) -> pd.DataFrame:
        """Parse raw FYERS candle list into a clean, deduplicated IST DataFrame.

        FYERS occasionally returns duplicate minute bars which cause
        ``df.loc[ts]`` to return a DataFrame instead of a Series, crashing
        any downstream ``float()`` conversion.  This method eliminates
        duplicates and invalid rows before data touches any strategy.
        """
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(candles, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
        df.set_index("datetime", inplace=True)
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
        # Deduplicate: keep last bar for each timestamp
        df = df[~df.index.duplicated(keep="last")]
        # Drop bars with zero/negative/non-finite prices (FYERS anomalies)
        df = df[(df["close"] > 0) & (df["open"] > 0) & df["close"].apply(math.isfinite)]
        return df

    def fetch_historical_daily(self, days: int = 180) -> pd.DataFrame:
        """Fetch daily closing prices for cointegration tests via FYERS API."""
        print(f"[DataFeed] Fetching daily FYERS exchange data for pairs calibration ({days} days)...")
        access_token = get_access_token()
        fyers = fyersModel.FyersModel(
            client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path=""
        )
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)
        all_prices = {}

        for sym in self.symbols:
            try:
                response = self._history_request(fyers, {
                    "symbol": UniverseManager.to_fyers_symbol(sym),
                    "resolution": "D", "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1",
                })
                if response.get("s") == "ok" and response.get("candles"):
                    df = self._parse_fyers_candles(response["candles"])
                    all_prices[sym] = df["close"]
            except Exception as exc:
                logger.warning("FYERS daily backfill warning for %s: %s", sym, exc)

        prices_df = pd.DataFrame(all_prices).dropna(how="all")
        if not prices_df.empty:
            prices_df = prices_df.loc[prices_df.index.date < datetime.now().date()]
        return prices_df.iloc[-days:] if not prices_df.empty else prices_df

    def fetch_quote_snapshot(self):
        """Last available FYERS quotes; exchange timestamps are never made current."""
        client = fyersModel.FyersModel(client_id=FYERS_CONFIG.client_id, token=get_access_token(), is_async=False, log_path="")
        quotes = []
        for offset in range(0, len(self.symbols), 50):
            response = client.quotes({"symbols": ",".join(UniverseManager.to_fyers_symbol(s) for s in self.symbols[offset:offset+50])})
            if response.get("s") != "ok":
                self.last_error = f"FYERS quote snapshot rejected (code {response.get('code')})"
                continue
            for item in response.get("d", []):
                value = item.get("v", {})
                try:
                    price = float(value.get("lp", 0))
                    epoch = float(value.get("tt", 0))
                    if price <= 0 or epoch <= 0 or not math.isfinite(price):
                        continue
                    stamp = pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None)
                    quotes.append((UniverseManager.to_clean_symbol(item["n"]), price, stamp))
                except (ValueError, TypeError, KeyError):
                    continue
        return quotes

    def backfill_intraday_candles(self, days: int = 3, symbols: Optional[List[str]] = None) -> None:
        print("[DataFeed] Pre-filling indicator buffers from FYERS servers...")
        access_token = get_access_token()
        if not access_token:
            return
        fyers = fyersModel.FyersModel(
            client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path=""
        )
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        pending = symbols if symbols is not None else self.symbols

        for number, sym in enumerate(pending, 1):
            self.history_status = f"Loading candles {number}/{len(pending)}: {sym}"
            try:
                response = self._history_request(fyers, {
                    "symbol": UniverseManager.to_fyers_symbol(sym),
                    "resolution": "5", "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1",
                })
                if response.get("s") == "ok" and response.get("candles"):
                    df = self._parse_fyers_candles(response["candles"])
                    self.aggregators[sym].load_history(df)
                else:
                    logger.warning("Intraday history unavailable for %s: %s", sym,
                                   response.get("message", response.get("s")))
            except Exception as exc:
                logger.warning("FYERS intraday warning for %s: %s", sym, exc)

        self.history_status = "History requests complete; missing symbols wait for sufficient candles"

    def register_tick_callback(self, callback: Callable[[str, float, float, datetime], None]) -> None:
        self.on_tick_callbacks.append(callback)

    def add_symbols(self, symbols: List[str]) -> None:
        """Warm new symbols before subscribing; existing/open symbols stay watched."""
        additions = [UniverseManager.to_clean_symbol(s) for s in symbols
                     if UniverseManager.to_clean_symbol(s) not in self.aggregators]
        if not additions:
            return
        for symbol in additions:
            self.aggregators[symbol] = CandleAggregator(symbol)
        self.symbols.extend(additions)
        self.backfill_intraday_candles(days=10, symbols=additions)
        if self.ws:
            self.ws.subscribe(symbols=[UniverseManager.to_fyers_symbol(s) for s in additions], data_type="SymbolUpdate")

    def process_tick(self, symbol: str, price: float, volume: float,
                     ts: Optional[datetime] = None) -> None:
        """Validate, stamp, and dispatch a single tick.

        Anomalous ticks (price <= 0, non-finite, out-of-order) are silently
        dropped before touching any indicator state.
        """
        try:
            price = float(price)
            volume = float(volume)
        except (TypeError, ValueError):
            return
        if not math.isfinite(price) or price <= 0:
            return
        if not math.isfinite(volume) or volume < 0:
            return

        clean_sym = UniverseManager.to_clean_symbol(symbol)
        now = ts or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime()
        if now.tzinfo is not None:
            now = pd.Timestamp(now).tz_convert("Asia/Kolkata").tz_localize(None).to_pydatetime()
        if pd.isna(now):
            return

        previous = self._last_symbol_time.get(clean_sym)
        if previous is not None and now < previous:
            return
        self._last_symbol_time[clean_sym] = now

        if clean_sym in self.aggregators:
            self.aggregators[clean_sym].add_tick(price, volume, now)

        self.tick_count += 1
        self.last_tick_time = now

        # Live mode: dispatch via worker queue (non-blocking)
        # Sim mode: call callbacks inline (single-threaded replay)
        item = (clean_sym, price, volume, now)
        if self._worker_thread and self._worker_thread.is_alive():
            try:
                self._tick_queue.put_nowait(item)
            except queue.Full:
                try:
                    self._tick_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._tick_queue.put_nowait(item)
                except queue.Full:
                    pass
        else:
            for cb in self.on_tick_callbacks:
                try:
                    cb(clean_sym, price, volume, now)
                except Exception as exc:
                    self.callback_errors += 1
                    self.last_error = f"{clean_sym}: {type(exc).__name__}: {exc}"
                    logger.exception("Strategy callback failed for %s", clean_sym)

    def process_live_message(self, msg) -> None:
        """Normalize FYERS cumulative volume to per-update traded volume."""
        if not isinstance(msg, dict):
            return
        symbol = msg.get("symbol") or msg.get("name") or msg.get("n")
        price = msg.get("ltp", msg.get("lp", msg.get("last_price")))
        if symbol is None or price is None:
            return
        sym = UniverseManager.to_clean_symbol(str(symbol))
        epoch = msg.get("exch_feed_time")
        ts = (pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None).to_pydatetime()
              if epoch else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime())
        previous_time = self._last_symbol_time.get(sym)
        if previous_time is not None and ts < previous_time:
            return
        if not math.isfinite(float(price)) or float(price) <= 0:
            return
        if "vol_traded_today" in msg:
            cumulative = float(msg["vol_traded_today"])
            if not math.isfinite(cumulative) or cumulative < 0:
                return
            previous = self._cumulative_volumes.get(sym)
            volume = max(0.0, cumulative - previous[1]) if previous and previous[0] == ts.date() else 0.0
            self._cumulative_volumes[sym] = (ts.date(), max(cumulative, previous[1]) if previous and previous[0] == ts.date() else cumulative)
        else:
            volume = float(msg.get("v") or msg.get("volume") or msg.get("vol") or 0)
        self.process_tick(sym, float(price), volume, ts)

    def start_live_websocket(self) -> None:
        print("[DataFeed] Initializing Live WebSockets...")
        self._start_worker()  # Strategy callbacks run on worker thread, not socket thread
        access_token = get_access_token()

        def onmessage(message):
            try:
                if isinstance(message, list):
                    for item in message:
                        self.process_live_message(item)
                elif isinstance(message, dict):
                    self.process_live_message(message)
                    if "feeds" in message and isinstance(message["feeds"], dict):
                        for item in message["feeds"].values():
                            self.process_live_message(item)
            except Exception as exc:
                self.last_error = f"WebSocket message: {type(exc).__name__}: {exc}"
                logger.exception("Invalid WebSocket message")

        def onerror(message):
            self.last_error = str(message)
            print(f"[WebSocket Error]: {message}")

        def onclose(message):
            print(f"[WebSocket Closed]: {message}")

        def onopen():
            print(f"[DataFeed] WebSocket Connected! Subscribing to {len(self.symbols)} live stocks...")
            if self.ws:
                try:
                    self.ws.subscribe(
                        symbols=[UniverseManager.to_fyers_symbol(s) for s in self.symbols],
                        data_type="SymbolUpdate"
                    )
                except Exception as exc:
                    print(f"[WebSocket Subscription Error]: {exc}")

        self.ws = data_ws.FyersDataSocket(
            access_token=f"{FYERS_CONFIG.client_id}:{access_token}",
            log_path="", litemode=False, write_to_file=False, reconnect=True,
            on_connect=onopen, on_close=onclose, on_error=onerror, on_message=onmessage,
        )
        self.is_running = True
        self.socket_thread = threading.Thread(target=self.ws.connect, daemon=True)
        self.socket_thread.start()

    def start_simulation_feed(self, tick_interval_sec: float = 0.2) -> None:
        self.is_running = True
        self.replay_status = "Downloading history"
        self.replay_completed = 0
        self.replay_total = 0
        print("[DataFeed] Connecting to FYERS for historical 1-min playback...")

        access_token = get_access_token()
        if not access_token:
            print("[DataFeed] ERROR: No FYERS access token. Run with --auth first.")
            self.is_running = False
            self.replay_status = "Failed: authenticate with --auth first"
            return

        fyers = fyersModel.FyersModel(
            client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path=""
        )
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)  # 10 days for better indicator warm-up
        hist_data: Dict[str, pd.DataFrame] = {}

        for number, sym in enumerate(self.symbols, 1):
            print(f"[DataFeed] Replay history {number}/{len(self.symbols)}: {sym}", flush=True)
            try:
                response = self._history_request(fyers, {
                    "symbol": UniverseManager.to_fyers_symbol(sym),
                    "resolution": "1", "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1",
                })
                if response.get("s") == "ok" and response.get("candles"):
                    # _parse_fyers_candles deduplicates — guarantees loc[ts] returns a Series
                    df = self._parse_fyers_candles(response["candles"])
                    if not df.empty:
                        hist_data[sym] = df
                    else:
                        print(f"[DataFeed] No valid candles after dedup/validation for {sym}")
                else:
                    print(f"[DataFeed] No replay candles for {sym} (status: {response.get('s', 'unknown')})")
            except Exception as exc:
                print(f"[DataFeed] Failed to download FYERS data for {sym}: {exc}")

        def _sim_loop():
            if not hist_data:
                print("[DataFeed] No historical data found. Halting simulation.")
                self.is_running = False
                self.replay_status = "Failed: no historical candles returned"
                return

            all_timestamps: set = set()
            for df in hist_data.values():
                all_timestamps.update(df.index.tolist())
            common_index = sorted(all_timestamps)

            # Pre-seed aggregators with historical candle data for instant indicator warm-up
            for sym, df in hist_data.items():
                if sym in self.aggregators:
                    self.aggregators[sym].load_history(df)

            self.replay_total = len(common_index)
            self.replay_status = "Running (indicators pre-warmed)"
            print(
                f"[DataFeed] Replaying {len(common_index)} minute timestamps across "
                f"{len(hist_data)} stocks; processing time is additional to playback delay."
            )

            last_gc_hour = -1
            for ts in common_index:
                if not self.is_running:
                    break
                for sym in self.symbols:
                    if sym not in hist_data or ts not in hist_data[sym].index:
                        continue
                    row = hist_data[sym].loc[ts]
                    # After dedup, loc[ts] always returns a Series — guard for safety
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    self.process_tick(sym, float(row["close"]), float(row["volume"]), ts=ts)
                time.sleep(tick_interval_sec)
                self.replay_completed += 1
                # Hourly GC: release closed-position dicts and stale indicator frames
                if hasattr(ts, "hour") and ts.hour != last_gc_hour:
                    gc.collect()
                    last_gc_hour = ts.hour

            print("[DataFeed] Simulation playback completed.")
            self.replay_status = "Completed" if self.replay_completed == self.replay_total else "Stopped"
            self.is_running = False

        def guarded_replay():
            try:
                _sim_loop()
            except Exception as exc:
                self.last_error = f"Replay failed: {type(exc).__name__}: {exc}"
                self.replay_status = "Failed"
                logger.exception("Historical replay failed")
            finally:
                self.is_running = False

        self.socket_thread = threading.Thread(target=guarded_replay, daemon=True)
        self.socket_thread.start()

    def stop(self) -> None:
        self.is_running = False
        self._stop_worker()
        if self.ws:
            ws_conn = self.ws
            self.ws = None
            def safe_close():
                try:
                    ws_conn.close_connection()
                except Exception as exc:
                    logger.warning("WebSocket close failed: %s", type(exc).__name__)
            close_thread = threading.Thread(target=safe_close, daemon=True)
            close_thread.start()
            close_thread.join(timeout=0.5)
        print("[DataFeed] Data Feed stopped.")
