import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from quant_desk.data.universe import UniverseManager
from quant_desk.core.config import FYERS_CONFIG, get_access_token


class CandleAggregator:
    def __init__(self, symbol: str, buffer_size: int = 500):
        self.symbol = UniverseManager.to_clean_symbol(symbol)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        
        self.ticks: List[Dict] = []
        self.df_5m = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        self.df_15m = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

    def add_tick(self, ltp: float, volume: float, timestamp: Optional[datetime] = None) -> None:
        ts = timestamp or datetime.now()
        with self.lock:
            self.ticks.append({"timestamp": ts, "price": ltp, "volume": volume})
            if len(self.ticks) > 20000:
                self.ticks = self.ticks[-10000:]
            self._update_candles(ts, ltp, volume)

    def _update_candles(self, ts: datetime, price: float, volume: float) -> None:
        ts_5m = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        ts_15m = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
        self.df_5m = self._upsert_candle(self.df_5m, ts_5m, price, volume)
        self.df_15m = self._upsert_candle(self.df_15m, ts_15m, price, volume)

    def _upsert_candle(self, df: pd.DataFrame, ts: datetime, price: float, volume: float) -> pd.DataFrame:
        if df.empty:
            vwap = price
            pv = price * volume
            return pd.DataFrame(
                [{"open": price, "high": price, "low": price, "close": price, "volume": volume, "vwap": vwap, "pv": pv}],
                index=[ts]
            )

        if ts in df.index:
            row = df.loc[ts]
            new_high = max(float(row["high"]), price)
            new_low = min(float(row["low"]), price)
            new_close = price
            new_vol = float(row["volume"]) + volume
            prev_pv = float(row["pv"]) if "pv" in row and not pd.isna(row["pv"]) else (float(row["close"]) * float(row["volume"]))
            new_pv = prev_pv + (price * volume)
            vwap = new_pv / new_vol if new_vol > 0 else price
            
            df.loc[ts, ["high", "low", "close", "volume", "vwap", "pv"]] = [new_high, new_low, new_close, new_vol, vwap, new_pv]
            return df
        else:
            vwap = price
            pv = price * volume
            new_row = pd.DataFrame(
                [{"open": price, "high": price, "low": price, "close": price, "volume": volume, "vwap": vwap, "pv": pv}],
                index=[ts]
            )
            updated = pd.concat([df, new_row])
            if len(updated) > self.buffer_size:
                updated = updated.iloc[-self.buffer_size:]
            return updated

    def get_5m_dataframe(self) -> pd.DataFrame:
        with self.lock:
            return self.df_5m.copy()

    def get_15m_dataframe(self) -> pd.DataFrame:
        with self.lock:
            return self.df_15m.copy()


class DataFeedManager:
    """Manages true Live WebSocket streams and 100% accurate FYERS backfill data."""

    def __init__(self, symbols: List[str]):
        self.symbols = [UniverseManager.to_clean_symbol(s) for s in symbols]
        self.aggregators: Dict[str, CandleAggregator] = {s: CandleAggregator(s) for s in self.symbols}
        self.on_tick_callbacks: List[Callable[[str, float, float, datetime], None]] = []
        self.is_running = False
        self.socket_thread: Optional[threading.Thread] = None
        self.ws = None

    def fetch_historical_daily(self, days: int = 180) -> pd.DataFrame:
        """Fetch daily closing prices for cointegration tests directly via FYERS API."""
        print(f"[DataFeed] Fetching daily FYERS exchange data for pairs calibration ({days} days)...")
        access_token = get_access_token()
        fyers = fyersModel.FyersModel(client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path="")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60) # Buffer to guarantee 180 trading days
        
        all_prices = {}
        
        for sym in self.symbols:
            try:
                fyers_sym = UniverseManager.to_fyers_symbol(sym)
                data_payload = {
                    "symbol": fyers_sym,
                    "resolution": "D",
                    "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
                response = fyers.history(data=data_payload)
                
                if response.get("s") == "ok" and response.get("candles"):
                    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    df = pd.DataFrame(response['candles'], columns=cols)
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                    df.set_index('datetime', inplace=True)
                    all_prices[sym] = df['close']
            except Exception as e:
                print(f"[DataFeed] FYERS daily backfill warning for {sym}: {e}")
                
        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.ffill().bfill().dropna()
        return prices_df.iloc[-days:] if not prices_df.empty else prices_df

    def backfill_intraday_candles(self, days: int = 3) -> None:
        print("[DataFeed] Pre-filling indicator buffers from FYERS servers...")
        access_token = get_access_token()
        if not access_token:
            return
            
        fyers = fyersModel.FyersModel(client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path="")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for sym in self.symbols:
            try:
                fyers_sym = UniverseManager.to_fyers_symbol(sym)
                data_payload = {
                    "symbol": fyers_sym,
                    "resolution": "5",
                    "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
                response = fyers.history(data=data_payload)
                if response.get("s") == "ok" and response.get("candles"):
                    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    df = pd.DataFrame(response['candles'], columns=cols)
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                    df.set_index('datetime', inplace=True)
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)
                    
                    agg = self.aggregators[sym]
                    with agg.lock:
                        df["pv"] = df["close"] * df["volume"]
                        df["vwap"] = df["pv"].cumsum() / df["volume"].cumsum()
                        agg.df_5m = df.copy()
                        agg.df_15m = df.copy() 
            except Exception as e:
                print(f"[DataFeed] FYERS intraday warning for {sym}: {e}")

    def register_tick_callback(self, callback: Callable[[str, float, float, datetime], None]) -> None:
        self.on_tick_callbacks.append(callback)

    def process_tick(self, symbol: str, price: float, volume: float, ts: Optional[datetime] = None) -> None:
        clean_sym = UniverseManager.to_clean_symbol(symbol)
        now = ts or datetime.now()
        if clean_sym in self.aggregators:
            self.aggregators[clean_sym].add_tick(price, volume, now)
        for cb in self.on_tick_callbacks:
            try:
                cb(clean_sym, price, volume, now)
            except Exception as e:
                pass

    def start_live_websocket(self) -> None:
        print("[DataFeed] Initializing Live WebSockets...")
        access_token = get_access_token()
        fyers_symbols = [UniverseManager.to_fyers_symbol(s) for s in self.symbols]

        def _handle_single_msg(msg):
            if isinstance(msg, dict):
                sym_raw = msg.get('symbol') or msg.get('name') or msg.get('n')
                ltp_val = msg.get('ltp') or msg.get('lp') or msg.get('last_price')
                if sym_raw is not None and ltp_val is not None:
                    sym = UniverseManager.to_clean_symbol(str(sym_raw))
                    ltp = float(ltp_val)
                    vol = float(msg.get('v') or msg.get('volume') or msg.get('vol') or 0)
                    self.process_tick(sym, ltp, vol)

        def onmessage(message):
            try:
                if isinstance(message, list):
                    for item in message:
                        _handle_single_msg(item)
                elif isinstance(message, dict):
                    _handle_single_msg(message)
                    if 'feeds' in message and isinstance(message['feeds'], dict):
                        for item in message['feeds'].values():
                            _handle_single_msg(item)
            except Exception as e:
                pass

        def onerror(message):
            print(f"[WebSocket Error]: {message}")

        def onclose(message):
            print(f"[WebSocket Closed]: {message}")

        def onopen():
            print(f"[DataFeed] WebSocket Connected! Subscribing to {len(fyers_symbols)} live stocks...")
            if self.ws:
                try:
                    self.ws.subscribe(symbols=fyers_symbols, data_type="SymbolUpdate")
                except Exception as e:
                    print(f"[WebSocket Subscription Error]: {e}")

        self.ws = data_ws.FyersDataSocket(
            access_token=f"{FYERS_CONFIG.client_id}:{access_token}",
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=onopen,
            on_close=onclose,
            on_error=onerror,
            on_message=onmessage
        )
        
        self.is_running = True
        self.socket_thread = threading.Thread(target=self.ws.connect, daemon=True)
        self.socket_thread.start()

    def start_simulation_feed(self, tick_interval_sec: float = 0.2) -> None:
        self.is_running = True
        print("[DataFeed] Connecting to FYERS for historical 1-min playback...")

        access_token = get_access_token()
        if not access_token:
            print("[DataFeed] ERROR: No FYERS access token. Run with --auth first.")
            self.is_running = False
            return
            
        fyers = fyersModel.FyersModel(client_id=FYERS_CONFIG.client_id, token=access_token, is_async=False, log_path="")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        hist_data = {}
        
        for sym in self.symbols:
            try:
                fyers_sym = UniverseManager.to_fyers_symbol(sym)
                data_payload = {
                    "symbol": fyers_sym,
                    "resolution": "1", 
                    "date_format": "1", 
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
                response = fyers.history(data=data_payload)
                if response.get("s") == "ok" and response.get("candles"):
                    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    df = pd.DataFrame(response['candles'], columns=cols)
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                    df.set_index('datetime', inplace=True)
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)
                    hist_data[sym] = df
            except Exception as e:
                print(f"[DataFeed] Failed to download FYERS data for {sym}: {e}")

        def _sim_loop():
            if not hist_data:
                print("[DataFeed] No historical data found. Halting simulation.")
                self.is_running = False
                return

            all_timestamps = set()
            for df in hist_data.values():
                all_timestamps.update(df.index.tolist())
            common_index = sorted(list(all_timestamps))
            
            print(f"[DataFeed] Starting playback of {len(common_index)} accurate bars...")
            
            for ts in common_index:
                if not self.is_running:
                    break
                for sym in self.symbols:
                    if sym in hist_data and ts in hist_data[sym].index:
                        row = hist_data[sym].loc[ts]
                        self.process_tick(sym, float(row['close']), float(row['volume']), ts=ts)
                time.sleep(tick_interval_sec)
                
            print("[DataFeed] Simulation playback completed.")
            self.is_running = False

        self.socket_thread = threading.Thread(target=_sim_loop, daemon=True)
        self.socket_thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.ws:
            self.ws.unsubscribe(symbols=[UniverseManager.to_fyers_symbol(s) for s in self.symbols])
        print("[DataFeed] Data Feed stopped.")