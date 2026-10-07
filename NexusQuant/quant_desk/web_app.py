"""Local paper dashboard: python -m quant_desk.web_app. No real-order routes."""
import argparse
import json
import threading
import time
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from quant_desk.main import MasterQuantDesk
from quant_desk.engines.research_strategies import STYLES

ROOT = Path(__file__).resolve().parent


class Dashboard:
    def __init__(self):
        self.desk = None
        self.worker = None
        self.status = "Ready"
        self.error = ""
        self.guard = threading.Lock()
        self.cancel = threading.Event()
        self.curve = []

    def start(self, mode, strategy):
        if mode not in ("live", "sim") or strategy not in (*STYLES, "observe"):
            raise ValueError("Invalid feed or strategy")
        with self.guard:
            if self.worker and self.worker.is_alive():
                raise ValueError("A session is already running; stop it before starting another")
            self.cancel.clear()
            self.desk, self.curve, self.error = None, [], ""
            self.status = "Loading NSE universe and FYERS history"
            def run():
                try:
                    from quant_desk.core.config import get_access_token
                    if not get_access_token():
                        raise ValueError("FYERS login required. Run python -m quant_desk.main --auth in PowerShell.")
                    desk = MasterQuantDesk(mode)
                    desk.momentum.config = replace(desk.momentum.config, entry_style=strategy)
                    # Isolate candidates for interpretable paper results. Original
                    # pairs remain available in the CLI; they have no validation here.
                    desk.stat_arb.config = replace(desk.stat_arb.config, enabled=False)
                    self.desk = desk
                    if self.cancel.is_set():
                        self.status = "Stopped"
                        return
                    self.status = "Running"
                    desk.start(headless=True, stop_event=self.cancel)
                    self.status = (desk.data_feed.replay_status if mode == "sim" else "Stopped")
                except (Exception, SystemExit) as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.status = "Failed"
                finally:
                    if self.cancel.is_set():
                        self.status = "Stopped"
            self.worker = threading.Thread(target=run, daemon=True)
            self.worker.start()

    def stop(self):
        self.cancel.set()
        self.status = "Stopping; waiting for any in-flight data request"
        if self.desk:
            self.desk.is_running = False
            self.desk.data_feed.is_running = False

    def snapshot(self):
        report = {"status": "unavailable"}
        try:
            report = json.loads((ROOT / "strategy_report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        result = {"status": self.status, "error": self.error, "research": report,
                  "mode": None, "positions": [], "trades": [], "quotes": [], "curve": [],
                  "summary": {}, "strategy": "observe", "running": bool(self.worker and self.worker.is_alive())}
        desk = self.desk
        if not desk:
            return result
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        with desk.lock:
            summary = desk.broker.get_portfolio_summary(desk.latest_prices)
            stamp = str(desk.data_feed.last_tick_time or "Waiting for feed")
            if not self.curve or self.curve[-1]["time"] != stamp:
                self.curve.append({"time": stamp, "equity": summary["total_equity"]})
                self.curve = self.curve[-1200:]
            positions = []
            for symbol, pos in desk.broker.open_positions.items():
                price = desk.latest_prices.get(symbol, pos.entry_price)
                positions.append({**asdict(pos), "ltp": price,
                                  "unrealized": (price-pos.entry_price)*pos.quantity*(1 if pos.side == "LONG" else -1)-pos.entry_friction})
            quotes = []
            for symbol in sorted(desk.momentum_symbols):
                ts = desk.latest_price_times.get(symbol)
                age = max(0, (now-pd.Timestamp(ts)).total_seconds()) if ts is not None else None
                signal = desk.latest_mom_signals.get(symbol)
                quotes.append({"symbol": symbol, "price": desk.latest_prices.get(symbol),
                               "timestamp": str(ts) if ts is not None else None,
                               "stale": desk.mode == "live" and (age is None or age > 60),
                               "signal": signal.action if signal else "WAIT",
                               "reason": desk.execution_status.get(symbol, signal.reason if signal else "Waiting for feed / candles")})
            result.update(mode=desk.mode, strategy=desk.momentum.config.entry_style,
                          summary=summary, positions=positions,
                          trades=[asdict(r) for r in desk.broker.trade_history[-100:]][::-1],
                          quotes=quotes, curve=list(self.curve), ticks=desk.data_feed.tick_count,
                          last_tick=stamp, feed_error=desk.data_feed.last_error,
                          history_status=desk.data_feed.history_status,
                          universe=desk.universe.status, universe_error=desk.universe_error,
                          replay={"completed": desk.data_feed.replay_completed,
                                  "total": desk.data_feed.replay_total, "status": desk.data_feed.replay_status})
        return result


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, default=str, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def trusted_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.trusted_host():
                return self.respond(403, {"error": "Local access only"})
            if self.path == "/":
                return self.respond(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self.respond(200, app.snapshot())
            return self.respond(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.trusted_host() or origin != f"http://{self.headers.get('Host')}" or self.headers.get("Content-Type") != "application/json":
                return self.respond(403, {"error": "Same-origin local JSON requests required"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2048:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/start":
                    app.start(body.get("mode"), body.get("strategy"))
                elif self.path == "/api/stop":
                    app.stop()
                else:
                    return self.respond(404, {"error": "Not found"})
                return self.respond(200, {"ok": True})
            except (ValueError, TypeError, AttributeError) as exc:
                return self.respond(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = Dashboard()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Nexus Quant dashboard: {url}\nKeep this terminal open. Ctrl+C stops the local server.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.stop()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
