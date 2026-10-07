"""NSE-origin equity universe snapshots with explicit coverage and freshness."""
import argparse
import io
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import requests


class NSEDataError(RuntimeError):
    pass


@dataclass
class NSESnapshot:
    source: str
    as_of: str
    session_date: str
    coverage: str
    global_ranking: bool
    universe_size: int
    stocks: List[Dict]
    warnings: List[str]


class NSEMarketClient:
    BASE = "https://www.nseindia.com"
    WATCH = "/api/NextApi/apiClient/marketWatchApi"
    SECTOR_INDICES = {
        "NIFTY BANK": "BANK", "NIFTY METAL": "METALS", "NIFTY PHARMA": "PHARMA",
        "NIFTY IT": "IT", "NIFTY AUTO": "AUTO", "NIFTY FMCG": "FMCG",
        "NIFTY OIL AND GAS": "ENERGY", "NIFTY REALTY": "REALTY",
        "NIFTY FIN SERVICE": "FINANCE", "NIFTY HEALTHCARE": "HEALTHCARE",
        "NIFTY CONSR DURBL": "CONSUMER", "NIFTY MEDIA": "MEDIA",
    }

    def __init__(self, session=None, timeout=15):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9",
            "Referer": self.BASE + "/market-data/live-equity-market",
        })
        self._master = None
        self.sector_map = {}

    def _get(self, url, params=None):
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise NSEDataError(f"NSE request unavailable: {url}: {type(exc).__name__}") from exc

    def _json(self, path, params=None):
        try:
            payload = self._get(self.BASE + path, params).json()
        except ValueError as exc:
            raise NSEDataError("NSE returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise NSEDataError("Unexpected NSE response shape")
        return payload

    def equity_master(self):
        if self._master is None:
            text = self._get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv").text
            try:
                frame = pd.read_csv(io.StringIO(text), skipinitialspace=True)
                frame.columns = frame.columns.str.strip()
                self._master = set(frame.loc[frame["SERIES"].str.strip().eq("EQ"), "SYMBOL"].str.strip())
            except (ValueError, KeyError) as exc:
                raise NSEDataError("Invalid NSE equity security master") from exc
            if len(self._master) < 50:
                raise NSEDataError("NSE equity master is incomplete")
        return self._master

    @staticmethod
    def _stamp(value):
        if not value:
            raise NSEDataError("NSE data has no exchange timestamp")
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
            return stamp
        except (ValueError, TypeError) as exc:
            raise NSEDataError(f"Invalid NSE timestamp: {value}") from exc

    @staticmethod
    def rank_rows(rows, master, top_n, excluded=()):
        """Rank shares traded, independent of return, sector, or index membership."""
        valid = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).strip()
            if symbol not in master or symbol in excluded or row.get("priority") == 1:
                continue
            if row.get("series", "EQ") != "EQ":
                continue
            try:
                volume = float(str(row.get("totalTradedVolume", 0)).replace(",", ""))
                price = float(str(row.get("lastPrice", 0)).replace(",", ""))
                if not math.isfinite(volume) or not math.isfinite(price) or volume <= 0 or price <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            cleaned = {"symbol": symbol, "volume": int(volume), "ltp": price,
                       "change_pct": row.get("pChange"), "sector": row.get("sector", "UNCLASSIFIED")}
            if symbol not in valid or volume > valid[symbol]["volume"]:
                valid[symbol] = cleaned
        ranked = sorted(valid.values(), key=lambda r: (-r["volume"], r["symbol"]))
        return [{"rank": i + 1, **row} for i, row in enumerate(ranked[:top_n])], len(valid)

    def daily_snapshot(self, session_date: date, top_n=50, excluded=()):
        """Exact EOD volume ranking across NSE's current EQ equity master."""
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{session_date:%d%m%Y}.csv"
        try:
            frame = pd.read_csv(io.StringIO(self._get(url).text), skipinitialspace=True)
            frame.columns = frame.columns.str.strip()
            frame = frame.loc[frame["SERIES"].str.strip().eq("EQ")]
            dates = {self._stamp(d).date() for d in frame["DATE1"].unique()}
            if dates != {session_date}:
                raise NSEDataError("NSE bhavcopy date does not match the requested session")
            rows = [{"symbol": r["SYMBOL"], "series": r["SERIES"], "lastPrice": r["CLOSE_PRICE"],
                     "totalTradedVolume": r["TTL_TRD_QNTY"],
                     "pChange": (float(r["CLOSE_PRICE"]) / float(r["PREV_CLOSE"]) - 1) * 100 if float(r["PREV_CLOSE"]) > 0 else None}
                    for r in frame.to_dict("records")]
        except (ValueError, KeyError) as exc:
            raise NSEDataError("Invalid NSE full price-volume bhavcopy") from exc
        ranked, count = self.rank_rows(rows, self.equity_master(), top_n, excluded)
        if len(ranked) < top_n:
            raise NSEDataError(f"NSE report has only {len(ranked)} eligible stocks; requested {top_n}")
        return NSESnapshot(url, f"{session_date} 15:30:00", str(session_date),
                           "NSE EQ equities in current NSE equity master; ETFs/SME/other series excluded", True, count, ranked, [])

    def index_rows(self, code):
        payload = self._json(self.WATCH, {"functionName": "getIndicesData", "symbol": code})
        nested = payload.get("data", {})
        rows = nested.get("data", []) if isinstance(nested, dict) else []
        if not rows:
            raise NSEDataError(f"NSE index snapshot unavailable: {code}")
        return rows

    def load_sectors(self):
        warnings = []
        industry_groups = {
            "Financial Services": "FINANCE", "Information Technology": "IT",
            "Healthcare": "HEALTHCARE", "Metals & Mining": "METALS",
            "Automobile and Auto Components": "AUTO", "Fast Moving Consumer Goods": "FMCG",
            "Oil Gas & Consumable Fuels": "ENERGY", "Power": "POWER",
            "Telecommunication": "TELECOM", "Capital Goods": "CAPITAL_GOODS",
            "Consumer Durables": "CONSUMER", "Consumer Services": "SERVICES",
            "Construction Materials": "CEMENT", "Construction": "INFRA",
            "Chemicals": "CHEMICALS", "Realty": "REALTY", "Services": "SERVICES",
            "Textiles": "TEXTILES", "Media Entertainment & Publication": "MEDIA",
        }
        try:
            url = "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
            frame = pd.read_csv(io.StringIO(self._get(url).text))
            for row in frame.to_dict("records"):
                industry = str(row["Industry"]).strip()
                self.sector_map[row["Symbol"]] = industry_groups.get(industry, industry.upper())
        except (NSEDataError, ValueError, KeyError) as exc:
            warnings.append(f"NSE industry classifications unavailable: {type(exc).__name__}")
        specific = {}
        for code, sector in self.SECTOR_INDICES.items():
            try:
                for row in self.index_rows(code):
                    if row.get("series") == "EQ":
                        # BANK and PHARMA take precedence over broader groups.
                        specific.setdefault(row["symbol"], sector)
            except NSEDataError as exc:
                warnings.append(str(exc))
        self.sector_map.update(specific)
        return warnings

    def live_snapshot(self, top_n=50, now=None, excluded=()):
        """Broader website snapshot; NEVER represents itself as all-market coverage."""
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        if now.tzinfo is not None:
            now = now.tz_convert("Asia/Kolkata").tz_localize(None)
        rows, warnings, sources = [], [], []
        # Main indices are not the universe restriction: these broad lists provide
        # about a thousand symbols, plus the public most-active list outside them.
        for code in ("NIFTY TOTAL MKT", "NIFTY SMALLCAP 500", "PERMITTED TO TRADE"):
            group = self.index_rows(code)
            rows.extend(group)
            sources.append(code)
        active = self._json("/api/live-analysis-most-active-securities", {"index": "volume"})
        active_rows = active.get("data", [])
        if not isinstance(active_rows, list):
            raise NSEDataError("Invalid NSE most-active response")
        rows.extend(active_rows)
        timestamps = [self._stamp(r["lastUpdateTime"]) for r in rows if r.get("series", "EQ") == "EQ" and r.get("lastUpdateTime")]
        if not timestamps:
            raise NSEDataError("NSE snapshot has no valid timestamps")
        latest = max(timestamps)
        if latest.date() != now.date() or latest > now + pd.Timedelta(minutes=5):
            raise NSEDataError(f"NSE snapshot is not today's data: {latest}")
        if time(9, 15) <= now.time() <= time(15, 30) and now - latest > pd.Timedelta(minutes=15):
            raise NSEDataError(f"NSE snapshot is stale: {latest}")
        # Discard stale per-security quotes during the session.
        fresh_rows = []
        for row in rows:
            if not row.get("lastUpdateTime"):
                continue
            stamp = self._stamp(row["lastUpdateTime"])
            if stamp.date() != now.date():
                continue
            if time(9, 15) <= now.time() <= time(15, 30) and now - stamp > pd.Timedelta(minutes=15):
                continue
            fresh_rows.append({**row, "sector": self.sector_map.get(row.get("symbol"), "UNCLASSIFIED")})
        ranked, count = self.rank_rows(fresh_rows, self.equity_master(), top_n, excluded)
        if len(ranked) < top_n:
            raise NSEDataError(f"NSE live snapshot contains only {len(ranked)} eligible stocks")
        warnings.append("Intraday ranking covers fetched NSE lists, not every NSE listing. The public most-active endpoint is capped at 20.")
        return NSESnapshot(self.BASE + self.WATCH, str(latest), str(latest.date()),
                           "; ".join(sources) + "; NSE Most Active", False, count, ranked, warnings)

    def select_snapshot(self, top_n=50, now=None, excluded=()):
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        if now.tzinfo is not None:
            now = now.tz_convert("Asia/Kolkata").tz_localize(None)
        if now.time() >= time(15, 30):
            try:
                return self.daily_snapshot(now.date(), top_n, excluded)
            except NSEDataError:
                pass  # Published full report may not be ready immediately.
        return self.live_snapshot(top_n, now, excluded)


def main():
    parser = argparse.ArgumentParser(description="Download an NSE top-volume universe snapshot")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=Path("quant_desk/nse_universe_snapshot.json"))
    args = parser.parse_args()
    client = NSEMarketClient()
    snapshot = client.daily_snapshot(args.date, args.top) if args.date else client.select_snapshot(args.top)
    snapshot.warnings.extend(client.load_sectors())
    for row in snapshot.stocks:
        row["sector"] = client.sector_map.get(row["symbol"], "UNCLASSIFIED")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
    pd.DataFrame(snapshot.stocks).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"NSE {snapshot.session_date}: {len(snapshot.stocks)} ranked stocks from {snapshot.universe_size} equities")
    print(f"Coverage: {snapshot.coverage}; all-market ranking: {snapshot.global_ranking}")
    print(pd.DataFrame(snapshot.stocks).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
