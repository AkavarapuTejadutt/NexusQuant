from typing import List, Tuple, Optional
from datetime import datetime, timedelta

from quant_desk.data.nse_market import NSEMarketClient, NSEDataError
from pathlib import Path
from dataclasses import asdict
import json
import itertools
import pandas as pd

# Core Banking Lists
NIFTY_BANK_SYMBOLS = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "PNB", "BANKBARODA", "INDUSINDBK", "AUBANK", "IDFCFIRSTB"
]

BANK_PAIRS = [
    ("HDFCBANK", "ICICIBANK"),
    ("SBIN", "PNB"),
    ("SBIN", "BANKBARODA"),
    ("AXISBANK", "KOTAKBANK"),
    ("ICICIBANK", "AXISBANK")
]

# Standard Nifty 50 Lists
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", 
    "BHARTIARTL", "BAJFINANCE", "LT", "HINDUNILVR", "AXISBANK", "KOTAKBANK", 
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "TATAMOTORS", "NTPC", 
    "TATASTEEL", "POWERGRID", "ASIANPAINT", "BAJAJFINSV", "ADANIENT", 
    "M&M", "HCLTECH", "ADANIPORTS", "WIPRO", "ONGC", "JSWSTEEL", 
    "GRASIM", "COALINDIA", "HINDALCO", "TECHM", "DRREDDY"
]

NIFTY_PAIRS = [
    ("HDFCBANK", "ICICIBANK"), ("TCS", "INFY"), ("RELIANCE", "ONGC"), 
    ("TATASTEEL", "JSWSTEEL"), ("AXISBANK", "KOTAKBANK")
]

# Broad Liquid F&O Universe for Dynamic Scanners
BROAD_FO_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL", 
    "BAJFINANCE", "LT", "AXISBANK", "KOTAKBANK", "TATAMOTORS", "SUNPHARMA", "NTPC", 
    "TATASTEEL", "POWERGRID", "ASIANPAINT", "ADANIENT", "M&M", "HCLTECH", "WIPRO", 
    "ONGC", "JSWSTEEL", "HINDALCO", "COALINDIA", "TECHM", "DRREDDY", "PNB", "BANKBARODA", 
    "INDUSINDBK", "AUBANK", "TVSMOTOR", "BAJAJ-AUTO", "EICHERMOT", "DIVISLAB", "CIPLA", 
    "APOLLOHOSP", "ZEEL", "VEDL", "AMBUJACEM", "DLF", "GAIL", "BHEL", "HAL", "BEL", 
    "IRCTC", "PFC", "RECLTD", "TRENT", "CHOLAFIN", "INDIGO", "ZOMATO"
]


SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "TECHM": "IT", "HCLTECH": "IT",
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "AXISBANK": "BANK",
    "KOTAKBANK": "BANK", "PNB": "BANK", "BANKBARODA": "BANK", "INDUSINDBK": "BANK",
    "AUBANK": "BANK", "IDFCFIRSTB": "BANK",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "POWER", "POWERGRID": "POWER",
    "COALINDIA": "ENERGY", "GAIL": "ENERGY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS", "VEDL": "METALS",
    "TATAMOTORS": "AUTO", "MARUTI": "AUTO", "M&M": "AUTO", "TVSMOTOR": "AUTO",
    "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA", "CIPLA": "PHARMA",
    "APOLLOHOSP": "HEALTHCARE",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "ASIANPAINT": "FMCG", "TITAN": "CONSUMER",
    "BHARTIARTL": "TELECOM", "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "CHOLAFIN": "FINANCE", "PFC": "FINANCE", "RECLTD": "FINANCE",
    "LT": "INFRA", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA", "ULTRACEMCO": "CEMENT",
    "AMBUJACEM": "CEMENT", "DLF": "REALTY", "GRASIM": "CONGLOMERATE",
    "BHEL": "CAPITAL_GOODS", "HAL": "DEFENCE", "BEL": "DEFENCE", "IRCTC": "SERVICES",
    "TRENT": "RETAIL", "INDIGO": "AVIATION", "ZOMATO": "TECH", "ZEEL": "MEDIA"
}


class UniverseManager:
    """Selects stocks from NSE and maps symbols for FYERS market data."""

    def __init__(self, screener_csv_path: Optional[str] = None, nse_client=None):
        self.screener_csv_path = screener_csv_path
        self.excluded_symbols = set()
        self.fo_ban_list = set()
        self.nse_client = nse_client or NSEMarketClient()
        self.latest_snapshot = None
        self.status = "NSE universe not loaded"

    @classmethod
    def get_sector(cls, symbol: str) -> str:
        clean = cls.to_clean_symbol(symbol)
        return SECTOR_MAP.get(clean, "MISC")

    def get_nse_top_volume(self, top_n: int = 50, now=None) -> List[str]:
        """Today's share-volume leaders from NSE; never substitute a static list."""
        snapshot = self.nse_client.select_snapshot(top_n, now, self.excluded_symbols)
        if not self.nse_client.sector_map:
            snapshot.warnings.extend(self.nse_client.load_sectors())
        for stock in snapshot.stocks:
            sector = self.nse_client.sector_map.get(stock["symbol"])
            if sector:
                SECTOR_MAP[stock["symbol"]] = sector
                stock["sector"] = sector
        self.latest_snapshot = snapshot
        qualifier = "full EQ ranking" if snapshot.global_ranking else "partial intraday coverage"
        self.status = f"NSE | {snapshot.session_date} | {len(snapshot.stocks)} stocks | {snapshot.universe_size} equities covered | {qualifier}"
        output = Path(__file__).resolve().parent.parent / "nse_universe_snapshot.json"
        output.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
        pd.DataFrame(snapshot.stocks).to_csv(output.with_suffix(".csv"), index=False)
        print(f"[UniverseManager] {self.status}")
        return [s["symbol"] for s in snapshot.stocks]

    def discover_pairs(self, daily_prices: pd.DataFrame, symbols: List[str], max_pairs=5):
        """Generate same-sector candidates; cointegration still controls admission."""
        candidates = []
        for a, b in itertools.combinations(symbols, 2):
            sector = self.get_sector(a)
            if sector in ("MISC", "UNCLASSIFIED") or sector != self.get_sector(b):
                continue
            if a not in daily_prices or b not in daily_prices:
                continue
            aligned = daily_prices[[a, b]].dropna()
            if len(aligned) < 60:
                continue
            correlation = aligned.pct_change(fill_method=None).dropna().corr().iloc[0, 1]
            if pd.notna(correlation) and correlation >= .7:
                candidates.append((float(correlation), a, b))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(a, b) for _, a, b in candidates[:max_pairs]]

    @staticmethod
    def to_yfinance_symbol(symbol: str) -> str:
        clean = symbol.replace("NSE:", "").replace("-EQ", "").strip()
        if clean in ["L&T", "LT"]:
            return "LT.NS"
        if not clean.endswith(".NS") and not clean.endswith(".BO"):
            return f"{clean}.NS"
        return clean

    @staticmethod
    def to_fyers_symbol(symbol: str) -> str:
        clean = symbol.replace(".NS", "").replace(".BO", "").replace("NSE:", "").replace("-EQ", "").strip()
        if clean in ["L&T", "LT"]:
            return "NSE:LT-EQ"
        return f"NSE:{clean}-EQ"

    @staticmethod
    def to_clean_symbol(symbol: str) -> str:
        return symbol.replace("NSE:", "").replace("-EQ", "").replace(".NS", "").replace(".BO", "").strip()

    def get_eligible_symbols(self, base_list: Optional[List[str]] = None) -> List[str]:
        symbols = base_list or NIFTY_50_SYMBOLS
        return [self.to_clean_symbol(s) for s in symbols]

    def get_eligible_pairs(self, pairs_list: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        pairs = pairs_list or NIFTY_PAIRS
        return [(self.to_clean_symbol(a), self.to_clean_symbol(b)) for a, b in pairs]

    def get_dynamic_volume_shockers(self, top_n: int = 50) -> List[str]:
        """Compatibility alias: now ranks today's NSE share volume."""
        return self.get_nse_top_volume(top_n)
