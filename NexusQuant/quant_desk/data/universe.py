from typing import List, Tuple, Optional
from datetime import datetime, timedelta

from fyers_apiv3 import fyersModel
from quant_desk.core.config import FYERS_CONFIG, get_access_token

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
    """Manages ticker mapping and strictly uses FYERS for dynamic volume scanning."""

    def __init__(self, screener_csv_path: Optional[str] = None):
        self.screener_csv_path = screener_csv_path
        self.excluded_symbols = set()
        self.fo_ban_list = set()

    @classmethod
    def get_sector(cls, symbol: str) -> str:
        clean = cls.to_clean_symbol(symbol)
        return SECTOR_MAP.get(clean, "MISC")

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

    def get_dynamic_volume_shockers(self, top_n: int = 30) -> List[str]:
        """Scans broad universe for volume shockers directly using the FYERS History API."""
        print(f"[UniverseManager] Scanning {len(BROAD_FO_UNIVERSE)} F&O stocks for Volume Spikes via FYERS API...")
        
        access_token = get_access_token()
        if not access_token:
            print("[UniverseManager] ERROR: No FYERS access token. Reverting to base highly-liquid list.")
            return [self.to_clean_symbol(s) for s in BROAD_FO_UNIVERSE[:top_n]]

        fyers = fyersModel.FyersModel(
            client_id=FYERS_CONFIG.client_id, 
            token=access_token, 
            is_async=False, 
            log_path=""
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=10) # 10 days to ensure we capture 5 trading sessions
        
        stock_scores = []
        
        for sym in BROAD_FO_UNIVERSE:
            try:
                fyers_sym = self.to_fyers_symbol(sym)
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
                    candles = response["candles"]
                    if len(candles) >= 2:
                        # Extract volumes (Index 5 in the FYERS payload)
                        volumes = [c[5] for c in candles]
                        yesterday_vol = float(volumes[-1])
                        
                        # Calculate average of previous days
                        prev_vols = volumes[:-1]
                        prev_avg_vol = sum(prev_vols) / len(prev_vols)
                        
                        if prev_avg_vol > 0:
                            vol_spike = yesterday_vol / prev_avg_vol
                            stock_scores.append((sym, vol_spike))
            except Exception:
                # Silently skip individual FYERS fetching errors to keep the scanner moving fast
                continue
        
        if not stock_scores:
            print("[UniverseManager] No volume data retrieved. Reverting to base list.")
            return [self.to_clean_symbol(s) for s in BROAD_FO_UNIVERSE[:top_n]]

        # Sort by the highest volume spike multiple
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        selected_symbols = [s[0] for s in stock_scores[:top_n]]
        
        print(f"[UniverseManager] Selected top {len(selected_symbols)} FYERS-verified volume shockers.")
        return selected_symbols