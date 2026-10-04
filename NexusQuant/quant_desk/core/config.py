import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


@dataclass(frozen=True)
class FyersConfig:
    client_id: str = os.getenv("FYERS_CLIENT_ID", "cKNPUI0IK79-100")
    secret_key: str = os.getenv("FYERS_SECRET_KEY", "KNPUI0IK79-100")
    redirect_uri: str = os.getenv("FYERS_REDIRECT_URI", "http://127.0.0.1")
    access_token: str = os.getenv("FYERS_ACCESS_TOKEN", "")
    grant_type: str = "authorization_code"
    response_type: str = "code"
    state: str = "quant_desk_state"


@dataclass(frozen=True)
class RiskConfig:
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", 1000000.0))
    daily_kill_switch_pct: float = float(os.getenv("DAILY_KILL_SWITCH_PCT", -0.02))
    half_kelly_fraction: float = float(os.getenv("HALF_KELLY_FRACTION", 0.5))
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", 0.065))
    max_position_size_pct: float = 0.15           # Max 15% capital in a single trade
    max_risk_per_trade_pct: float = 0.0075        # Max 0.75% portfolio risk per trade
    max_open_positions: int = 3                   # Max 3 total concurrent positions
    max_positions_per_sector: int = 1             # Max 1 position per sector
    max_portfolio_open_risk_pct: float = 0.015    # Max 1.5% total open risk across portfolio


@dataclass(frozen=True)
class FrictionConfig:
    slippage_pct: float = float(os.getenv("SLIPPAGE_PCT", 0.0003))      # 0.03%
    stt_pct: float = float(os.getenv("STT_PCT", 0.00025))              # 0.025%
    exchange_txn_pct: float = float(os.getenv("EXCHANGE_TXN_PCT", 0.00003)) # 0.003%
    sebi_fee_pct: float = 0.000001                                     # ₹10 per crore (0.0001%)
    stamp_duty_pct: float = 0.00003                                    # 0.003% for intraday buy
    friction_payoff_mult: float = float(os.getenv("FRICTION_PAYOFF_MULT", 3.0))


@dataclass(frozen=True)
class StatArbConfig:
    lookback_days: int = 180
    p_value_threshold: float = 0.05
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    hard_stop_zscore: float = 3.5
    half_life_time_mult: float = 2.5


@dataclass(frozen=True)
class MomentumConfig:
    ema_fast: int = 6
    ema_slow: int = 30
    adx_period: int = 14
    adx_trigger: float = 25.0       # Require strong ADX > 25.0 for entries
    adx_mute: float = 20.0          # Mute chop when ADX <= 20.0
    volume_spike_mult: float = 1.5   # Require 1.5x volume spike
    volume_sma_period: int = 20
    atr_stop_mult: float = 1.5       # Stop Loss = 1.5 x ATR_14
    atr_target_mult: float = 2.5     # Target Take Profit = 2.5 x ATR_14 (1.67:1 Reward-to-Risk)
    partial_tp_pct: float = 0.50     # Book 50% profit @ Stage 1 target, trail rest


@dataclass(frozen=True)
class RLConfig:
    state_dim: int = 6
    action_dim: int = 4  # 0: HOLD, 1: BUY, 2: SELL, 3: EXIT
    learning_rate: float = 0.005
    discount_factor: float = 0.95
    epsilon_start: float = 0.20       # Calibrated starting exploration
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    memory_capacity: int = 10000
    batch_size: int = 32
    friction_penalty_weight: float = 2.0
    drawdown_penalty_weight: float = 1.5
    confidence_floor: float = 0.48    # Require RL confidence >= 48%


# Singleton Instances
FYERS_CONFIG = FyersConfig()
RISK_CONFIG = RiskConfig()
FRICTION_CONFIG = FrictionConfig()
STAT_ARB_CONFIG = StatArbConfig()
MOMENTUM_CONFIG = MomentumConfig()
RL_CONFIG = RLConfig()


def get_access_token() -> str:
    """Read current access token from environment or file."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    return os.getenv("FYERS_ACCESS_TOKEN", "")


def update_access_token_in_env(token: str) -> None:
    """Save newly generated access token into .env file."""
    token_str = token.strip()
    if not ENV_PATH.exists():
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write(f"FYERS_ACCESS_TOKEN={token_str}\n")
        return

    lines = []
    found = False
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("FYERS_ACCESS_TOKEN="):
                lines.append(f"FYERS_ACCESS_TOKEN={token_str}\n")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"\nFYERS_ACCESS_TOKEN={token_str}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    os.environ["FYERS_ACCESS_TOKEN"] = token_str
