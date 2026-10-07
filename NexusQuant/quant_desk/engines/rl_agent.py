import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from quant_desk.core.config import RL_CONFIG, RLConfig
from quant_desk.data.universe import UniverseManager


@dataclass
class RLState:
    z_score: float
    adx_14: float
    ema_spread_ratio: float
    vwap_distance_pct: float
    volume_spike_ratio: float
    return_5m_pct: float

    def to_vector(self) -> np.ndarray:
        """Normalize state features into 6-dimensional numpy vector [-1.0, +1.0] using Z-scores & log returns."""
        v0 = np.clip(self.z_score / 3.5, -1.0, 1.0)
        v1 = np.clip((self.adx_14 - 20.0) / 30.0, -1.0, 1.0)
        v2 = np.clip(self.ema_spread_ratio * 50.0, -1.0, 1.0)
        v3 = np.clip(self.vwap_distance_pct * 50.0, -1.0, 1.0)
        v4 = np.clip((self.volume_spike_ratio - 1.0) / 2.0, -1.0, 1.0)
        v5 = np.clip(self.return_5m_pct * 100.0, -1.0, 1.0)
        return np.array([v0, v1, v2, v3, v4, v5], dtype=np.float32)


@dataclass
class RLActionDecision:
    action: int  # 0: HOLD, 1: BUY, 2: SELL, 3: EXIT
    action_name: str
    confidence_weight: float  # 0.0 to 1.0
    q_values: List[float]
    is_exploration: bool
    reason: str


class ReplayMemory:
    """Experience Replay Buffer for Reinforcement Learning Agent."""

    def __init__(self, capacity: int = RL_CONFIG.memory_capacity):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class QLearningAgent:
    """Adaptive Linear Q-Learning Agent with Epsilon-Greedy Policy & Bellman Optimality Updates."""

    def __init__(self, config: RLConfig = RL_CONFIG):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon_start

        # Weights matrix (action_dim x state_dim) and bias (action_dim)
        # Initialize small symmetric weights so action values start balanced
        self.W = np.random.default_rng(42).normal(0.0, 0.05, (self.action_dim, self.state_dim))
        self.b = np.zeros(self.action_dim, dtype=np.float32)

        self.memory = ReplayMemory(capacity=config.memory_capacity)
        self.training_steps = 0

    def predict_q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """Predict Q(S, a) for all actions."""
        return np.dot(self.W, state_vec) + self.b

    def select_action(self, state_vec: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, bool]:
        """Select action using Epsilon-Greedy policy."""
        q_vals = self.predict_q_values(state_vec)
        
        if not eval_mode and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            is_exp = True
        else:
            action = int(np.argmax(q_vals))
            is_exp = False

        # Calculate confidence weight w_RL in range [0.0, 1.0]
        q_max = float(q_vals[action])
        # Softmax / Sigmoid confidence scaling
        confidence = float(1.0 / (1.0 + np.exp(-np.clip(q_max, -5.0, 5.0))))
        return action, confidence, is_exp

    def learn_from_step(self, state_vec: np.ndarray, action: int, reward: float, next_state_vec: np.ndarray, done: bool) -> float:
        """Update Q-function weights using Bellman Optimality Equation: Q(S,A) += lr * [R + gamma * max Q(S',a') - Q(S,A)]."""
        self.memory.push(state_vec, action, reward, next_state_vec, done)
        
        # Predict Q(S, A)
        q_curr = np.dot(self.W[action], state_vec) + self.b[action]
        
        # Target Q value
        if done:
            q_target = reward
        else:
            q_next_max = np.max(self.predict_q_values(next_state_vec))
            q_target = reward + self.gamma * q_next_max

        td_error = q_target - q_curr

        # Gradient step for action row
        self.W[action] += self.lr * td_error * state_vec
        self.b[action] += self.lr * td_error

        # Experience Replay Mini-Batch Update
        if len(self.memory) >= self.config.batch_size:
            batch = self.memory.sample(self.config.batch_size)
            for b_s, b_a, b_r, b_s_next, b_done in batch:
                b_q_curr = np.dot(self.W[b_a], b_s) + self.b[b_a]
                b_q_target = b_r if b_done else (b_r + self.gamma * np.max(self.predict_q_values(b_s_next)))
                b_td = b_q_target - b_q_curr
                self.W[b_a] += (self.lr * 0.5) * b_td * b_s
                self.b[b_a] += (self.lr * 0.5) * b_td

        # Epsilon Decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        self.training_steps += 1
        return float(abs(td_error))


class RLMetaFilter:
    """Reinforcement Learning Meta-Filter: Evaluates market state and provides learned confidence weights."""

    ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "EXIT"}

    def __init__(self, config: RLConfig = RL_CONFIG):
        self.agent = QLearningAgent(config=config)
        self.last_states: Dict[str, np.ndarray] = {}
        self.last_actions: Dict[str, int] = {}
        self.pending_entries: Dict[str, Tuple[np.ndarray, int]] = {}
        self.negative_return_history: List[float] = []

    @staticmethod
    def extract_state_from_candles(
        symbol: str,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        z_score: float = 0.0
    ) -> RLState:
        """Extract 6-dimensional RL state vector from indicators."""
        if df_5m.empty:
            return RLState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        price = float(df_5m["close"].iloc[-1])
        prev_price = float(df_5m["close"].iloc[-2]) if len(df_5m) > 1 else price
        return_5m = (price - prev_price) / prev_price if prev_price > 0 else 0.0

        if "ema_fast" in df_5m.columns and "ema_slow" in df_5m.columns:
            ema_fast = float(df_5m["ema_fast"].iloc[-1])
            ema_slow = float(df_5m["ema_slow"].iloc[-1])
        else:
            ema_fast = float(df_5m["close"].ewm(span=6, adjust=False).mean().iloc[-1])
            ema_slow = float(df_5m["close"].ewm(span=30, adjust=False).mean().iloc[-1])
        ema_spread_ratio = (ema_fast - ema_slow) / price if price > 0 else 0.0

        if "vwap" in df_5m.columns:
            vwap = float(df_5m["vwap"].iloc[-1])
        else:
            from quant_desk.engines.momentum import MomentumEngine
            vwap = float(MomentumEngine.calculate_vwap(df_5m).iloc[-1])
        vwap_dist = (price - vwap) / vwap if vwap > 0 else 0.0

        if "vol_sma" in df_5m.columns:
            vol_sma = float(df_5m["vol_sma"].iloc[-1])
        else:
            vol_sma = float(df_5m["volume"].shift(1).rolling(20).mean().iloc[-1]) if len(df_5m) >= 21 else 1.0
        vol_curr = float(df_5m["volume"].iloc[-1])
        vol_spike = vol_curr / vol_sma if (vol_sma > 0 and not pd.isna(vol_sma)) else 1.0

        # ADX 14 on 15m
        if "adx_14" in df_15m.columns:
            adx_val = float(df_15m["adx_14"].iloc[-1])
        else:
            from quant_desk.engines.momentum import MomentumEngine
            adx_val = float(MomentumEngine.calculate_adx(df_15m).iloc[-1]) if not df_15m.empty else 0.0

        return RLState(
            z_score=z_score,
            adx_14=adx_val,
            ema_spread_ratio=ema_spread_ratio,
            vwap_distance_pct=vwap_dist,
            volume_spike_ratio=vol_spike,
            return_5m_pct=return_5m
        )

    def evaluate_decision(
        self,
        symbol: str,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        z_score: float = 0.0,
        proposed_action: str = "HOLD"
    ) -> RLActionDecision:
        """Evaluate RL Policy decision and return confidence weight w_RL."""
        sym = UniverseManager.to_clean_symbol(symbol)
        state_obj = self.extract_state_from_candles(sym, df_5m, df_15m, z_score)
        state_vec = state_obj.to_vector()

        action_idx = {"BUY": 1, "SELL": 2, "EXIT": 3}.get(proposed_action, 0)
        is_exp = False
        ready = self.agent.config.enabled and self.agent.training_steps >= self.agent.config.min_training_steps
        q = self.agent.predict_q_values(state_vec)
        confidence = float(1 / (1 + np.exp(-np.clip(q[action_idx], -5, 5)))) if ready else 1.0
        q_values = [float(q) for q in self.agent.predict_q_values(state_vec)]
        action_name = self.ACTION_MAP.get(action_idx, "HOLD")

        # Save state for reward update
        self.pending_entries[sym] = (state_vec, action_idx)

        reason = (
            f"RL Agent Policy: Selected {action_name} (Confidence: {confidence*100:.1f}%, "
            f"Epsilon: {self.agent.epsilon:.3f})"
        )
        if not ready:
            reason = "RL gate bypassed: disabled or insufficient training; deterministic strategy used."

        return RLActionDecision(
            action=action_idx,
            action_name=action_name,
            confidence_weight=confidence,
            q_values=q_values,
            is_exploration=is_exp,
            reason=reason
        )

    def record_entry(self, symbol: str) -> None:
        """Associate rewards with the executed entry, not a later random action."""
        sym = UniverseManager.to_clean_symbol(symbol)
        if sym in self.pending_entries:
            state, action = self.pending_entries[sym]
            self.last_states[sym] = state.copy()
            self.last_actions[sym] = action

    def process_reward_step(
        self,
        symbol: str,
        net_pnl: float,
        friction_cost: float,
        is_closed: bool = False,
        next_df_5m: Optional[pd.DataFrame] = None,
        next_df_15m: Optional[pd.DataFrame] = None
    ) -> float:
        """Compute Stepwise Sortino / Risk-Adjusted reward & execute Q-learning step update."""
        sym = UniverseManager.to_clean_symbol(symbol)
        if not self.agent.config.enabled or sym not in self.last_states or sym not in self.last_actions:
            return 0.0

        state_vec = self.last_states[sym]
        action = self.last_actions[sym]

        # 1. Stepwise Downside Volatility Tracking for Sortino Ratio calculation
        if net_pnl < 0:
            self.negative_return_history.append(net_pnl)
            if len(self.negative_return_history) > 100:
                self.negative_return_history.pop(0)

        downside_dev = np.std(self.negative_return_history) if len(self.negative_return_history) >= 2 else 100.0
        if downside_dev < 1e-4:
            downside_dev = 1.0

        # Stepwise Risk-Adjusted Reward (Sortino metric)
        sortino_reward = (net_pnl / downside_dev)

        # 2. Drawdown & Friction Penalties
        drawdown_penalty = abs(min(0.0, net_pnl)) * self.agent.config.drawdown_penalty_weight
        friction_penalty = friction_cost * self.agent.config.friction_penalty_weight

        # 3. Exploration / Mode Collapse Inactivity Penalty: Penalize perpetual HOLD output
        inactivity_penalty = self.agent.config.hold_inactivity_penalty if action == 0 else 0.0

        reward = sortino_reward - (friction_penalty / 1000.0) - (drawdown_penalty / 1000.0) - inactivity_penalty
        reward = float(np.clip(reward, -5.0, 5.0))

        if next_df_5m is not None and next_df_15m is not None:
            next_state_obj = self.extract_state_from_candles(sym, next_df_5m, next_df_15m)
            next_state_vec = next_state_obj.to_vector()
        else:
            next_state_vec = state_vec

        td_error = self.agent.learn_from_step(state_vec, action, reward, next_state_vec, is_closed)
        return reward
