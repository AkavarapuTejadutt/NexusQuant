"""Fixed, causal research hypotheses; parameters are not profitability claims."""
from datetime import time
import pandas as pd

STYLES = ("trend", "breakout", "opening_range", "pullback", "range_reversion", "adaptive_alpha")


def research_signal(engine, symbol, five, fifteen):
    from quant_desk.engines.momentum import MomentumSignal
    price = float(five.close.iloc[-1]) if len(five) else 0.0
    def result(action="HOLD", reason="Warming up completed candles", adx=0., relative=0., vwap=0.):
        return MomentumSignal(symbol, action, price, 0., 0., adx, relative, vwap, reason)
    style = engine.config.entry_style
    if style == "observe":
        return result(reason="Observe only: no strategy approved for deployment")
    if len(five) < 50 or len(fifteen) < 35:
        return result()
    # Eligibility is a hypothesis shared by these new candidates, not a change
    # to NSE's volume ranking. Excludes very low priced / thin intraday stocks.
    turnover = float((five.close * five.volume).iloc[-21:-1].mean())
    if price < 50 or turnover < 1_000_000:
        return result(reason="Research liquidity filter: price >= Rs50, mean 5m turnover >= Rs10 lakh")
    ts = five.index[-1]
    today = five.loc[five.index.date == ts.date()]
    vwap = float(five["vwap"].iloc[-1]) if "vwap" in five.columns else float(engine.calculate_vwap(five).iloc[-1])
    adx = float(fifteen["adx_14"].iloc[-1]) if "adx_14" in fifteen.columns else float(engine.calculate_adx(fifteen).iloc[-1])
    mean_vol = float(five.volume.iloc[-21:-1].mean())
    relative = float(five.volume.iloc[-1]) / mean_vol if mean_vol > 0 else 0.
    fast = five["ema_fast"] if "ema_fast" in five.columns else engine.calculate_ema(five.close, 20)
    slow = five["ema_slow"] if "ema_slow" in five.columns else engine.calculate_ema(five.close, 50)
    higher = fifteen["higher_ema"] if "higher_ema" in fifteen.columns else engine.calculate_ema(fifteen.close, 30)
    tr = pd.concat([five.high-five.low, (five.high-five.close.shift()).abs(),
                    (five.low-five.close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-15:-1].mean())
    if atr <= 0 or float(tr.iloc[-1]) > 3 * atr:
        return result(reason="Volatility shock: wait", adx=adx, relative=relative, vwap=vwap)
    up = fast.iloc[-1] > slow.iloc[-1] and higher.iloc[-1] > higher.iloc[-4]
    down = fast.iloc[-1] < slow.iloc[-1] and higher.iloc[-1] < higher.iloc[-4]
    buy = sell = False
    if style == "opening_range":
        opening = today.between_time("09:15", "09:25")
        if len(opening) != 3 or not time(9, 30) <= ts.time() <= time(11, 30):
            return result(reason="Opening-range window: 09:30–11:30; requires all three opening bars")
        high, low = float(opening.high.max()), float(opening.low.min())
        buy = five.close.iloc[-2] <= high < price and price > vwap and up
        sell = five.close.iloc[-2] >= low > price and price < vwap and down
        buy, sell = buy and relative >= 1.5, sell and relative >= 1.5
    elif style == "pullback":
        # Fresh reclaim of the fast EMA after a pullback, in a confirmed trend.
        buy = up and adx >= 25 and five.close.iloc[-2] <= fast.iloc[-2] and price > fast.iloc[-1] and price > vwap
        sell = down and adx >= 25 and five.close.iloc[-2] >= fast.iloc[-2] and price < fast.iloc[-1] and price < vwap
        buy, sell = buy and relative >= 1., sell and relative >= 1.
    elif style == "range_reversion":
        # A return inside a prior band, not an unconfirmed falling-knife entry.
        center = five.close.shift().rolling(20).mean()
        spread = five.close.shift().rolling(20).std()
        lower, upper = center - 2 * spread, center + 2 * spread
        if adx < 20 and relative <= 2 and time(10) <= ts.time() < time(14, 30):
            buy = five.close.iloc[-2] < lower.iloc[-2] and price > lower.iloc[-1] and price < vwap
            sell = five.close.iloc[-2] > upper.iloc[-2] and price < upper.iloc[-1] and price > vwap
    elif style == "adaptive_alpha":
        # Institutional Dual-Regime Model:
        # Regime 1 (ADX >= 22 & Efficiency > 0.25): Intraday VWAP Pullback / Breakout Continuation
        # Regime 2 (ADX < 22 & Low Noise): VWAP Band Mean-Reversion Dip Buying / Rally Shorting
        vwap_series = engine.calculate_vwap(five)
        vwap_curr = float(vwap_series.iloc[-1])
        vwap_prev = float(vwap_series.iloc[-2]) if len(vwap_series) > 1 else vwap_curr
        vwap_slope = (vwap_curr - vwap_prev) / vwap_prev if vwap_prev > 0 else 0.0

        std_20 = float(five.close.iloc[-21:-1].std()) if len(five) >= 21 else 1.0
        vwap_upper = vwap_curr + 1.8 * std_20
        vwap_lower = vwap_curr - 1.8 * std_20

        change = abs(float(five.close.iloc[-1]) - float(five.close.iloc[-11])) if len(five) >= 11 else 0.0
        volatility = float((five.close - five.close.shift(1)).abs().iloc[-10:].sum()) if len(five) >= 11 else 1.0
        er = change / volatility if volatility > 0 else 0.0

        if time(9, 30) <= ts.time() <= time(15, 0):
            if adx >= 22 and er > 0.25:
                ema8 = float(engine.calculate_ema(five.close, 8).iloc[-1])
                prev_p = float(five.close.iloc[-2])
                buy = (price > vwap_curr) and (vwap_slope > 0) and (prev_p <= ema8) and (price > ema8) and (relative >= 1.2)
                sell = (price < vwap_curr) and (vwap_slope < 0) and (prev_p >= ema8) and (price < ema8) and (relative >= 1.2)
            else:
                prev_p = float(five.close.iloc[-2])
                buy = (prev_p <= vwap_lower) and (price > vwap_lower) and (price < vwap_curr)
                sell = (prev_p >= vwap_upper) and (price < vwap_upper) and (price > vwap_curr)
    else:
        raise ValueError(f"Unknown research strategy: {style}")
    action = "BUY" if buy else "SELL" if sell else "HOLD"
    return result(action, f"{style}: {action}", adx, relative, vwap)
