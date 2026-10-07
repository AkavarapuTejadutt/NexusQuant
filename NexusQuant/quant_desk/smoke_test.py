"""Offline integration soak using cached real prices; NOT a strategy backtest."""
import contextlib
import io
import json
import math
from dataclasses import replace
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unittest.mock import patch
import pandas as pd
from quant_desk.main import MasterQuantDesk
from quant_desk.engines.research_strategies import STYLES
from quant_desk.evaluate_strategies import read_candles


def main():
    root=Path(__file__).resolve().parent
    datasets={p.stem:read_candles(p) for p in sorted((root/'research_data').glob('*.csv'))}
    if not datasets:
        raise RuntimeError('Download research candles first')
    sessions=sorted(set.intersection(*(set(df.index.date) for df in datasets.values())))[-5:]
    events=[]
    for symbol,df in datasets.items():
        for ts,row in df.loc[pd.Index(df.index.date).isin(sessions)].iterrows():
            events.append((ts,symbol,float(row.close),float(row.volume)))
    events.sort()
    report={'purpose':'Integration soak only: 5m close replay does not simulate intrabar execution', 'results':{}}
    for style in (*STYLES,'observe'):
        print(f'Integration replay: {style}, {len(events)} real price events',flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch('quant_desk.main.UniverseManager.get_nse_top_volume',return_value=list(datasets)):
                desk=MasterQuantDesk('sim')
            desk.pairs=[]
            desk.momentum.config=replace(desk.momentum.config,entry_style=style)
            desk.data_feed.register_tick_callback(desk.on_tick_update)
            for ts,symbol,price,volume in events:
                desk.data_feed.process_tick(symbol,price,volume,ts)
            summary=desk.broker.get_portfolio_summary(desk.latest_prices)
        if desk.data_feed.callback_errors or not all(math.isfinite(v) for v in summary.values() if isinstance(v,(int,float))):
            raise RuntimeError(f'{style} failed integration invariants')
        if style=='observe' and (desk.broker.trade_history or desk.broker.open_positions):
            raise RuntimeError('Observe mode placed an order')
        if desk.broker.open_positions:
            raise RuntimeError('End-of-day positions left open')
        report['results'][style]={'ticks':desk.data_feed.tick_count,'callback_errors':desk.data_feed.callback_errors,
                                  'closed_execution_records':len(desk.broker.trade_history),'end_positions':len(desk.broker.open_positions)}
    (root/'integration_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
