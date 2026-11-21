# 今後 strategy_no2.py を増やした際は、①`from core.strategies import`以降に追加分を、②STRATEGIES 辞書に1行足す

import pandas as pd
from core.strategies import strategy_no1, strategy_no2
from core.strategies.generate_signals import generate_signals

STRATEGIES = {
    "strategy_no1": strategy_no1.evaluate_conditions,
    "strategy_no2": strategy_no2.evaluate_conditions,
}


def run_strategy(
    df: pd.DataFrame,
    strategy_name: str = "strategy_no1",
    mode: str = "both",
) -> pd.DataFrame:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    return generate_signals(df, STRATEGIES[strategy_name], mode=mode)