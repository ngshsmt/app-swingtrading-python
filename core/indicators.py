# indicators.py

"""
`screener.py` の TradingView 完全互換 RMA・Supertrend ロジックをベースにし、テクニカル指標の計算に必要な `numpy` と `pandas` のみをインポートしています。
# ※Supertrendの方向（`ST_Direction`）は、インジケーター側の `stDirection < 0`（`-1: Buy`, `1: Sell`）と一致させています。
"""

import numpy as np
import pandas as pd


def calculate_rma(series: pd.Series, period: int) -> pd.Series:
    """TradingView ta.rma (Wilder's Smoothing) と完全互換の計算"""
    rma = np.full(len(series), np.nan)
    vals = series.values
    if len(vals) < period:
        return pd.Series(rma, index=series.index)

    # 最初はSMAで初期化
    rma[period - 1] = np.mean(vals[:period])
    alpha = 1.0 / period
    for i in range(period, len(vals)):
        rma[i] = alpha * vals[i] + (1.0 - alpha) * rma[i - 1]
    return pd.Series(rma, index=series.index)


def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    TradingView ta.supertrend と完全互換の計算
    戻り値: (ST_Direction [-1: Buy, 1: Sell], ST_Value)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range の計算
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # RMAによるATR計算
    atr = calculate_rma(tr, period)

    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    final_ub = np.full(len(df), np.nan)
    final_lb = np.full(len(df), np.nan)
    st_dir = np.full(len(df), 1)  # 初期値 1 (Sell)

    first_valid_idx = period - 1
    if first_valid_idx >= len(df):
        return pd.Series(st_dir, index=df.index), pd.Series(final_ub, index=df.index)

    # 初期設定
    final_ub[first_valid_idx] = basic_ub.iloc[first_valid_idx]
    final_lb[first_valid_idx] = basic_lb.iloc[first_valid_idx]
    st_dir[first_valid_idx] = (
        -1 if close.iloc[first_valid_idx] > final_ub[first_valid_idx] else 1
    )

    for i in range(first_valid_idx + 1, len(df)):
        # Upper Band 更新
        if basic_ub.iloc[i] < final_ub[i - 1] or close.iloc[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Lower Band 更新
        if basic_lb.iloc[i] > final_lb[i - 1] or close.iloc[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Direction 判定 (-1: Buy, 1: Sell)
        prev_dir = st_dir[i - 1]
        if prev_dir == -1:
            st_dir[i] = 1 if close.iloc[i] < final_lb[i] else -1
        else:
            st_dir[i] = -1 if close.iloc[i] > final_ub[i] else 1

    st_val = np.where(st_dir == -1, final_lb, final_ub)
    return pd.Series(st_dir, index=df.index), pd.Series(st_val, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameにSMA(5, 10, 20, 50)とSupertrend(10, 3.0)を追加する"""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    df["SMA5"] = close.rolling(window=5).mean()
    df["SMA10"] = close.rolling(window=10).mean()
    df["SMA20"] = close.rolling(window=20).mean()
    df["SMA50"] = close.rolling(window=50).mean()

    df["ST_Direction"], df["ST_Value"] = calculate_supertrend(
        df, period=10, multiplier=3.0
    )
    return df