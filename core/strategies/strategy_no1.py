# strategy_no1（買いと空売りの条件はそれぞれ反転されたもの）
# 旧`strategy.py`　前半の売買条件のみ残し、後半の条件に基づく状態管理は`generate_signals`へ移行した

import numpy as np
import pandas as pd
from core.indicators import add_indicators


def evaluate_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    テクニカル指標を付与し、各日の売買条件（ブール値フラグ）を計算する。
    """
    df = add_indicators(df)
    close = df["Close"]

    # Supertrend 方向判定 (-1: Buy, 1: Sell)
    df["isStBuy"] = df["ST_Direction"] == -1
    df["isStSell"] = df["ST_Direction"] == 1

    # パーフェクトオーダー条件（買い）
    df["isSimplePO"] = (
        (df["SMA5"] > df["SMA10"])
        & (df["SMA10"] > df["SMA20"])
        & (close > df["SMA50"])
    )

    # 転換・クロス条件（買い）
    # 1. SupertrendがSell(1)からBuy(-1)に転換した日
    df["st_turn_buy"] = (df["ST_Direction"].shift(1) == 1) & (df["ST_Direction"] == -1)
    # 2. SMA5がSMA10をゴールデンクロスした日
    df["sma_cross_buy"] = (df["SMA5"].shift(1) <= df["SMA10"].shift(1)) & (
        df["SMA5"] > df["SMA10"]
    )

    # 買いエントリー / 手仕舞い条件
    df["longEntryCond"] = (
        df["isStBuy"] & df["isSimplePO"] & (df["st_turn_buy"] | df["sma_cross_buy"])
    )
    df["longExitCond"] = (df["SMA5"] < df["SMA20"]) | (~df["isStBuy"])

    # 逆パーフェクトオーダー条件（空売り）
    df["isSimpleInversePO"] = (
        (df["SMA5"] < df["SMA10"])
        & (df["SMA10"] < df["SMA20"])
        & (close < df["SMA50"])
    )

    # 転換・クロス条件（空売り）
    # 1. SupertrendがBuy(-1)からSell(1)に転換した日
    df["st_turn_sell"] = (df["ST_Direction"].shift(1) == -1) & (df["ST_Direction"] == 1)
    # 2. SMA5がSMA10をデッドクロスした日
    df["sma_cross_sell"] = (df["SMA5"].shift(1) >= df["SMA10"].shift(1)) & (
        df["SMA5"] < df["SMA10"]
    )

    # 空売りエントリー / 手仕舞い条件
    df["shortEntryCond"] = (
        df["isStSell"]
        & df["isSimpleInversePO"]
        & (df["st_turn_sell"] | df["sma_cross_sell"])
    )
    df["shortExitCond"] = (df["SMA5"] > df["SMA20"]) | (~df["isStSell"])

    return df