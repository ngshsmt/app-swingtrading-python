# strategy_no2
# 戻り売りエントリー ＋ 1.5×ATRストップロス（SL） ＋ 改善案1エグジット（Close > SMA5）

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
    df["st_turn_buy"] = (df["ST_Direction"].shift(1) == 1) & (df["ST_Direction"] == -1)
    df["sma_cross_buy"] = (df["SMA5"].shift(1) <= df["SMA10"].shift(1)) & (
        df["SMA5"] > df["SMA10"]
    )

    # 買いエントリー / 手仕舞い条件
    df["longEntryCond"] = (
        df["isStBuy"] & df["isSimplePO"] & (df["st_turn_buy"] | df["sma_cross_buy"])
    )
    df["longExitCond"] = (df["SMA5"] < df["SMA20"]) | (~df["isStBuy"])

    # --- 空売り条件（strategy_no2: 戻り売り ＋ 改善案1エグジット ＋ 1.5×ATR SL） ---
    # 戻り売り条件
    # 1. 大局環境: Close < SMA50 かつ isStSell (Supertrend Sell継続)
    is_downtrend = (close < df["SMA50"]) & df["isStSell"]
    # 2. 戻り (リバウンド): High >= SMA5 (当日の高値が5日線に接触/上抜け)
    rebound_touch = df["High"] >= df["SMA5"]
    # 3. 陰線転落 (当日トリガー): Close < Open (陰線) かつ Close < SMA5 (5日線を下回って引け)
    bearish_turn = (close < df["Open"]) & (close < df["SMA5"])

    # 空売りエントリー条件 (戻り売り)
    df["shortEntryCond"] = is_downtrend & rebound_touch & bearish_turn

    # 空売り手仕舞い条件 (改善案1: 5日線を終値で上抜け、またはSupertrendのSell終了)
    df["shortExitCond"] = (close > df["SMA5"]) | (~df["isStSell"])

    # 初期ストップロス (SL) および 1Rリスク幅: 1.5 × ATR
    df["shortSL"] = close + (1.5 * df["ATR"])
    df["shortRisk1R"] = 1.5 * df["ATR"]

    return df