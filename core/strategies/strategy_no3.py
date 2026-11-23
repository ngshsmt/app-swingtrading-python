# strategy_no3
# 戻り売りエントリー（タイムスパン0〜3日） ＋ SMA10エグジット ＋ 1.5×ATR SL

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

    # --- 空売り条件（strategy_no3: 戻り売り[0〜3日スパン] ＋ SMA10エグジット ＋ 1.5×ATR SL） ---
    # 1. 大局環境: 50日線の下 かつ Supertrend Sell継続
    is_downtrend = (close < df["SMA50"]) & df["isStSell"]

    # 2. 戻り検知 (Setup: 当日を含む直近4営業日[0〜3日前]の間に10日線タッチ実績)
    touch_sma10 = df["High"] >= df["SMA10"]
    rebound_valid = touch_sma10.rolling(window=4, min_periods=1).max().astype(bool)

    # 3. 陰線転落トリガー (Trigger: 当日陰線 かつ 5日線割れ)
    bearish_turn = (close < df["Open"]) & (close < df["SMA5"])

    # 空売りエントリー条件 (戻り売り)
    df["shortEntryCond"] = is_downtrend & rebound_valid & bearish_turn

    # 空売り手仕舞い条件 (10日線を終値で上抜け、またはSupertrendのSell終了)
    df["shortExitCond"] = (close > df["SMA10"]) | (~df["isStSell"])

    # 初期ストップロス (SL) および 1Rリスク幅: 1.5 × ATR
    df["shortSL"] = close + (1.5 * df["ATR"])
    df["shortRisk1R"] = 1.5 * df["ATR"]

    return df