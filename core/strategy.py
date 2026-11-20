# strategy.py

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

    # 逆パーフェクトオーダー条件（空売り）
    df["isSimpleInversePO"] = (
        (df["SMA5"] < df["SMA10"])
        & (df["SMA10"] < df["SMA20"])
        & (close < df["SMA50"])
    )

    # 転換・クロス条件（空売り）
    df["st_turn_sell"] = (df["ST_Direction"].shift(1) == -1) & (df["ST_Direction"] == 1)
    df["sma_cross_sell"] = (df["SMA5"].shift(1) >= df["SMA10"].shift(1)) & (
        df["SMA5"] < df["SMA10"]
    )

    # 空売りエントリー / 手仕舞い条件（ベースライン：完全ミラー）
    df["shortEntryCond"] = (
        df["isStSell"]
        & df["isSimpleInversePO"]
        & (df["st_turn_sell"] | df["sma_cross_sell"])
    )
    df["shortExitCond"] = (df["SMA5"] > df["SMA20"]) | (~df["isStSell"])

    return df


def generate_signals(df: pd.DataFrame, mode: str = "both") -> pd.DataFrame:
    """
    3値状態管理（simPosition: 0=ノーポジ, 1=買い, -1=空売り）とドテン対応により、
    時系列のポジション推移と発生シグナル（Long / Exit L / Short / Exit S）を算出する。
    TradingViewインジケーター (Signal & Alert Indicator_v1.0.0) と完全互換。
    mode: "both" (両方向), "long_only" (買いのみ), "short_only" (空売りのみ)
    """
    df = evaluate_conditions(df)

    n = len(df)
    positions = np.zeros(n, dtype=int)
    signals: list[str | None] = [None] * n

    # 個別イベントフラグ（ドテン時に Exit L と Short の両方が True になるケースに対応）
    long_entry_sig = np.zeros(n, dtype=bool)
    long_exit_sig = np.zeros(n, dtype=bool)
    short_entry_sig = np.zeros(n, dtype=bool)
    short_exit_sig = np.zeros(n, dtype=bool)

    long_entry_cond = df["longEntryCond"].values.copy()
    long_exit_cond = df["longExitCond"].values.copy()
    short_entry_cond = df["shortEntryCond"].values.copy()
    short_exit_cond = df["shortExitCond"].values.copy()

    if mode == "short_only":
        long_entry_cond[:] = False
    elif mode == "long_only":
        short_entry_cond[:] = False

    sim_position = 0

    for i in range(n):
        if sim_position == 0:
            if long_entry_cond[i]:
                long_entry_sig[i] = True
                sim_position = 1
                signals[i] = "Long"
            elif short_entry_cond[i]:
                short_entry_sig[i] = True
                sim_position = -1
                signals[i] = "Short"

        elif sim_position == 1:
            if short_entry_cond[i]:
                # ドテン空売り（買い手仕舞い ＋ 空売りエントリー）
                long_exit_sig[i] = True
                short_entry_sig[i] = True
                sim_position = -1
                signals[i] = "Short"
            elif long_exit_cond[i]:
                long_exit_sig[i] = True
                sim_position = 0
                signals[i] = "Exit L"

        elif sim_position == -1:
            if long_entry_cond[i]:
                # ドテン買い（空売り手仕舞い ＋ 買いエントリー）
                short_exit_sig[i] = True
                long_entry_sig[i] = True
                sim_position = 1
                signals[i] = "Long"
            elif short_exit_cond[i]:
                short_exit_sig[i] = True
                sim_position = 0
                signals[i] = "Exit S"

        positions[i] = sim_position

    df["Position"] = positions
    df["Signal"] = signals
    df["longEntrySignal"] = long_entry_sig
    df["longExitSignal"] = long_exit_sig
    df["shortEntrySignal"] = short_entry_sig
    df["shortExitSignal"] = short_exit_sig

    return df