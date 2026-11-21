# `core/strategies/strategy_no1.py`などの売買条件の評価からシグナル生成を分離し、`generate_signals`関数を定義する。

from typing import Callable
import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    evaluate_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mode: str = "both",
) -> pd.DataFrame:
    df = evaluate_fn(df)

from typing import Callable

def generate_signals(
    df: pd.DataFrame,
    evaluate_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mode: str = "both",
) -> pd.DataFrame:
    df = evaluate_fn(df)
    """
    3値状態管理（simPosition: 0=ノーポジ, 1=買い, -1=空売り）とドテン対応により、
    時系列のポジション推移と発生シグナル（Long / Exit L / Short / Exit S）を算出する。
    TradingViewインジケーター (Signal & Alert Indicator_v1.0.0) と完全互換。
    mode: "both" (両方向), "long_only" (買いのみ), "short_only" (空売りのみ)
    """

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