# nikkei225.py
"""
日経平均株価（^N225）の日足株価データを yfinance 経由で取得・成形する共通モジュール。
単一責任: 日足データの取得とクレンジングのみを行い、指標計算やストラテジロジックは含めない。
"""

import pandas as pd
import yfinance as yf


def fetch_nikkei225(
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """
    yfinance 経由で日経平均（^N225）の日足株価データを取得する。

    Parameters
    ----------
    start_date : str | None
        取得開始日（'YYYY-MM-DD' 形式）
    end_date : str | None
        取得終了日（'YYYY-MM-DD' 形式）
    period : str | None
        取得期間（例: '5y'）。start_date / end_date が指定されている場合はそちらを優先。

    Returns
    -------
    pd.DataFrame
        日付（タイムゾーンなし DatetimeIndex）をインデックスとし、
        ['Open', 'High', 'Low', 'Close', 'Volume'] を含む成形済み DataFrame。
        取得失敗時は空の DataFrame を返す。
    """
    ticker_symbol = "^N225"
    try:
        ticker = yf.Ticker(ticker_symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        elif period:
            df = ticker.history(period=period, auto_adjust=True)
        else:
            df = ticker.history(period="max", auto_adjust=True)

        if df.empty:
            return pd.DataFrame()

        # MultiIndex カラムの平坦化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 必要なカラムの存在確認と抽出
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # タイムゾーンを除去して個別株データと日付結合可能にする
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"[nikkei225] エラー: 日経平均データの取得に失敗しました ({e})")
        return pd.DataFrame()
