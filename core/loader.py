# loader.py
# ※現在使用中のファイル名が `watchlist_20261004.csv` のままであれば、デフォルトファイル名を変更するか、`data/` に移動する際にリネームしてください。

"""
`screener.py` 内の読み込み処理をベースにし、標準ライブラリ（`os`）と `pandas` のみで動作する共通ローダーです。`operations/` 配下から呼び出してもファイルが見つかるよう、デフォルトパスをリポジトリのルート基準（`data/watchlist_jpx400.csv`）で解決できるようにしています。
"""

import os
from pathlib import Path
import pandas as pd


def load_watchlist(csv_path: str | Path | None = None) -> list[str]:
    """
    共通ウォッチリストCSVを読み込み、銘柄コード（例: '7203'）のリストを返す。
    パス未指定時はリポジトリルート直下の data/watchlist_jpx400.csv を参照する。
    """
    if csv_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        csv_path = base_dir / "data" / "watchlist_jpx400.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {csv_path}")

    df_watchlist = pd.read_csv(csv_path)
    code_col = "code" if "code" in df_watchlist.columns else df_watchlist.columns[0]

    # '.T' が含まれていても純粋な4桁コード文字列に正規化して返す
    codes = (
        df_watchlist[code_col]
        .astype(str)
        .str.replace(".T", "", regex=False)
        .str.strip()
        .tolist()
    )
    return codes