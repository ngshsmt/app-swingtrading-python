# backtest.py_v5

import concurrent.futures
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

# リポジトリのルート階層をパスに追加（operations/ 配下から core/ を import するため）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.strategy import generate_signals

warnings.filterwarnings('ignore', category=UserWarning)


# =========================================================
# 1. Backtesting.py 用 Strategy クラス
# =========================================================
class CustomSignalStrategy(Strategy):
    mode = "short_only"  # 空売り単独検証用 ("short_only", "long_only", "both")

    def init(self):
        # 共通ストラテジーモジュールで指標計算＆シグナル判定を実行
        df_with_signals = generate_signals(self.data.df, mode=self.mode)

        # Backtesting.py の self.I() 用に数値コードへ変換
        #  1: Long (買いエントリー)
        # -1: Exit L (買い手仕舞い)
        #  2: Short (空売りエントリー)
        # -2: Exit S (空売り手仕舞い)
        sig_code = np.zeros(len(df_with_signals), dtype=int)
        sig_code[df_with_signals['longExitSignal']] = -1
        sig_code[df_with_signals['shortExitSignal']] = -2
        sig_code[df_with_signals['longEntrySignal']] = 1
        sig_code[df_with_signals['shortEntrySignal']] = 2

        # ドテン発生日（同日に手仕舞いと逆方向エントリーが両方 True になる日）の識別
        is_doten_short = df_with_signals['longExitSignal'] & df_with_signals['shortEntrySignal']
        is_doten_long = df_with_signals['shortExitSignal'] & df_with_signals['longEntrySignal']
        sig_code[is_doten_short] = -3  # 買い決済 -> ドテン空売り
        sig_code[is_doten_long] = 3    # 空売り決済 -> ドテン買い

        self.signal = self.I(lambda: sig_code, name='Signal')

    def next(self):
        sig = self.signal[-1]

        if sig == 1 and not self.position:
            self.buy()
        elif sig == -1 and self.position.is_long:
            self.position.close()
        elif sig == 2 and not self.position:
            self.sell()
        elif sig == -2 and self.position.is_short:
            self.position.close()
        elif sig == -3:
            if self.position:
                self.position.close()
            self.sell()
        elif sig == 3:
            if self.position:
                self.position.close()
            self.buy()


# =========================================================
# 2. 1銘柄ごとのデータ取得 ＆ バックテストワーカー関数
# =========================================================
def process_single_ticker(item):
    """
    yfinance経由で株価データを取得し、バックテストを実行
    item: tuple (code, company_name, period, start_date, end_date)
    """
    code, company_name, period, start_date, end_date = item
    
    # 日本株対応用ティッカーシンボル (例: 1662 -> 1662.T)
    ticker_symbol = f"{code}.T" if not str(code).endswith('.T') else str(code)

    try:
        # yfinance で株価取得
        ticker = yf.Ticker(ticker_symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        else:
            df = ticker.history(period=period, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # データ取得失敗または件数不足のチェック
        if df.empty or len(df) < 60:
            return None

        # --- バックテストに必要なカラムの存在チェック・成形 ---
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            return None

        # Volume列がない場合は0で埋める
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # データが削られた結果、件数が減った場合の再チェック
        if len(df) < 60:
            return None

        # バックテスト実行（元本は"cash"。デフォルトは100万円）
        bt = Backtest(df, CustomSignalStrategy, cash=1_000_000, commission=0, finalize_trades=True)
        stats = bt.run()
        trades = stats['_trades'].copy()

        start_str = stats['Start'].strftime('%Y-%m-%d') if pd.notnull(stats['Start']) else None
        end_str = stats['End'].strftime('%Y-%m-%d') if pd.notnull(stats['End']) else None
        num_trades = int(stats['# Trades'])

        # トレード未発生時
        if num_trades == 0:
            summary_res = {
                'コード': code,
                '会社名': company_name,
                'テスト対象開始日': start_str,
                'テスト対象終了日': end_str,
                'トレード回数': 0,
                '勝率[%]': 0.0,
                '平均利益[%]': 0.0,
                '平均損失[%]': 0.0,
                'リスクリワード': np.nan,
                'プロフィットファクター': np.nan,
                '最大ドローダウン[%]': 0.0,
                '最大連敗数': 0,
                '平均含み益保有日数': 0.0,
                '平均含み損保有日数': 0.0
            }
            return summary_res, pd.DataFrame()

        # -------------------------------------------------------------
        # 機能①・②：含み益日数・含み損日数の計算
        # -------------------------------------------------------------
        unrealized_profit_days = []
        unrealized_loss_days = []

        for _, trade in trades.iterrows():
            entry_time = trade['EntryTime']
            exit_time = trade['ExitTime']
            entry_price = trade['EntryPrice']
            is_long = trade['Size'] > 0

            # エントリーから手仕舞いまでの日次株価データを抽出
            trade_df = df.loc[entry_time:exit_time]

            if is_long:
                profit_mask = trade_df['Close'] > entry_price
                loss_mask = trade_df['Close'] < entry_price
            else:
                profit_mask = trade_df['Close'] < entry_price
                loss_mask = trade_df['Close'] > entry_price

            unrealized_profit_days.append(int(profit_mask.sum()))
            unrealized_loss_days.append(int(loss_mask.sum()))

        # -------------------------------------------------------------
        # 全トレード詳細データの整形と追加カラムの作成
        # -------------------------------------------------------------
        # 1. バーの本数の計算 (ExitBar - EntryBar)
        trades['バーの本数'] = trades['ExitBar'] - trades['EntryBar']

        # 2. 日時フォーマットの変更 (パターンA: 2020-01-15 09:00 形式 / 秒数なし・時差なし)
        for col in ['EntryTime', 'ExitTime']:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col]).dt.strftime('%Y-%m-%d %H:%M')

        # 3. 損益率[%] の100倍換算
        if 'ReturnPct' in trades.columns:
            trades['ReturnPct'] = (trades['ReturnPct'] * 100).round(2)

        # 4. 追加項目の格納と既存列の日本語化
        trades['コード'] = code
        trades['会社名'] = company_name
        trades['含み益保有日数'] = unrealized_profit_days
        trades['含み損保有日数'] = unrealized_loss_days

        rename_dict = {
            'Size': '取引株数',
            'EntryPrice': '新規建価格',
            'ExitPrice': '決済価格',
            'SL': '損切り価格',
            'TP': '利確価格',
            'PnL': '損益金額',
            'ReturnPct': '損益率[%]',
            'Commission': '手数料',
            'EntryTime': '新規建日時',
            'ExitTime': '決済日時',
            'Duration': '保有期間'
        }
        trades = trades.rename(columns=rename_dict)

        # 5. 指定された順番でのカラムの並べ替え（不要な列は除外）
        desired_columns = [
            'コード',
            '会社名',
            '取引株数',
            '新規建価格',
            '決済価格',
            '損切り価格',
            '利確価格',
            '損益金額',
            '損益率[%]',
            '手数料',
            '新規建日時',
            '決済日時',
            'バーの本数',
            '保有期間',
            '含み益保有日数',
            '含み損保有日数'
        ]

        existing_cols = [c for c in desired_columns if c in trades.columns]
        trades = trades[existing_cols]

        # -------------------------------------------------------------
        # サマリー指標の集計（win_rateを最初に定義）
        # -------------------------------------------------------------
        win_rate = float(stats['Win Rate [%]']) if pd.notnull(stats['Win Rate [%]']) else 0.0

        # 平均利益[%]・平均損失[%]の計算
        winning_returns = trades['損益率[%]'][trades['損益率[%]'] > 0]
        losing_returns = trades['損益率[%]'][trades['損益率[%]'] < 0]

        avg_gain_pct = float(winning_returns.mean()) if len(winning_returns) > 0 else 0.0
        avg_loss_pct = float(abs(losing_returns.mean())) if len(losing_returns) > 0 else 0.0

        if avg_loss_pct > 0:
            risk_reward = avg_gain_pct / avg_loss_pct
        else:
            risk_reward = np.nan if avg_gain_pct == 0 else np.inf

        pf = float(stats['Profit Factor']) if pd.notnull(stats['Profit Factor']) else np.nan

        # ドローダウン(%) & 最大連敗数
        max_dd_pct = float(stats['Max. Drawdown [%]']) if pd.notnull(stats['Max. Drawdown [%]']) else 0.0

        is_loss = stats['_trades']['PnL'] < 0
        max_losing_streak = 0
        current_streak = 0
        for loss in is_loss:
            if loss:
                current_streak += 1
                if current_streak > max_losing_streak:
                    max_losing_streak = current_streak
            else:
                current_streak = 0

        summary_res = {
            'コード': code,
            '会社名': company_name,
            'テスト対象開始日': start_str,
            'テスト対象終了日': end_str,
            'トレード回数': num_trades,
            '勝率[%]': round(win_rate, 2),
            '平均利益[%]': round(avg_gain_pct, 2),
            '平均損失[%]': round(avg_loss_pct, 2),
            'リスクリワード': round(risk_reward, 2) if np.isfinite(risk_reward) else risk_reward,
            'プロフィットファクター': round(pf, 2) if pd.notnull(pf) else pf,
            '最大ドローダウン[%]': round(abs(max_dd_pct), 2),
            '最大連敗数': max_losing_streak,
            '平均含み益保有日数': round(float(np.mean(unrealized_profit_days)), 2),
            '平均含み損保有日数': round(float(np.mean(unrealized_loss_days)), 2)
        }

        # (サマリー結果, 個別トレード詳細) のタプルで返す
        return summary_res, trades

    except Exception as e:
        # pfなどの未定義変数は出力せず、発生した具体的なエラー内容のみを表示
        print(f"Error executing {code} ({company_name}): {type(e).__name__} - {e}")
        return None

# =========================================================
# 3. メイン処理
# =========================================================
import datetime
import os
from core.loader import load_watchlist

if __name__ == '__main__':
    WATCHLIST_PATH = ROOT_DIR / "data" / "watchlist_jpx400.csv"
    OUTPUT_DIR = Path("/Users/natsuikeigo/app-swingtrading-python/outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 取得期間の設定
    # 変動させたい場合 FETCH_PERIOD = "5y"←直近5年を見る場合, START_DATE = None, END_DATE = None
    # 固定させたい場合は下記の通り
    FETCH_PERIOD = None
    START_DATE = "2021-09-01"
    END_DATE = "2026-09-01"

    codes = load_watchlist(WATCHLIST_PATH)

    tasks = [
        (code, str(code), FETCH_PERIOD, START_DATE, END_DATE)
        for code in codes
    ]

    print(f"銘柄マスタから {len(tasks)} 件の銘柄を読み込みました。")
    period_info = f"{START_DATE} 〜 {END_DATE}" if START_DATE and END_DATE else f"FETCH_PERIOD={FETCH_PERIOD}"
    print(f"バックテスト期間: {period_info}")
    print("yfinance経由で株価データの取得とバックテスト（Short Only）を並列で実行します...")

    summary_results = []
    all_trades_list = []

    # ProcessPoolExecutor による並列処理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_single_ticker, tasks, chunksize=5):
            if res is not None:
                summary_res, trades_df = res
                summary_results.append(summary_res)
                if not trades_df.empty:
                    all_trades_list.append(trades_df)

    # DataFrame化
    df_summary = pd.DataFrame(summary_results)
    df_all_trades = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    # プレビュー表示
    print("\n--- バックテスト結果サマリー（上位10件） ---")
    if not df_summary.empty:
        print(df_summary.head(10))

    # -------------------------------------------------------------
    # 2つのCSVファイルに分けて出力
    # -------------------------------------------------------------
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = OUTPUT_DIR / f'{now_str}_jpx400_supertrend_results.csv'
    all_trades_filename = OUTPUT_DIR / f'{now_str}_jpx400_supertrend_all_trades.csv'

    # 1. 銘柄別サマリー結果を出力
    df_summary.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"\n[出力完了] サマリー結果を '{summary_filename}' に保存しました。")

    # 2. 全トレード詳細を出力
    df_all_trades.to_csv(all_trades_filename, index=False, encoding='utf-8-sig')
    print(f"[出力完了] 全トレード詳細を '{all_trades_filename}' に保存しました。")