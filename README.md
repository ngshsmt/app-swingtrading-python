# リポジトリのルート階層
├── core/                         ◀ 【頭脳】ロジックと共通読み込み処理
│   ├── __init__.py               (空ファイル)
│   ├── indicators.py             ・Supertrend, SMA計算（共通利用・変更なし）
│   ├── loader.py                 ・ウォッチリスト読込（共通利用・変更なし）
│   └── strategies/               ★【新設】ストラテジ・プラグイン置き場
│       ├── __init__.py           ・登録されたストラテジを名前で取得するレジストリ
│       ├── generate_signals.py   ・共通の状態管理（simPosition・ドテン処理）エンジン
│       ├── strategy_no1.py       ・現行ストラテジ（売買条件の定義のみ）　デフォルト
│       └── strategy_no2.py       ・2つ目以降のストラテジ
│
├── data/                         ◀ 【共通データ】CSVファイル置き場
│   └── watchlist_jpx400.csv      ・JPX400などの共通ウォッチリスト
│
├── operations/                   ◀ 【実働部隊】実行スクリプト群
│   ├── screener.py               ・アラート＆Notion登録（環境変数等で稼働ストラテジを指定）
│   ├── backtest.py               ・バックテスト（CLI引数でストラテジ・モード・期間を指定）
│   ├── tracker.py                ・寄り付き追跡（変更なし）[cite: 2]
│   ├── montecarlo.py             ・モンテカルロ（変更なし）
│   └── app.py                    ・ダッシュボード（変更なし）
│
└── .github/workflows/            ◀ GitHub Actions設定

---