# Pythonメインスクリプト（operations/screener.py）_2.0.0

import datetime
import os
import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# リポジトリのルート階層をパスに追加（operations/ 配下から core/ を import するため）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.loader import load_watchlist
from core.strategy import generate_signals

# ==========================================
# 1. 環境変数・設定値の読み込み
# ==========================================
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
MAIL_TO = os.environ.get("MAIL_TO")


# ==========================================
# 2. Notion過去ステータス照合関数
# ==========================================
def get_latest_signal_from_notion(code: str) -> str | None:
    """Notion DBから対象銘柄の最新レコードのシグナルを取得する"""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return None

    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    payload = {
        "filter": {
            "property": "銘柄",
            "title": {
                "starts_with": code,
            },
        },
        "sorts": [
            {
                "property": "日付",
                "direction": "descending",
            }
        ],
        "page_size": 1,
    }

    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                props = results[0].get("properties", {})
                signal_prop = props.get("シグナル", {}).get("select")
                if signal_prop:
                    return signal_prop.get("name")
    except Exception as e:
        print(f"Error querying Notion for {code}: {e}")

    return None


# ==========================================
# 3. 銘柄データ取得 & シグナル評価関数
# ==========================================
def analyze_ticker(ticker_symbol: str) -> dict | None:
    try:
        # ウォームアップ確保のため300日分取得
        df = yf.download(ticker_symbol, period="300d", interval="1d", progress=False)
        if df.empty or len(df) < 100:
            return None

        # 共通ストラテジーモジュールで指標計算＆シグナル判定を実行
        df = generate_signals(df)
        latest = df.iloc[-1]

        code = ticker_symbol.replace(".T", "")
        raw_signal = latest["Signal"]

        # Notionの過去ステータスと照合して一対化（交互化）を強制
        final_signal = None
        if raw_signal:
            latest_notion_signal = get_latest_signal_from_notion(code)

            if raw_signal == "Long":
                if latest_notion_signal != "Long":
                    final_signal = "Long"
                else:
                    print(f"Skip Long for {code}: Already in LONG position.")
            elif raw_signal == "Exit L":
                if latest_notion_signal == "Long":
                    final_signal = "Exit L"
                else:
                    print(f"Skip Exit L for {code}: No active LONG position.")
            elif raw_signal == "Short":
                if latest_notion_signal != "Short":
                    final_signal = "Short"
                else:
                    print(f"Skip Short for {code}: Already in SHORT position.")
            elif raw_signal == "Exit S":
                if latest_notion_signal == "Short":
                    final_signal = "Exit S"
                else:
                    print(f"Skip Exit S for {code}: No active SHORT position.")

        return {
            "symbol": ticker_symbol,
            "code": code,
            "close": float(latest["Close"]),
            "st_val": float(latest["ST_Value"]),
            "signal": final_signal,
        }
    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")
        return None


# ==========================================
# 4. Notion DB登録関数
# ==========================================
def push_to_notion(result_data: dict, company_name: str = "") -> None:
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        print("Notion API credentials not provided. Skipping Notion push.")
        return

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    today_str = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ).strftime("%Y-%m-%d")
    title_text = f"{result_data['code']} {company_name}".strip()

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "銘柄": {"title": [{"text": {"content": title_text}}]},
            "日付": {"date": {"start": today_str}},
            "シグナル": {"select": {"name": result_data["signal"]}},
            "終値": {"number": result_data["close"]},
            "ステータス": {"status": {"name": "未対応"}},
            "想定損切り価格": {"number": result_data["st_val"]},
        },
    }

    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 200:
        print(f"Failed to push to Notion: {res.text}")


# ==========================================
# 5. メール送信関数
# ==========================================
def send_email(signals_dict: dict) -> None:
    if not SMTP_USER or not SMTP_PASS or not MAIL_TO:
        print("SMTP credentials not provided. Skipping email dispatch.")
        return

    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now_jst.strftime("%Y/%m/%d")
    weekday_str = weekdays_jp[now_jst.weekday()]

    subject = f"{date_str}({weekday_str})の対象銘柄コード"

    long_list = "\n".join(signals_dict["Long"]) if signals_dict["Long"] else ""
    short_list = "\n".join(signals_dict["Short"]) if signals_dict["Short"] else ""
    exit_l_list = "\n".join(signals_dict["Exit L"]) if signals_dict["Exit L"] else ""
    exit_s_list = "\n".join(signals_dict["Exit S"]) if signals_dict["Exit S"] else ""

    body = f"""①エントリー
【買い】
{long_list}

【空売り】
{short_list}

②手仕舞い
【買い】
{exit_l_list}

【空売り】
{exit_s_list}"""

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = SMTP_USER
    msg["To"] = MAIL_TO

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# ==========================================
# 6. メイン実行フロー
# ==========================================
def main() -> None:
    try:
        codes = load_watchlist()
    except FileNotFoundError as e:
        print(e)
        return

    signals_summary = {
        "Long": [],
        "Short": [],
        "Exit L": [],
        "Exit S": [],
    }

    print(f"Starting analysis for {len(codes)} stocks...")

    for code in codes:
        ticker = code if code.endswith(".T") else f"{code}.T"
        res = analyze_ticker(ticker)

        if res and res["signal"]:
            print(f"Signal detected: {res['code']} -> {res['signal']}")
            push_to_notion(res)
            if res["signal"] in signals_summary:
                signals_summary[res["signal"]].append(res["code"])

    send_email(signals_summary)
    print("Processing completed.")


if __name__ == "__main__":
    main()