import os
import time
import datetime
import threading
import requests
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
from groq import Groq
import db

# --- Config & Environment Variables ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080))

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

ALL_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "BTC/USD": "BTC-USD"
}

db.init_db(list(ALL_PAIRS.keys()))
active_pairs = db.get_saved_watchlist(list(ALL_PAIRS.keys()))
TIMEFRAMES = ["5m", "15m"]

app = FastAPI()

# --- 0. Safe yfinance Download (Retry + Backoff on Rate Limit) ---
def safe_yf_download(ticker: str, period: str, interval: str, max_retries: int = 3):
    """
    ห่อ yf.download ด้วย retry + exponential backoff
    เพื่อรับมือกับ Yahoo Finance rate limit (HTTP 429 / Too Many Requests)
    แทนที่จะปล่อยให้ล้มเหลวทันทีหรือได้ข้อมูลว่างเปล่ากลับมา
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is not None and not df.empty:
                return df
            last_error = "empty dataframe"
        except Exception as e:
            last_error = str(e)
            is_rate_limited = "429" in last_error or "Too Many Requests" in last_error or "rate" in last_error.lower()
            if is_rate_limited:
                print(f"yfinance rate-limited for {ticker} (attempt {attempt + 1}/{max_retries}): {last_error}")
            else:
                print(f"yfinance error for {ticker} (attempt {attempt + 1}/{max_retries}): {last_error}")

        # รอก่อน retry แบบ exponential backoff (2s, 4s, 8s, ...)
        if attempt < max_retries - 1:
            time.sleep(2 ** (attempt + 1))

    print(f"yfinance download failed for {ticker} after {max_retries} attempts: {last_error}")
    return pd.DataFrame()

# --- 1. ZigZag Calculation Engine ---
def calculate_zigzag(df: pd.DataFrame, deviation_pct: float):
    threshold = deviation_pct / 100.0
    pivots = np.zeros(len(df))
    last_pivot_type = 0
    last_pivot_val = df['Close'].iloc[0]
    last_pivot_idx = 0

    for i in range(1, len(df)):
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        if last_pivot_type == 0:
            if (current_high - last_pivot_val) / last_pivot_val >= threshold:
                last_pivot_type = 1
                last_pivot_val = current_high
                last_pivot_idx = i
            elif (last_pivot_val - current_low) / last_pivot_val >= threshold:
                last_pivot_type = -1
                last_pivot_val = current_low
                last_pivot_idx = i
        elif last_pivot_type == 1:
            if current_high > last_pivot_val:
                last_pivot_val = current_high
                last_pivot_idx = i
            elif (last_pivot_val - current_low) / last_pivot_val >= threshold:
                pivots[last_pivot_idx] = 1
                last_pivot_type = -1
                last_pivot_val = current_low
                last_pivot_idx = i
        elif last_pivot_type == -1:
            if current_low < last_pivot_val:
                last_pivot_val = current_low
                last_pivot_idx = i
            elif (current_high - last_pivot_val) / last_pivot_val >= threshold:
                pivots[last_pivot_idx] = -1
                last_pivot_type = 1
                last_pivot_val = current_high
                last_pivot_idx = i

    if last_pivot_type == 1:
        pivots[last_pivot_idx] = 1
    elif last_pivot_type == -1:
        pivots[last_pivot_idx] = -1

    return pivots

# --- 2. Setup Analysis ---
def analyze_zigzag_setup(df: pd.DataFrame):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=10).rsi()
    pivots_fast = calculate_zigzag(df, deviation_pct=0.05)
    pivots_slow = calculate_zigzag(df, deviation_pct=0.15)

    closed = df.iloc[-1]
    
    slow_window = pivots_slow[-3:]
    has_slow_bottom = -1 in slow_window
    has_slow_top = 1 in slow_window

    has_fast_bottom = (pivots_fast[-1] == -1 or pivots_fast[-2] == -1)
    has_fast_top = (pivots_fast[-1] == 1 or pivots_fast[-2] == 1)

    total_range = closed['High'] - closed['Low']
    if total_range == 0:
        return closed, False, False

    upper_wick = closed['High'] - max(closed['Close'], closed['Open'])
    lower_wick = min(closed['Close'], closed['Open']) - closed['Low']

    call_candidate = (
        (has_slow_bottom and has_fast_bottom) and
        (lower_wick / total_range >= 0.20 or closed['Close'] > closed['Open']) and
        (closed['rsi'] <= 45)
    )

    put_candidate = (
        (has_slow_top and has_fast_top) and
        (upper_wick / total_range >= 0.20 or closed['Close'] < closed['Open']) and
        (closed['rsi'] >= 55)
    )

    return closed, call_candidate, put_candidate

def ask_groq_ai_zigzag(pair: str, tf: str, closed_row, setup_type: str):
    if not groq_client:
        return None
    prompt = f"""
    Binary Options Price Action Analysis.
    Asset: {pair} | TF: {tf} | Setup: {setup_type}
    Snapshot: Close: {closed_row['Close']:.5f}, RSI(10): {closed_row['rsi']:.2f}
    Respond ONLY in JSON:
    {{"signal": "{setup_type}", "confidence": <float 0-100>, "reason": "<1-sentence reasoning>"}}
    """
    try:
        res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

# --- 3. Detailed Backtest Engine (Trade-by-Trade History) ---
def execute_backtest(pair: str, tf: str, days: int):
    ticker = ALL_PAIRS.get(pair)
    if not ticker:
        return {"error": "ไม่พบคู่เงินนี้"}
    
    df = safe_yf_download(ticker, period=f"{days}d", interval=tf)
    if df.empty or len(df) < 50:
        return {"error": f"ข้อมูลย้อนหลังมีไม่เพียงพอ (ได้มา {len(df)} แท่งเทียน)"}
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=10).rsi()
    pivots_fast = calculate_zigzag(df, deviation_pct=0.05)
    pivots_slow = calculate_zigzag(df, deviation_pct=0.15)

    wins, losses, draws = 0, 0, 0
    trade_logs = []
    current_win_streak = 0
    max_win_streak = 0
    current_loss_streak = 0
    max_loss_streak = 0

    trade_count = 0
    for i in range(20, len(df) - 1):
        slow_win = pivots_slow[i-2 : i+1]
        has_slow_b = -1 in slow_win
        has_slow_t = 1 in slow_win

        has_fast_b = (pivots_fast[i] == -1 or pivots_fast[i-1] == -1)
        has_fast_t = (pivots_fast[i] == 1 or pivots_fast[i-1] == 1)

        closed = df.iloc[i]
        next_candle = df.iloc[i + 1]

        total_range = closed['High'] - closed['Low']
        if total_range == 0:
            continue

        upper_wick = closed['High'] - max(closed['Close'], closed['Open'])
        lower_wick = min(closed['Close'], closed['Open']) - closed['Low']

        call_cond = (
            (has_slow_b and has_fast_b) and
            (lower_wick / total_range >= 0.20 or closed['Close'] > closed['Open']) and
            (closed['rsi'] <= 45)
        )
        
        put_cond = (
            (has_slow_t and has_fast_t) and
            (upper_wick / total_range >= 0.20 or closed['Close'] < closed['Open']) and
            (closed['rsi'] >= 55)
        )

        signal = "CALL" if call_cond else ("PUT" if put_cond else None)

        if signal:
            trade_count += 1
            entry_p = float(closed['Close'])
            expiry_p = float(next_candle['Close'])
            entry_time = df.index[i].strftime('%Y-%m-%d %H:%M')

            if signal == "CALL":
                res = "WIN" if expiry_p > entry_p else ("LOSS" if expiry_p < entry_p else "DRAW")
            else:
                res = "WIN" if expiry_p < entry_p else ("LOSS" if expiry_p > entry_p else "DRAW")

            if res == "WIN":
                wins += 1
                current_win_streak += 1
                current_loss_streak = 0
                if current_win_streak > max_win_streak:
                    max_win_streak = current_win_streak
            elif res == "LOSS":
                losses += 1
                current_loss_streak += 1
                current_win_streak = 0
                if current_loss_streak > max_loss_streak:
                    max_loss_streak = current_loss_streak
            else:
                draws += 1
                current_win_streak = 0
                current_loss_streak = 0

            trade_logs.append({
                "no": trade_count,
                "time": entry_time,
                "direction": signal,
                "entry_price": entry_p,
                "expiry_price": expiry_p,
                "result": res
            })

    total = wins + losses + draws
    winrate = round((wins / (wins + losses) * 100), 2) if (wins + losses) > 0 else 0.0

    return {
        "pair": pair,
        "timeframe": tf,
        "days": days,
        "total": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": winrate,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "trade_logs": trade_logs
    }

# --- 4. Web Dashboard UI ---
@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    stats = db.fetch_winrate()
    saved_list = db.get_saved_watchlist(list(ALL_PAIRS.keys()))
    
    total_trades = sum(row.get('total_trades', 0) for row in stats)
    total_wins = sum(row.get('wins', 0) for row in stats)
    total_losses = sum(row.get('losses', 0) for row in stats)
    overall_wr = round((total_wins / (total_wins + total_losses) * 100), 2) if (total_wins + total_losses) > 0 else 0.0

    pair_checkboxes = ""
    for p in ALL_PAIRS.keys():
        checked = "checked" if p in saved_list else ""
        pair_checkboxes += f"""
        <label style="margin-right: 15px; display: inline-flex; align-items: center; cursor: pointer;">
            <input type="checkbox" name="pairs" value="{p}" {checked} style="margin-right: 6px; width: 16px; height: 16px;"> {p}
        </label>
        """

    pair_options = "".join([f"<option value='{p}'>{p}</option>" for p in ALL_PAIRS.keys()])

    stats_rows = ""
    for row in stats:
        wr = row.get('win_rate_percentage') or 0
        stats_rows += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #334155;"><b>{row['pair']}</b></td>
            <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['timeframe']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['total_trades']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #334155; color: #4ade80;">{row['wins']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #334155; color: #f87171;">{row['losses']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #334155; font-weight: bold;">{wr}%</td>
        </tr>
        """
    if not stats_rows:
        stats_rows = "<tr><td colspan='6' style='text-align: center; padding: 20px; color: #94a3b8;'>ยังไม่มีข้อมูลสถิติที่บันทึกผลแล้ว</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Binary AI Control Center</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: auto; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .stat-box {{ background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
            .btn {{ background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; }}
            .btn-green {{ background: #10b981; }}
            .btn:hover {{ opacity: 0.9; }}
            select {{ background: #0f172a; border: 1px solid #334155; color: white; padding: 8px 12px; border-radius: 6px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th {{ background: #0f172a; padding: 12px; border-bottom: 2px solid #334155; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⚡ Binary AI Pro: Control & Backtest Engine</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">Win Rate สด (Live)</div>
                    <div style="font-size: 28px; font-weight: bold; color: {'#4ade80' if overall_wr >= 60 else '#f87171'};">{overall_wr}%</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">จำนวนไม้สดทั้งหมด</div>
                    <div style="font-size: 28px; font-weight: bold;">{total_trades}</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">ชนะสด (WIN)</div>
                    <div style="font-size: 28px; font-weight: bold; color: #4ade80;">{total_wins}</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">แพ้สด (LOSS)</div>
                    <div style="font-size: 28px; font-weight: bold; color: #f87171;">{total_losses}</div>
                </div>
            </div>

            <!-- กล่องทดสอบ Backtest -->
            <div class="card" style="border: 1px solid #3b82f6;">
                <h3>🧪 ทดสอบย้อนหลังพร้อมดู Pattern แพ้-ชนะ (Backtest Simulator)</h3>
                <form action="/run-backtest" method="post" style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
                    <div>
                        <label>เลือกคู่เงิน:</label>
                        <select name="pair">{pair_options}</select>
                    </div>
                    <div>
                        <label>Timeframe:</label>
                        <select name="tf">
                            <option value="5m">5 นาที (5m)</option>
                            <option value="15m">15 นาที (15m)</option>
                        </select>
                    </div>
                    <div>
                        <label>ย้อนหลัง:</label>
                        <select name="days">
                            <option value="7">7 วันล่าสุด</option>
                            <option value="30" selected>30 วันล่าสุด (1 เดือน)</option>
                            <option value="59">60 วันล่าสุด (2 เดือน)</option>
                        </select>
                    </div>
                    <div style="margin-top: 18px;">
                        <button type="submit" class="btn btn-green">🚀 เริ่มจำลอง Backtest ทันที</button>
                    </div>
                </form>
            </div>

            <!-- กล่องตั้งค่า Watchlist -->
            <div class="card">
                <h3>🎯 ตั้งค่าโฟกัสคู่เงิน (บันทึกลง Database อัตโนมัติ)</h3>
                <form action="/update-watchlist" method="post">
                    <div style="margin: 15px 0;">
                        {pair_checkboxes}
                    </div>
                    <button type="submit" class="btn">บันทึกการตั้งค่าคู่เงิน</button>
                </form>
            </div>

            <!-- กล่องตารางสถิติ Real-time -->
            <div class="card">
                <h3>📊 สถิติการรันจริง (Live Signals)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>คู่เงิน</th>
                            <th>Timeframe</th>
                            <th>เทรดทั้งหมด</th>
                            <th>ชนะ (WIN)</th>
                            <th>แพ้ (LOSS)</th>
                            <th>Win Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stats_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- 5. Backtest Result View (With Table & Pattern Badges) ---
@app.post("/run-backtest")
def handle_backtest(pair: str = Form(...), tf: str = Form(...), days: int = Form(...)):
    res = execute_backtest(pair, tf, days)
    if "error" in res:
        return HTMLResponse(content=f"""
            <script>
                alert("เกิดข้อผิดพลาด: {res['error']}");
                window.location.href = "/";
            </script>
        """)
    
    wr_color = "#4ade80" if res['winrate'] >= 60 else "#f87171"

    # สร้างแถบ Visual Pattern Badge
    pattern_badges = ""
    for log in res["trade_logs"]:
        if log["result"] == "WIN":
            pattern_badges += f'<span style="background:#15803d; color:#fff; padding:3px 7px; border-radius:4px; font-size:12px; font-weight:bold; margin:2px;" title="ไม้ที่ {log["no"]}: WIN">W</span>'
        elif log["result"] == "LOSS":
            pattern_badges += f'<span style="background:#b91c1c; color:#fff; padding:3px 7px; border-radius:4px; font-size:12px; font-weight:bold; margin:2px;" title="ไม้ที่ {log["no"]}: LOSS">L</span>'
        else:
            pattern_badges += f'<span style="background:#64748b; color:#fff; padding:3px 7px; border-radius:4px; font-size:12px; font-weight:bold; margin:2px;" title="ไม้ที่ {log["no"]}: DRAW">D</span>'

    # สร้างตารางแจกแจงทุกไม้
    trade_rows = ""
    for log in reversed(res["trade_logs"]):  # แสดงไม้ล่าสุดขึ้นก่อน
        res_color = "#4ade80" if log["result"] == "WIN" else ("#f87171" if log["result"] == "LOSS" else "#94a3b8")
        dir_color = "#38bdf8" if log["direction"] == "CALL" else "#f472b6"
        trade_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">#{log['no']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #94a3b8;">{log['time']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; font-weight: bold; color: {dir_color};">{log['direction']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{log['entry_price']:.5f}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{log['expiry_price']:.5f}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; font-weight: bold; color: {res_color};">{log['result']}</td>
        </tr>
        """

    result_html = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ผลทดสอบ Backtest {res['pair']}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: white; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 25px; margin-bottom: 20px; border: 1px solid #334155; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 15px 0; }}
            .stat-box {{ background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
            .btn {{ display: inline-block; background: #3b82f6; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; margin-top: 15px; }}
            th {{ background: #0f172a; padding: 10px 12px; border-bottom: 2px solid #334155; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2>📈 ผลการทดสอบ Backtest: {res['pair']} ({res['timeframe']})</h2>
                <a href="/" class="btn">⬅️ กลับหน้าหลัก Dashboard</a>
            </div>

            <div class="card">
                <div class="grid">
                    <div class="stat-box">
                        <div style="color: #94a3b8; font-size: 13px;">Win Rate รวม</div>
                        <div style="font-size: 30px; font-weight: bold; color: {wr_color};">{res['winrate']}%</div>
                    </div>
                    <div class="stat-box">
                        <div style="color: #94a3b8; font-size: 13px;">เข้าเงื่อนไขทั้งหมด</div>
                        <div style="font-size: 26px; font-weight: bold;">{res['total']} ไม้</div>
                    </div>
                    <div class="stat-box">
                        <div style="color: #94a3b8; font-size: 13px;">ชนะ (WIN) / แพ้ (LOSS)</div>
                        <div style="font-size: 22px; font-weight: bold;"><span style="color:#4ade80;">{res['wins']}</span> / <span style="color:#f87171;">{res['losses']}</span></div>
                    </div>
                    <div class="stat-box">
                        <div style="color: #94a3b8; font-size: 13px;">ชนะติดกันสูงสุด (Streak)</div>
                        <div style="font-size: 24px; font-weight: bold; color: #4ade80;">🔥 {res['max_win_streak']} ไม้</div>
                    </div>
                    <div class="stat-box">
                        <div style="color: #94a3b8; font-size: 13px;">แพ้ติดกันสูงสุด (Max DD)</div>
                        <div style="font-size: 24px; font-weight: bold; color: #f87171;">⚠️ {res['max_loss_streak']} ไม้</div>
                    </div>
                </div>

                <h4 style="margin: 20px 0 10px 0; color: #94a3b8;">🧩 ผัง Pattern ผลลัพธ์ตามลำดับเวลา (เรียงจากอดีต -> ปัจจุบัน):</h4>
                <div style="background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; line-height: 2.2; word-wrap: break-word;">
                    {pattern_badges}
                </div>
            </div>

            <div class="card">
                <h3>📋 ตารางแจกแจงประวัติทุกไม้แบบละเอียด (เรียงจากไม้ล่าสุดลงไป)</h3>
                <div style="max-height: 500px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>ไม้ที่</th>
                                <th>วัน-เวลาที่เข้า</th>
                                <th>คำสั่ง</th>
                                <th>ราคาเปิดแท่ง</th>
                                <th>ราคาปิดแท่ง</th>
                                <th>ผลลัพธ์</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trade_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=result_html)

@app.post("/update-watchlist")
def update_watchlist(pairs: list[str] = Form(default=[])):
    global active_pairs
    active_pairs = [p for p in pairs if p in ALL_PAIRS] if pairs else list(ALL_PAIRS.keys())
    db.update_saved_watchlist(active_pairs)
    send_telegram(f"⚙️ *มีการบันทึก Watchlist ใหม่ลง Database:*\n`{', '.join(active_pairs)}`")
    return HTMLResponse(content="""<script>alert("บันทึกสำเร็จ!"); window.location.href = "/";</script>""")

# --- 6. Telegram & Timing Helpers ---
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram send error: {e}")

def send_telegram_direct(chat_id, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram direct error: {e}")

def sleep_until_next_5m_candle():
    now = datetime.datetime.now(datetime.timezone.utc)
    minutes_to_next = 5 - (now.minute % 5)
    target_time = (now + datetime.timedelta(minutes=minutes_to_next)).replace(second=2, microsecond=0)
    sleep_seconds = (target_time - now).total_seconds()
    if sleep_seconds < 0:
        sleep_seconds += 300
    return sleep_seconds

# --- 7. Threads ---
def synchronized_trading_loop():
    time.sleep(5)
    send_telegram("🚀 *ระบบ Binary ZigZag Confluence Engine เริ่มทำงานแล้ว (24/7)*")
    
    while True:
        try:
            wait_time = sleep_until_next_5m_candle()
            time.sleep(wait_time)
            
            current_watchlist = db.get_saved_watchlist(list(ALL_PAIRS.keys()))
            current_focus = [p for p in current_watchlist if p in ALL_PAIRS]
            
            current_utc = datetime.datetime.now(datetime.timezone.utc)
            current_minute = current_utc.minute
            
            timeframes_to_check = ["5m"]
            if current_minute % 15 == 0:
                timeframes_to_check.append("15m")

            for name in current_focus:
                ticker = ALL_PAIRS[name]
                for tf in timeframes_to_check:
                    df = safe_yf_download(ticker, period="5d", interval=tf)
                    if df.empty or len(df) < 50:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    closed_candle, is_call, is_put = analyze_zigzag_setup(df)
                    setup = "CALL" if is_call else ("PUT" if is_put else None)
                    
                    if setup:
                        ai_res = ask_groq_ai_zigzag(name, tf, closed_candle, setup)
                        if ai_res and ai_res.get("confidence", 0) >= 70:
                            entry_price = float(closed_candle['Close'])
                            conf = float(ai_res['confidence'])
                            reason = ai_res.get('reason', '')
                            
                            sig_id = db.save_signal(name, tf, setup, entry_price, conf, reason)
                            
                            duration_mins = 5 if tf == "5m" else 15
                            expiry_time = (current_utc + datetime.timedelta(minutes=duration_mins)).strftime('%H:%M')
                            
                            icon = "🟢" if setup == "CALL" else "🔴"
                            msg = (
                                f"{icon} *สัญญาณเข้าเทรดแท่งใหม่ ({setup})*\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"📐 **ตัวบ่งชี้:** `ZigZag Confluence Engine`\n"
                                f"📊 **คู่เงิน:** `{name}`\n"
                                f"⏱ **Timeframe:** `{tf}`\n"
                                f"🏁 **หมดเวลาที่:** `{expiry_time} UTC`\n"
                                f"💵 **ราคาเปิดแท่ง:** `{entry_price:.5f}`\n"
                                f"🎯 **ความมั่นใจ:** `{conf}%`\n"
                                f"💡 **วิเคราะห์ AI:** {reason}\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"🆔 `Ref ID: #{sig_id}`"
                            )
                            send_telegram(msg)
        except Exception as e:
            print(f"Scanner error: {e}")
            time.sleep(10)

def outcome_checker_loop():
    time.sleep(15)
    while True:
        try:
            pending = db.get_pending_signals()
            now = datetime.datetime.now(datetime.timezone.utc)
            
            for sig in pending:
                duration_min = 5 if sig["timeframe"] == "5m" else 15
                target_expiry = sig["created_at"].replace(tzinfo=datetime.timezone.utc) if sig["created_at"].tzinfo is None else sig["created_at"]
                target_expiry += datetime.timedelta(minutes=duration_min)
                
                if now >= target_expiry:
                    ticker = ALL_PAIRS.get(sig["pair"])
                    if not ticker:
                        continue
                    
                    df = safe_yf_download(ticker, period="1d", interval="1m")
                    if df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # หาแท่งเทียนที่เวลาตรง (หรือใกล้ที่สุด) กับเวลาหมดอายุจริง
                    # แทนการหยิบแท่งล่าสุดที่ดึงมาได้เสมอ ซึ่งอาจดีเลย์จากเวลาหมดอายุจริง
                    candle_index = df.index
                    if candle_index.tz is None:
                        candle_index = candle_index.tz_localize("UTC")
                    else:
                        candle_index = candle_index.tz_convert("UTC")

                    time_diffs = np.abs((candle_index - target_expiry).total_seconds())
                    closest_pos = int(time_diffs.argmin())
                    closest_diff_seconds = time_diffs[closest_pos]

                    # ถ้าแท่งที่ใกล้ที่สุดยังห่างจากเวลาหมดอายุจริงเกิน 2 นาที
                    # แปลว่าข้อมูลยังมาไม่ถึง (data lag) รอรอบถัดไปแทนที่จะตัดสินผลด้วยราคาผิดเวลา
                    if closest_diff_seconds > 120:
                        continue

                    expiry_price = float(df.iloc[closest_pos]['Close'])
                    entry_price = float(sig['entry_price'])
                    
                    if sig['direction'] == "CALL":
                        result = "WIN" if expiry_price > entry_price else ("LOSS" if expiry_price < entry_price else "DRAW")
                    else:
                        result = "WIN" if expiry_price < entry_price else ("LOSS" if expiry_price > entry_price else "DRAW")
                    
                    db.save_result(sig['id'], expiry_price, result)
                    
                    res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "⚪ DRAW")
                    msg = (
                        f"🔔 *สรุปผลการเทรด: {res_icon}*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🆔 `Ref ID: #{sig['id']}`\n"
                        f"📊 **คู่เงิน:** `{sig['pair']}` ({sig['timeframe']})\n"
                        f"🎯 **คำสั่ง:** `{sig['direction']}`\n"
                        f"💵 **ราคาเปิด:** `{entry_price:.5f}`\n"
                        f"🏁 **ราคาปิดแท่ง:** `{expiry_price:.5f}`"
                    )
                    send_telegram(msg)
            time.sleep(20)
        except Exception as e:
            print(f"Outcome checker error: {e}")
            time.sleep(20)

def telegram_polling_loop():
    global active_pairs
    last_update_id = 0
    while True:
        try:
            if not TELEGRAM_BOT_TOKEN:
                time.sleep(10)
                continue
                
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=30"
            res = requests.get(url, timeout=35).json()
            
            if res.get("ok"):
                for item in res.get("result", []):
                    last_update_id = item["update_id"]
                    msg = item.get("message", {})
                    text = msg.get("text", "").strip()
                    sender_chat_id = msg.get("chat", {}).get("id")
                    
                    if not sender_chat_id:
                        continue

                    if text.startswith("/focus"):
                        parts = text.split(" ", 1)
                        if len(parts) < 2 or parts[1].strip().lower() == "all":
                            active_pairs = list(ALL_PAIRS.keys())
                            db.update_saved_watchlist(active_pairs)
                            send_telegram_direct(sender_chat_id, "✅ *ตั้งค่าโฟกัส:* เฝ้าดู **ทุกคู่เงิน**")
                        else:
                            selected = [p.strip().upper() for p in parts[1].split(",") if p.strip().upper() in ALL_PAIRS]
                            if selected:
                                active_pairs = selected
                                db.update_saved_watchlist(active_pairs)
                                pairs_text = ", ".join(active_pairs)
                                send_telegram_direct(sender_chat_id, f"🎯 *ตั้งค่าโฟกัสสำเร็จ!*\nกำลังเฝ้าเฉพาะ: `{pairs_text}`")
                            else:
                                send_telegram_direct(sender_chat_id, "⚠️ ไม่พบคู่เงินที่ระบุ กรุณาพิมพ์เช่น `/focus EUR/USD,GBP/USD`")

                    elif text == "/watchlist":
                        saved = db.get_saved_watchlist(list(ALL_PAIRS.keys()))
                        send_telegram_direct(sender_chat_id, f"📋 *คู่เงินที่กำลังเฝ้าดู:*\n`{', '.join(saved)}`")

                    elif text == "/stat":
                        try:
                            stats = db.fetch_winrate()
                            if not stats:
                                send_telegram_direct(sender_chat_id, "📊 *ยังไม่มีข้อมูลสถิติการเทรดที่สรุปผลแล้ว*")
                            else:
                                report = "📈 *สรุปสถิติ Win Rate รวม:*\n━━━━━━━━━━━━━━━\n"
                                for row in stats:
                                    rate = row.get('win_rate_percentage') or 0
                                    report += f"🔹 *{row['pair']} ({row['timeframe']})*: Win Rate *{rate}%* (รวม {row['total_trades']} ไม้)\n"
                                send_telegram_direct(sender_chat_id, report)
                        except Exception as db_err:
                            send_telegram_direct(sender_chat_id, f"❌ Database Error: `{db_err}`")

                    elif text in ["/help", "/start"]:
                        msg_help = (
                            f"📌 *บอท ZigZag Confluence พร้อมทำงาน!*\nChat ID: `{sender_chat_id}`\n\n"
                            f"• `/focus EUR/USD` - เลือกคู่เงิน\n"
                            f"• `/focus all` - เฝ้าทุกคู่\n"
                            f"• `/watchlist` - ดูคู่เงินที่กำลังเฝ้า\n"
                            f"• `/stat` - ดูสถิติ Win Rate\n"
                            f"• `/help` - คู่มือ"
                        )
                        send_telegram_direct(sender_chat_id, msg_help)
            time.sleep(1)
        except Exception as e:
            time.sleep(5)

# --- 8. Start Background Threads ---
threading.Thread(target=synchronized_trading_loop, daemon=True).start()
threading.Thread(target=outcome_checker_loop, daemon=True).start()
threading.Thread(target=telegram_polling_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
