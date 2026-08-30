import os
import time
import datetime
import threading
import requests
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

# --- 1. ระบบ Economic News Filter (Forex Factory Free Feed) ---
high_impact_news = []
last_news_fetch = None

def fetch_economic_calendar():
    """ดึงปฏิทินข่าว High Impact สัปดาห์นี้"""
    global high_impact_news, last_news_fetch
    now = datetime.datetime.now(datetime.timezone.utc)
    if last_news_fetch and (now - last_news_fetch).total_seconds() < 3600:
        return
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, timeout=10).json()
        high_impact = []
        for item in res:
            if item.get("impact") == "High":
                # แปลงเวลาเป็น UTC
                news_time = datetime.datetime.fromisoformat(item["date"].replace("Z", "+00:00"))
                high_impact.append({
                    "title": item.get("title"),
                    "currency": item.get("country"),
                    "time": news_time
                })
        high_impact_news = high_impact
        last_news_fetch = now
        print(f"Loaded {len(high_impact_news)} High Impact news events.")
    except Exception as e:
        print(f"News fetch error: {e}")

def is_near_news(pair: str) -> tuple[bool, str]:
    """ตรวจสอบว่าคู่เงินนี้อยู่ในช่วงข่าว High Impact ก่อน/หลัง 15 นาที หรือไม่"""
    fetch_economic_calendar()
    now = datetime.datetime.now(datetime.timezone.utc)
    currencies = pair.replace("/", " ").split()
    
    for event in high_impact_news:
        if event["currency"] in currencies:
            time_diff = (event["time"] - now).total_seconds() / 60.0
            # บล็อกก่อนข่าวออก 15 นาที และหลังข่าวออก 15 นาที
            if -15 <= time_diff <= 15:
                return True, f"⚠️ ใกล้ช่วงข่าวกล่องแดง: {event['title']} ({event['currency']}) เวลา {event['time'].strftime('%H:%M')} UTC"
    return False, ""

# --- 2. Advanced Technical Analysis (EMA 200 + ADX + Key Levels) ---
def analyze_advanced_technical(df: pd.DataFrame):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    
    closed = df.iloc[-1]
    prev = df.iloc[-2]
    
    # คำนวณแนวรับ/แนวต้าน (Swing High / Swing Low ย้อนหลัง 60 แท่ง)
    recent_window = df.iloc[-60:-1]
    support_level = float(recent_window['Low'].min())
    resistance_level = float(recent_window['High'].max())
    
    # ตัวกรองเทรนด์รุนแรง: ถ้า ADX > 32 ถือว่าเทรนด์แรงมาก ห้ามสวน
    is_ranging_or_healthy = closed['adx'] < 32
    
    # เงื่อนไข CALL (Mean Reversion ร่วมกับ Trend Alignment):
    # 1. ต้องอยู่เหนือ EMA 200 (เทรนด์ใหญ่เป็นขาขึ้น)
    # 2. หลุดกรอบ BB ล่างแล้วดีดกลับ
    # 3. RSI Overbought/Oversold ฟื้นตัว
    # 4. ตลาดไม่มีเทรนด์แรงเกินไป (ADX < 32)
    call_candidate = (
        is_ranging_or_healthy and
        (closed['Close'] > closed['ema_200']) and
        (prev['Close'] <= prev['bb_lower']) and
        (closed['Close'] > closed['bb_lower']) and
        (closed['rsi'] < 40)
    )
    
    # เงื่อนไข PUT:
    # 1. ต้องอยู่ใต้ EMA 200 (เทรนด์ใหญ่เป็นขาลง)
    # 2. หลุดกรอบ BB บนแล้วกดตัวกลับ
    # 3. RSI Overbought ลดระดับลง
    put_candidate = (
        is_ranging_or_healthy and
        (closed['Close'] < closed['ema_200']) and
        (prev['Close'] >= prev['bb_upper']) and
        (closed['Close'] < closed['bb_upper']) and
        (closed['rsi'] > 60)
    )
    
    return closed, call_candidate, put_candidate, support_level, resistance_level

# --- 3. Prompt จูนขั้นสูงสำหรับ Groq (Llama-3) ---
def ask_groq_ai_advanced(pair: str, tf: str, closed_row, setup_type: str, support: float, resistance: float):
    if not groq_client:
        return None
    prompt = f"""
    You are an elite Binary Options Price Action Specialist analyzing a 1-Candle Reversal.
    Asset: {pair} | Timeframe: {tf} | Signal Setup: Potential {setup_type}
    Closed Candle Data:
    - Close: {closed_row['Close']:.5f}
    - RSI(14): {closed_row['rsi']:.2f}
    - ADX(14): {closed_row['adx']:.2f} (Trend Strength)
    - EMA 50: {closed_row['ema_50']:.5f} | EMA 200: {closed_row['ema_200']:.5f}
    - Key Support (60-bars): {support:.5f} | Key Resistance: {resistance:.5f}

    Evaluation Rules:
    1. Confirm if the setup respects the EMA 200 macro trend and is near Key Support/Resistance.
    2. Strictly filter out weak reversals or strong trending momentum.
    
    Respond ONLY in strict JSON format:
    {{"signal": "{setup_type}", "confidence": <float 0-100>, "reason": "<1-sentence clear reason mentioning Key Level or Trend>"}}
    """
    try:
        res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

# --- Web Dashboard UI ---
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
            .container {{ max-width: 1000px; margin: auto; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .stat-box {{ background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
            .btn {{ background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; }}
            .btn:hover {{ background: #2563eb; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th {{ background: #0f172a; padding: 12px; border-bottom: 2px solid #334155; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⚡ Binary AI Pro (News Filter + Trend Shield)</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">Win Rate รวม</div>
                    <div style="font-size: 28px; font-weight: bold; color: {'#4ade80' if overall_wr >= 60 else '#f87171'};">{overall_wr}%</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">จำนวนไม้ทั้งหมด</div>
                    <div style="font-size: 28px; font-weight: bold;">{total_trades}</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">ชนะ (WIN)</div>
                    <div style="font-size: 28px; font-weight: bold; color: #4ade80;">{total_wins}</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 14px;">แพ้ (LOSS)</div>
                    <div style="font-size: 28px; font-weight: bold; color: #f87171;">{total_losses}</div>
                </div>
            </div>

            <div class="card">
                <h3>🎯 ตั้งค่าโฟกัสคู่เงิน (บันทึกลง Database อัตโนมัติ)</h3>
                <form action="/update-watchlist" method="post">
                    <div style="margin: 15px 0;">
                        {pair_checkboxes}
                    </div>
                    <button type="submit" class="btn">บันทึกการตั้งค่าคู่เงิน</button>
                </form>
            </div>

            <div class="card">
                <h3>📊 สถิติ Win Rate แยกตามคู่เงิน & Timeframe</h3>
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

@app.post("/update-watchlist")
def update_watchlist(pairs: list[str] = Form(default=[])):
    global active_pairs
    active_pairs = [p for p in pairs if p in ALL_PAIRS] if pairs else list(ALL_PAIRS.keys())
    db.update_saved_watchlist(active_pairs)
    send_telegram(f"⚙️ *มีการบันทึก Watchlist ใหม่ลง Database:*\n`{', '.join(active_pairs)}`")
    return HTMLResponse(content="""<script>alert("บันทึกสำเร็จ!"); window.location.href = "/";</script>""")

# --- Telegram Helper Functions ---
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

# --- Background Threads ---
def synchronized_trading_loop():
    time.sleep(5)
    fetch_economic_calendar()
    send_telegram("🚀 *ระบบ Binary AI Pro (News Shield + EMA 200 + ADX) เริ่มทำงานแล้ว (24/7)*")
    
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
                # ตรวจสอบข่าวก่อนสแกนกราฟ
                near_news, news_reason = is_near_news(name)
                if near_news:
                    continue  # ข้ามคู่นี้ทันทีหากอยู่ในช่วงข่าว High Impact

                ticker = ALL_PAIRS[name]
                for tf in timeframes_to_check:
                    # ดึงข้อมูลย้อนหลัง 5 วันเพื่อคำนวณ EMA 200 ได้แม่นยำ
                    df = yf.download(ticker, period="5d", interval=tf, progress=False)
                    if df.empty or len(df) < 205:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    closed_candle, is_call, is_put, support, resistance = analyze_advanced_technical(df)
                    setup = "CALL" if is_call else ("PUT" if is_put else None)
                    
                    if setup:
                        ai_res = ask_groq_ai_advanced(name, tf, closed_candle, setup, support, resistance)
                        if ai_res and ai_res.get("confidence", 0) >= 80:
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
                                f"📊 **คู่เงิน:** `{name}`\n"
                                f"⏱ **Timeframe:** `{tf}` (เข้าแท่งนี้ทันที)\n"
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
                    
                    df = yf.download(ticker, period="1d", interval="1m", progress=False)
                    if df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    expiry_price = float(df.iloc[-1]['Close'])
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
                            send_telegram_direct(sender_chat_id, "✅ *ตั้งค่าโฟกัส:* เฝ้าดู **ทุกคู่เงิน** (บันทึกลง Database แล้ว)")
                        else:
                            selected = [p.strip().upper() for p in parts[1].split(",") if p.strip().upper() in ALL_PAIRS]
                            if selected:
                                active_pairs = selected
                                db.update_saved_watchlist(active_pairs)
                                pairs_text = ", ".join(active_pairs)
                                send_telegram_direct(sender_chat_id, f"🎯 *ตั้งค่าโฟกัสสำเร็จ!*\nกำลังเฝ้าเฉพาะ: `{pairs_text}` (บันทึกลง Database แล้ว)")
                            else:
                                send_telegram_direct(sender_chat_id, "⚠️ ไม่พบคู่เงินที่ระบุ กรุณาพิมพ์เช่น `/focus EUR/USD,GBP/USD`")

                    elif text == "/watchlist":
                        saved = db.get_saved_watchlist(list(ALL_PAIRS.keys()))
                        current = ", ".join(saved)
                        available = ", ".join(ALL_PAIRS.keys())
                        send_telegram_direct(sender_chat_id, f"📋 *คู่เงินที่กำลังเฝ้าดู (จาก Database):*\n`{current}`\n\n🔹 *คู่เงินทั้งหมดที่มี:* `{available}`")

                    elif text == "/stat":
                        try:
                            stats = db.fetch_winrate()
                            if not stats:
                                send_telegram_direct(sender_chat_id, "📊 *ยังไม่มีข้อมูลสถิติการเทรดที่สรุปผลแล้ว*")
                            else:
                                report = "📈 *สรุปสถิติ Win Rate รวม:*\n━━━━━━━━━━━━━━━\n"
                                for row in stats:
                                    rate = row.get('win_rate_percentage') or 0
                                    report += (
                                        f"🔹 *{row['pair']} ({row['timeframe']})*\n"
                                        f"   • ทั้งหมด: `{row['total_trades']}` ไม้\n"
                                        f"   • ชนะ: `{row['wins']}` | แพ้: `{row['losses']}` | เสมอ: `{row['draws']}`\n"
                                        f"   • Win Rate: *{rate}%*\n\n"
                                    )
                                send_telegram_direct(sender_chat_id, report)
                        except Exception as db_err:
                            send_telegram_direct(sender_chat_id, f"❌ Database Error: `{db_err}`")

                    elif text in ["/help", "/start"]:
                        msg_help = (
                            f"📌 *บอทพร้อมทำงาน (News Filter + Trend Protection)*\n"
                            f"Chat ID: `{sender_chat_id}`\n\n"
                            f"• `/focus EUR/USD,GBP/USD` - เลือกคู่เงินที่ต้องการ\n"
                            f"• `/focus all` - เฝ้าทุกคู่เงิน\n"
                            f"• `/watchlist` - ดูคู่เงินที่กำลังเฝ้าอยู่\n"
                            f"• `/stat` - ดูอัตรา Win Rate รวม\n"
                            f"• `/help` - แสดงเมนูช่วยเหลือ"
                        )
                        send_telegram_direct(sender_chat_id, msg_help)
            
            time.sleep(1)
        except Exception as e:
            time.sleep(5)

# --- Start Threads ---
threading.Thread(target=synchronized_trading_loop, daemon=True).start()
threading.Thread(target=outcome_checker_loop, daemon=True).start()
threading.Thread(target=telegram_polling_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
