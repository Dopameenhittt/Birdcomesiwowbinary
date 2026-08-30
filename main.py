import os
import time
import datetime
import threading
import requests
import pandas as pd
import ta
import yfinance as yf
from fastapi import FastAPI
import uvicorn
from groq import Groq
import db

# --- Config & Environment Variables ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080))

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

WATCHLIST = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X"
}
TIMEFRAMES = ["5m", "15m"]

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "active", "service": "Binary AI Bot 24/7"}

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

# --- Technical Analysis & Groq ---
def analyze_technical(df: pd.DataFrame):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    call_candidate = (prev['Close'] <= prev['bb_lower']) and (last['Close'] > last['bb_lower']) and (last['rsi'] < 38)
    put_candidate = (prev['Close'] >= prev['bb_upper']) and (last['Close'] < last['bb_upper']) and (last['rsi'] > 62)
    return last, call_candidate, put_candidate

def ask_groq_ai(pair: str, tf: str, last_row, setup_type: str):
    if not groq_client:
        return None
    prompt = f"""
    You are an elite Binary Options Price Action Specialist.
    Asset: {pair}, Timeframe: {tf}, Setup: Potential {setup_type}
    Current Price: {last_row['Close']}
    RSI(14): {last_row['rsi']:.2f}, BB Upper: {last_row['bb_upper']:.2f}, BB Lower: {last_row['bb_lower']:.2f}, EMA 50: {last_row['ema_50']:.2f}

    Evaluate strength for 1-candle expiry. Respond ONLY in JSON:
    {{"signal": "{setup_type}", "confidence": <float 0-100>, "reason": "<short sentence>"}}
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

# --- Thread 1: Signal Scanner Loop ---
def trading_bot_loop():
    time.sleep(5)
    db.init_db()
    send_telegram("🚀 *ระบบ Binary AI Scanner พร้อมทำงานแล้ว (24/7)*")
    
    while True:
        try:
            for name, ticker in WATCHLIST.items():
                for tf in TIMEFRAMES:
                    period = "1d" if tf == "5m" else "5d"
                    df = yf.download(ticker, period=period, interval=tf, progress=False)
                    if df.empty or len(df) < 50:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    last, is_call, is_put = analyze_technical(df)
                    setup = "CALL" if is_call else ("PUT" if is_put else None)
                    
                    if setup:
                        ai_res = ask_groq_ai(name, tf, last, setup)
                        if ai_res and ai_res.get("confidence", 0) >= 80:
                            entry_price = float(last['Close'])
                            conf = float(ai_res['confidence'])
                            reason = ai_res.get('reason', '')
                            
                            sig_id = db.save_signal(name, tf, setup, entry_price, conf, reason)
                            
                            icon = "🟢" if setup == "CALL" else "🔴"
                            msg = (
                                f"{icon} *สัญญาณเข้าเทรด ({setup})*\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"📊 **คู่เงิน:** `{name}`\n"
                                f"⏱ **Expiry:** `{tf}`\n"
                                f"💵 **ราคาเข้า:** `{entry_price:.5f}`\n"
                                f"🎯 **ความมั่นใจ:** `{conf}%`\n"
                                f"💡 **เหตุผล:** {reason}\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"🆔 `Ref ID: #{sig_id}`"
                            )
                            send_telegram(msg)
            time.sleep(60)
        except Exception as e:
            print(f"Scanner error: {e}")
            time.sleep(30)

# --- Thread 2: Outcome Checker Loop (ตรวจผล WIN/LOSS) ---
def outcome_checker_loop():
    time.sleep(10)
    while True:
        try:
            pending = db.get_pending_signals()
            now = datetime.datetime.utcnow()
            
            for sig in pending:
                duration_min = 5 if sig["timeframe"] == "5m" else 15
                target_expiry = sig["created_at"] + datetime.timedelta(minutes=duration_min)
                
                # เช็คเฉพาะสัญญาณที่เวลาผ่านไปจนหมดแท่งเทียนแล้ว
                if now >= target_expiry:
                    ticker = WATCHLIST.get(sig["pair"])
                    if not ticker:
                        continue
                    
                    # ดึงราคาปัจจุบันเพื่อเทียบผลลัพธ์
                    df = yf.download(ticker, period="1d", interval="1m", progress=False)
                    if df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    expiry_price = float(df.iloc[-1]['Close'])
                    entry_price = float(sig['entry_price'])
                    
                    # ประเมินผลลัพธ์
                    if sig['direction'] == "CALL":
                        result = "WIN" if expiry_price > entry_price else ("LOSS" if expiry_price < entry_price else "DRAW")
                    else: # PUT
                        result = "WIN" if expiry_price < entry_price else ("LOSS" if expiry_price > entry_price else "DRAW")
                    
                    # บันทึกผลลง Neon
                    db.save_result(sig['id'], expiry_price, result)
                    
                    # แจ้งเตือนผลลัพธ์เข้า Telegram
                    res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "⚪ DRAW")
                    msg = (
                        f"🔔 *สรุปผลการเทรด: {res_icon}*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🆔 `Ref ID: #{sig['id']}`\n"
                        f"📊 **คู่เงิน:** `{sig['pair']}` ({sig['timeframe']})\n"
                        f"🎯 **คำสั่ง:** `{sig['direction']}`\n"
                        f"💵 **ราคาเข้า:** `{entry_price:.5f}`\n"
                        f"🏁 **ราคาปิด:** `{expiry_price:.5f}`"
                    )
                    send_telegram(msg)
            
            time.sleep(30)
        except Exception as e:
            print(f"Outcome checker error: {e}")
            time.sleep(30)

# --- Thread 3: Telegram Command Handler (/stat, /help) ---
def telegram_polling_loop():
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
                    text = msg.get("text", "")
                    
                    if text == "/stat":
                        stats = db.fetch_winrate()
                        if not stats:
                            send_telegram("📊 *ยังไม่มีข้อมูลสถิติการเทรดที่สรุปผลแล้ว*")
                        else:
                            report = "📈 *สรุปสถิติ Win Rate รวม:*\n━━━━━━━━━━━━━━━\n"
                            for row in stats:
                                rate = row['win_rate_percentage'] or 0
                                report += (
                                    f"🔹 *{row['pair']} ({row['timeframe']})*\n"
                                    f"   • ทั้งหมด: `{row['total_trades']}` ไม้\n"
                                    f"   • ชนะ: `{row['wins']}` | แพ้: `{row['losses']}` | เสมอ: `{row['draws']}`\n"
                                    f"   • Win Rate: *{rate}%*\n\n"
                                )
                            send_telegram(report)
                    elif text == "/help":
                        send_telegram("📌 *คำสั่งที่ใช้งานได้:*\n`/stat` - ดูอัตรา Win Rate ปัจจุบัน\n`/help` - ดูคำอธิบายการใช้งาน")
            
            time.sleep(1)
        except Exception as e:
            time.sleep(5)

# --- เริ่มรันทุก Thread ---
threading.Thread(target=trading_bot_loop, daemon=True).start()
threading.Thread(target=outcome_checker_loop, daemon=True).start()
threading.Thread(target=telegram_polling_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
