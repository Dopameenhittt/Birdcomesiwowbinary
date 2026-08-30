import os
import time
import threading
import requests
import pandas as pd
import ta
import yfinance as yf
from fastapi import FastAPI
import uvicorn
from groq import Groq
import db

# 1. โหลด Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 8080))

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# รายชื่อคู่เงินที่ต้องการเฝ้า (Forex Ticker บน Yahoo Finance)
WATCHLIST = [
    {"name": "EUR/USD", "ticker": "EURUSD=X"},
    {"name": "GBP/USD", "ticker": "GBPUSD=X"},
    {"name": "USD/JPY", "ticker": "JPY=X"},
    {"name": "AUD/USD", "ticker": "AUDUSD=X"}
]
TIMEFRAMES = ["5m", "15m"]

# 2. FastAPI Web Server ป้องกัน Render หลับ
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "running", "message": "Binary AI Bot is active 24/7"}

# 3. ฟังก์ชันส่งแจ้งเตือน Telegram
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# 4. คำนวณ Technical Indicators & Pre-filter
def analyze_technical(df: pd.DataFrame):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Pre-filter Rules
    call_candidate = (prev['Close'] <= prev['bb_lower']) and (last['Close'] > last['bb_lower']) and (last['rsi'] < 38)
    put_candidate = (prev['Close'] >= prev['bb_upper']) and (last['Close'] < last['bb_upper']) and (last['rsi'] > 62)
    
    return last, call_candidate, put_candidate

# 5. วิเคราะห์และตัดสินใจด้วย Groq AI
def ask_groq_ai(pair: str, tf: str, last_row, setup_type: str):
    if not groq_client:
        return None
    prompt = f"""
    You are an elite Binary Options Price Action Specialist.
    Asset: {pair}, Timeframe: {tf}
    Setup Type: Potential {setup_type}
    Current Price: {last_row['Close']}
    RSI(14): {last_row['rsi']:.2f}
    BB Upper: {last_row['bb_upper']:.2f}, BB Lower: {last_row['bb_lower']:.2f}
    EMA 50: {last_row['ema_50']:.2f}

    Evaluate the strength of this mean-reversion setup.
    Respond ONLY in strict JSON format:
    {{"signal": "{setup_type}", "confidence": <float 0-100>, "reason": "<1 sentence reasoning>"}}
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

# 6. Background Loop สำหรับเฝ้ากราฟ
def trading_bot_loop():
    time.sleep(5)
    db.init_db()
    send_telegram("🚀 *บอทวิเคราะห์ Binary AI เริ่มทำงานบน Render แล้ว (24/7)*")
    
    while True:
        try:
            for item in WATCHLIST:
                for tf in TIMEFRAMES:
                    # ดึงข้อมูลย้อนหลัง
                    period = "1d" if tf == "5m" else "5d"
                    df = yf.download(item["ticker"], period=period, interval=tf, progress=False)
                    
                    if df.empty or len(df) < 50:
                        continue
                        
                    # ปรับแต่ง MultiIndex ของ yfinance หากมี
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    last, is_call, is_put = analyze_technical(df)
                    setup = "CALL" if is_call else ("PUT" if is_put else None)
                    
                    if setup:
                        ai_res = ask_groq_ai(item["name"], tf, last, setup)
                        if ai_res and ai_res.get("confidence", 0) >= 80:
                            entry_price = float(last['Close'])
                            conf = float(ai_res['confidence'])
                            reason = ai_res.get('reason', '')
                            
                            # บันทึกลง Neon Database
                            sig_id = db.save_signal(item["name"], tf, setup, entry_price, conf, reason)
                            
                            # ส่งสัญญาณเข้า Telegram
                            icon = "🟢" if setup == "CALL" else "🔴"
                            msg = (
                                f"{icon} *สัญญาณเข้าเทรด ({setup})*\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"📊 **คู่เงิน:** `{item['name']}`\n"
                                f"⏱ **Timeframe:** `{tf}`\n"
                                f"💵 **ราคาปัจจุบัน:** `{entry_price:.5f}`\n"
                                f"🎯 **ความมั่นใจ:** `{conf}%`\n"
                                f"💡 **เหตุผล:** {reason}\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"🆔 `Ref ID: #{sig_id}`"
                            )
                            send_telegram(msg)
                            
            # รอ 60 วินาทีก่อนตรวจแท่งเทียนรอบถัดไป
            time.sleep(60)
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(30)

# รัน Loop ใน Thread แยก
threading.Thread(target=trading_bot_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)