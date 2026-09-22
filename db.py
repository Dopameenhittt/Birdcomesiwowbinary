import os
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

# --- Connection Pool ---
# ใช้ pool แทนการเปิด/ปิด connection ใหม่ทุกครั้ง เพื่อลด overhead
# และป้องกัน connection ค้าง (leak) เวลามีการเรียกใช้ถี่ๆ จากหลาย thread พร้อมกัน
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set!")
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    return _pool

def get_connection():
    return _get_pool().getconn()

def release_connection(conn):
    if conn is not None:
        _get_pool().putconn(conn)

def init_db(default_pairs=None):
    if default_pairs is None:
        default_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "BTC/USD"]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 1. ตาราง signals
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    pair VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    direction VARCHAR(10) NOT NULL,
                    entry_price NUMERIC(12, 5) NOT NULL,
                    confidence NUMERIC(5, 2) NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. ตาราง trade_results
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade_results (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER REFERENCES signals(id) ON DELETE CASCADE UNIQUE,
                    expiry_price NUMERIC(12, 5) NOT NULL,
                    result VARCHAR(10) NOT NULL,
                    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. ตาราง bot_settings สำหรับจำค่า Watchlist ถาวร
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key VARCHAR(50) PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # บันทึกค่าเริ่มต้นถ้ายังไม่มี
            cur.execute("""
                INSERT INTO bot_settings (key, value)
                VALUES ('active_watchlist', %s)
                ON CONFLICT (key) DO NOTHING;
            """, (json.dumps(default_pairs),))

            # 3b. ตาราง signal_candidates — เก็บ "ทุก" สัญญาณที่ผ่านเงื่อนไข ZigZag+RSI
            # ไม่ว่า Groq AI จะอนุมัติ (confidence >= 70) หรือไม่ก็ตาม
            # แยกจากตาราง signals โดยเจตนา เพราะ signals คือตารางที่ AI-Iqoption-running
            # โพลไปยิงออเดอร์เงินจริง — ห้ามเอาไม้ที่ Groq ไม่ผ่านไปปนในนั้นเด็ดขาด
            # ตารางนี้มีไว้เทียบสถิติภายหลังเท่านั้นว่า Groq filter ช่วยจริงหรือเปล่า
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_candidates (
                    id SERIAL PRIMARY KEY,
                    pair VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    direction VARCHAR(10) NOT NULL,
                    entry_price NUMERIC(12, 5) NOT NULL,
                    confidence NUMERIC(5, 2),
                    ai_approved BOOLEAN NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS candidate_results (
                    id SERIAL PRIMARY KEY,
                    candidate_id INTEGER REFERENCES signal_candidates(id) ON DELETE CASCADE UNIQUE,
                    expiry_price NUMERIC(12, 5) NOT NULL,
                    result VARCHAR(10) NOT NULL,
                    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. View สรุปสถิติ
            cur.execute("""
                CREATE OR REPLACE VIEW winrate_summary AS
                SELECT 
                    s.pair,
                    s.timeframe,
                    COUNT(r.id) AS total_trades,
                    COUNT(CASE WHEN r.result = 'WIN' THEN 1 END) AS wins,
                    COUNT(CASE WHEN r.result = 'LOSS' THEN 1 END) AS losses,
                    COUNT(CASE WHEN r.result = 'DRAW' THEN 1 END) AS draws,
                    ROUND(
                        (COUNT(CASE WHEN r.result = 'WIN' THEN 1 END)::NUMERIC / 
                        NULLIF(COUNT(CASE WHEN r.result IN ('WIN', 'LOSS') THEN 1 END), 0)) * 100, 
                        2
                    ) AS win_rate_percentage
                FROM signals s
                INNER JOIN trade_results r ON s.id = r.signal_id
                GROUP BY s.pair, s.timeframe;
            """)

            # View เทียบ win rate ของไม้ที่ Groq "อนุมัติ" กับ "ปฏิเสธ" —
            # ใช้ตอบคำถามว่า Groq filter ช่วยจริงหรือแค่สุ่มตัดโอกาสทิ้ง
            cur.execute("""
                CREATE OR REPLACE VIEW candidate_winrate_summary AS
                SELECT
                    c.ai_approved,
                    COUNT(r.id) AS total_trades,
                    COUNT(CASE WHEN r.result = 'WIN' THEN 1 END) AS wins,
                    COUNT(CASE WHEN r.result = 'LOSS' THEN 1 END) AS losses,
                    COUNT(CASE WHEN r.result = 'DRAW' THEN 1 END) AS draws,
                    ROUND(
                        (COUNT(CASE WHEN r.result = 'WIN' THEN 1 END)::NUMERIC /
                        NULLIF(COUNT(CASE WHEN r.result IN ('WIN', 'LOSS') THEN 1 END), 0)) * 100,
                        2
                    ) AS win_rate_percentage
                FROM signal_candidates c
                INNER JOIN candidate_results r ON c.id = r.candidate_id
                GROUP BY c.ai_approved;
            """)
            conn.commit()
    finally:
        release_connection(conn)

def get_saved_watchlist(default_pairs):
    """ดึงรายชื่อคู่เงินที่บันทึกไว้ใน Neon"""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM bot_settings WHERE key = 'active_watchlist';")
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception as e:
        print(f"Error reading watchlist from DB: {e}")
    finally:
        release_connection(conn)
    return default_pairs

def update_saved_watchlist(pairs_list):
    """บันทึกรายชื่อคู่เงินชุดใหม่ลง Neon"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_settings (key, value, updated_at)
                VALUES ('active_watchlist', %s, CURRENT_TIMESTAMP)
                ON CONFLICT (key) 
                DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;
            """, (json.dumps(pairs_list),))
            conn.commit()
    finally:
        release_connection(conn)

def save_signal(pair: str, timeframe: str, direction: str, entry_price: float, confidence: float, reason: str) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (pair, timeframe, direction, entry_price, confidence, reason)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (pair, timeframe, direction, entry_price, confidence, reason))
            signal_id = cur.fetchone()[0]
            conn.commit()
        return signal_id
    finally:
        release_connection(conn)

def get_pending_signals():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.id, s.pair, s.timeframe, s.direction, s.entry_price, s.created_at
                FROM signals s
                LEFT JOIN trade_results r ON s.id = r.signal_id
                WHERE r.id IS NULL;
            """)
            results = cur.fetchall()
        return results
    finally:
        release_connection(conn)

def save_result(signal_id: int, expiry_price: float, result: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_results (signal_id, expiry_price, result)
                VALUES (%s, %s, %s)
                ON CONFLICT (signal_id) DO NOTHING;
            """, (signal_id, expiry_price, result))
            conn.commit()
    finally:
        release_connection(conn)

def fetch_winrate():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM winrate_summary;")
            results = cur.fetchall()
        return results
    finally:
        release_connection(conn)

# --- signal_candidates: บันทึกทุกสัญญาณ ZigZag+RSI ไม่ว่า Groq จะอนุมัติหรือไม่ ---
# (สำหรับวิเคราะห์ย้อนหลังเท่านั้น ไม่เกี่ยวกับ signals ที่ใช้เทรดเงินจริง)

def save_candidate(pair: str, timeframe: str, direction: str, entry_price: float,
                    confidence, ai_approved: bool, reason: str) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signal_candidates (pair, timeframe, direction, entry_price, confidence, ai_approved, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (pair, timeframe, direction, entry_price, confidence, ai_approved, reason))
            candidate_id = cur.fetchone()[0]
            conn.commit()
        return candidate_id
    finally:
        release_connection(conn)

def get_pending_candidates():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT c.id, c.pair, c.timeframe, c.direction, c.entry_price, c.created_at
                FROM signal_candidates c
                LEFT JOIN candidate_results r ON c.id = r.candidate_id
                WHERE r.id IS NULL;
            """)
            results = cur.fetchall()
        return results
    finally:
        release_connection(conn)

def save_candidate_result(candidate_id: int, expiry_price: float, result: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO candidate_results (candidate_id, expiry_price, result)
                VALUES (%s, %s, %s)
                ON CONFLICT (candidate_id) DO NOTHING;
            """, (candidate_id, expiry_price, result))
            conn.commit()
    finally:
        release_connection(conn)

def fetch_candidate_winrate():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM candidate_winrate_summary ORDER BY ai_approved DESC;")
            results = cur.fetchall()
        return results
    finally:
        release_connection(conn)
