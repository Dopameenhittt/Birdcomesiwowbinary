import os
import psycopg2
from psycopg2.extras import RealDictCursor

# ดึง URL จาก Environment Variable บน Render
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล Neon Postgres"""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set!")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """สร้างตารางและ View อัตโนมัติเมื่อ Service เริ่มทำงาน"""
    conn = get_connection()
    with conn.cursor() as cur:
        # ตาราง signals
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

        # ตาราง trade_results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_results (
                id SERIAL PRIMARY KEY,
                signal_id INTEGER REFERENCES signals(id) ON DELETE CASCADE,
                expiry_price NUMERIC(12, 5) NOT NULL,
                result VARCHAR(10) NOT NULL,
                resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # View สรุปสถิติ Win Rate รวม
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
        conn.commit()
    conn.close()
    print("Database tables initialized successfully.")

def save_signal(pair: str, timeframe: str, direction: str, entry_price: float, confidence: float, reason: str) -> int:
    """บันทึกสัญญาณเข้าเทรด และคืนค่า signal_id"""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO signals (pair, timeframe, direction, entry_price, confidence, reason)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (pair, timeframe, direction, entry_price, confidence, reason))
        signal_id = cur.fetchone()[0]
        conn.commit()
    conn.close()
    return signal_id

def save_result(signal_id: int, expiry_price: float, result: str):
    """บันทึกผลการเทรด (WIN / LOSS / DRAW)"""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trade_results (signal_id, expiry_price, result)
            VALUES (%s, %s, %s);
        """, (signal_id, expiry_price, result))
        conn.commit()
    conn.close()

def fetch_winrate():
    """ดึงสถิติ Win Rate ล่าสุดไปแสดงใน Telegram"""
    conn = get_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM winrate_summary;")
        results = cur.fetchall()
    conn.close()
    return results