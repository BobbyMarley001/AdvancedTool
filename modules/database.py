import sqlite3
from datetime import datetime
import os

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                file TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def log_action(self, action, file_path):
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO logs (action, file, timestamp) VALUES (?, ?, ?)", (action, file_path, timestamp))
            self.conn.commit()
        except Exception as e:
            print(f"خطا در ثبت لاگ: {str(e)}")  # این خطا فقط تو کنسول نمایش داده می‌شه

    def get_logs(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM logs")
            rows = cursor.fetchall()
            return [{"id": row[0], "action": row[1], "file": row[2], "timestamp": row[3]} for row in rows]
        except Exception as e:
            print(f"خطا در دریافت لاگ‌ها: {str(e)}")
            return []