import sqlite3
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = 'engagement.db'

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def init_db():
    try:
        conn = get_db_connection()
        if not conn:
            return
        
        c = conn.cursor()
        
        # Table for Users
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL
            )
        ''')
        
        # Table for Sessions
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                start_time REAL NOT NULL,
                end_time REAL,
                avg_engagement INTEGER,
                status_summary TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Check if user_id column exists in sessions table (for backward compatibility)
        cursor = c.execute("PRAGMA table_info(sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'user_id' not in columns:
            c.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users (id)")
        
        # Table for Engagement Logs
        c.execute('''
            CREATE TABLE IF NOT EXISTS engagement_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                score INTEGER,
                emotion TEXT,
                engagement_level TEXT,
                people_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def create_session(start_time, user_id=None):
    try:
        conn = get_db_connection()
        if not conn: return None
        
        c = conn.cursor()
        if user_id:
            c.execute('INSERT INTO sessions (start_time, user_id) VALUES (?, ?)', (start_time, user_id))
        else:
            c.execute('INSERT INTO sessions (start_time) VALUES (?)', (start_time,))
        session_id = c.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Created session {session_id} at {start_time}")
        return session_id
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return None

def end_session(session_id, end_time, avg_engagement, status_summary):
    try:
        conn = get_db_connection()
        if not conn: return
        
        c = conn.cursor()
        c.execute('''
            UPDATE sessions 
            SET end_time = ?, avg_engagement = ?, status_summary = ?
            WHERE id = ?
        ''', (end_time, avg_engagement, status_summary, session_id))
        conn.commit()
        conn.close()
        logger.info(f"Ended session {session_id}. Avg Score: {avg_engagement}")
    except Exception as e:
        logger.error(f"Error ending session {session_id}: {e}")

def log_engagement(session_id, timestamp, score, emotion, people_count, engagement_level="Unknown"):
    try:
        conn = get_db_connection()
        if not conn: return
        
        c = conn.cursor()
        c.execute('''
            INSERT INTO engagement_logs (session_id, timestamp, score, emotion, people_count, engagement_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, timestamp, score, emotion, people_count, engagement_level))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging engagement for session {session_id}: {e}")

def get_all_sessions(user_id=None):
    try:
        conn = get_db_connection()
        if not conn: return []
        
        if user_id:
            sessions = conn.execute('SELECT * FROM sessions WHERE user_id = ? ORDER BY start_time DESC', (user_id,)).fetchall()
        else:
            sessions = conn.execute('SELECT * FROM sessions ORDER BY start_time DESC').fetchall()
        conn.close()
        return [dict(s) for s in sessions]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return []

def get_session_details(session_id):
    try:
        conn = get_db_connection()
        if not conn: return None
        
        session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
        logs = conn.execute('SELECT * FROM engagement_logs WHERE session_id = ? ORDER BY timestamp ASC', (session_id,)).fetchall()
        conn.close()
        
        if session:
            return {'session': dict(session), 'logs': [dict(log) for log in logs]}
        return None
    except Exception as e:
        logger.error(f"Error fetching session details {session_id}: {e}")
        return None

def create_user(email, password_hash, name):
    try:
        conn = get_db_connection()
        if not conn: return False
        c = conn.cursor()
        c.execute('INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)', (email, password_hash, name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False

def get_user_by_email(email):
    try:
        conn = get_db_connection()
        if not conn: return None
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        logger.error(f"Error getting user by email: {e}")
        return None

def get_user_by_id(user_id):
    try:
        conn = get_db_connection()
        if not conn: return None
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        logger.error(f"Error getting user by id: {e}")
        return None
