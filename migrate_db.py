import sqlite3
import os

DB_NAME = 'engagement.db'

def migrate():
    if not os.path.exists(DB_NAME):
        print("Database does not exist. No migration needed.")
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        # Check if column exists
        c.execute("PRAGMA table_info(engagement_logs)")
        columns = [info[1] for info in c.fetchall()]
        
        if 'engagement_level' not in columns:
            print("Adding engagement_level column to engagement_logs...")
            c.execute("ALTER TABLE engagement_logs ADD COLUMN engagement_level TEXT")
            conn.commit()
            print("Migration successful.")
        else:
            print("Column 'engagement_level' already exists.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
