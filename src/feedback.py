import sqlite3
from datetime import datetime
import uuid


def init_feedback_db():
    conn = sqlite3.connect('feedback.db')
    conn.execute("PRAGMA foreign_keys = ON")
    c = conn.cursor()

    # Create chat_history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            feedback_value INTEGER, -- 1 for like, 0 for dislike
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chat_history(id)
        )
    ''')

    conn.commit()
    conn.close()


init_feedback_db()


def save_chat_history(user_id, query, response):
    try:
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Chuyển đổi response thành chuỗi nếu cần
        if not isinstance(response, str):
            response = str(response)

        print(
            f"Lưu chat_history: user_id={user_id}, query={query}, response={response}, chat_id={chat_id}")

        # Thực hiện câu lệnh SQL
        c.execute('''
            INSERT INTO chat_history (id, user_id, query, response, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (chat_id, user_id, query, response, timestamp))

        conn.commit()
        print(f"Đã lưu chat_history với chat_id: {chat_id}")
        return chat_id
    except Exception as e:
        print(f"Lỗi khi lưu chat_history: {e}")
        return None
    finally:
        conn.close()


def save_feedback(chat_id: str, feedback_value: int) -> bool:
    print(f"\n=== Starting save_feedback ===")
    print(f"Input - chat_id: {chat_id}, feedback_value: {feedback_value}")

    if not chat_id:
        print("Error: chat_id is None or empty")
        return False

    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()

        # Kiểm tra chat_id trong chat_history
        cursor.execute("SELECT id FROM chat_history WHERE id = ?", (chat_id,))
        chat_exists = cursor.fetchone()
        print(f"Chat history check result: {chat_exists}")

        if not chat_exists:
            print(f"Error: Chat ID {chat_id} not found in chat_history")
            return False

        # Kiểm tra feedback hiện tại
        cursor.execute("SELECT id FROM feedback WHERE chat_id = ?", (chat_id,))
        existing = cursor.fetchone()
        print(f"Existing feedback check result: {existing}")

        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if existing:
            query = """
                UPDATE feedback 
                SET feedback_value = ?, timestamp = ?
                WHERE chat_id = ?
            """
            params = (feedback_value, timestamp, chat_id)
            print("Executing UPDATE query")
        else:
            query = """
                INSERT INTO feedback (id, chat_id, feedback_value, timestamp)
                VALUES (?, ?, ?, ?)
            """
            params = (feedback_id, chat_id, feedback_value, timestamp)
            print("Executing INSERT query")

        cursor.execute(query, params)
        conn.commit()

        # Verify the operation
        cursor.execute("SELECT * FROM feedback WHERE chat_id = ?", (chat_id,))
        result = cursor.fetchone()
        print(f"Verification query result: {result}")

        return True

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        print("=== Ending save_feedback ===\n")
        conn.close()


if __name__ == "__main__":
    init_feedback_db()
