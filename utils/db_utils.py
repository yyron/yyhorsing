import sqlite3

# Create or connect to the database
DB_PATH = "uploaded_files.db"

def init_db():
    """Initialize the database with the required table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_file_to_db(file_name, file_content):
    """Save a file to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO uploaded_files (name, content) VALUES (?, ?)", (file_name, file_content))
    conn.commit()
    conn.close()

def get_all_files():
    """Retrieve all files from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM uploaded_files")
    files = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return files

def fetch_file_content(file_id):
    """Fetch the content of a file by its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM uploaded_files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None