import sqlite3

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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if the file already exists in the database
    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE name = ?", (file_name,))
    file_exists = cursor.fetchone()[0] > 0

    if file_exists:
        conn.close()
        raise ValueError(f"A file with the name '{file_name}' already exists.")

    # If the file doesn't exist, insert it into the database
    cursor.execute("INSERT INTO uploaded_files (name, content) VALUES (?, ?)", (file_name, file_content))
    conn.commit()
    conn.close()

def update_file_in_db(file_id, file_content):
    """Update the content of an existing file in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Update the file content
    cursor.execute("UPDATE uploaded_files SET content = ? WHERE id = ?", (file_content, file_id))
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

def delete_file_from_db(file_id):
    """Delete a file from the database by its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
    conn.commit()  # Ensure the changes are saved to the database
    conn.close()