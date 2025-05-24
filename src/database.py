import sqlite3

# Connect to SQLite database (creates file if not exists)
conn = sqlite3.connect("patient_history.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL
    )
''')

conn.commit()
conn.close()
