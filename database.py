import sqlite3, os
DB_PATH = os.path.join("instance", "app.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    full_name     TEXT,
    created_at    DATETIME DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS predictions_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sqft        INTEGER NOT NULL,
    bedrooms    INTEGER NOT NULL,
    bathrooms   INTEGER NOT NULL,
    year_built  INTEGER NOT NULL,
    zip_code    INTEGER NOT NULL,
    predicted_price REAL NOT NULL,
    created_at  DATETIME DEFAULT (datetime('now'))
);
"""
def get_db():
    os.makedirs("instance", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
    print(f"Database initialised at {DB_PATH}")
if __name__ == "__main__":
    init_db()
