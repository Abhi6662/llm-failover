"""SQLite-based telemetry tracking."""

import sqlite3
import time
from pathlib import Path
from typing import Optional


class Telemetry:
    """Track token usage and latency in SQLite."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize telemetry.
        
        Args:
            db_path: Path to SQLite database (default: ~/.llm-failover.db)
        """
        if db_path is None:
            db_path = str(Path.home() / ".llm-failover.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Create telemetry table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()
    
    def log(self, endpoint: str, tokens: int, latency_ms: int) -> None:
        """Log a request.
        
        Args:
            endpoint: Endpoint URL
            tokens: Total tokens used
            latency_ms: Request latency in milliseconds
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO requests (endpoint, tokens, latency_ms, timestamp) VALUES (?, ?, ?, ?)",
                (endpoint, tokens, latency_ms, time.time())
            )
            conn.commit()
    
    def get_stats(self, endpoint: Optional[str] = None) -> dict:
        """Get telemetry statistics.
        
        Args:
            endpoint: Optional endpoint to filter by
            
        Returns:
            Dictionary with total_tokens, avg_latency_ms, request_count
        """
        with sqlite3.connect(self.db_path) as conn:
            if endpoint:
                cursor = conn.execute(
                    "SELECT SUM(tokens), AVG(latency_ms), COUNT(*) FROM requests WHERE endpoint = ?",
                    (endpoint,)
                )
            else:
                cursor = conn.execute(
                    "SELECT SUM(tokens), AVG(latency_ms), COUNT(*) FROM requests"
                )
            
            row = cursor.fetchone()
            return {
                "total_tokens": row[0] or 0,
                "avg_latency_ms": int(row[1] or 0),
                "request_count": row[2] or 0
            }
