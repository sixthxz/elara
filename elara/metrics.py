"""
MetricsLogger — per-cycle RRG observables and aggregate statistics.

Usage (in-memory):
    ml = MetricsLogger()
    compressed, metrics = gk.evaluate(sa, sb)
    ml.record(metrics, compressed)
    print(f"lock_frac={ml.lock_frac:.3f}")
    ml.save("metrics.json")

Usage (SQLite-backed):
    ml = MetricsLogger(db_path="elara.db", session_id="sess-001")
    ml.record(metrics, compressed)
    ml.save("metrics.json")  # exports to JSON
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _init_db(db_path: str) -> Tuple[sqlite3.Connection, str]:
    """Initialize database and sessions table. Returns (conn, session_id)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            model TEXT,
            params_json TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            delta REAL NOT NULL,
            cusum REAL NOT NULL,
            rho_star REAL NOT NULL,
            d_rho REAL NOT NULL,
            d_rho_series TEXT NOT NULL,
            d_rho_meta REAL NOT NULL,
            reff REAL NOT NULL,
            regime TEXT NOT NULL,
            juncture INTEGER NOT NULL,
            compressed INTEGER NOT NULL,
            lyapunov_v REAL NOT NULL DEFAULT 0.0,
            bytes_saved INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Migrations: add columns to pre-existing databases
    for col_sql in [
        "ALTER TABLE records ADD COLUMN lyapunov_v REAL DEFAULT 0.0",
        "ALTER TABLE records ADD COLUMN bytes_saved INTEGER DEFAULT 0",
    ]:
        try:
            cursor.execute(col_sql)
        except Exception:
            pass  # column already exists

    conn.commit()
    return conn, None


class MetricsLogger:
    """Records per-cycle RRG observables and computes aggregate stats.

    Standalone — no dependency on Gatekeeper. Feed it the metrics dict
    returned by gk.evaluate() and the compression decision.

    lock_frac is the primary KPI: fraction of cycles where delta <= 0
    and regime is Sufficient (i.e., the gatekeeper said compress=True).
    Expected from COT-01 pilot: FAITHFUL ≈ 0.667, WRONG ≈ 0.533, UNFAITHFUL ≈ 0.300.

    Can be backed by SQLite (when db_path provided) or in-memory (default).
    """

    def __init__(self, db_path: Optional[str] = None, session_id: Optional[str] = None,
                 model: Optional[str] = None, params: Optional[dict] = None) -> None:
        self._records: List[dict] = []
        self._db_path = db_path
        self._session_id = session_id or str(uuid.uuid4())
        self._conn: Optional[sqlite3.Connection] = None

        if db_path:
            self._conn, _ = _init_db(db_path)
            self._conn.row_factory = sqlite3.Row

            now = datetime.utcnow().isoformat()
            params_json = json.dumps(params) if params else None

            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, started_at, model, params_json) "
                "VALUES (?, ?, ?, ?)",
                (self._session_id, now, model, params_json)
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, metrics: dict, compressed: bool) -> None:
        """Append one cycle.

        Args:
            metrics: dict returned by Gatekeeper.evaluate()
            compressed: True iff the gatekeeper decided to compress
        """
        d_rho    = metrics.get("d_rho", 0.0)
        rho_star = metrics.get("rho_star", 0.0)
        record = {
            "cycle":        len(self._records),
            "timestamp":    time.time(),
            "delta":        metrics.get("delta", 0.0),
            "cusum":        metrics.get("cusum", 0.0),
            "rho_star":     rho_star,
            "d_rho":        d_rho,
            "d_rho_series": metrics.get("d_rho_series", []),
            "d_rho_meta":   metrics.get("d_rho_meta", 0.0),
            "reff":         metrics.get("reff", 2.0),
            "regime":       metrics.get("regime", "Background"),
            "juncture":     bool(metrics.get("juncture", False)),
            "compressed":   compressed,
            # Lyapunov potential V(t) = d_rho + (1 - |rho*|).  V* ~= 0 at Sufficient attractor.
            "lyapunov_v":   metrics.get("lyapunov_v", d_rho + (1.0 - abs(rho_star))),
            "bytes_saved":  0,
        }
        self._records.append(record)

        if self._conn:
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT INTO records "
                "(session_id, cycle, timestamp, delta, cusum, rho_star, d_rho, "
                "d_rho_series, d_rho_meta, reff, regime, juncture, compressed, lyapunov_v, bytes_saved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._session_id,
                    record["cycle"],
                    record["timestamp"],
                    record["delta"],
                    record["cusum"],
                    record["rho_star"],
                    record["d_rho"],
                    json.dumps(record["d_rho_series"]),
                    record["d_rho_meta"],
                    record["reff"],
                    record["regime"],
                    int(record["juncture"]),
                    int(record["compressed"]),
                    record["lyapunov_v"],
                    0,
                )
            )
            self._conn.commit()

    def update_last_bytes_saved(self, bytes_saved: int) -> None:
        """Set bytes_saved on the most-recently recorded cycle.

        Called by the proxy after the compression guard confirms the candidate
        body is actually smaller than the original.
        """
        if not self._records:
            return
        self._records[-1]["bytes_saved"] = bytes_saved
        if self._conn:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE records SET bytes_saved = ? "
                "WHERE id = (SELECT MAX(id) FROM records WHERE session_id = ?)",
                (bytes_saved, self._session_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Aggregate KPIs
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return len(self._records)

    @property
    def lock_frac(self) -> float:
        """Fraction of cycles where compressed=True."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r["compressed"]) / len(self._records)

    @property
    def total_bytes_saved(self) -> int:
        return sum(r.get("bytes_saved", 0) for r in self._records)

    @property
    def juncture_count(self) -> int:
        return sum(1 for r in self._records if r["juncture"])

    def by_regime(self) -> Dict[str, dict]:
        """Per-regime aggregate stats.

        Returns a dict keyed by regime name, each value containing:
          count, lock_frac, mean_delta, mean_rho_star, mean_d_rho
        """
        buckets: Dict[str, List[dict]] = {}
        for r in self._records:
            regime = r["regime"]
            buckets.setdefault(regime, []).append(r)

        result = {}
        for regime, records in buckets.items():
            n = len(records)
            result[regime] = {
                "count":       n,
                "lock_frac":   sum(1 for r in records if r["compressed"]) / n,
                "mean_delta":  sum(r["delta"] for r in records) / n,
                "mean_rho_star": sum(r["rho_star"] for r in records) / n,
                "mean_d_rho":  sum(r["d_rho"] for r in records) / n,
            }
        return result

    def summary(self) -> dict:
        """Full aggregate summary dict."""
        return {
            "cycle_count":      self.cycle_count,
            "lock_frac":        self.lock_frac,
            "juncture_count":   self.juncture_count,
            "total_bytes_saved": self.total_bytes_saved,
            "by_regime":        self.by_regime(),
        }

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def report(self) -> str:
        """One-page text summary suitable for printing or logging."""
        lines = [
            f"MetricsLogger: {self.cycle_count} cycles  "
            f"lock_frac={self.lock_frac:.3f}  "
            f"junctures={self.juncture_count}  "
            f"bytes_saved={self.total_bytes_saved}"
        ]
        for regime, stats in sorted(self.by_regime().items()):
            lines.append(
                f"  {regime:<14} "
                f"n={stats['count']:>3}  "
                f"lock={stats['lock_frac']:.0%}  "
                f"delta={stats['mean_delta']:+.4f}  "
                f"rho*={stats['mean_rho_star']:+.3f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize all cycle records to JSON.

        If backed by SQLite, exports from database.
        Otherwise exports from in-memory records.
        """
        records = self._records

        if self._conn:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM records WHERE session_id = ? ORDER BY cycle ASC",
                (self._session_id,)
            )
            rows = cursor.fetchall()
            records = []
            for row in rows:
                record = dict(row)
                record["d_rho_series"] = json.loads(record["d_rho_series"])
                record["juncture"] = bool(record["juncture"])
                record["compressed"] = bool(record["compressed"])
                records.append(record)

        Path(path).write_text(
            json.dumps({"records": records}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "MetricsLogger":
        """Deserialize from a file produced by save().

        Loads into in-memory storage (for backward compatibility).
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ml = cls()
        ml._records = payload["records"]
        return ml

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._records = []
        if self._conn:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM records WHERE session_id = ?", (self._session_id,))
            self._conn.commit()

    def close(self) -> None:
        """Close database connection. Safe to call even if not using SQLite."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __len__(self) -> int:
        return len(self._records)

    def __del__(self) -> None:
        """Ensure connection is closed on cleanup."""
        self.close()
