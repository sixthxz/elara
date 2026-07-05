"""
Phase B — proxy under real API conditions.

Runs 2 coherent sessions (10 turns total) through the proxy at localhost:8877,
then reads elara_proxy_metrics.db and prints lock_frac and tokens_saved.
"""

import os
import time
import sqlite3
import json
from pathlib import Path

import anthropic

DB_PATH = Path(__file__).parent / "elara_proxy_metrics.db"

SESSIONS = [
    {
        "id": "B1", "topic": "algebra_lineal",
        "turns": [
            "¿Qué es un espacio vectorial y por qué es útil?",
            "¿Cómo se relacionan los vectores con las transformaciones lineales?",
            "¿Qué significa que una transformación sea invertible?",
            "¿Cómo conecta esto con los valores y vectores propios?",
            "¿Para qué sirven los valores propios en aplicaciones reales?",
        ]
    },
    {
        "id": "B2", "topic": "termodinamica",
        "turns": [
            "¿Qué dice la primera ley de la termodinámica?",
            "¿Por qué la segunda ley implica que el desorden siempre aumenta?",
            "¿Qué es la entropía en términos prácticos?",
            "¿Cómo funcionan los motores de calor a partir de estas leyes?",
            "¿Cuál es el límite teórico de eficiencia de un motor?",
        ]
    },
]


def run_session(client, session):
    sid, topic = session["id"], session["topic"]
    print(f"\n{'='*50}")
    print(f"Session {sid} — {topic}")
    print(f"{'='*50}")

    history = []
    for i, question in enumerate(session["turns"]):
        print(f"  Turn {i+1}: {question[:70]}")
        history.append({"role": "user", "content": question})
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=history,
        )
        answer = response.content[0].text
        history.append({"role": "assistant", "content": answer})
        print(f"  > {answer[:80]}...")
        time.sleep(0.8)

    print(f"  Session {sid} done — {len(session['turns'])} turns")


def report_metrics():
    if not DB_PATH.exists():
        print(f"\n[WARN] DB not found at {DB_PATH}")
        return

    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()

    # Total records
    cur.execute("SELECT COUNT(*) FROM records")
    total = cur.fetchone()[0]

    # Compressed count
    cur.execute("SELECT COUNT(*) FROM records WHERE compressed=1")
    compressed = cur.fetchone()[0]

    # delta stats for compressed rows
    cur.execute("SELECT AVG(delta), MIN(delta), MAX(delta) FROM records WHERE compressed=1")
    row = cur.fetchone()
    avg_delta, min_delta, max_delta = row if row[0] is not None else (None, None, None)

    # rho_star stats
    cur.execute("SELECT AVG(rho_star) FROM records")
    avg_rho = cur.fetchone()[0]

    # regime distribution
    cur.execute("SELECT regime, COUNT(*) FROM records GROUP BY regime")
    regimes = dict(cur.fetchall())

    # lyapunov
    cur.execute("SELECT AVG(lyapunov_v) FROM records WHERE lyapunov_v IS NOT NULL")
    avg_lyap = cur.fetchone()[0]

    con.close()

    lock_frac = compressed / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print("Phase B Results")
    print(f"{'='*50}")
    print(f"  total records  : {total}")
    print(f"  compressed     : {compressed}")
    print(f"  lock_frac      : {lock_frac:.3f}  ({compressed}/{total})")
    if avg_delta is not None:
        print(f"  delta (compressed): avg={avg_delta:+.4f}  min={min_delta:+.4f}  max={max_delta:+.4f}")
    print(f"  avg rho*       : {avg_rho:.4f}" if avg_rho else "  avg rho*       : n/a")
    print(f"  avg lyapunov_v : {avg_lyap:.4f}" if avg_lyap else "  avg lyapunov_v : n/a")
    print(f"  regimes        : {regimes}")
    print()


if __name__ == "__main__":
    os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:8877"
    print(f"Phase B — routing through proxy at {os.environ['ANTHROPIC_BASE_URL']}")
    print(f"Sessions: {len(SESSIONS)}  Turns: {sum(len(s['turns']) for s in SESSIONS)}")

    client = anthropic.Anthropic()

    for session in SESSIONS:
        run_session(client, session)
        time.sleep(1.5)

    print("\nAll sessions complete.")
    report_metrics()
