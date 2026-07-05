"""
run_phase_b.py — Phase B (tokens_saved)

Runs a 12-turn coherent conversation through the Elara proxy (started
in-process via ElaraClient). Measures:
  - total_bytes_saved  (from proxy metrics, SQLite-backed)
  - total_uncompressed_baseline  (cumulative full-history body sizes)
  - savings_pct = total_bytes_saved / total_uncompressed * 100

12 turns → 10 gatekeeper-eligible turns (W=3 warmup on first 2 turns).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from elara import ElaraClient
from elara.proxy import _metrics as proxy_metrics, LISTEN_PORT

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300

# 14-turn coherent conversation — Python list comprehensions (validated topic,
# lock_frac≈0.750 in prior end-to-end runs). 12 gatekeeper-eligible turns (W=3).
TURNS = [
    "Explain how Python list comprehensions work.",
    "Show me a list comprehension that filters even numbers from 1 to 20.",
    "How do nested list comprehensions work? Give a short example.",
    "What are the performance implications of list comprehensions vs for-loops?",
    "How can I use list comprehensions to build a dictionary?",
    "Can list comprehensions handle multiple if conditions in one expression?",
    "How do list comprehensions compare to the map() and filter() functions?",
    "Can you use list comprehensions with enumerate or zip?",
    "How do you write a conditional (ternary) expression inside a list comprehension?",
    "What are the memory implications of very large list comprehensions?",
    "Can list comprehensions be used for matrix operations or 2-D data?",
    "How do list comprehensions work with string manipulation and splitting?",
    "What are best practices for keeping list comprehensions readable?",
    "When should you avoid list comprehensions and use a plain for-loop instead?",
]


def _body_bytes(messages: list) -> int:
    """JSON body size (bytes) that the client sends to the proxy for this turn."""
    payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": messages}
    return len(json.dumps(payload).encode())


def main() -> None:
    print("Phase B (tokens_saved) — Long coherent session through Elara proxy")
    print(f"  Model: {MODEL}   Turns: {len(TURNS)}   W=3")
    print()

    history:               list = []
    actual_input_tokens:   list = []
    uncompressed_body_sizes: list = []

    with ElaraClient() as client:
        print(f"  Proxy started on http://localhost:{LISTEN_PORT}")
        print()
        print(f"  {'Turn':>4}  {'in_tok':>7}  {'body_B':>8}  Question")
        print(f"  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*48}")

        for i, user_text in enumerate(TURNS):
            history.append({"role": "user", "content": user_text})

            # Measure full uncompressed body (what the proxy receives from us)
            uncompressed_body_sizes.append(_body_bytes(history))

            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=history,
            )
            asst_text = resp.content[0].text
            history.append({"role": "assistant", "content": asst_text})
            actual_input_tokens.append(resp.usage.input_tokens)

            print(
                f"  {i+1:4d}  {resp.usage.input_tokens:7d}  "
                f"{uncompressed_body_sizes[-1]:8d}  {user_text[:50]}"
            )
            time.sleep(0.5)

    # Proxy is stopped; read accumulated metrics
    summary            = proxy_metrics.summary()
    total_bytes_saved  = summary["total_bytes_saved"]
    lock_frac          = summary["lock_frac"]
    cycle_count        = summary["cycle_count"]

    total_uncompressed = sum(uncompressed_body_sizes)
    total_actual_tok   = sum(actual_input_tokens)
    savings_pct        = (
        100.0 * total_bytes_saved / total_uncompressed
        if total_uncompressed > 0 else 0.0
    )

    print()
    print("=" * 64)
    print("  Phase B — Results")
    print("=" * 64)
    print(f"  Turns total:              {len(TURNS)}")
    print(f"  Gatekeeper eligible:      {cycle_count}  (turns 3–{len(TURNS)}, W=3)")
    print(f"  Lock fraction:            {lock_frac:.3f}")
    print(f"  Actual input tokens:      {total_actual_tok:,}")
    print()
    print(f"  Uncompressed baseline:    {total_uncompressed:,} bytes")
    print(f"  Total bytes saved:        {total_bytes_saved:,} bytes")
    print(f"  Savings:                  {savings_pct:.1f}%")
    print()
    print(proxy_metrics.report())


if __name__ == "__main__":
    main()
