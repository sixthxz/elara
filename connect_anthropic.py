# -*- coding: utf-8 -*-
"""
connect_anthropic.py
====================
End-to-end: Anthropic API + Elara Gatekeeper + PROXY-01 signals + MetricsLogger.

Runs 3 conversation categories (COHERENT / MIXED / VARIED) through the real
Anthropic API. After each turn, computes sa/sb proxy signals via
all-MiniLM-L6-v2 embeddings, evaluates the Gatekeeper, and records whether
the cycle would have been compressed. Reports lock_frac and simulated token
savings vs raw-history baseline.

Usage
-----
    python connect_anthropic.py                     # reads ANTHROPIC_API_KEY
    python connect_anthropic.py --api-key sk-ant-...
    python connect_anthropic.py --W 3 --turns 6 --out my_results.json
    python connect_anthropic.py --quiet             # suppress per-turn output

Expected output (matches calibrated ratios: COHERENT >> VARIED):
    COHERENT  lock_frac ~ 0.60-0.80   (strong topic coupling)
    MIXED     lock_frac ~ 0.30-0.55
    VARIED    lock_frac ~ 0.10-0.25   (near-zero topic coupling)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from elara.gatekeeper import Gatekeeper
from elara.metrics import MetricsLogger
from elara.adapters import from_api_stream


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "claude-haiku-4-5-20251001"
DEFAULT_W      = 3      # continuous drho (W=2 gives binary drho)
DEFAULT_TURNS  = 6      # turns per conversation
TAU_RHO        = 0.35   # PROXY-01 calibrated value

# When the gatekeeper fires (compress=True) at turn t, the next call would
# send a seed summary instead of the full history.  The seed context is
# ~40-60 chars → ~15 tokens.  Add a small per-turn overhead: 40 tokens total.
SEED_TOKEN_ESTIMATE = 40


# ---------------------------------------------------------------------------
# Conversation templates (3 categories × 6 turns)
# ---------------------------------------------------------------------------

CONVERSATIONS: Dict[str, List[str]] = {
    "COHERENT": [
        "Explain how Python list comprehensions work.",
        "Show me a list comprehension that filters even numbers from 1 to 20.",
        "How do nested list comprehensions work? Give a short example.",
        "What are the performance implications of list comprehensions vs for-loops?",
        "How can I use list comprehensions to build a dictionary?",
        "Can list comprehensions handle multiple if conditions in one expression?",
    ],
    "MIXED": [
        "Tell me about neural networks in machine learning.",
        "What activation functions are commonly used and why?",
        "How does backpropagation compute gradients?",
        "What are convolutional neural networks used for?",
        "How does transfer learning reduce training time?",
        "What is the difference between batch and online learning?",
    ],
    "VARIED": [
        "What year did the Western Roman Empire fall?",
        "Explain quantum entanglement in simple terms.",
        "What is the best technique to cook pasta al dente?",
        "How does a transistor work as a switch in a circuit?",
        "Who wrote the play Hamlet, and when?",
        "What causes inflation in an economy?",
    ],
}


# ---------------------------------------------------------------------------
# Per-category runner
# ---------------------------------------------------------------------------

def run_category(
    category:   str,
    user_turns: List[str],
    client:     anthropic.Anthropic,
    W:          int,
    model:      str,
    verbose:    bool = True,
) -> dict:
    """Run one conversation category; return metrics + token counts."""
    gk = Gatekeeper(W=W, tau_rho=TAU_RHO)
    ml = MetricsLogger()

    history:           List[dict]              = []
    turns:             List[Tuple[str, str]]   = []
    baseline_tokens:   List[int]               = []
    output_tokens:     List[int]               = []
    compressed:        List[bool]              = []
    metrics_per_turn:  List[dict]              = []

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Category: {category}   W={W}   model={model}")
        print(f"{'='*62}")

    for t, user_text in enumerate(user_turns):
        history.append({"role": "user", "content": user_text})
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            messages=history,
        )
        asst_text = resp.content[0].text
        history.append({"role": "assistant", "content": asst_text})
        turns.append((user_text, asst_text))

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        baseline_tokens.append(in_tok)
        output_tokens.append(out_tok)

        # Evaluate gatekeeper once we have >= W turns (from_api_stream needs >= 2)
        if len(turns) >= W:
            sa, sb = from_api_stream(turns)
            do_compress, m = gk.evaluate(sa.tolist(), sb.tolist())
            ml.record(m, do_compress)
            compressed.append(do_compress)
            metrics_per_turn.append({
                "turn":                 t,
                "rho_star":             round(m.get("rho_star",    0.0), 4),
                "d_rho":                round(m.get("d_rho",       0.0), 4),
                "delta":                round(m.get("delta",       0.0), 6),
                "regime":               m.get("regime", ""),
                "juncture":             bool(m.get("juncture", False)),
                "compressed":           do_compress,
                "baseline_input_tokens": in_tok,
            })
            if verbose:
                flag = "COMPRESS" if do_compress else "pass    "
                print(
                    f"  turn {t+1:2d} [{flag}]  "
                    f"rho*={m.get('rho_star', 0.0):+.3f}  "
                    f"delta={m.get('delta', 0.0):+.6f}  "
                    f"regime={m.get('regime', ''):<12}  "
                    f"in={in_tok:4d} tok"
                )
        else:
            compressed.append(False)
            metrics_per_turn.append({"turn": t, "baseline_input_tokens": in_tok,
                                      "note": f"warmup (W={W})"})
            if verbose:
                print(
                    f"  turn {t+1:2d} [warmup  ]  "
                    f"in={in_tok:4d} tok  (accumulating W={W} turns)"
                )

        time.sleep(0.25)  # light rate-limit courtesy

    # Simulate token savings: when compress=True at turn t, the NEXT call
    # (turn t+1) would send a seed summary (~SEED_TOKEN_ESTIMATE tokens)
    # instead of the full growing history.
    total_baseline = sum(baseline_tokens)
    simulated_savings = 0
    for t in range(len(compressed)):
        if compressed[t] and t + 1 < len(baseline_tokens):
            simulated_savings += max(0, baseline_tokens[t + 1] - SEED_TOKEN_ESTIMATE)

    return {
        "category":               category,
        "W":                      W,
        "model":                  model,
        "n_turns":                len(user_turns),
        "n_eval_turns":           ml.cycle_count,
        "baseline_tokens_total":  total_baseline,
        "simulated_savings":      simulated_savings,
        "simulated_savings_pct":  round(
            100.0 * simulated_savings / total_baseline if total_baseline > 0 else 0.0, 2
        ),
        "lock_frac":              round(ml.lock_frac, 4),
        "juncture_count":         ml.juncture_count,
        "metrics_summary":        ml.summary(),
        "metrics_per_turn":       metrics_per_turn,
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_report(results: List[dict]) -> None:
    print()
    print("=" * 72)
    print("  Elara PROXY-01 — End-to-End Results")
    print("=" * 72)
    hdr = (
        f"  {'Category':10}  {'lock_frac':>10}  {'eval_turns':>10}  "
        f"{'baseline':>10}  {'saved':>8}  {'saved_%':>8}  {'junctures':>9}"
    )
    print(hdr)
    print("  " + "-" * 68)
    for r in results:
        print(
            f"  {r['category']:10}  {r['lock_frac']:>10.4f}  "
            f"{r['n_eval_turns']:>10d}  "
            f"{r['baseline_tokens_total']:>10d}  "
            f"{r['simulated_savings']:>8d}  "
            f"{r['simulated_savings_pct']:>7.1f}%  "
            f"{r['juncture_count']:>9d}"
        )
    print()

    cat_lf = {r["category"]: r["lock_frac"] for r in results}
    if "COHERENT" in cat_lf and "VARIED" in cat_lf:
        lf_c = cat_lf["COHERENT"]
        lf_v = cat_lf["VARIED"]
        ratio = lf_c / lf_v if lf_v > 0 else float("inf")
        verdict = "PASS" if lf_c > lf_v else "FAIL"
        print(
            f"  COHERENT lock_frac={lf_c:.3f}  VARIED lock_frac={lf_v:.3f}  "
            f"ratio={ratio:.2f}x  [{verdict}]"
        )
        print(
            "  (target: COHERENT >> VARIED; calibrated ratio ~2-3x on synthetic data)"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Anthropic API + Elara Gatekeeper + PROXY-01 metrics."
    )
    parser.add_argument("--api-key", default=None, metavar="KEY",
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env var).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--W", type=int, default=DEFAULT_W,
                        help=f"Gatekeeper coherence window (default: {DEFAULT_W}).")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS,
                        help=f"Turns per conversation (default: {DEFAULT_TURNS}).")
    parser.add_argument("--out", default="connect_anthropic_results.json",
                        metavar="PATH",
                        help="Output JSON path (default: connect_anthropic_results.json).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-turn output.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  or pass --api-key sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    n_turns = min(args.turns, 6)  # cap at template length

    all_results = []
    for category, user_turns in CONVERSATIONS.items():
        result = run_category(
            category   = category,
            user_turns = user_turns[:n_turns],
            client     = client,
            W          = args.W,
            model      = args.model,
            verbose    = not args.quiet,
        )
        all_results.append(result)

    print_report(all_results)

    output = {
        "_meta": {
            "model":               args.model,
            "W":                   args.W,
            "tau_rho":             TAU_RHO,
            "turns":               n_turns,
            "seed_token_estimate": SEED_TOKEN_ESTIMATE,
        },
        "results": all_results,
    }
    Path(args.out).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
