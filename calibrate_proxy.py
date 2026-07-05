# -*- coding: utf-8 -*-
"""
calibrate_proxy.py
==================
Calibration for PROXY-01: empirical tau_lock_dr for Anthropic API conversations.

Step 1 — collect data (choose one):
    python calibrate_proxy.py --synthetic           # no API/embeddings required
    python calibrate_proxy.py --collect             # real API + sentence-transformers

Step 2 — calibrate (runs automatically after collection, or standalone):
    python calibrate_proxy.py                       # uses proxy_activations.json
    python calibrate_proxy.py --raw proxy_activations.json

What it computes
----------------
Same algorithm as calibrate.py, adapted for PROXY-01 signals:
  sa(t) = L2 norm of all-MiniLM-L6-v2 embedding(user_turn_t), norm01
  sb(t) = L2 norm of all-MiniLM-L6-v2 embedding(claude_turn_t), norm01
  checkpoint = each conversation turn boundary

Output: proxy_activations.json + proxy_calibration.json

W=2 note
--------
With W=2, Pearson of two points is +/-1 (sign of co-movement), so drho
is binary: 0 (consecutive windows agree in sign) or 1 (disagree).
Any tau_lock_dr in (0, 1) gives the same gate — same as COT-01.
Use --W 3 for continuous drho and a more informative calibration.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from elara.engine import RRGObservables, RRGParams
from elara.adapters import PROXY_PARAMS, _norm01

# Reuse domain-agnostic calibration utilities from calibrate.py
from calibrate import (
    collect_per_category,
    lock_frac_at_threshold,
    percentile_table,
    recommend_threshold,
    sweep_thresholds,
)


# ---------------------------------------------------------------------------
# Baseline parameters — matches PROXY_PARAMS in adapters.py
# ---------------------------------------------------------------------------

_BASE_PARAMS = dict(
    tau_lock_rho = 0.35,
    tau_lock_dr  = 0.02,   # placeholder — this is what we fit
    tau_meta     = 0.015,
    tau_exit     = 0.04,
)


def _make_params(W: int) -> RRGParams:
    return RRGParams(W=W, **_BASE_PARAMS)


# ---------------------------------------------------------------------------
# Synthetic data generation (no sentence-transformers required)
# ---------------------------------------------------------------------------

def _norm01_np(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _gen_coherent(rng: np.random.Generator, n_turns: int) -> Tuple[list, list]:
    """High latent coupling — COHERENT (user + assistant stay on topic)."""
    latent = np.cumsum(rng.standard_normal(n_turns) * 0.4)
    sa = latent + 0.08 * rng.standard_normal(n_turns)
    sb = latent + 0.08 * rng.standard_normal(n_turns)
    return _norm01_np(sa).tolist(), _norm01_np(sb).tolist()


def _gen_mixed(rng: np.random.Generator, n_turns: int) -> Tuple[list, list]:
    """Moderate coupling — MIXED (topic partially maintained)."""
    latent = np.cumsum(rng.standard_normal(n_turns) * 0.4)
    sa = 0.55 * latent + 0.84 * rng.standard_normal(n_turns)
    sb = 0.55 * latent + 0.84 * rng.standard_normal(n_turns)
    return _norm01_np(sa).tolist(), _norm01_np(sb).tolist()


def _gen_varied(rng: np.random.Generator, n_turns: int) -> Tuple[list, list]:
    """Alternating co-movement — VARIED (topic switching per turn)."""
    latent = np.cumsum(rng.standard_normal(n_turns) * 0.4)
    sa = latent + 0.15 * rng.standard_normal(n_turns)
    # Flip sign every other turn to force rho_ab to alternate +1/-1
    flip = np.where(np.arange(n_turns) % 2 == 0, 1.0, -1.0)
    sb = flip * latent + 0.15 * rng.standard_normal(n_turns)
    return _norm01_np(sa).tolist(), _norm01_np(sb).tolist()


_SYNTH_GENERATORS = {
    "COHERENT": _gen_coherent,
    "MIXED":    _gen_mixed,
    "VARIED":   _gen_varied,
}


def generate_synthetic_activations(
    n_per_cat: int = 20,
    n_turns:   int = 12,
    seed:      int = 42,
) -> dict:
    """
    Generate synthetic proxy activations without any API or embeddings.

    Categories
    ----------
    COHERENT — strong latent coupling; models an on-topic conversation.
               Target lock_frac ≈ 0.75–0.90.
    MIXED    — moderate coupling; models a typical multi-topic session.
               Target lock_frac ≈ 0.50–0.65.
    VARIED   — alternating co-movement; models abrupt topic switches.
               Target lock_frac ≈ 0.10–0.30.

    Returns data dict in the same format as proxy_activations.json.
    """
    rng = np.random.default_rng(seed)

    categories: Dict[str, list] = {}
    for label, gen_fn in _SYNTH_GENERATORS.items():
        samples = []
        for i in range(n_per_cat):
            sa, sb = gen_fn(rng, n_turns)
            samples.append({
                "index":         i,
                "label":         label,
                "sa":            sa,
                "sb":            sb,
                "n_checkpoints": n_turns,
            })
        categories[label] = samples

    return {
        "_meta": {
            "description": "Synthetic PROXY-01 activations — no sentence-transformers",
            "model":       "synthetic",
            "sa":          "norm01(controlled latent-factor scalar), proxy for embed norm",
            "sb":          "norm01(controlled latent-factor scalar), proxy for embed norm",
            "n_per_cat":   n_per_cat,
            "n_turns":     n_turns,
            "seed":        seed,
        },
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Real API collection (requires: anthropic + sentence-transformers)
# ---------------------------------------------------------------------------

# Conversation templates: each entry is a list of user turns.
# The assistant responds in-context; turns form a coherent session.
_COLLECT_CONVERSATIONS_PROSE = {
    "COHERENT": [
        # Single topic — embeddings should stay correlated
        ["Explain how Python list comprehensions work.",
         "Show me a list comprehension that filters even numbers.",
         "How do nested list comprehensions work?",
         "What are the performance implications vs a regular for-loop?"],
        ["What is quicksort and how does it work?",
         "What is its average time complexity?",
         "How does it compare to mergesort?",
         "Can you write a Python implementation?"],
        ["Explain gradient descent in machine learning.",
         "What is the learning rate and how does it affect training?",
         "What is stochastic gradient descent?",
         "How do adaptive learning rate methods like Adam differ?"],
    ],
    "MIXED": [
        # Related but shifting subtopics
        ["Tell me about neural networks.",
         "What activation functions are commonly used?",
         "How does backpropagation work?",
         "What are convolutional neural networks?"],
        ["Explain recursion in programming.",
         "What is a base case?",
         "Show me a recursive Fibonacci function.",
         "What is memoization?"],
        ["What is the difference between a list and a tuple in Python?",
         "When should I use each one?",
         "How do dictionaries differ from both?",
         "What are Python sets?"],
    ],
    "VARIED": [
        # Unrelated topic switch each turn — embeddings should diverge
        ["What year did the Roman Empire fall?",
         "Explain quantum entanglement.",
         "What is the best way to cook pasta al dente?",
         "How does a transistor work?"],
        ["What is photosynthesis?",
         "Name the largest planet in the solar system.",
         "What is Fibonacci sequence?",
         "How do airplanes generate lift?"],
        ["Who wrote Hamlet?",
         "What is the Pythagorean theorem?",
         "Describe the water cycle.",
         "What causes inflation?"],
    ],
}

# Coding-domain conversation templates.
# These approximate real coding sessions (text turns only — tool calls are
# stripped by _extract_text_turns in proxy.py, so calibration targets the
# semantic signal carried by question/explanation turns).
_COLLECT_CONVERSATIONS_CODING = {
    "COHERENT": [
        # Single coding topic, each turn deepens the same concept
        ["How do I implement a binary search tree in Python?",
         "What insert and search methods should the BST class have?",
         "How do I implement in-order traversal recursively?",
         "How do I delete a node while preserving BST properties?"],
        ["Explain how Python decorators work.",
         "Show me a simple timing decorator example.",
         "How do I write a decorator that accepts its own arguments?",
         "How does functools.wraps preserve the wrapped function's metadata?"],
        ["How does async/await work in Python?",
         "What is the event loop and how does it schedule coroutines?",
         "How do I run multiple coroutines concurrently with asyncio.gather?",
         "What is the difference between asyncio.gather and asyncio.wait?"],
        ["What is the difference between __str__ and __repr__ in Python?",
         "When is each one called automatically?",
         "How should I implement them for a custom class?",
         "How do f-strings interact with __str__ vs __repr__?"],
        ["Explain Python's descriptor protocol.",
         "What methods does a descriptor need to implement?",
         "How do property objects use the descriptor protocol internally?",
         "How does __set_name__ help with descriptor classes?"],
        ["How does Python's GIL affect multithreaded programs?",
         "When does the GIL get released during execution?",
         "How does multiprocessing sidestep the GIL?",
         "What workloads benefit from threading despite the GIL?"],
    ],
    "MIXED": [
        # Coding sessions that shift between related but distinct subtopics
        ["What is big-O notation?",
         "What is the time complexity of binary search?",
         "How does space complexity relate to time complexity?",
         "What is amortized analysis and when is it useful?"],
        ["How does Python manage memory with reference counting?",
         "What is cyclic garbage collection in Python?",
         "How can I profile memory usage in a Python program?",
         "What are common causes of memory leaks in Python?"],
        ["What is the difference between a stack and a queue?",
         "How do I implement a stack in Python?",
         "What is a deque and when should I use collections.deque?",
         "How does a priority queue differ from a regular queue?"],
        ["How does Python's import system work?",
         "What is the difference between a module and a package?",
         "How do relative imports work?",
         "What is __init__.py used for?"],
    ],
    "VARIED": [
        # Abrupt topic change each turn — embeddings should diverge
        ["How do Python generators work?",
         "Explain the difference between JOIN types in SQL.",
         "What is the purpose of a Makefile?",
         "How does HTTP keep-alive work?"],
        ["What is a hash table and how does it handle collisions?",
         "Explain CSS flexbox layout.",
         "How does Git's three-way merge algorithm work?",
         "What is the decorator pattern in object-oriented design?"],
        ["How does the CPython bytecode interpreter work?",
         "What is a foreign key constraint in a relational database?",
         "How do you center a div in CSS?",
         "What is JWT authentication?"],
    ],
}

_DOMAIN_CONVERSATIONS = {
    "prose":  _COLLECT_CONVERSATIONS_PROSE,
    "coding": _COLLECT_CONVERSATIONS_CODING,
}

# Default for backward-compat (callers that don't pass domain)
_COLLECT_CONVERSATIONS = _COLLECT_CONVERSATIONS_PROSE


def collect_from_api(
    n_per_cat:  int = 10,
    api_key:    Optional[str] = None,
    embed_model: str = "all-MiniLM-L6-v2",
    llm_model:  str = "claude-haiku-4-5-20251001",
    domain:     str = "prose",
) -> dict:
    """
    Collect PROXY-01 activations from real Anthropic API conversations.

    Requires
    --------
    - anthropic package:          pip install anthropic
    - sentence-transformers:      pip install sentence-transformers
    - ANTHROPIC_API_KEY env var or --api-key argument

    Strategy per category
    ---------------------
    COHERENT — 4-turn single-topic conversations.
    MIXED    — 4-turn conversations with shifting subtopics.
    VARIED   — 4-turn conversations with unrelated topic-switch each turn.
    """
    try:
        import anthropic as ant
    except ImportError as exc:
        raise ImportError(
            "anthropic package required for --collect.\n"
            "Install: pip install anthropic"
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers required for --collect.\n"
            "Install: pip install sentence-transformers"
        ) from exc

    from elara.adapters import from_api_stream

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Pass --api-key or: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    conversations = _DOMAIN_CONVERSATIONS.get(domain, _COLLECT_CONVERSATIONS_PROSE)

    client  = ant.Anthropic(api_key=key)
    encoder = SentenceTransformer(embed_model)

    def _embed_vecs(texts: List[str]) -> np.ndarray:
        return encoder.encode(texts, show_progress_bar=False)  # (T, D)

    categories: Dict[str, list] = {}

    for label, convs in conversations.items():
        samples = []
        idx = 0
        conv_pool = convs * (n_per_cat // len(convs) + 1)  # tile to cover n_per_cat

        while idx < n_per_cat:
            user_turns = conv_pool[idx % len(conv_pool)]
            history: List[dict] = []
            collected_turns: List[Tuple[str, str]] = []

            print(f"  {label} [{idx + 1}/{n_per_cat}] — {user_turns[0][:40]}...")
            for user_text in user_turns:
                history.append({"role": "user", "content": user_text})
                resp = client.messages.create(
                    model=llm_model,
                    max_tokens=300,
                    messages=history,
                )
                assistant_text = resp.content[0].text
                history.append({"role": "assistant", "content": assistant_text})
                collected_turns.append((user_text, assistant_text))

            sa, sb = from_api_stream(
                collected_turns,
                model_name=embed_model,
                _embed_fn=_embed_vecs,
            )

            samples.append({
                "index":         idx,
                "label":         label,
                "sa":            sa.tolist(),
                "sb":            sb.tolist(),
                "n_checkpoints": len(collected_turns),
                "turns":         [[u, a] for u, a in collected_turns],
            })
            idx += 1

        categories[label] = samples

    return {
        "_meta": {
            "description": f"PROXY-01 activations from Anthropic API — domain={domain}",
            "model":       llm_model,
            "embedding":   embed_model,
            "domain":      domain,
            "sa":          "norm01(L2 norm of all-MiniLM-L6-v2 embed(user_turn_t))",
            "sb":          "norm01(L2 norm of all-MiniLM-L6-v2 embed(claude_turn_t))",
        },
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_proxy_report(
    data:     dict,
    per_cat:  dict,
    sweep:    list,
    rec:      dict,
    params:   RRGParams,
) -> None:
    meta = data.get("_meta", {})
    print()
    print("=" * 62)
    print("PROXY-01 Calibration Report")
    print("=" * 62)
    print(f"Model     : {meta.get('model', 'unknown')}")
    print(f"W         : {params.W}")
    print(f"tau_rho   : {params.tau_lock_rho}")
    print(f"Baseline tau_lock_dr : {params.tau_lock_dr}")
    print()

    # Per-category drho stats
    all_drho = np.concatenate([v["drho"] for v in per_cat.values()])
    print(f"{'Category':12} | {'n_windows':>9} | {'drho_min':>9} | "
          f"{'drho_mean':>10} | {'drho_max':>9} | {'drho_p95':>9}")
    print("-" * 72)
    for label, arrs in per_cat.items():
        dr   = arrs["drho"]
        ptbl = percentile_table(dr)
        print(f"{label:12} | {len(dr):>9} | {dr.min():>9.4f} | "
              f"{dr.mean():>10.4f} | {dr.max():>9.4f} | {ptbl['p95']:>9.4f}")

    # W=2 binary detection
    unique_vals = np.unique(np.round(all_drho, 4))
    if len(unique_vals) <= 3:
        print()
        print(f"  NOTE: drho is near-binary with W={params.W}.")
        print(f"  Unique values: {unique_vals.tolist()}")
        print("  With W=2, Pearson of 2 points is +/-1, so drho in {0, 1}.")
        print("  Any tau_lock_dr in (0, 1) gives the same AND gate.")
        print("  For continuous drho, rerun with --W 3.")

    # Threshold sweep around recommendation
    print()
    rec_tau    = rec["tau_lock_dr"]
    categories = list(per_cat.keys())
    header = f"{'tau_lock_dr':>12} | " + " | ".join(
        f"{'lf_' + c[:8]:>12}" for c in categories
    )
    print("Threshold sweep (rows near recommendation):")
    print(header)
    print("-" * len(header))
    shown_taus = sorted({
        row["tau_lock_dr"] for row in sweep
        if abs(row["tau_lock_dr"] - rec_tau) < 0.12
    })[:8]
    for row in sweep:
        if row["tau_lock_dr"] in shown_taus:
            marker = " <--" if abs(row["tau_lock_dr"] - rec_tau) < 1e-9 else "    "
            fracs  = " | ".join(
                f"{row.get(f'lock_frac_{c}', 0.0):>12.4f}" for c in categories
            )
            print(f"{row['tau_lock_dr']:>12.6f} | {fracs}{marker}")

    # Recommendation
    print()
    print("=" * 62)
    print("Recommendation")
    print("=" * 62)
    print(f"  tau_lock_dr = {rec['tau_lock_dr']:.6f}")
    print(f"  -> {rec['top_category']} lock_frac = {rec['lock_frac']:.4f}  "
          f"(target ~ {rec['target_frac']:.3f})")
    print()
    print("  Update PROXY_PARAMS in elara/adapters.py:")
    print()
    print("  PROXY_PARAMS = RRGParams(")
    print(f"      W            = {params.W},")
    print(f"      tau_lock_rho = {params.tau_lock_rho},")
    print(f"      tau_lock_dr  = {rec['tau_lock_dr']:.6f},   # calibrated")
    print(f"      tau_meta     = {params.tau_meta},")
    print(f"      tau_exit     = {params.tau_exit},")
    print("  )")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate PROXY-01 tau_lock_dr from conversation activations."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic proxy_activations.json (no API required)."
    )
    mode.add_argument(
        "--collect", action="store_true",
        help="Collect from real Anthropic API (requires anthropic + sentence-transformers)."
    )

    parser.add_argument(
        "--raw", default="proxy_activations.json",
        metavar="PATH",
        help="Input raw activations file (default: proxy_activations.json)."
    )
    parser.add_argument(
        "--raw-out", default="proxy_activations.json",
        metavar="PATH",
        help="Output path for collected/synthetic activations (default: proxy_activations.json)."
    )
    parser.add_argument(
        "--out", default="proxy_calibration.json",
        metavar="PATH",
        help="Output calibration file (default: proxy_calibration.json)."
    )
    parser.add_argument(
        "--W", type=int, default=2,
        help="Rolling window size (default: 2). Use 3 for continuous drho."
    )
    parser.add_argument(
        "--n-per-cat", type=int, default=20,
        help="Samples per category for --synthetic (default: 20)."
    )
    parser.add_argument(
        "--n-turns", type=int, default=12,
        help="Turns per conversation for --synthetic (default: 12)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for --synthetic (default: 42)."
    )
    parser.add_argument(
        "--target", type=float, default=0.667,
        help="Target lock_frac for top category (default: 0.667)."
    )
    parser.add_argument(
        "--api-key", default=None,
        metavar="KEY",
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)."
    )
    parser.add_argument(
        "--domain", default="prose", choices=list(_DOMAIN_CONVERSATIONS.keys()),
        help="Conversation domain for --collect (default: prose). Options: prose, coding."
    )
    args = parser.parse_args()

    # --- Step 1: collect / load raw data ---

    raw_path = args.raw_out if (args.synthetic or args.collect) else args.raw

    if args.synthetic:
        print(f"Generating synthetic activations "
              f"({args.n_per_cat} samples/cat, {args.n_turns} turns, seed={args.seed})...")
        data = generate_synthetic_activations(
            n_per_cat = args.n_per_cat,
            n_turns   = args.n_turns,
            seed      = args.seed,
        )
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {raw_path}")

    elif args.collect:
        print(f"Collecting from Anthropic API (domain={args.domain})...")
        data = collect_from_api(
            n_per_cat = args.n_per_cat,
            api_key   = args.api_key,
            domain    = args.domain,
        )
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {raw_path}")

    else:
        if not Path(raw_path).exists():
            print(f"Error: {raw_path} not found.")
            print("Generate data first:")
            print("  python calibrate_proxy.py --synthetic")
            print("  python calibrate_proxy.py --collect")
            sys.exit(1)
        print(f"Loading: {raw_path}")
        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)

    # Report loaded data
    meta    = data.get("_meta", {})
    n_total = sum(len(v) for v in data.get("categories", {}).values())
    print(f"Model  : {meta.get('model', 'unknown')}")
    print(f"Samples: {n_total}")

    # --- Step 2: calibrate ---

    params  = _make_params(args.W)
    per_cat = collect_per_category(data, params)

    if not per_cat:
        print("Error: no valid windows found. Check proxy_activations.json format.")
        sys.exit(1)

    sweep = sweep_thresholds(per_cat, tau_rho=params.tau_lock_rho)
    rec   = recommend_threshold(sweep, list(per_cat.keys()), target_frac=args.target)

    print_proxy_report(data, per_cat, sweep, rec, params)

    # --- Save calibration.json ---

    result = {
        "_meta": {
            "raw":                   raw_path,
            "model":                 meta.get("model", "unknown"),
            "W":                     params.W,
            "tau_lock_rho":          params.tau_lock_rho,
            "baseline_tau_lock_dr":  params.tau_lock_dr,
        },
        "recommendation": {
            "tau_lock_dr":  rec["tau_lock_dr"],
            "top_category": rec["top_category"],
            "lock_frac":    rec["lock_frac"],
            "target_frac":  rec["target_frac"],
        },
        "percentiles": {
            label: percentile_table(arrs["drho"])
            for label, arrs in per_cat.items()
        },
        "percentiles_pooled": percentile_table(
            np.concatenate([v["drho"] for v in per_cat.values()])
        ),
        "sweep": sweep,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
