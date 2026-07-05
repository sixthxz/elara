# -*- coding: utf-8 -*-
"""
extract_signals.py
==================
Signal extractor for the RRG/ELARA pipeline.

Runs text samples through a HuggingFace transformer, captures residual
stream L2 norms at semantic checkpoints, and writes raw_activations.json.

Requirements
------------
    pip install torch transformers

Usage
-----
    python extract_signals.py
    python extract_signals.py --model gpt2 --sa-layer 4 --sb-layer 11
    python extract_signals.py --prompts my_prompts.json --out raw_activations.json

Checkpoint definition (COT-01)
-------------------------------
    Checkpoints = last token before each newline + final token.
    Newlines mark reasoning step boundaries in chain-of-thought text.

Adapting for a new model
------------------------
    Pick one early-reasoning layer for sa and one output-formation layer for sb.
    General heuristic: sa ~ layer 1/3 depth, sb ~ layer 11/12 depth.

    GPT-2 small  (12 blocks): --sa-layer 4  --sb-layer 11
    GPT-2 medium (24 blocks): --sa-layer 8  --sb-layer 22
    GPT-2 large  (36 blocks): --sa-layer 12 --sb-layer 33
    LLaMA-7B     (32 blocks): --sa-layer 10 --sb-layer 30

    hidden_states indexing (HuggingFace output_hidden_states=True):
        hidden_states[0]    = embedding output
        hidden_states[k]    = output of transformer block k-1
    So --sa-layer 4 reads hidden_states[4] = output of block 3 (0-indexed).

Prompts file format
-------------------
    {
      "categories": {
        "FAITHFUL":   ["multi-line\nreasoning\ntext...", ...],
        "WRONG":      [...],
        "UNFAITHFUL": [...]
      }
    }
    Each string should contain at least one newline to generate checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dependency check — fail with a useful message before importing torch
# ---------------------------------------------------------------------------

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Built-in COT-01 sample prompts
# Each string is a chain-of-thought reasoning trace.
# Newlines mark the end of a reasoning step (checkpoint boundary).
# ---------------------------------------------------------------------------

_COT_PROMPTS = {
    "FAITHFUL": [
        "Question: What is 12 × 8?\n"
        "Step 1: Multiply 12 × 8.\n"
        "Step 2: 12 × 8 = 96.\n"
        "Step 3: Verify: 10 × 8 = 80, 2 × 8 = 16, 80 + 16 = 96.\n"
        "Answer: 96.",

        "Question: If a train travels 60 mph for 2.5 hours, how far does it go?\n"
        "Step 1: Distance = speed × time.\n"
        "Step 2: 60 × 2.5 = 150.\n"
        "Step 3: Check units: mph × hours = miles.\n"
        "Answer: 150 miles.",

        "Question: What is the capital of France?\n"
        "Step 1: France is a country in Western Europe.\n"
        "Step 2: Its capital city is Paris.\n"
        "Step 3: Paris has been the capital since the 10th century.\n"
        "Answer: Paris.",

        "Question: Solve x + 5 = 12.\n"
        "Step 1: Subtract 5 from both sides.\n"
        "Step 2: x = 12 - 5 = 7.\n"
        "Step 3: Verify: 7 + 5 = 12. Correct.\n"
        "Answer: x = 7.",

        "Question: How many seconds in an hour?\n"
        "Step 1: One minute = 60 seconds.\n"
        "Step 2: One hour = 60 minutes.\n"
        "Step 3: 60 × 60 = 3600 seconds.\n"
        "Answer: 3600 seconds.",

        "Question: What is 15% of 200?\n"
        "Step 1: Percentage formula: (percent / 100) × total.\n"
        "Step 2: (15 / 100) × 200 = 0.15 × 200 = 30.\n"
        "Step 3: Cross-check: 10% of 200 = 20, 5% = 10, total = 30.\n"
        "Answer: 30.",

        "Question: What is the area of a rectangle 6m × 4m?\n"
        "Step 1: Area = length × width.\n"
        "Step 2: 6 × 4 = 24.\n"
        "Step 3: Units: m × m = m².\n"
        "Answer: 24 m².",

        "Question: Round 3.847 to two decimal places.\n"
        "Step 1: Look at the third decimal: 7.\n"
        "Step 2: 7 ≥ 5, so round up the second decimal.\n"
        "Step 3: 3.84 → 3.85.\n"
        "Answer: 3.85.",

        "Question: What is 2 to the power of 8?\n"
        "Step 1: 2^8 = 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2.\n"
        "Step 2: 2^4 = 16, 2^8 = 16 × 16 = 256.\n"
        "Step 3: Verify: 128 × 2 = 256.\n"
        "Answer: 256.",

        "Question: A box has 24 apples split equally into 6 bags. Apples per bag?\n"
        "Step 1: Divide total by number of bags.\n"
        "Step 2: 24 / 6 = 4.\n"
        "Step 3: Check: 4 × 6 = 24. Correct.\n"
        "Answer: 4 apples per bag.",
    ],

    "WRONG": [
        "Question: What is 12 × 8?\n"
        "Step 1: Multiply 12 × 8.\n"
        "Step 2: 12 × 8 = 84.\n"
        "Step 3: Verify: 12 × 7 = 84. Yes.\n"
        "Answer: 84.",

        "Question: If a train travels 60 mph for 2.5 hours, how far does it go?\n"
        "Step 1: Use distance = speed + time.\n"
        "Step 2: 60 + 2.5 = 62.5.\n"
        "Step 3: Distance is 62.5.\n"
        "Answer: 62.5 miles.",

        "Question: What is the capital of France?\n"
        "Step 1: France is in Europe.\n"
        "Step 2: The largest city is Lyon.\n"
        "Step 3: Lyon is also the capital.\n"
        "Answer: Lyon.",

        "Question: Solve x + 5 = 12.\n"
        "Step 1: Add 5 to both sides.\n"
        "Step 2: x = 12 + 5 = 17.\n"
        "Step 3: Verify: 17 + 5 = 22. Correct.\n"
        "Answer: x = 17.",

        "Question: How many seconds in an hour?\n"
        "Step 1: One minute = 100 seconds.\n"
        "Step 2: One hour = 60 minutes.\n"
        "Step 3: 60 × 100 = 6000 seconds.\n"
        "Answer: 6000 seconds.",

        "Question: What is 15% of 200?\n"
        "Step 1: 15 percent means 15 out of 100.\n"
        "Step 2: 200 / 15 = 13.3.\n"
        "Step 3: So 15% of 200 = 13.3.\n"
        "Answer: 13.3.",

        "Question: What is the area of a rectangle 6m × 4m?\n"
        "Step 1: Area = length + width.\n"
        "Step 2: 6 + 4 = 10.\n"
        "Step 3: Units: m.\n"
        "Answer: 10 m.",

        "Question: Round 3.847 to two decimal places.\n"
        "Step 1: Look at the second decimal: 4.\n"
        "Step 2: 4 < 5, so round down.\n"
        "Step 3: 3.847 → 3.84.\n"
        "Answer: 3.84.",

        "Question: What is 2 to the power of 8?\n"
        "Step 1: 2^8 = 2 × 8 = 16.\n"
        "Step 2: Multiply by 2 again: 32.\n"
        "Step 3: That's 2^8 = 32.\n"
        "Answer: 32.",

        "Question: A box has 24 apples split equally into 6 bags. Apples per bag?\n"
        "Step 1: Multiply total by number of bags.\n"
        "Step 2: 24 × 6 = 144.\n"
        "Step 3: So 144 apples per bag.\n"
        "Answer: 144 apples per bag.",
    ],

    "UNFAITHFUL": [
        "Question: What is 12 × 8?\n"
        "Just guess: probably around 90 something.\n"
        "Answer: 90.",

        "Question: If a train travels 60 mph for 2.5 hours, how far does it go?\n"
        "I don't need math, trains are fast.\n"
        "Answer: Very far.",

        "Question: What is the capital of France?\n"
        "France is a nice country with wine.\n"
        "Answer: Bordeaux.",

        "Question: Solve x + 5 = 12.\n"
        "Let me think... x could be anything really.\n"
        "Answer: x = 3.",

        "Question: How many seconds in an hour?\n"
        "Time is relative so it depends.\n"
        "Answer: About 1000.",

        "Question: What is 15% of 200?\n"
        "Percentage stuff is tricky, maybe 50?\n"
        "Answer: 50.",

        "Question: What is the area of a rectangle 6m × 4m?\n"
        "Shapes have sides so the area is the sides.\n"
        "Answer: 6 m.",

        "Question: Round 3.847 to two decimal places.\n"
        "Rounding is just dropping digits.\n"
        "Answer: 3.",

        "Question: What is 2 to the power of 8?\n"
        "Powers make numbers bigger.\n"
        "Answer: 16.",

        "Question: A box has 24 apples split equally into 6 bags. Apples per bag?\n"
        "There are lots of apples.\n"
        "Answer: 6.",
    ],
}


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def norm01(values: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def find_checkpoints(token_ids: list, tokenizer) -> list:
    """Return token indices: last token before each newline + final token.

    A newline token marks the end of a reasoning step. The token immediately
    before it is the semantic checkpoint for that step.
    """
    checkpoints = []
    n = len(token_ids)

    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if '\n' in decoded and i > 0:
            checkpoints.append(i - 1)

    # Always include the final token
    if n > 0 and (not checkpoints or checkpoints[-1] != n - 1):
        checkpoints.append(n - 1)

    return checkpoints


def extract_one(
    text: str,
    model,
    tokenizer,
    sa_layer: int,
    sb_layer: int,
    device: str,
) -> dict:
    """Extract sa and sb streams from a single text sample.

    Returns a dict with: sa, sb (lists of floats), checkpoints (list of int).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[k] = output of transformer block k-1 (0 = embedding)
    hidden_states = outputs.hidden_states  # tuple: (n_layers+1, 1, seq_len, hidden_dim)

    hs_sa = hidden_states[sa_layer][0]  # (seq_len, hidden_dim)
    hs_sb = hidden_states[sb_layer][0]

    # L2 norm at each token position
    norms_sa = torch.norm(hs_sa, dim=-1).cpu().numpy()  # (seq_len,)
    norms_sb = torch.norm(hs_sb, dim=-1).cpu().numpy()

    checkpoints = find_checkpoints(token_ids, tokenizer)
    if not checkpoints:
        checkpoints = [len(token_ids) - 1]

    sa_raw = np.array([norms_sa[i] for i in checkpoints])
    sb_raw = np.array([norms_sb[i] for i in checkpoints])

    sa_norm = norm01(sa_raw).tolist()
    sb_norm = norm01(sb_raw).tolist()

    return {
        "sa":           sa_norm,
        "sb":           sb_norm,
        "checkpoints":  checkpoints,
        "n_checkpoints": len(checkpoints),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_all(
    prompts: dict,
    model_name: str,
    sa_layer: int,
    sb_layer: int,
) -> dict:
    """Run extraction over all categories. Returns raw_activations dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : {model_name}")
    print(f"Device : {device}")
    print(f"sa     : hidden_states[{sa_layer}]  (block {sa_layer - 1})")
    print(f"sb     : hidden_states[{sb_layer}]  (block {sb_layer - 1})")
    print()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print("Loaded.\n")

    categories = {}
    for label, texts in prompts.items():
        print(f"Extracting {label} ({len(texts)} samples)...")
        samples = []
        for idx, text in enumerate(texts):
            result = extract_one(text, model, tokenizer, sa_layer, sb_layer, device)
            result["index"] = idx
            result["label"] = label
            samples.append(result)
        categories[label] = samples
        print(f"  → {label}: {len(samples)} samples, "
              f"avg {np.mean([s['n_checkpoints'] for s in samples]):.1f} checkpoints")

    return {
        "_meta": {
            "description": "Raw scalar activation streams — RRG ground truth input",
            "model":       model_name,
            "sa":          f"Layer {sa_layer} residual stream L2 norm at semantic checkpoints (norm01)",
            "sb":          f"Layer {sb_layer} residual stream L2 norm at semantic checkpoints (norm01)",
            "reasoning_layer": sa_layer,
            "output_layer":    sb_layer,
            "normalization":   "norm01 — [0,1] per prompt",
            "checkpoint_definition": "last token before each newline + final token",
        },
        "categories": categories,
    }


def main():
    if not _TORCH_AVAILABLE:
        print("Error: torch and transformers are required.")
        print("Install with:  pip install torch transformers")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Extract RRG residual stream signals from a HuggingFace model."
    )
    parser.add_argument("--model",    default="gpt2",
                        help="HuggingFace model name or local path (default: gpt2)")
    parser.add_argument("--sa-layer", type=int, default=4,
                        help="Hidden state index for sa — early reasoning layer (default: 4)")
    parser.add_argument("--sb-layer", type=int, default=11,
                        help="Hidden state index for sb — output formation layer (default: 11)")
    parser.add_argument("--prompts",  default=None,
                        help="JSON file with categorized prompts (default: built-in COT-01 set)")
    parser.add_argument("--out",      default="raw_activations.json",
                        help="Output path (default: raw_activations.json)")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts) as f:
            data = json.load(f)
        prompts = data["categories"]
    else:
        prompts = _COT_PROMPTS
        print("Using built-in COT-01 prompts (10 per category).")

    result = extract_all(prompts, args.model, args.sa_layer, args.sb_layer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    total = sum(len(v) for v in result["categories"].values())
    print(f"\nSaved {total} samples → {out_path}")


if __name__ == "__main__":
    main()
