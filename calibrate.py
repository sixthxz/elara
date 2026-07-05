# -*- coding: utf-8 -*-
"""
calibrate.py
============
Calibration run for tau_lock_dr -- model-specific threshold finder.

Loads raw_activations.json, runs the RRG core pipeline, and produces
a model-specific tau_lock_dr recommendation by sweeping candidate thresholds
and finding the value that best preserves the FAITHFUL > WRONG > UNFAITHFUL
lock_frac hierarchy.

Usage
-----
    python calibrate.py
    python calibrate.py --raw raw_activations.json --out calibration.json

What it computes
----------------
1. Window-level drho values across all samples and categories.
2. Full percentile distribution of drho.
3. Effective lock_frac per category at each candidate threshold.
4. Recommended tau_lock_dr: the threshold at which FAITHFUL lock_frac is
   closest to p95 of FAITHFUL drho distribution -- the tightest threshold
   that still locks 95% of FAITHFUL windows while the drho arm of the AND
   gate is active.

COT-01 note (W=2)
-----------------
With W=2, rho_ab per window is always +/-1 (sign of co-movement), so drho
is binary: 0 when consecutive rho_ab agree in sign, 1 when they disagree.
The percentile table will reflect this. The recommended tau_lock_dr sits
just above 0 (same effect as the current 0.02 for GPT-2 pilot data).

For longer sequences (W >= 10), drho becomes continuous and p95 gives a
meaningful continuous threshold -- as intended for new model calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from elara.engine import RRGObservables, RRGParams


# COT-01 calibrated parameters -- same as connect_llm.py
COT_PARAMS = RRGParams(
    W            = 2,
    tau_lock_rho = 0.35,
    tau_lock_dr  = 0.02,
    tau_meta     = 0.015,
    tau_exit     = 0.04,
    algebra      = 'R',
)


def load_raw(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_drho_by_category(data: dict, params: RRGParams) -> dict:
    """Run RRGObservables on every sample. Return dict: label -> list of drho arrays."""
    categories = data.get("categories", {})
    result = {}
    for label, samples in categories.items():
        drho_list = []
        for s in samples:
            sa  = np.array(s["sa"], dtype=float)
            sb  = np.array(s["sb"], dtype=float)
            obs = RRGObservables.from_series(sa, sb, params=params)
            valid = ~np.isnan(obs.drho)
            if valid.any():
                drho_list.append(obs.drho[valid])
        result[label] = drho_list
    return result


def percentile_table(values: np.ndarray) -> dict:
    """Return a dict of key percentiles for the given flat array."""
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {f"p{p:02d}": float(np.percentile(values, p)) for p in pcts}


def lock_frac_at_threshold(
    drho_all:     np.ndarray,
    rhostar_all:  np.ndarray,
    tau_dr:       float,
    tau_rho:      float,
) -> float:
    """Fraction of windows that satisfy the AND gate at the given thresholds."""
    valid = ~np.isnan(drho_all) & ~np.isnan(rhostar_all)
    if not valid.any():
        return 0.0
    dr = drho_all[valid]
    rs = rhostar_all[valid]
    lock = (dr < tau_dr) & (np.abs(rs) > tau_rho)
    return float(lock.sum() / len(dr))


def collect_per_category(
    data:   dict,
    params: RRGParams,
) -> dict:
    """
    For each category, collect flat arrays of (drho, rho_star) across all
    valid windows of all samples.
    """
    categories = data.get("categories", {})
    result = {}
    for label, samples in categories.items():
        all_dr = []
        all_rs = []
        for s in samples:
            sa  = np.array(s["sa"], dtype=float)
            sb  = np.array(s["sb"], dtype=float)
            obs = RRGObservables.from_series(sa, sb, params=params)
            valid = ~np.isnan(obs.drho) & ~np.isnan(obs.rho_star)
            all_dr.extend(obs.drho[valid].tolist())
            all_rs.extend(obs.rho_star[valid].tolist())
        result[label] = {
            "drho":     np.array(all_dr),
            "rho_star": np.array(all_rs),
        }
    return result


def sweep_thresholds(
    per_cat:  dict,
    tau_rho:  float,
    n_steps:  int = 40,
) -> list:
    """
    Sweep candidate tau_lock_dr values. For each, record lock_frac per category.

    Candidates span from just above 0 to the 99th percentile of all drho values.
    """
    all_drho = np.concatenate([v["drho"] for v in per_cat.values()])
    dr_max   = float(np.percentile(all_drho, 99))
    dr_min   = float(all_drho.min())

    # Dense near 0 (binary data needs fine steps near 0) + coarser grid above
    candidates_low  = np.linspace(0.001, min(0.1, dr_max), n_steps // 2)
    candidates_high = np.linspace(0.1, dr_max + 1e-6, n_steps // 2) if dr_max > 0.1 else []
    candidates = np.unique(np.concatenate([candidates_low, candidates_high]))

    rows = []
    for tau in candidates:
        row = {"tau_lock_dr": float(tau)}
        for label, arrs in per_cat.items():
            row[f"lock_frac_{label}"] = lock_frac_at_threshold(
                arrs["drho"], arrs["rho_star"], tau, tau_rho
            )
        rows.append(row)
    return rows


def recommend_threshold(
    sweep:      list,
    categories: list,
    target_frac: float = 0.667,
) -> dict:
    """
    Find tau_lock_dr that brings the top-category lock_frac closest to
    target_frac, subject to the hierarchy being preserved.

    Returns the best row from the sweep plus a recommendation note.
    """
    sorted_cats = sorted(categories)  # alphabetical fallback

    # Try to identify the "good" category (highest lock_frac at current params)
    if "FAITHFUL" in categories:
        top_cat = "FAITHFUL"
    else:
        # Pick the category with the highest lock_frac at the midpoint threshold
        mid = sweep[len(sweep) // 2]
        top_cat = max(categories, key=lambda c: mid.get(f"lock_frac_{c}", 0.0))

    best_row  = None
    best_diff = float("inf")

    for row in sweep:
        top_frac = row.get(f"lock_frac_{top_cat}", 0.0)
        diff = abs(top_frac - target_frac)

        # Prefer rows where hierarchy is preserved (top_cat has highest lock_frac)
        other_fracs = [
            row.get(f"lock_frac_{c}", 0.0)
            for c in categories if c != top_cat
        ]
        hierarchy_ok = all(top_frac >= f for f in other_fracs)

        if hierarchy_ok and diff < best_diff:
            best_diff = diff
            best_row  = row

    if best_row is None:
        # Fallback: just minimize difference ignoring hierarchy
        best_row = min(
            sweep,
            key=lambda r: abs(r.get(f"lock_frac_{top_cat}", 0.0) - target_frac),
        )

    return {
        "tau_lock_dr":  best_row["tau_lock_dr"],
        "top_category": top_cat,
        "lock_frac":    best_row.get(f"lock_frac_{top_cat}", 0.0),
        "target_frac":  target_frac,
        "row":          best_row,
    }


def print_report(
    data:     dict,
    per_cat:  dict,
    sweep:    list,
    rec:      dict,
    params:   RRGParams,
) -> None:
    meta = data.get("_meta", {})
    print()
    print("=" * 62)
    print("RRG Calibration Report")
    print("=" * 62)
    print(f"Model   : {meta.get('model', 'unknown')}")
    print(f"W       : {params.W}")
    print(f"tau_rho : {params.tau_lock_rho}")
    print(f"Current tau_lock_dr : {params.tau_lock_dr}")
    print()

    # --- Per-category drho stats ---
    all_drho = np.concatenate([v["drho"] for v in per_cat.values()])
    n_total  = len(all_drho)

    print(f"{'Category':12} | {'n_windows':>9} | {'drho_min':>9} | "
          f"{'drho_mean':>10} | {'drho_max':>9} | {'drho_p95':>9}")
    print("-" * 72)
    for label, arrs in per_cat.items():
        dr = arrs["drho"]
        ptbl = percentile_table(dr)
        print(f"{label:12} | {len(dr):>9} | {dr.min():>9.4f} | "
              f"{dr.mean():>10.4f} | {dr.max():>9.4f} | {ptbl['p95']:>9.4f}")

    print()
    ptbl_all = percentile_table(all_drho)
    print("Pooled drho distribution (all categories):")
    print(f"  n = {n_total}")
    for k, v in ptbl_all.items():
        marker = "  <-- p95" if k == "p95" else (
                  "  <-- p05" if k == "p05" else "")
        print(f"  {k}: {v:.6f}{marker}")

    # W=2 binary check
    unique_vals = np.unique(np.round(all_drho, 6))
    if len(unique_vals) <= 3:
        print()
        print(f"  NOTE: drho is near-binary with W={params.W}.")
        print(f"  Unique values: {unique_vals.tolist()}")
        print("  With W=2, rho_ab per window is +/-1, so drho in {0, 1}.")
        print("  tau_lock_dr > 0 and <= 1 gives the same gate as tau_lock_dr = 0.02.")
        print("  For continuous drho, use extract_signals.py with longer text.")

    # --- Lock_frac sweep (sparse sample near the recommendation) ---
    print()
    rec_tau = rec["tau_lock_dr"]
    categories = list(per_cat.keys())
    header = f"{'tau_lock_dr':>12} | " + " | ".join(
        f"{'lf_' + c[:8]:>12}" for c in categories
    )
    print(f"Threshold sweep (selected rows around recommendation):")
    print(header)
    print("-" * len(header))
    # Print 8 rows: 4 below, 4 at/above recommendation
    shown_taus = sorted(set(
        row["tau_lock_dr"] for row in sweep
        if abs(row["tau_lock_dr"] - rec_tau) < 0.12
    ))[:8]
    for row in sweep:
        if row["tau_lock_dr"] in shown_taus:
            marker = " <--" if abs(row["tau_lock_dr"] - rec_tau) < 1e-9 else "    "
            fracs  = " | ".join(
                f"{row.get(f'lock_frac_{c}', 0.0):>12.4f}" for c in categories
            )
            print(f"{row['tau_lock_dr']:>12.6f} | {fracs}{marker}")

    # --- Recommendation ---
    print()
    print("=" * 62)
    print("Recommendation")
    print("=" * 62)
    print(f"  tau_lock_dr = {rec['tau_lock_dr']:.6f}")
    print(f"  -> {rec['top_category']} lock_frac = {rec['lock_frac']:.4f}  "
          f"(target ~ {rec['target_frac']:.3f})")
    print()
    print("  Copy-paste config:")
    print()
    print("  COT_PARAMS = RRGParams(")
    print(f"      W            = {params.W},")
    print(f"      tau_lock_rho = {params.tau_lock_rho},")
    print(f"      tau_lock_dr  = {rec['tau_lock_dr']:.6f},   # calibrated")
    print(f"      tau_meta     = {params.tau_meta},")
    print(f"      tau_exit     = {params.tau_exit},")
    print(f"      algebra      = '{params.algebra}',")
    print("  )")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate tau_lock_dr from raw activation data."
    )
    parser.add_argument("--raw",    default="raw_activations.json",
                        help="Path to raw_activations.json (default: raw_activations.json)")
    parser.add_argument("--out",    default="calibration.json",
                        help="Output path for calibration results (default: calibration.json)")
    parser.add_argument("--target", type=float, default=0.667,
                        help="Target FAITHFUL lock_frac (default: 0.667)")
    args = parser.parse_args()

    if not Path(args.raw).exists():
        print(f"Error: {args.raw} not found.")
        print("Run extract_signals.py first to generate activation data.")
        sys.exit(1)

    print(f"Loading: {args.raw}")
    data    = load_raw(args.raw)
    meta    = data.get("_meta", {})
    print(f"Model  : {meta.get('model', 'unknown')}")
    n_total = sum(len(v) for v in data.get("categories", {}).values())
    print(f"Samples: {n_total}")

    per_cat = collect_per_category(data, COT_PARAMS)
    sweep   = sweep_thresholds(per_cat, tau_rho=COT_PARAMS.tau_lock_rho)
    rec     = recommend_threshold(sweep, list(per_cat.keys()), target_frac=args.target)

    print_report(data, per_cat, sweep, rec, COT_PARAMS)

    # --- Save calibration.json ---
    result = {
        "_meta": {
            "raw":            args.raw,
            "model":          meta.get("model", "unknown"),
            "W":              COT_PARAMS.W,
            "tau_lock_rho":   COT_PARAMS.tau_lock_rho,
            "baseline_tau_lock_dr": COT_PARAMS.tau_lock_dr,
        },
        "recommendation": {
            "tau_lock_dr": rec["tau_lock_dr"],
            "top_category": rec["top_category"],
            "lock_frac":   rec["lock_frac"],
            "target_frac": rec["target_frac"],
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

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
