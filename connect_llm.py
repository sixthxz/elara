# -*- coding: utf-8 -*-
"""
connect_llm.py
==============
Connects raw_activations.json (COT-01 domain) to the RRG/ELARA pipeline.

Usage
-----
    python connect_llm.py --raw raw_activations.json

What this does
--------------
1. Loads raw_activations.json — sa, sb per sample per category
2. Feeds each sample through RRG core (windowed_corr, ρ*, dρ, FSM)
3. Feeds each RRGObservables through ELARA driver (cost, reward, convergence)
4. Prints summary table and saves results to llm_rrg_results.json

This is the correct pipeline per the paper:
  sa, sb (raw) → core.py → ρ*, dρ → elara_driver.py → reward
"""

import json
import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from elara.engine import RRGObservables, RRGParams
from elara.elara_driver import drive, group_summary


# COT-01 domain-local parameters (from rrg_cross_domain_index.json)
COT_PARAMS = RRGParams(
    W            = 2,      # W_SEMANTIC = 2 reasoning steps (paper primary)
    tau_lock_rho = 0.35,   # TAU_RHO
    tau_lock_dr  = 0.02,   # TAU_LOCK
    tau_meta     = 0.015,  # TAU_META
    tau_exit     = 0.04,   # TAU_EXIT
    algebra      = 'R',    # A0 — scalar residual norms
)


def load_raw(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def process_sample(sa: np.ndarray, sb: np.ndarray) -> dict:
    """
    Run full RRG + ELARA pipeline on one sample.

    Parameters
    ----------
    sa, sb : np.ndarray — raw scalar streams, shape (N_checkpoints,)

    Returns
    -------
    dict with RRG observables + ELARA summary
    """
    obs    = RRGObservables.from_series(sa, sb, params=COT_PARAMS)
    driver = drive(obs, step_size=0.05)
    return driver.summary()


def process_category(samples: list, label: str) -> list:
    results = []
    for s in samples:
        sa = np.array(s['sa'], dtype=float)
        sb = np.array(s['sb'], dtype=float)
        r  = process_sample(sa, sb)
        r['label'] = label
        r['index'] = s['index']
        r['n_checkpoints'] = s['n_checkpoints']
        results.append(r)
    return results


def print_summary(all_results: dict) -> None:
    print()
    print(f"{'Category':12} | {'n':>4} | {'ρ* mean':>9} | {'dρ mean':>9} | "
          f"{'lock_frac':>9} | {'cost':>7} | {'reward':>7} | {'converging':>10}")
    print("-" * 85)

    for label, results in all_results.items():
        n          = len(results)
        rs_mean    = np.mean([r['rho_star_mean'] for r in results])
        dr_mean    = np.mean([r['drho_mean']     for r in results])
        lf_mean    = np.mean([r['lock_frac']     for r in results])
        cost_mean  = np.mean([r['cost_mean']     for r in results])
        rew_mean   = np.mean([r['reward_mean']   for r in results])
        conv_frac  = np.mean([r['is_converging'] for r in results])

        print(f"{label:12} | {n:>4} | {rs_mean:>+9.3f} | {dr_mean:>9.4f} | "
              f"{lf_mean:>9.1%} | {cost_mean:>7.3f} | {rew_mean:>7.3f} | "
              f"{conv_frac:>9.1%}")

    print()
    print("Expected hierarchy (fiel al paper):")
    print("  FAITHFUL   -- highest reward, highest lock_frac, lowest cost")
    print("  WRONG      -- medium (locks but to wrong attractor)")
    print("  UNFAITHFUL -- lowest reward, lowest lock_frac, highest cost")

    # Per-step break detection
    print()
    print(f"{'Category':12} | {'break%':>7} | {'break_idx':>9} | {'pre_lock':>9} | {'post_lock':>9}")
    print("-" * 57)
    for label, results in all_results.items():
        breaks    = [r for r in results if r.get('break_found')]
        break_pct = len(breaks) / len(results)
        idx_mean  = np.mean([r['break_index']    for r in breaks]) if breaks else float('nan')
        pre_mean  = np.mean([r['pre_break_lock']  for r in results])
        post_mean = np.mean([r['post_break_lock'] for r in results])
        print(f"{label:12} | {break_pct:>7.1%} | {idx_mean:>9.1f} | {pre_mean:>9.3f} | {post_mean:>9.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='rrg_output/raw_activations.json',
                        help='Path to raw_activations.json')
    parser.add_argument('--out', default='rrg_output/llm_rrg_results.json',
                        help='Output path for results')
    args = parser.parse_args()

    print(f"Loading: {args.raw}")
    data = load_raw(args.raw)

    meta = data.get('_meta', {})
    print(f"Model : {meta.get('model', 'unknown')}")
    print(f"sa    : {meta.get('sa', '?')}")
    print(f"sb    : {meta.get('sb', '?')}")
    print()

    all_results = {}
    for label, samples in data['categories'].items():
        print(f"Processing {label} ({len(samples)} samples)...")
        all_results[label] = process_category(samples, label)

    print_summary(all_results)

    # --- Second AND gate — group-level classifier ---
    # Build drivers dict: label -> list of ELARADriver
    drivers = {}
    for label, samples in data['categories'].items():
        drv_list = []
        for s in samples:
            sa  = np.array(s['sa'], dtype=float)
            sb  = np.array(s['sb'], dtype=float)
            obs = RRGObservables.from_series(sa, sb, params=COT_PARAMS)
            drv_list.append(drive(obs))
        drivers[label] = drv_list

    # FAITHFUL is the reference (stable) group for CoT domain
    gs = group_summary(drivers, reference_label='FAITHFUL')

    print("\nSecond AND gate — group-level classification:")
    print(f"{'Category':12} | {'lock_var':>9} | {'zeros':>5} | {'ones':>5} | {'bimodal':>7} | {'gate':>14}")
    print("-" * 70)
    for label, r in gs.items():
        print(f"{label:12} | {r['lock_frac_var']:>9.3f} | {r['lock_frac_zeros']:>5} | "
              f"{r['lock_frac_ones']:>5} | {str(r['bimodal']):>7} | {r['group_lock_class']:>14}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
