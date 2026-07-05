# -*- coding: utf-8 -*-
"""
rrg/elara_driver.py
===================
ELARA — Emergent Lock via Attractor Relational Analysis
Layer 2: Driver / Optimizer

ELARA is defined in two layers:

  Layer 1 (core.py)     — Certifier. Detects when a trajectory has entered
                          the Lock attractor. Passive. Domain-agnostic.

  Layer 2 (this file)   — Driver. Uses the Rank-Collapse Theorem as a
                          cost function and guides state transitions toward
                          the attractor via Pontryagin's Minimum Principle.
                          Active. Also domain-agnostic.

The key insight
---------------
The Rank-Collapse Theorem biconditional:

    Σ̂ ≈ rank-1  ⟺  dρ → 0  AND  |ρ*| → 1

is not only a detector — it is a cost function.

Minimizing L(ρ*, dρ) = dρ + (1 - |ρ*|) is equivalent to pushing the system
toward the Lock attractor. Pontryagin's Minimum Principle gives the optimal
control policy for that minimization over a finite horizon.

What ELARA does NOT do
----------------------
- It does not know the domain
- It does not know what sa, sb represent
- It does not interpret the meaning of Lock in any domain
- It does not generate sa, sb — those come from the domain

What ELARA does
---------------
- Receives the current phase state (ρ*, dρ) from RRG core
- Computes the cost gradient
- Returns the optimal control direction: how the state should move
- Evaluates whether a trajectory is converging toward the attractor

Reference: Calderas Cervantes, J.D. — RRG Formal Core (2026)
           Pontryagin, L.S. et al. — The Mathematical Theory of Optimal Processes
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from rrg.core import RRGObservables, RRGParams, LOCK, EMERGENCE, EXIT, NEUTRAL


# ---------------------------------------------------------------------------
# Cost function — the Rank-Collapse Theorem as objective
# ---------------------------------------------------------------------------

def lock_cost(rho_star: float, drho: float) -> float:
    """
    L(ρ*, dρ) = dρ + (1 - |ρ*|)

    Cost is zero only at the Lock attractor: dρ=0, |ρ*|=1.
    Cost increases as the system moves away from Lock.

    This is the cost function that Pontryagin minimizes.
    It is derived directly from the Rank-Collapse Theorem biconditional —
    not an arbitrary choice.

    Parameters
    ----------
    rho_star : float — current ρ* value
    drho     : float — current dρ value

    Returns
    -------
    float — cost ∈ [0, 2], lower is closer to Lock
    """
    return float(drho + (1.0 - abs(rho_star)))


def lock_cost_trajectory(obs: RRGObservables) -> np.ndarray:
    """
    Compute lock_cost at every timestep of an RRGObservables trajectory.

    Returns
    -------
    cost : np.ndarray, shape (T,), NaN where observables are undefined
    """
    rs = obs.rho_star
    dr = obs.drho
    valid = ~np.isnan(rs) & ~np.isnan(dr)
    cost = np.full(len(rs), np.nan)
    cost[valid] = dr[valid] + (1.0 - np.abs(rs[valid]))
    return cost


# ---------------------------------------------------------------------------
# Cost gradient — direction of steepest descent toward Lock
# ---------------------------------------------------------------------------

def cost_gradient(rho_star: float, drho: float) -> Tuple[float, float]:
    """
    ∇L(ρ*, dρ) — gradient of the cost with respect to (ρ*, dρ).

    ∂L/∂ρ* = -sign(ρ*)   (push ρ* toward ±1)
    ∂L/∂dρ =  1           (always reduce dρ)

    The optimal control direction is -∇L (steepest descent).

    Returns
    -------
    (grad_rho_star, grad_drho) : tuple of floats
    """
    grad_rs = -np.sign(rho_star) if rho_star != 0 else 0.0
    grad_dr = 1.0
    return float(grad_rs), float(grad_dr)


def optimal_direction(rho_star: float, drho: float) -> Tuple[float, float]:
    """
    Optimal control direction: -∇L(ρ*, dρ).

    This is the direction in phase space (ρ*, dρ) that most reduces cost.
    A system following this direction will converge to Lock.

    Returns
    -------
    (d_rho_star, d_drho) : unit vector pointing toward Lock
    """
    g_rs, g_dr = cost_gradient(rho_star, drho)
    direction = np.array([-g_rs, -g_dr])
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return 0.0, 0.0
    direction = direction / norm
    return float(direction[0]), float(direction[1])


# ---------------------------------------------------------------------------
# Hamiltonian — Pontryagin's Minimum Principle
# ---------------------------------------------------------------------------

@dataclass
class PontryaginState:
    """
    State for the Pontryagin optimal control problem.

    State variables  x = (ρ*, dρ)
    Costate variables λ = (λ_ρ*, λ_dρ)
    Control          u ∈ U  (admissible set, domain-defined)

    The Hamiltonian:
        H(x, u, λ) = L(x) + λᵀ f(x, u)

    where f(x, u) is the state dynamics (how ρ* and dρ evolve under control u).
    Pontryagin's Minimum Principle: optimal u* minimizes H at each instant.
    """
    rho_star : float   # current ρ*
    drho     : float   # current dρ
    lam_rs   : float   # costate for ρ*  (initialized to ∂L/∂ρ*)
    lam_dr   : float   # costate for dρ  (initialized to ∂L/∂dρ)
    t        : int     # current timestep

    @classmethod
    def initialize(cls, rho_star: float, drho: float, t: int = 0) -> "PontryaginState":
        """
        Initialize costate variables from the cost gradient at (ρ*, dρ).
        λ₀ = ∇L(x₀)
        """
        lam_rs, lam_dr = cost_gradient(rho_star, drho)
        return cls(
            rho_star = rho_star,
            drho     = drho,
            lam_rs   = lam_rs,
            lam_dr   = lam_dr,
            t        = t,
        )

    def hamiltonian(self, u_rs: float, u_dr: float) -> float:
        """
        H(x, u, λ) = L(x) + λ_ρ* · u_ρ* + λ_dρ · u_dρ

        where u = (u_ρ*, u_dρ) is the proposed control increment.

        Lower H → better control action.
        """
        L = lock_cost(self.rho_star, self.drho)
        return L + self.lam_rs * u_rs + self.lam_dr * u_dr

    def optimal_control(self, step_size: float = 0.1) -> Tuple[float, float]:
        """
        u* = argmin H over admissible controls.

        For linear H in u, the minimum is achieved by moving maximally
        in the direction of -λ (bang-bang control).

        Returns
        -------
        (u_rho_star, u_drho) : optimal control increment
        """
        direction = np.array([-self.lam_rs, -self.lam_dr])
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return 0.0, 0.0
        direction = direction / norm * step_size
        return float(direction[0]), float(direction[1])

    def step(self, step_size: float = 0.1) -> "PontryaginState":
        """
        Apply one step of optimal control.

        State update:   x_{t+1} = x_t + u*
        Costate update: λ_{t+1} = ∇L(x_{t+1})  (recompute from new state)

        Returns
        -------
        New PontryaginState after one optimal step.
        """
        u_rs, u_dr = self.optimal_control(step_size)

        new_rs = float(np.clip(self.rho_star + u_rs, -1.0, 1.0))
        new_dr = float(max(0.0, self.drho + u_dr))

        new_lam_rs, new_lam_dr = cost_gradient(new_rs, new_dr)

        return PontryaginState(
            rho_star = new_rs,
            drho     = new_dr,
            lam_rs   = new_lam_rs,
            lam_dr   = new_lam_dr,
            t        = self.t + 1,
        )


# ---------------------------------------------------------------------------
# ELARA Driver — the main interface
# ---------------------------------------------------------------------------

@dataclass
class ELARADriver:
    """
    ELARA Layer 2 — the driver.

    Given an RRGObservables trajectory (from Layer 1), ELARA:
      1. Evaluates the cost trajectory L(ρ*, dρ)
      2. Determines whether the system is converging toward Lock
      3. Computes the optimal control sequence via Pontryagin
      4. Returns the reward signal for RL (reward ∝ -cost)

    This is the bridge from detection to generation:
      - RRG core detects where the attractor is
      - ELARA driver guides trajectories toward it

    Parameters
    ----------
    obs       : RRGObservables from core.py
    step_size : float — control step size (learning rate analog)
    gamma     : float — discount factor for trajectory reward
    """
    obs       : RRGObservables
    step_size : float = 0.05
    gamma     : float = 0.95

    def cost_trajectory(self) -> np.ndarray:
        """L(t) = dρ(t) + (1 - |ρ*(t)|) at each timestep."""
        return lock_cost_trajectory(self.obs)

    def reward_trajectory(self) -> np.ndarray:
        """
        Reward signal for RL: reward(t) = -L(t), clipped to [0, 1].

        reward → 1 when system is in Lock (cost → 0)
        reward → 0 when system is far from Lock (cost → 2)

        This is the reward landscape described in the paper:
          Lock genuino      → compact alpha-shape, high reward
          Resonance-Lock    → medium reward
          Oscillating-Sign  → dispersed, low reward
        """
        cost = self.cost_trajectory()
        valid = ~np.isnan(cost)
        reward = np.full(len(cost), np.nan)
        reward[valid] = np.clip(1.0 - cost[valid] / 2.0, 0.0, 1.0)
        return reward

    def is_converging(self, window: int = 20) -> bool:
        """
        Returns True if the cost is decreasing over the last `window` steps.
        A converging trajectory is moving toward Lock.
        """
        cost = self.cost_trajectory()
        valid = cost[~np.isnan(cost)]
        if len(valid) < window:
            return False
        recent = valid[-window:]
        slope = np.polyfit(np.arange(window), recent, 1)[0]
        return bool(slope < 0)

    def optimal_trajectory(
        self,
        n_steps : int = 50,
    ) -> np.ndarray:
        """
        Simulate the optimal trajectory from the current phase state
        using Pontryagin's Minimum Principle.

        Starts from the last valid (ρ*, dρ) in the observed trajectory
        and runs n_steps of optimal control.

        Returns
        -------
        trajectory : np.ndarray, shape (n_steps, 3)
            Columns: [ρ*, dρ, cost]
        """
        # Start from last valid observed state
        rs = self.obs.rho_star
        dr = self.obs.drho
        valid = ~np.isnan(rs) & ~np.isnan(dr)
        if not valid.any():
            raise ValueError("No valid observations to initialize from.")

        last = np.where(valid)[0][-1]
        state = PontryaginState.initialize(
            rho_star = float(rs[last]),
            drho     = float(dr[last]),
            t        = 0,
        )

        trajectory = np.zeros((n_steps, 3))
        for i in range(n_steps):
            trajectory[i] = [
                state.rho_star,
                state.drho,
                lock_cost(state.rho_star, state.drho),
            ]
            state = state.step(self.step_size)

        return trajectory

    def summary(self) -> dict:
        """
        Full ELARA summary for a trajectory.
        """
        cost    = self.cost_trajectory()
        reward  = self.reward_trajectory()
        obs_sum = self.obs.summary()
        brk     = find_break_checkpoint(self.obs)

        return {
            # From RRG core (Layer 1)
            **obs_sum,
            # From ELARA driver (Layer 2)
            'cost_mean'        : float(np.nanmean(cost)),
            'cost_final'       : float(cost[~np.isnan(cost)][-1]) if (~np.isnan(cost)).any() else np.nan,
            'reward_mean'      : float(np.nanmean(reward)),
            'reward_final'     : float(reward[~np.isnan(reward)][-1]) if (~np.isnan(reward)).any() else np.nan,
            'is_converging'    : self.is_converging(),
            # Per-step break detection (open problem from paper)
            'break_found'      : brk['break_found'],
            'break_index'      : brk['break_index'],
            'break_drho'       : brk['break_drho'],
            'pre_break_lock'   : brk['pre_break_lock'],
            'post_break_lock'  : brk['post_break_lock'],
        }


# ---------------------------------------------------------------------------
# Per-step dρ trajectory — break point detection
# ---------------------------------------------------------------------------

def find_break_checkpoint(obs: RRGObservables) -> dict:
    """
    Find the specific timestep where coupling broke mid-sequence.

    'Break' = dρ crosses from below τ_lock_dr to above it after having
    been in Lock or Emergence. This is the open problem from the paper —
    the gate tells you what type of lock happened across the group, but
    this tells you which checkpoint broke the coupling while it was happening.

    Domain-agnostic — operates on any RRGObservables.

    Parameters
    ----------
    obs : RRGObservables

    Returns
    -------
    dict with:
        'break_found'     : bool
        'break_index'     : int or None  — timestep of the break
        'break_drho'      : float or None — dρ value at break
        'break_regime'    : str or None  — regime label at break
        'pre_break_lock'  : float — fraction of time in Lock before break
        'post_break_lock' : float — fraction of time in Lock after break
        'drho_trajectory' : list of float — full dρ series (NaN-clean)
        'regime_trajectory': list of str  — full regime series
    """
    dr     = obs.drho
    regime = obs.regime
    tau    = obs.params.tau_lock_dr

    valid  = ~np.isnan(dr)
    idx    = np.where(valid)[0]

    if len(idx) < 2:
        return {
            'break_found'      : False,
            'break_index'      : None,
            'break_drho'       : None,
            'break_regime'     : None,
            'pre_break_lock'   : 0.0,
            'post_break_lock'  : 0.0,
            'drho_trajectory'  : [],
            'regime_trajectory': [],
        }

    # Find break: dρ rises significantly after its minimum.
    # break_thr = min_val + 10% of the trajectory's dρ range.
    # This is relative to the trajectory — not to τ_lock_dr which is a
    # global threshold and may never be crossed in short sequences.
    break_index = None

    dr_valid  = dr[idx]
    min_pos   = int(np.argmin(dr_valid))
    min_val   = float(dr_valid[min_pos])
    dr_range  = float(dr_valid.max() - min_val)
    break_thr = min_val + max(0.1 * dr_range, tau * 0.1)

    for i in range(min_pos + 1, len(idx)):
        t_curr = idx[i]
        if dr[t_curr] > break_thr:
            break_index = int(t_curr)
            break

    # Pre/post break lock fractions
    lock_mask = obs.lock_mask
    if break_index is not None:
        pre  = lock_mask[:break_index]
        post = lock_mask[break_index:]
        pre_frac  = float(pre.sum()  / max(len(pre),  1))
        post_frac = float(post.sum() / max(len(post), 1))
    else:
        pre_frac  = float(lock_mask.sum() / max(valid.sum(), 1))
        post_frac = pre_frac

    return {
        'break_found'      : break_index is not None,
        'break_index'      : break_index,
        'break_drho'       : float(dr[break_index]) if break_index is not None else None,
        'break_regime'     : str(regime[break_index]) if break_index is not None else None,
        'pre_break_lock'   : pre_frac,
        'post_break_lock'  : post_frac,
        'drho_trajectory'  : [float(v) for v in dr[valid]],
        'regime_trajectory': [str(regime[i]) for i in idx],
    }

def classify_group_lock(
    lock_frac_array    : np.ndarray,
    reference_variance : float,
) -> str:
    """
    Second AND gate — group-level Lock classifier.

    The first AND gate (in core.py) operates per-timestep:
        dρ < τ_lock  AND  |ρ*| > τ_rho

    This second gate operates at the population level — it distinguishes
    genuine-lock from resonance-lock by examining the distribution of
    lock_frac across a group of samples.

    How the gap was found
    ---------------------
    Cross-referencing paper summary statistics against ELARA's raw JSON
    output revealed bimodal distribution in UNFAITHFUL category
    (variance=0.233, 7 zeros and 3 perfect locks). The paper's second AND
    gate classifies this as resonance-lock. ELARA had no group-level
    classifier — it reported means and moved on. This function closes that gap.

    Parameters
    ----------
    lock_frac_array    : np.ndarray
        lock_frac per sample in the group. Shape (N,), values in [0, 1].
    reference_variance : float
        Variance of lock_frac in the reference (stable) group for this domain.
        Supplied by the domain adapter — NOT hardcoded.
        Examples:
          CoT domain    → variance of FAITHFUL lock_frac distribution
          EEG domain    → variance of resting-state lock_frac
          SKR domain    → variance of pre-equinox baseline epoch

    Returns
    -------
    str — one of:
        'genuine-lock'   — majority lock, low variance, no bimodality
        'resonance-lock' — locks but bimodal or variance > 2x reference
        'no-lock'        — minority of samples lock
    """
    arr      = np.asarray(lock_frac_array, dtype=float)
    var      = float(np.var(arr))
    zeros    = int(np.sum(arr == 0))
    ones     = int(np.sum(arr >= 0.99))
    bimodal  = (zeros > 0) and (ones > 0)
    majority = np.sum(arr > 0) >= len(arr) / 2

    if not majority:
        return "no-lock"
    if var > 2.0 * reference_variance or bimodal:
        return "resonance-lock"
    return "genuine-lock"


def group_summary(
    drivers            : dict,           # label -> list of ELARADriver
    reference_label    : str,            # which group is the stable reference
) -> dict:
    """
    Run the second AND gate across a set of groups.

    Parameters
    ----------
    drivers         : dict mapping label -> list of ELARADriver
                      One driver per sample, grouped by category.
    reference_label : str
                      The label of the reference (stable) group.
                      Its lock_frac variance becomes reference_variance.

    Returns
    -------
    dict mapping label -> {
        'n'                  : int,
        'lock_frac_mean'     : float,
        'lock_frac_var'      : float,
        'lock_frac_zeros'    : int,
        'lock_frac_ones'     : int,
        'bimodal'            : bool,
        'group_lock_class'   : str,   ← second AND gate result
        'reward_mean'        : float,
        'cost_mean'          : float,
    }
    """
    # Compute lock_frac per sample per group
    group_fracs = {}
    for label, drv_list in drivers.items():
        group_fracs[label] = np.array([
            d.summary()['lock_frac'] for d in drv_list
        ])

    # Reference variance from the stable group
    if reference_label not in group_fracs:
        raise ValueError(
            f"reference_label '{reference_label}' not in drivers. "
            f"Available: {list(group_fracs.keys())}"
        )
    reference_variance = float(np.var(group_fracs[reference_label]))

    result = {}
    for label, fracs in group_fracs.items():
        drv_list   = drivers[label]
        summaries  = [d.summary() for d in drv_list]
        gate_class = classify_group_lock(fracs, reference_variance)

        result[label] = {
            'n'                : len(fracs),
            'lock_frac_mean'   : float(np.mean(fracs)),
            'lock_frac_var'    : float(np.var(fracs)),
            'lock_frac_zeros'  : int(np.sum(fracs == 0)),
            'lock_frac_ones'   : int(np.sum(fracs >= 0.99)),
            'bimodal'          : bool((np.sum(fracs == 0) > 0) and (np.sum(fracs >= 0.99) > 0)),
            'group_lock_class' : gate_class,
            'reward_mean'      : float(np.mean([s['reward_mean'] for s in summaries])),
            'cost_mean'        : float(np.mean([s['cost_mean']   for s in summaries])),
        }

    return result


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def drive(
    obs       : RRGObservables,
    step_size : float = 0.05,
    gamma     : float = 0.95,
) -> ELARADriver:
    """
    Create an ELARADriver from an RRGObservables bundle.

    Usage
    -----
        from rrg.core import generate_regime
        from rrg.elara_driver import drive

        obs    = generate_regime('lock')
        driver = drive(obs)
        print(driver.summary())
        traj   = driver.optimal_trajectory(n_steps=100)
    """
    return ELARADriver(obs=obs, step_size=step_size, gamma=gamma)
