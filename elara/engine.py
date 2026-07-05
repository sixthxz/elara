"""
RRG Engine — Layer 1-5 pure math.
All formulas traced directly from the paper (Calderas Cervantes 2026).

No IO, no state, no side effects. Every function maps inputs → output.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Layer 1 — Primary relational process
# ---------------------------------------------------------------------------

def _pearson(a: List[float], b: List[float]) -> float:
    """Pearson correlation for two windows of equal length.
    Returns 0.0 on degenerate input (constant series).
    """
    n = len(a)
    if n < 2:
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((x - mean_b) ** 2 for x in b))
    if den_a < 1e-10 or den_b < 1e-10:
        return 0.0
    r = num / (den_a * den_b)
    # Clamp numerical noise
    return max(-1.0, min(1.0, r))


def compute_rho_ab(sa: List[float], sb: List[float], W: int) -> List[float]:
    """Rolling windowed Pearson correlation. Eq 3.

    rho_ab(t, W) = Corr(sa[t,W], sb[t,W]) in [-1, 1]

    Returns a list of length max(0, T - W + 1).
    Layer 1 → feeds all subsequent layers.
    """
    T = len(sa)
    if T != len(sb):
        raise ValueError("sa and sb must have the same length")
    rho = []
    for t in range(W - 1, T):
        win_a = sa[t - W + 1 : t + 1]
        win_b = sb[t - W + 1 : t + 1]
        rho.append(_pearson(win_a, win_b))
    return rho


# ---------------------------------------------------------------------------
# Layer 2 — First-order statistics of rho_ab
# ---------------------------------------------------------------------------

def compute_rho_star(rho_ab: List[float]) -> float:
    """E[rho_ab] — mean correlation over the window. Layer 2.

    rho* = E[rho_ab]

    Used in: reff, delta, AND gate condition |rho*| > tau_rho.
    """
    if not rho_ab:
        return 0.0
    return sum(rho_ab) / len(rho_ab)


def compute_d_rho(rho_ab: List[float]) -> float:
    """Var(rho_ab) — first-order stability. Eq 7. Layer 2.

    d_rho(t, W) = Var({ rho_ab(s) : s in [t-W+1, t] })

    Primary observable of RRG: measures fluctuation of coupling stability.
    Seed-independent under weak stationarity (Section 4).
    """
    if len(rho_ab) < 2:
        return 0.0
    mu = sum(rho_ab) / len(rho_ab)
    return sum((r - mu) ** 2 for r in rho_ab) / len(rho_ab)


def compute_d_rho_series(rho_ab: List[float]) -> List[float]:
    """Per-checkpoint cumulative d_rho. Phase D early-warning readout.

    d_rho_series[k] = Var(rho_ab[0:k+1])  for k = 0 .. K-1

    series[0] == 0.0 always (single value).
    series[-1] == compute_d_rho(rho_ab) exactly.
    Rising values at checkpoint k signal geometric instability before
    the final aggregate is resolved.
    """
    return [compute_d_rho(rho_ab[:k + 1]) for k in range(len(rho_ab))]


# ---------------------------------------------------------------------------
# Layer 3 — Meta-signal and effective rank
# ---------------------------------------------------------------------------

def compute_d_rho_meta(sa: List[float], sb: List[float], W: int) -> float:
    """Second-order stability / early-warning signal. Eq 8. Layer 3.

    d_rho_meta(t, W) = Var( Corr(Δsa[t,W], Δsb[t,W]) )

    where Δsa(t) = sa(t) - sa(t-1).

    Detects deceleration of coupling before Sufficient is lost.
    Pre-juncture signal: dρ,meta ↑ → δ approaches 0 → juncture.
    """
    if len(sa) < W + 1:
        return 0.0
    # First differences
    dsa = [sa[i] - sa[i - 1] for i in range(1, len(sa))]
    dsb = [sb[i] - sb[i - 1] for i in range(1, len(sb))]
    # Rolling correlation of differences
    rho_meta = compute_rho_ab(dsa, dsb, W)
    if len(rho_meta) < 2:
        return 0.0
    return compute_d_rho(rho_meta)


def compute_reff(rho_star: float) -> float:
    """Effective rank of joint covariance. Eq 6. Layer 3.

    reff = 2 / (1 + rho*^2)  ∈ (1, 2]

    reff = 2  → isotropic, no coupling
    reff → 1  → rank-1 collapse, full coupling
    """
    return 2.0 / (1.0 + rho_star ** 2)


# ---------------------------------------------------------------------------
# Layer 5 — Self-diagnostic (temporal discrepancy)
# ---------------------------------------------------------------------------

def fisher_bound(rho_star: float, W: int) -> float:
    """Upper limit of d_rho under stationarity. Eq 16.

    Fisher bound = (1 - rho*^2)^2 / W

    Derived internally — no external calibration needed.
    This is the validity boundary in Eq 29.
    """
    return (1.0 - rho_star ** 2) ** 2 / W


def compute_delta(d_rho: float, rho_star: float, W: int) -> float:
    """Temporal discrepancy / self-diagnostic. Eq 22. Layer 5.

    δ(t) = d_rho(t) - (1 - rho*^2)^2 / W

    Semantic fixed point (Proposition 9.5 / Theorem 1.2 iii):
      δ ≤ 0  →  within stationary envelope → seed valid, compressible
      δ > 0  →  structural drift → seed invalid, juncture imminent

    δ is simultaneously:
      (a) the detection output (juncture of observed pair)
      (b) the validity certificate (A3.1 violated when > 0)
      (c) the internal monitoring signal of the reflexive instrument
    """
    return d_rho - fisher_bound(rho_star, W)


def compute_cusum(delta_series: List[float]) -> float:
    """CUSUM of delta values. Eq 26. Proposition 5.8.

    C(t) = max(0, C(t-1) + δ(t)),  C(t0) = 0

    Tracks positive excursions of accumulated δ.
    Resets to 0 when channel re-enters stationary envelope.
    Unlike standard CUSUM, the null hypothesis is the Fisher bound —
    derived internally, not from a competing parametric model.
    """
    C = 0.0
    for d in delta_series:
        C = max(0.0, C + d)
    return C


# ---------------------------------------------------------------------------
# Convenience: compute all observables in one pass
# ---------------------------------------------------------------------------

def compute_all(
    sa: List[float],
    sb: List[float],
    W: int,
    delta_history: Optional[List[float]] = None,
) -> dict:
    """Compute the full RRG observable set for a window.

    Returns a dict with all Layer 1-5 quantities.
    delta_history, if provided, is used for CUSUM accumulation.
    """
    rho_ab_full = compute_rho_ab(sa, sb, W)
    # Eq 7: d_rho(t,W) = Var(rho_ab[t-W+1:t+1]) — last W values only.
    # Without this window, variance grows as history accumulates and Gate 1
    # fires only at turn W (d_rho=0 by single-value variance), then never again.
    rho_ab = rho_ab_full[-W:] if len(rho_ab_full) >= W else rho_ab_full

    rho_star = compute_rho_star(rho_ab)
    d_rho = compute_d_rho(rho_ab)
    d_rho_meta = compute_d_rho_meta(sa, sb, W)
    reff = compute_reff(rho_star)
    delta = compute_delta(d_rho, rho_star, W)
    fb = fisher_bound(rho_star, W)

    history = (delta_history or []) + [delta]
    cusum = compute_cusum(history)

    # Lyapunov potential — Proposition 5.4. V* ~= 0 is the Sufficient attractor.
    lyapunov_v = d_rho + (1.0 - abs(rho_star))

    return {
        "rho_ab": rho_ab_full,
        "rho_ab_window": rho_ab,
        "rho_star": rho_star,
        "d_rho": d_rho,
        "d_rho_series": compute_d_rho_series(rho_ab),
        "d_rho_meta": d_rho_meta,
        "reff": reff,
        "delta": delta,
        "fisher_bound": fb,
        "cusum": cusum,
        "lyapunov_v": lyapunov_v,
    }


# ---------------------------------------------------------------------------
# Numpy-based classes and functions (merged from core.py)
# ---------------------------------------------------------------------------

AlgebraLevel = Literal['R', 'C', 'H']


@dataclass
class RRGParams:
    """Instrument parameters — domain-local (Appendix A of the paper)."""
    W            : int          = 40
    tau_lock_rho : float        = 0.80
    tau_lock_dr  : float        = 0.12
    tau_meta     : float        = 0.08
    tau_exit     : float        = 0.18
    algebra      : AlgebraLevel = 'R'


def _corr_R(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-10 or sb < 1e-10:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _corr_C(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.real(np.dot(np.conj(a), b)) / (na * nb))


def _corr_H(a: np.ndarray, b: np.ndarray) -> float:
    def _qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    A = np.mean(a, axis=0)
    B = np.mean(b, axis=0)
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    AB = _qmul(A, B)
    BA = _qmul(B, A)
    return float(np.linalg.norm(AB - BA) / (na * nb))


_CORR_OPS = {'R': _corr_R, 'C': _corr_C, 'H': _corr_H}


def windowed_corr(
    sa      : np.ndarray,
    sb      : np.ndarray,
    W       : int,
    algebra : AlgebraLevel = 'R',
) -> np.ndarray:
    """Rolling correlation ρab(t) with window W and algebra level."""
    corr_fn = _CORR_OPS.get(algebra)
    if corr_fn is None:
        raise ValueError(f"Unknown algebra '{algebra}'. Use 'R', 'C', or 'H'.")
    T      = sa.shape[0]
    rho_ab = np.full(T, np.nan)
    for i in range(W - 1, T):
        rho_ab[i] = corr_fn(sa[i - W + 1 : i + 1], sb[i - W + 1 : i + 1])
    return rho_ab


def rolling_mean(x: np.ndarray, W: int) -> np.ndarray:
    """Rolling mean of x with window W, ignoring NaN."""
    T   = len(x)
    out = np.full(T, np.nan)
    for i in range(W - 1, T):
        window = x[i - W + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) > 1:
            out[i] = np.mean(valid)
    return out


def rolling_var(x: np.ndarray, W: int) -> np.ndarray:
    """Rolling variance dρ(t) = Var(ρab, W) — primary RRG observable."""
    T   = len(x)
    out = np.full(T, np.nan)
    for i in range(W - 1, T):
        window = x[i - W + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) > 1:
            out[i] = np.var(valid)
    return out


def reff(rho_star: np.ndarray) -> np.ndarray:
    """Effective rank: r_eff = 2 / (1 + ρ*²) ∈ (1, 2]. Numpy array version."""
    return 2.0 / (1.0 + np.where(np.isnan(rho_star), np.nan, rho_star) ** 2)


# Regime label constants
LOCK       = "LOCK"
EMERGENCE  = "EMERGENCE"
EXIT       = "EXIT"
NEUTRAL    = "NEUTRAL"
BACKGROUND = "BACKGROUND"


def classify_regime(
    rho_star : np.ndarray,
    drho     : np.ndarray,
    params   : Optional[RRGParams] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep numpy regime classifier using the AND gate.

    Returns (regime labels array, lock_mask bool array).
    """
    if params is None:
        params = RRGParams()
    T      = len(rho_star)
    regime = np.full(T, BACKGROUND, dtype=object)
    valid      = ~np.isnan(rho_star) & ~np.isnan(drho)
    lock       = valid & (drho < params.tau_lock_dr) & (np.abs(rho_star) > params.tau_lock_rho)
    emergence  = valid & (drho < params.tau_meta) & ~lock
    exit_      = valid & (drho > params.tau_exit)
    neutral    = valid & ~lock & ~emergence & ~exit_
    regime[lock]      = LOCK
    regime[emergence] = EMERGENCE
    regime[exit_]     = EXIT
    regime[neutral]   = NEUTRAL
    return regime, lock


@dataclass
class RRGObservables:
    """Full observable bundle for a pair (sa, sb)."""
    sa        : np.ndarray
    sb        : np.ndarray
    rho_ab    : np.ndarray
    rho_star  : np.ndarray
    drho      : np.ndarray
    r_eff     : np.ndarray
    regime    : np.ndarray
    lock_mask : np.ndarray
    params    : RRGParams

    @classmethod
    def from_series(
        cls,
        sa     : np.ndarray,
        sb     : np.ndarray,
        params : Optional[RRGParams] = None,
    ) -> "RRGObservables":
        if params is None:
            params = RRGParams()
        W        = params.W
        rho_ab_  = windowed_corr(sa, sb, W, algebra=params.algebra)
        rho_star_= rolling_mean(rho_ab_, W)
        drho_    = rolling_var(rho_ab_, W)
        r_eff_   = reff(rho_star_)
        regime_, lock_mask_ = classify_regime(rho_star_, drho_, params)
        return cls(
            sa        = sa,
            sb        = sb,
            rho_ab    = rho_ab_,
            rho_star  = rho_star_,
            drho      = drho_,
            r_eff     = r_eff_,
            regime    = regime_,
            lock_mask = lock_mask_,
            params    = params,
        )

    def summary(self) -> dict:
        valid = ~np.isnan(self.drho) & ~np.isnan(self.rho_star)
        n     = valid.sum()
        return {
            "n_valid"       : int(n),
            "rho_star_mean" : float(np.nanmean(self.rho_star)),
            "drho_mean"     : float(np.nanmean(self.drho)),
            "lock_frac"     : float(self.lock_mask.sum() / n) if n > 0 else 0.0,
            "reff_mean"     : float(np.nanmean(self.r_eff)),
        }

    def as_dict(self) -> dict:
        return {
            "sa"       : self.sa,
            "sb"       : self.sb,
            "rho_ab"   : self.rho_ab,
            "rho_star" : self.rho_star,
            "drho"     : self.drho,
            "r_eff"    : self.r_eff,
            "regime"   : self.regime,
            "lock_mask": self.lock_mask,
        }


def sign_stability(sa: np.ndarray, sb: np.ndarray, W: int) -> np.ndarray:
    """Rolling fraction of timesteps where Δsa · Δsb > 0. Axiom A3.5."""
    T    = len(sa)
    frac = np.full(T, np.nan)
    da   = np.diff(sa)
    db   = np.diff(sb)
    for i in range(W, T):
        chunk_a = da[i - W : i]
        chunk_b = db[i - W : i]
        frac[i] = np.mean((chunk_a * chunk_b) > 0)
    return frac


def rank_collapse_oracle(
    obs    : "RRGObservables",
    strict : bool = True,
) -> np.ndarray:
    """Evaluates the Rank-Collapse Theorem biconditional at each timestep."""
    if strict:
        return obs.lock_mask
    return (obs.regime == LOCK) | (obs.regime == EMERGENCE)


def generate_regime(
    regime  : str,
    n_steps : int   = 600,
    W       : int   = 40,
    noise   : float = 0.15,
    seed    : int   = 0,
    params  : Optional[RRGParams] = None,
) -> "RRGObservables":
    """Generate a controlled synthetic trajectory for a given RRG regime."""
    if params is None:
        params = RRGParams(W=W)
    rng = np.random.default_rng(seed)
    n   = n_steps
    if regime == 'lock':
        latent = rng.standard_normal(n)
        sa = latent + noise * rng.standard_normal(n)
        sb = latent + noise * rng.standard_normal(n)
    elif regime == 'resonance':
        shared = rng.standard_normal(n)
        sa = 0.4 * shared + 0.9 * rng.standard_normal(n)
        sb = 0.4 * shared + 0.9 * rng.standard_normal(n)
    elif regime == 'oscillating':
        omega = 2 * np.pi / 50
        t_arr = np.arange(n)
        target = np.sin(omega * t_arr)
        sa = rng.standard_normal(n)
        sb = (
            target * sa
            + np.sqrt(np.maximum(1 - target**2, 0)) * rng.standard_normal(n)
        )
    else:
        raise ValueError(f"Unknown regime '{regime}'. Use 'lock', 'resonance', or 'oscillating'.")
    sa = sa / (np.std(sa) + 1e-12)
    sb = sb / (np.std(sb) + 1e-12)
    return RRGObservables.from_series(sa, sb, params=params)
