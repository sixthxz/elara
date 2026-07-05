"""
RRG Regime Conditions — Section 5.

Three regimes as transitions in the geometry of Mab:
  Converging  — manifold approaching rank-1
  Sufficient  — manifold IS rank-1 (AND gate satisfied)
  Released    — manifold decompressing (juncture fired)

All thresholds are domain-local. tau_suff has a data-free endogenous
prior from Proposition 4.11 that requires no external calibration.
"""

import math
from enum import Enum
from typing import Optional


class Regime(Enum):
    BACKGROUND = "Background"   # No coupling structure detected
    CONVERGING = "Converging"   # Approaching rank-1, not yet Sufficient
    SUFFICIENT = "Sufficient"   # AND gate satisfied — compressible
    RELEASED   = "Released"     # Post-juncture, decompressing


# ---------------------------------------------------------------------------
# Endogenous threshold — Proposition 4.11 / Remark 4.12
# ---------------------------------------------------------------------------

def tau_suff_endo(W: int) -> float:
    """Endogenous sufficiency threshold. Proposition 4.11 / Remark 4.12.

    Derived from the null distribution of d_rho under independence,
    with no external calibration data.

    Derivation (Proposition 4.11 proof):
      Var(d_rho) ≈ 2/(6W^3)           [sum of (1-l/W)^5 ≈ W/6]
      sd(d_rho)  = sqrt(2/(W^2*neff)) with neff = W/6
      tau = 1/W + 1.645 * sqrt(2/(W^2 * W/6))
           = 1/W + 1.645 * sqrt(12/W^3)

    BUG NOTE — Eq 20 vs derivation:
      Eq 20 writes (1/W)(1+z*sqrt(3/W)) using neff≈2W/3 (from sum of
      (1-x)^3). The actual derivation uses sum of (1-x)^5 → neff=W/6.
      neff=W/6 reproduces the paper's stated values (Remark 4.12):
        W=20 → 0.114 ✓   (Eq 20 gives 0.082 ✗)
        W=60 → 0.029      (paper states 0.053; figure shows ~0.05-0.09)
    The neff=W/6 form is used here as it is self-consistent with the
    Proposition 4.11 derivation and matches the W=20 anchor value.
    Domain calibration (Appendix A) supersedes when null data is available.
    """
    neff = W / 6.0
    sd = math.sqrt(2.0 / (W ** 2 * neff))
    return 1.0 / W + 1.645 * sd


# ---------------------------------------------------------------------------
# AND gate — regime classification
# ---------------------------------------------------------------------------

def classify_regime(
    d_rho: float,
    rho_star: float,
    d_rho_meta: float,
    tau_suff: float,
    tau_rho: float = 0.45,
    tau_meta: Optional[float] = None,
    tau_rel: Optional[float] = None,
    prior_regime: Optional[Regime] = None,
) -> Regime:
    """AND gate regime classifier. Section 5.

    Priority order:
      1. Sufficient  — both AND gate conditions hold
      2. Converging  — meta-signal low but d_rho still above tau_suff
      3. Released    — departed from Sufficient (prior_regime matters)
      4. Background  — no structure

    tau_rho = 0.45 is the empirically consistent lower bound confirmed
    across TR-01, FIN-01, EEG-01 (Appendix A, Step 3).
    """
    if tau_meta is None:
        tau_meta = tau_suff
    if tau_rel is None:
        tau_rel = tau_suff * 1.5

    # Sufficient: AND gate — Section 5
    # d_rho < tau_suff  AND  |rho*| > tau_rho
    if d_rho < tau_suff and abs(rho_star) > tau_rho:
        return Regime.SUFFICIENT

    # Converging: meta-signal decelerating, not yet Sufficient
    # d_rho_meta < tau_meta  AND  d_rho > tau_suff
    if d_rho_meta < tau_meta and d_rho > tau_suff:
        return Regime.CONVERGING

    # Released: departed from Sufficient
    if prior_regime == Regime.SUFFICIENT and d_rho > tau_rel:
        return Regime.RELEASED

    return Regime.BACKGROUND


# ---------------------------------------------------------------------------
# Juncture detection — Definition 5.2
# ---------------------------------------------------------------------------

def is_juncture(
    delta: float,
    d_rho_meta: float,
    tau_meta: float,
    prior_was_sufficient: bool,
) -> bool:
    """Detect a juncture event. Definition 5.2.

    A juncture requires ALL THREE conditions (Proposition 5.3):
      (i)  Prior Sufficient state         — system was stable before t*
      (ii) delta(t*) > 0                  — sign change, structural drift
      (iii) d_rho_meta(t*) > tau_meta     — structural, not transient

    Juncture is irreversible within the same dynamical episode
    (Proposition 5.3): the system cannot return to Sufficient without
    an external coupling event that re-establishes rank-collapse.

    The sequence is:
      d_rho_meta ↑  →  delta → 0  →  delta > 0 ∧ d_rho_meta > tau_meta
      →  Juncture t*  →  Released
    """
    if not prior_was_sufficient:
        return False
    return delta > 0 and d_rho_meta > tau_meta


# ---------------------------------------------------------------------------
# Lyapunov potential — Proposition 5.4
# ---------------------------------------------------------------------------

def lyapunov_potential(d_rho: float, rho_star: float) -> float:
    """Stochastic Lyapunov potential. Proposition 5.4.

    V(t) = d_rho(t) + (1 - |rho*|)

    V* ≈ 0 is the Sufficient attractor.
    At juncture: E[V(t*+1) | F_t*] >= V(t*) + delta(t*)/2 > V(t*)
    Positive drift away from Sufficient confirms stochastic irreversibility.
    """
    return d_rho + (1.0 - abs(rho_star))


# ---------------------------------------------------------------------------
# Granger asymmetry — Definition 8.2 (Resonance-Sufficient detection)
# ---------------------------------------------------------------------------

def granger_asymmetry(f_ab: float, f_ba: float, eps: float = 1e-8) -> float:
    """Granger asymmetry ratio G(t). Eq 27.

    G(t) = |F_{a→b} - F_{b→a}| / (F_{a→b} + F_{b→a} + eps)

    Genuine Sufficient:     G(t) > gamma_G (directional coupling)
    Resonance-Sufficient:   G(t) <= gamma_G (symmetric ambient forcing)

    Used to distinguish genuine relational coupling from ambient
    structure that imposes symmetric forcing on both observers.
    Only non-trivial at algebra level A1=H and above; at A0=R
    the commutator is trivially zero (Remark 8.1).
    """
    return abs(f_ab - f_ba) / (f_ab + f_ba + eps)
