"""
Gatekeeper — pre-API context decision layer.

Insertion point: called BEFORE query_engine.py sends to the LLM API.

The gatekeeper asks one question each cycle:
  "Is this window geometrically stable enough to compress?"

If δ(t) ≤ 0 → compress to seed, inject seed into context.
If δ(t) > 0 → pass tokens through raw, invalidate stale seeds.

This replaces the naive "truncate when full" strategy with a
criterion-based strategy: compress *when it's safe*, not *when
you're desperate*. The compression cost is paid once, at the moment
of geometric certainty.

Integration with query_engine.py (minimal change):

    # Before:
    response = api_call(context=raw_tokens)

    # After:
    gk = Gatekeeper()  # shared instance, lives for the session
    context = gk.build_context(sa, sb, raw_tokens, intent, decisions)
    response = api_call(context=context)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .engine import compute_all
from .regime import (
    Regime,
    classify_regime,
    is_juncture,
    tau_suff_endo,
)
from .seed import RRGSeed, Tier
from .store import SeedStore


GATE2_THRESHOLD      = 0.05   # Phase D validated — perfect separation between COHERENT/BREAK groups
RESONANCE_LOCK_FACTOR = 0.7   # reduce effective tau_lock_dr by 30% when resonance-lock active


class Gatekeeper:
    """Pre-API context gate with RRG-based compression decisions.

    One instance per session. Maintains delta history for CUSUM
    and prior regime for juncture detection (Proposition 5.3).

    W=3 is the default for COT-01 / CG-01 (reasoning steps).
    For continuous-time domains use W ∈ [20, 80] (Appendix A).
    """

    def __init__(
        self,
        W: int = 3,
        tau_rho: float = 0.45,       # empirically consistent lower bound (App. A)
        store: Optional[SeedStore] = None,
        gate2_threshold: float = GATE2_THRESHOLD,
        resonance_lock_factor: float = RESONANCE_LOCK_FACTOR,
        tau_lock_dr: Optional[float] = None,  # override endogenous tau_suff with empirical value
    ):
        self.W                     = W
        self.tau_suff              = tau_suff_endo(W)
        self.tau_rho               = tau_rho
        self.tau_meta              = self.tau_suff   # same scale for pilot
        self.tau_rel               = self.tau_suff * 1.5
        self.gate2_threshold       = gate2_threshold
        self.resonance_lock_factor = resonance_lock_factor
        # Base tau for d_rho Gate 1 check: empirical if provided, else endogenous
        self._base_tau_suff        = tau_lock_dr if tau_lock_dr is not None else self.tau_suff

        self.store = store or SeedStore()

        # State across cycles
        self._delta_history:        List[float] = []
        self._prior_regime:         Regime = Regime.BACKGROUND
        self._prior_was_sufficient: bool   = False
        self._resonance_lock_active: bool  = False

    # ------------------------------------------------------------------
    # Core evaluation (Algorithm 1, Step 2)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sa: List[float],
        sb: List[float],
    ) -> Tuple[bool, Dict]:
        """Evaluate whether the current window is compressible.

        Args:
            sa: observer signal a (e.g. cosine_sim(query, response[t]))
            sb: observer signal b (e.g. cosine_sim(response[t], response[t-1]))

        Returns:
            (should_compress, metrics)
            should_compress: True iff delta <= 0 AND regime is Sufficient
            metrics: full RRG observable dict for logging / debugging
        """
        if len(sa) < self.W or len(sb) < self.W:
            return False, {"reason": "window_too_small", "W": self.W}

        obs = compute_all(sa, sb, self.W, self._delta_history)
        self._delta_history.append(obs["delta"])

        # Effective tau: lowered when resonance-lock has fired (Gate 2 action)
        effective_tau = (
            self._base_tau_suff * self.resonance_lock_factor
            if self._resonance_lock_active
            else self._base_tau_suff
        )

        regime = classify_regime(
            d_rho      = obs["d_rho"],
            rho_star   = obs["rho_star"],
            d_rho_meta = obs["d_rho_meta"],
            tau_suff   = effective_tau,
            tau_rho    = self.tau_rho,
            tau_meta   = self.tau_meta,
            tau_rel    = self.tau_rel,
            prior_regime = self._prior_regime,
        )

        juncture = is_juncture(
            delta             = obs["delta"],
            d_rho_meta        = obs["d_rho_meta"],
            tau_meta          = self.tau_meta,
            prior_was_sufficient = self._prior_was_sufficient,
        )

        # Update state
        self._prior_regime         = regime
        self._prior_was_sufficient = (regime == Regime.SUFFICIENT)

        # On juncture: invalidate all active seeds and clear resonance-lock
        invalidated: List[str] = []
        if juncture:
            invalidated = self.store.invalidate_all_active()
            self._resonance_lock_active = False

        # Sweep any seeds whose delta would now be positive
        swept = self.store.sweep_delta_positive(obs["delta"])
        invalidated.extend(swept)

        # Gate 1: compress only in confirmed Sufficient state (delta <= 0)
        should_compress = (regime == Regime.SUFFICIENT) and (obs["delta"] <= 0)

        # Gate 2: resonance-lock detection on per-checkpoint d_rho series (Phase D)
        d_rho_series = obs.get("d_rho_series", [])
        gate2_max_d_rho = max(d_rho_series) if d_rho_series else 0.0
        resonance_lock = should_compress and (gate2_max_d_rho >= self.gate2_threshold)
        if resonance_lock:
            next_tau = self._base_tau_suff * self.resonance_lock_factor
            self._resonance_lock_active = True
            print(
                f"[Gate2] resonance-lock: max(d_rho_series)={gate2_max_d_rho:.4f}"
                f" >= {self.gate2_threshold}"
                f" — tau_lock_dr next={next_tau:.6f} (factor={self.resonance_lock_factor})"
            )

        metrics = {
            **obs,
            "regime":               regime.value,
            "juncture":             juncture,
            "tau_suff":             effective_tau,
            "tau_rho":              self.tau_rho,
            "W":                    self.W,
            "invalidated_seeds":    invalidated,
            "gate2_max_d_rho":      gate2_max_d_rho,
            "resonance_lock":       resonance_lock,
            "resonance_lock_active": self._resonance_lock_active,
            "tau_lock_dr_effective": effective_tau,
        }

        return should_compress, metrics

    # ------------------------------------------------------------------
    # Compression (Algorithm 2 — seed creation)
    # ------------------------------------------------------------------

    def compress(
        self,
        sa: List[float],
        sb: List[float],
        tokens: List[str],
        intent: str,
        decisions: List[str],
        code_state: Any = None,
        token_range: Tuple[int, int] = (0, 0),
    ) -> Optional[RRGSeed]:
        """Create and store a seed if the window is compressible.

        The compression cost (generating intent + decisions) is paid
        exactly once, at the moment delta <= 0 is confirmed.

        Returns the seed if created, None if not compressible.
        """
        should_compress, metrics = self.evaluate(sa, sb)
        if not should_compress:
            return None

        seed = RRGSeed(
            rho_star     = metrics["rho_star"],
            d_rho        = metrics["d_rho"],
            d_rho_meta   = metrics["d_rho_meta"],
            reff         = metrics["reff"],
            delta        = metrics["delta"],
            cusum        = metrics["cusum"],
            regime       = Regime(metrics["regime"]),
            W            = self.W,
            intent       = intent,
            decisions    = decisions,
            code_state   = code_state,
            token_range  = token_range,
            content_hash = RRGSeed.hash_tokens(tokens),
            tier         = Tier.HOT,
        )

        self.store.put(seed)
        return seed

    # ------------------------------------------------------------------
    # Context assembly (final step before API call)
    # ------------------------------------------------------------------

    def build_context(
        self,
        sa: List[float],
        sb: List[float],
        raw_tokens: List[str],
        intent: str = "",
        decisions: Optional[List[str]] = None,
        code_state: Any = None,
        token_range: Tuple[int, int] = (0, 0),
    ) -> str:
        """Build the final context string for the LLM API call.

        1. Evaluate the current window.
        2. If compressible, create a seed from intent + decisions.
        3. Assemble: [valid seed fragments] + [raw pass-through tokens].

        This is the single integration point with query_engine.py.
        """
        if intent:
            self.compress(
                sa, sb, raw_tokens, intent,
                decisions or [], code_state, token_range
            )
        else:
            # Still evaluate for juncture detection even without compression
            self.evaluate(sa, sb)

        seed_context = self.store.context_fragments()
        raw          = " ".join(raw_tokens)

        if seed_context:
            return f"{seed_context}\n\n---\n\n{raw}"
        return raw

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict:
        """Current state snapshot for logging and debugging."""
        effective_tau = (
            self._base_tau_suff * self.resonance_lock_factor
            if self._resonance_lock_active
            else self._base_tau_suff
        )
        return {
            "W":                      self.W,
            "tau_suff":               self.tau_suff,
            "tau_rho":                self.tau_rho,
            "prior_regime":           self._prior_regime.value,
            "prior_was_sufficient":   self._prior_was_sufficient,
            "delta_history_len":      len(self._delta_history),
            "last_delta":             self._delta_history[-1] if self._delta_history else None,
            "resonance_lock_active":  self._resonance_lock_active,
            "tau_lock_dr_effective":  effective_tau,
            "store":                  self.store.summary(),
        }

    def reset(self) -> None:
        """Reset cycle state (keep store). Use between independent episodes."""
        self._delta_history         = []
        self._prior_regime          = Regime.BACKGROUND
        self._prior_was_sufficient  = False
        self._resonance_lock_active = False
