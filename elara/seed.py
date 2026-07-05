"""
RRGSeed — autodescriptive, self-certifying seed.

A seed is valid if and only if delta <= 0 (Theorem 1.2 iii).
No external state is required to determine validity — the seed
carries its own validity certificate in the delta field.

Semantic fixed point (Proposition 9.5):
  delta ∈ closure(P)  AND  delta = ValidityCertificate(closure(P))

The seed never reconstructs original tokens. It reconstructs
*intent and state* — what the LLM needs to continue coherently.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from .regime import Regime


class Tier(Enum):
    HOT  = "Hot"   # Active, recently created
    WARM = "Warm"  # Aging, still valid
    COLD = "Cold"  # Invalidated or expired


@dataclass
class RRGSeed:
    """Self-certifying compressed context block.

    Geometry fields (decide validity automatically):
      delta <= 0  →  seed is valid, safe to inject into LLM context
      delta > 0   →  seed is stale/invalid, must not be injected

    Recreation fields (generated once at compression time, never recalculated):
      These are the LLM-readable summary of what the block contained.
      The compression cost is paid exactly once, at the moment of
      geometric certainty (delta <= 0 confirmed).
    """

    # ------------------------------------------------------------------
    # Geometry — RRG observables (self-certifying)
    # ------------------------------------------------------------------
    rho_star:    float   # E[rho_ab] — mean coupling
    d_rho:       float   # Var(rho_ab) — first-order stability
    d_rho_meta:  float   # Var(Corr(Δsa, Δsb)) — early warning
    reff:        float   # Effective rank = 2/(1+rho*^2)
    delta:       float   # Self-diagnostic: < 0 = valid, > 0 = invalid
    cusum:       float   # CUSUM accumulator C(t)
    regime:      Regime  # Sufficient | Converging | Released | Background
    W:           int     # Coherence window used

    # ------------------------------------------------------------------
    # Recreation content — generated once at compression time
    # ------------------------------------------------------------------
    intent:      str            # What objective was being pursued
    decisions:   List[str]      # Key decisions made in this block
    code_state:  Optional[Any]  # AST snapshot or None
    token_range: tuple          # (start_idx, end_idx) of covered tokens
    content_hash: str           # SHA-256[:16] of original tokens

    # ------------------------------------------------------------------
    # Lifecycle metadata
    # ------------------------------------------------------------------
    seed_id:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:     float = field(default_factory=time.time)
    invalidated_at: Optional[float] = None
    version:        int   = 0    # Increments on re-generation
    tier:           Tier  = Tier.HOT

    # ------------------------------------------------------------------
    # Self-certification (Theorem 1.2 iii)
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """Self-certifying validity check.

        No external state required. A seed is valid iff:
          1. delta <= 0 (within stationary envelope, Eq 22)
          2. Not explicitly invalidated (juncture occurred)

        Lcert < 1 for all W >= 2 (Proposition 1.3): estimation error
        in rho* contracts through the certificate — self-certification
        is most reliable near |rho*| ≈ 1, exactly where it's needed.
        """
        return self.delta <= 0.0 and self.invalidated_at is None

    def invalidate(self) -> None:
        """Mark as invalid. Called when delta > 0 (juncture detected).

        After invalidation: is_valid = False, tier moves to Cold.
        The seed is retained in the store (Cold tier) for historical
        reference but is not injected into LLM context.
        """
        self.invalidated_at = time.time()
        self.tier = Tier.COLD

    def regenerate(self, new_seed: "RRGSeed") -> "RRGSeed":
        """Produce a new seed for the same block, incrementing version.

        Called after a juncture when the block has re-stabilized.
        The old seed stays in Cold tier; this is its successor.
        """
        new_seed.version = self.version + 1
        return new_seed

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def to_context_fragment(self) -> str:
        """Render seed as compact LLM-context fragment.

        Returns empty string if invalid — gatekeeper checks this
        before injecting into the context window.
        """
        if not self.is_valid:
            return ""

        lines = [
            f"[SEED v{self.version} | {self.regime.value} | "
            f"reff={self.reff:.3f} | delta={self.delta:.4f}]",
            f"intent: {self.intent}",
        ]
        if self.decisions:
            lines.append("decisions: " + "; ".join(self.decisions))
        if self.code_state is not None:
            lines.append(f"code_state: {self.code_state}")
        lines.append(
            f"[tokens {self.token_range[0]}–{self.token_range[1]} | "
            f"hash={self.content_hash}]"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "seed_id":        self.seed_id,
            "rho_star":       self.rho_star,
            "d_rho":          self.d_rho,
            "d_rho_meta":     self.d_rho_meta,
            "reff":           self.reff,
            "delta":          self.delta,
            "cusum":          self.cusum,
            "regime":         self.regime.value,
            "W":              self.W,
            "intent":         self.intent,
            "decisions":      self.decisions,
            "code_state":     self.code_state,
            "token_range":    list(self.token_range),
            "content_hash":   self.content_hash,
            "created_at":     self.created_at,
            "invalidated_at": self.invalidated_at,
            "version":        self.version,
            "tier":           self.tier.value,
        }

    @staticmethod
    def from_dict(d: dict) -> "RRGSeed":
        return RRGSeed(
            seed_id        = d["seed_id"],
            rho_star       = d["rho_star"],
            d_rho          = d["d_rho"],
            d_rho_meta     = d["d_rho_meta"],
            reff           = d["reff"],
            delta          = d["delta"],
            cusum          = d["cusum"],
            regime         = Regime(d["regime"]),
            W              = d["W"],
            intent         = d["intent"],
            decisions      = d["decisions"],
            code_state     = d["code_state"],
            token_range    = tuple(d["token_range"]),
            content_hash   = d["content_hash"],
            created_at     = d["created_at"],
            invalidated_at = d["invalidated_at"],
            version        = d["version"],
            tier           = Tier(d["tier"]),
        )

    @staticmethod
    def hash_tokens(tokens: List[str]) -> str:
        """SHA-256 truncated to 16 hex chars for content integrity."""
        content = json.dumps(tokens, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
