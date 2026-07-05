"""
SeedStore — three-tier memory with geometric invalidation.

Unlike naive compaction (truncate when full), the store invalidates
by criterion (delta > 0), not by capacity. A seed is evicted when
its own geometry certifies it's stale — not when the buffer is full.

Tiers:
  Hot  — recently created, actively injected into context
  Warm — aging, still valid, injected but deprioritized
  Cold — invalidated or expired, retained for audit, never injected
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .seed import RRGSeed, Tier


class SeedStore:
    """Three-tier seed store with automatic geometric invalidation.

    Invalidation policy:
      - Explicit: store.invalidate(seed_id) — called by gatekeeper on juncture
      - Geometric: any seed with delta > 0 is auto-invalidated on access
      - Temporal: Hot seeds age to Warm; Warm seeds age to Cold (then evicted)

    Seeds in Cold tier are never injected into context but are kept
    in memory for version history and audit.
    """

    def __init__(
        self,
        hot_ttl_s: float  = 3_600,    # 1 hour
        warm_ttl_s: float = 86_400,   # 24 hours
        max_cold:   int   = 50,       # max cold seeds retained
    ):
        self._seeds: Dict[str, RRGSeed] = {}
        self._hot_ttl  = hot_ttl_s
        self._warm_ttl = warm_ttl_s
        self._max_cold = max_cold

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, seed: RRGSeed) -> str:
        """Store a seed. Returns seed_id."""
        self._seeds[seed.seed_id] = seed
        return seed.seed_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, seed_id: str) -> Optional[RRGSeed]:
        """Retrieve a seed (any tier). Returns None if not found."""
        return self._seeds.get(seed_id)

    def get_valid(self, seed_id: str) -> Optional[RRGSeed]:
        """Retrieve a seed only if currently valid (delta <= 0)."""
        seed = self._seeds.get(seed_id)
        if seed is None:
            return None
        self._age(seed)
        return seed if seed.is_valid else None

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, seed_id: str) -> bool:
        """Explicitly invalidate a seed. Returns True if found."""
        seed = self._seeds.get(seed_id)
        if seed and seed.is_valid:
            seed.invalidate()
            return True
        return False

    def invalidate_all_active(self) -> List[str]:
        """Invalidate all Hot and Warm seeds.

        Called by gatekeeper when a juncture is detected and the
        current block has changed direction irreversibly.
        Returns list of invalidated seed_ids.
        """
        invalidated = []
        for sid, seed in self._seeds.items():
            if seed.tier in (Tier.HOT, Tier.WARM) and seed.is_valid:
                seed.invalidate()
                invalidated.append(sid)
        return invalidated

    def sweep_delta_positive(self, current_delta: float) -> List[str]:
        """Invalidate seeds whose geometry is now stale (delta > 0).

        Called every cycle: if the channel has drifted (current_delta > 0),
        any seed that was based on a stationary assumption is now suspect.
        Returns invalidated seed_ids.
        """
        if current_delta <= 0:
            return []
        return self.invalidate_all_active()

    # ------------------------------------------------------------------
    # Active seeds (for context injection)
    # ------------------------------------------------------------------

    def active_seeds(self) -> List[RRGSeed]:
        """Return all valid seeds ordered: Hot first, then Warm.

        Performs tier aging before returning.
        """
        now = time.time()
        valid: List[RRGSeed] = []
        for seed in self._seeds.values():
            self._age(seed, now=now)
            if seed.is_valid and seed.tier != Tier.COLD:
                valid.append(seed)

        # Hot seeds first, then Warm; within each tier, newest first
        valid.sort(key=lambda s: (0 if s.tier == Tier.HOT else 1, -s.created_at))
        return valid

    def context_fragments(self) -> str:
        """Assemble all valid seed fragments for LLM context injection.

        Returns empty string if no valid seeds.
        """
        fragments = [s.to_context_fragment() for s in self.active_seeds()]
        non_empty = [f for f in fragments if f]
        return "\n\n".join(non_empty)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """Garbage-collect Cold seeds beyond max_cold. Returns count removed."""
        cold = [
            (sid, s) for sid, s in self._seeds.items()
            if s.tier == Tier.COLD
        ]
        cold.sort(key=lambda x: x[1].invalidated_at or 0)
        to_remove = cold[: max(0, len(cold) - self._max_cold)]
        for sid, _ in to_remove:
            del self._seeds[sid]
        return len(to_remove)

    def summary(self) -> dict:
        """Quick stats: count by tier and validity."""
        hot = warm = cold = invalid = 0
        for s in self._seeds.values():
            if s.tier == Tier.HOT:
                hot += 1
            elif s.tier == Tier.WARM:
                warm += 1
            else:
                cold += 1
            if not s.is_valid:
                invalid += 1
        return {
            "total":   len(self._seeds),
            "hot":     hot,
            "warm":    warm,
            "cold":    cold,
            "invalid": invalid,
            "active":  len(self.active_seeds()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _age(self, seed: RRGSeed, now: Optional[float] = None) -> None:
        """Age a seed through tiers based on elapsed time."""
        if not seed.is_valid:
            return
        now = now or time.time()
        age = now - seed.created_at
        if seed.tier == Tier.HOT and age > self._hot_ttl:
            seed.tier = Tier.WARM
        elif seed.tier == Tier.WARM and age > self._warm_ttl:
            seed.tier = Tier.COLD
            seed.invalidate()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the entire store to a JSON file."""
        payload = {
            "hot_ttl_s":  self._hot_ttl,
            "warm_ttl_s": self._warm_ttl,
            "max_cold":   self._max_cold,
            "seeds":      [s.to_dict() for s in self._seeds.values()],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SeedStore":
        """Deserialize a store from a JSON file produced by save()."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(
            hot_ttl_s  = payload["hot_ttl_s"],
            warm_ttl_s = payload["warm_ttl_s"],
            max_cold   = payload["max_cold"],
        )
        for d in payload["seeds"]:
            store._seeds[d["seed_id"]] = RRGSeed.from_dict(d)
        return store

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[RRGSeed]:
        return iter(self._seeds.values())

    def __len__(self) -> int:
        return len(self._seeds)
