# -*- coding: utf-8 -*-
"""
rrg/adapters.py
===============
RRG Adapter Layer — domain → standard format

Each adapter translates domain-specific data into the RRG standard bundle
(sa, sb → RRGObservables). The pipeline (core.py) is always identical.
Only the adapter changes per domain.

Standard contract
-----------------
Every adapter returns an RRGObservables (or dict of them, keyed by label).
Every adapter documents:
  - what sa and sb represent in that domain
  - the domain-local W and thresholds (RRGParams)
  - any augmentation required (e.g. quaternionic projection, pre-computed observables)

Domains implemented
-------------------
  from_timeseries     — generic scalar pair (EL, TR, FIN, EEG, SHM, GW, NT)
  from_cot            — Chain-of-Thought / LLM (pre-computed ρ*, dρ per checkpoint)
  from_conversation   — Conversation Geometry (turn-level scalars)

Usage
-----
    from elara.adapters import from_timeseries, from_cot
    from elara.engine import RRGParams

    # Generic domain
    obs = from_timeseries(sa, sb, params=RRGParams(W=60))

    # CoT / LLM domain
    bundles = from_cot(primary_results_dict)
    # bundles = {'FAITHFUL': RRGObservables, 'UNFAITHFUL': ..., 'WRONG': ...}
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from elara.engine import RRGObservables, RRGParams, rolling_mean, rolling_var, reff, classify_regime


# ---------------------------------------------------------------------------
# Adapter 1 — Generic scalar time series
# Domains: EL-01, TR-01, FIN-01, EEG-01, SHM-01, GW-01, NT-01
# ---------------------------------------------------------------------------

def from_timeseries(
    sa     : np.ndarray,
    sb     : np.ndarray,
    params : Optional[RRGParams] = None,
    label  : Optional[str] = None,
) -> RRGObservables:
    """
    Generic adapter for any domain that provides two scalar time series.

    sa and sb are the relational observers — what they represent is
    domain-specific (see index below), but the pipeline is identical.

    Domain index (from rrg_cross_domain_index.json):
    ─────────────────────────────────────────────────
    EL-01  Electricity Load
           sa = scalar load sensor A  |  sb = scalar load sensor B
           W  = 60 (1-hour windows)

    TR-01  Traffic (METR-LA)
           sa = speed reading node A  |  sb = speed reading node B
           W  = 12 (5-min sampling, 1-hour window)

    FIN-01 Financial Returns
           sa = asset return series A |  sb = asset return series B
           W  = 20 trading days

    EEG-01 Motor-Cortex EEG
           sa = C3 channel (scalar)   |  sb = C4 channel (scalar)
           W  = 160 (1 second at 160 Hz)

    SHM-01 Structural Health Monitoring
           sa = strain gauge A        |  sb = strain gauge B
           W  = 60 seconds (rosette) or W=24/6 (KW51 bridge, multi-scale)

    GW-01  Gravitational Waves (LIGO)
           sa = H1 whitened strain    |  sb = L1 whitened strain
           W  = multi-scale [5, 20, 480] seconds — call once per scale

    NT-01  Number Theory (Lagrange scalar)
           sa = depth-rate series A   |  sb = depth-rate series B
           W  = 60 sequence steps

    Parameters
    ----------
    sa, sb  : np.ndarray, shape (T,) — raw scalar observers
    params  : RRGParams with domain-local W and thresholds
    label   : optional string tag (e.g. 'EEG-subject-03')

    Returns
    -------
    RRGObservables
    """
    if params is None:
        params = RRGParams()

    obs = RRGObservables.from_series(sa, sb, params=params)

    if label is not None:
        obs.__dict__['label'] = label  # lightweight tag, not a dataclass field

    return obs


def from_timeseries_multiscale(
    sa       : np.ndarray,
    sb       : np.ndarray,
    windows  : List[int],
    params   : Optional[RRGParams] = None,
) -> Dict[int, RRGObservables]:
    """
    Multi-scale adapter — computes RRGObservables at each window scale.
    Used for GW-01 (LIGO: W = [5, 20, 480]) and SHM-01-B (KW51: W = [24, 6]).

    Parameters
    ----------
    sa, sb   : np.ndarray, shape (T,)
    windows  : list of int — window lengths to compute
    params   : RRGParams — thresholds applied at all scales (W overridden per scale)

    Returns
    -------
    dict mapping W -> RRGObservables
    """
    if params is None:
        params = RRGParams()

    result = {}
    for W in windows:
        p = RRGParams(
            W            = W,
            tau_lock_rho = params.tau_lock_rho,
            tau_lock_dr  = params.tau_lock_dr,
            tau_meta     = params.tau_meta,
            tau_exit     = params.tau_exit,
        )
        result[W] = RRGObservables.from_series(sa, sb, params=p)

    return result


# ---------------------------------------------------------------------------
# Adapter 2 — Chain-of-Thought / LLM
# Domain: COT-01
# ---------------------------------------------------------------------------

# CoT domain-local parameters (from rrg_cross_domain_index.json)
COT_PARAMS = RRGParams(
    W            = 3,      # W_SEMANTIC = 3 reasoning steps
    tau_lock_rho = 0.35,   # TAU_RHO
    tau_lock_dr  = 0.02,   # TAU_LOCK
    tau_meta     = 0.015,  # TAU_META
    tau_exit     = 0.04,   # TAU_EXIT
)

# CoT observables:
#   sa = Layer 4 residual norm at each reasoning step checkpoint
#   sb = Layer 11 residual norm at each reasoning step checkpoint
#
# The LLM domain is structurally different: instead of long time series,
# each sample is a SHORT trajectory (5 checkpoints for FAITHFUL/WRONG,
# 3 for UNFAITHFUL). ρ* and dρ are pre-computed in the JSON output.
# This adapter reconstructs the standard bundle from pre-computed observables.


def from_cot(
    primary_results : Dict[str, Any],
    params          : Optional[RRGParams] = None,
) -> Dict[str, "CoTBundle"]:
    """
    Adapter for Chain-of-Thought / LLM domain (COT-01).

    The CoT domain provides pre-computed ρ* and dρ per reasoning checkpoint,
    not raw sa/sb series. This adapter reconstructs RRG-compatible arrays
    from the JSON output of the CoT pipeline.

    Parameters
    ----------
    primary_results : dict
        Loaded from primary_results.json. Expected keys:
          'categories': dict mapping label -> list of sample dicts
          Each sample dict has:
            'rho_star'   : list of float|None, length = n_checkpoints
            'd_rho'      : list of float|None, length = n_checkpoints
            'states'     : list of str (regime labels)
            'checkpoints': list of int (token positions)
            'label'      : str ('FAITHFUL' | 'UNFAITHFUL' | 'WRONG')
            'lock_frac'  : float

    params : RRGParams — defaults to COT_PARAMS if None

    Returns
    -------
    dict mapping label -> CoTBundle
    Each CoTBundle has:
        .rho_star_matrix : np.ndarray shape (n_samples, n_checkpoints)
        .drho_matrix     : np.ndarray shape (n_samples, n_checkpoints)
        .lock_frac       : np.ndarray shape (n_samples,)
        .checkpoints     : list of int
        .label           : str
        .summary()       : dict of aggregate statistics
    """
    if params is None:
        params = COT_PARAMS

    categories = primary_results.get('categories', {})
    result     = {}

    for label, samples in categories.items():
        result[label] = CoTBundle.from_samples(samples, label, params)

    return result


class CoTBundle:
    """
    RRG observable bundle for the CoT / LLM domain.

    Unlike RRGObservables (which wraps a time series), CoTBundle wraps
    a population of short trajectories (one per reasoning sample).
    """

    def __init__(
        self,
        rho_star_matrix : np.ndarray,
        drho_matrix     : np.ndarray,
        lock_frac       : np.ndarray,
        checkpoints     : List[int],
        label           : str,
        params          : RRGParams,
    ):
        self.rho_star_matrix = rho_star_matrix  # (n_samples, n_checkpoints)
        self.drho_matrix     = drho_matrix       # (n_samples, n_checkpoints)
        self.lock_frac       = lock_frac         # (n_samples,)
        self.checkpoints     = checkpoints
        self.label           = label
        self.params          = params

    @classmethod
    def from_samples(
        cls,
        samples     : List[Dict],
        label       : str,
        params      : RRGParams,
    ) -> "CoTBundle":
        """Build a CoTBundle from the list of sample dicts in the JSON."""

        def _clean(arr):
            """Replace None with NaN, cast to float."""
            return np.array(
                [np.nan if v is None else float(v) for v in arr],
                dtype=float
            )

        # Pad all samples to the same length (max checkpoints across samples)
        all_rs   = [_clean(s['rho_star']) for s in samples]
        all_dr   = [_clean(s['d_rho'])    for s in samples]
        max_len  = max(len(x) for x in all_rs)

        def _pad(arr, length):
            out = np.full(length, np.nan)
            out[:len(arr)] = arr
            return out

        rs_matrix  = np.stack([_pad(x, max_len) for x in all_rs])
        dr_matrix  = np.stack([_pad(x, max_len) for x in all_dr])
        lock_frac  = np.array([s['lock_frac'] for s in samples], dtype=float)
        checkpoints = samples[0]['checkpoints']

        return cls(
            rho_star_matrix = rs_matrix,
            drho_matrix     = dr_matrix,
            lock_frac       = lock_frac,
            checkpoints     = checkpoints,
            label           = label,
            params          = params,
        )

    def summary(self) -> Dict[str, float]:
        """Aggregate statistics across all samples in this category."""
        return {
            'label'          : self.label,
            'n_samples'      : len(self.lock_frac),
            'lock_frac_mean' : float(np.nanmean(self.lock_frac)),
            'lock_frac_std'  : float(np.nanstd(self.lock_frac)),
            'rho_star_final' : float(np.nanmean(self.rho_star_matrix[:, -1])),
            'drho_final'     : float(np.nanmean(self.drho_matrix[:, -1])),
        }

    def phase_points(self) -> np.ndarray:
        """
        All valid (ρ*, dρ) points across all samples and checkpoints.
        Shape: (N, 2) — ready for alpha-shape or scatter plot.
        """
        rs   = self.rho_star_matrix.ravel()
        dr   = self.drho_matrix.ravel()
        valid = ~np.isnan(rs) & ~np.isnan(dr)
        return np.column_stack([rs[valid], dr[valid]])


# ---------------------------------------------------------------------------
# Adapter 3 — Conversation Geometry
# Domain: CG-01 (incomplete — awaiting domain paper)
# ---------------------------------------------------------------------------

def from_conversation(
    turn_scalars_ai    : np.ndarray,
    turn_scalars_human : np.ndarray,
    params             : Optional[RRGParams] = None,
) -> RRGObservables:
    """
    Adapter for Conversation Geometry (CG-01).

    sa = turn-level scalar per AI->Human exchange
    sb = turn-level scalar per Human->AI exchange

    The specific scalar feature (norm / entropy / embedding projection)
    is domain-specific and must be computed before calling this adapter.
    The pipeline is identical to from_timeseries.

    Note: CG-01 is currently incomplete in the domain index.
    W and thresholds are not yet calibrated. Defaults are used.

    Parameters
    ----------
    turn_scalars_ai    : np.ndarray, shape (T,)
    turn_scalars_human : np.ndarray, shape (T,)
    params             : RRGParams

    Returns
    -------
    RRGObservables
    """
    if params is None:
        # Placeholder — calibrate from domain paper when available
        params = RRGParams(W=5)

    return RRGObservables.from_series(turn_scalars_ai, turn_scalars_human, params=params)


# ---------------------------------------------------------------------------
# Adapter 4 — Anthropic API Proxy (PROXY-01)
# Embedding norms as sa/sb proxy for real API conversations
# ---------------------------------------------------------------------------

# PROXY-01 parameters (calibrated 2026-05-04 via calibrate_proxy.py --collect --W 3 --n-per-cat 6)
# Kept for backward-compat (tests, calibrate_proxy.py). proxy.py loads from the registry instead.
PROXY_PARAMS = RRGParams(
    W            = 3,
    tau_lock_rho = 0.35,
    tau_lock_dr  = 0.021842,
    tau_meta     = 0.015,
    tau_exit     = 0.04,
)


def load_proxy_params(
    entry_id: str,
    registry_path: Optional[str] = None,
) -> RRGParams:
    """Load RRGParams from calibration_registry.json by entry ID.

    Parameters
    ----------
    entry_id       : calibration entry ID, e.g. "CODING-01" or "PROXY-01"
    registry_path  : override path to calibration_registry.json; defaults to
                     <project-root>/calibration_registry.json
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(registry_path) if registry_path else (
        _Path(__file__).parent.parent / "calibration_registry.json"
    )
    with open(path, encoding="utf-8") as f:
        registry = _json.load(f)

    for entry in registry.get("entries", []):
        if entry["id"] == entry_id:
            return RRGParams(
                W            = entry["W"],
                tau_lock_rho = entry["tau_lock_rho"],
                tau_lock_dr  = entry["tau_lock_dr"],
                tau_meta     = entry.get("tau_meta", 0.015),
                tau_exit     = entry.get("tau_exit", 0.04),
            )
    raise ValueError(
        f"Calibration entry '{entry_id}' not found in {path}. "
        f"Available IDs: {[e['id'] for e in registry.get('entries', [])]}"
    )

_ENCODERS: Dict[str, Any] = {}  # lazy singletons keyed by model name


def _get_encoder(model_name: str) -> Any:
    """Lazy-load a sentence-transformers model (cached per model name)."""
    if model_name not in _ENCODERS:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the PROXY-01 adapter. "
                "Install with: pip install sentence-transformers"
            ) from exc
        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def _embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed texts with sentence-transformers and return unit-normalized matrix. Shape: (T, D)."""
    enc = _get_encoder(model_name)
    emb = enc.encode(texts, show_progress_bar=False)  # (T, D)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return emb / norms


def _cosim_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two unit-normalized matrices. Shape: (T,)."""
    return np.einsum('ij,ij->i', a, b)


def _norm01(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if the array is flat."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def from_api_stream(
    turns      : List[Tuple[str, str]],
    model_name : str = "all-MiniLM-L6-v2",
    _embed_fn  : Optional[Callable[[List[str]], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PROXY-01 adapter — cosine similarity signals from conversation turns.

    The previous L2-norm signals were degenerate: all-MiniLM-L6-v2 emits
    unit vectors, so all norms ≈ 1.0, _norm01 collapses to zeros, and
    Pearson(zeros, zeros) = 0 regardless of topic coherence.

    Signal definitions (PROXY-01, revised)
    ───────────────────────────────────────
    sa(t) = cosine_sim(user_t, asst_t)          — within-turn alignment
    sb(t) = cosine_sim(asst_t, asst_{t-1})      — consecutive response coherence
    sb(0) = sa(0)                                — bootstrap (no prior response)
    checkpoint = each conversation turn boundary

    Discriminates correctly:
      COHERENT → sa high, sb high (related responses) → both track → ρ* > 0
      VARIED   → sa high (on-topic answers), sb ≈ 0 (topic jumps) → diverge → ρ* ≈ 0

    Parameters
    ----------
    turns      : list of (user_text, assistant_text) tuples, length T ≥ 2
    model_name : sentence-transformers model (default: all-MiniLM-L6-v2)
    _embed_fn  : optional override for embedding (testing / custom models).
                 Must accept List[str] and return np.ndarray of shape (T, D).

    Returns
    -------
    (sa, sb) : np.ndarray pair, each shape (T,), values in [-1, 1]
    """
    if len(turns) < 2:
        raise ValueError(
            f"from_api_stream requires at least 2 turns, got {len(turns)}."
        )

    user_texts = [t[0] for t in turns]
    asst_texts = [t[1] for t in turns]
    T = len(turns)

    embed = _embed_fn or (lambda texts: _embed_texts(texts, model_name))

    emb_u = embed(user_texts)   # (T, D)
    emb_a = embed(asst_texts)   # (T, D)

    # Re-normalise to guard against _embed_fn returning non-unit vectors
    def _unit(e: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(e, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return e / n

    emb_u = _unit(emb_u)
    emb_a = _unit(emb_a)

    # sa(t) = cosine_sim(user_t, asst_t)
    sa = np.clip(_cosim_pairwise(emb_u, emb_a), -1.0, 1.0)  # (T,)

    # sb(t) = cosine_sim(asst_t, asst_{t-1}); sb(0) = sa(0) (bootstrap)
    sb = np.empty(T)
    sb[0] = sa[0]
    if T > 1:
        sb[1:] = np.clip(_cosim_pairwise(emb_a[1:], emb_a[:-1]), -1.0, 1.0)

    return sa, sb


# ---------------------------------------------------------------------------
# Registry — maps domain_id -> adapter function + default params
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: Dict[str, Dict] = {
    'EL-01': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=60),
        'sa'      : 'Scalar electricity load sensor A',
        'sb'      : 'Scalar electricity load sensor B',
    },
    'TR-01': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=12),
        'sa'      : 'Speed reading node A (METR-LA)',
        'sb'      : 'Speed reading node B (METR-LA)',
    },
    'FIN-01': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=20, tau_lock_rho=0.45, tau_lock_dr=0.1),
        'sa'      : 'Asset return series A',
        'sb'      : 'Asset return series B',
    },
    'EEG-01': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=160),
        'sa'      : 'C3 EEG channel (scalar)',
        'sb'      : 'C4 EEG channel (scalar)',
    },
    'SHM-01-A': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=60),
        'sa'      : 'Strain gauge A (rosette)',
        'sb'      : 'Strain gauge B (rosette)',
    },
    'SHM-01-B': {
        'adapter' : from_timeseries_multiscale,
        'params'  : RRGParams(W=24),
        'windows' : [24, 6],
        'sa'      : 'Modal frequency channel A (KW51)',
        'sb'      : 'Modal frequency channel B (KW51)',
    },
    'GW-01': {
        'adapter' : from_timeseries_multiscale,
        'params'  : RRGParams(W=20),
        'windows' : [5, 20, 480],
        'sa'      : 'H1 whitened strain (LIGO)',
        'sb'      : 'L1 whitened strain (LIGO)',
    },
    'NT-01': {
        'adapter' : from_timeseries,
        'params'  : RRGParams(W=60),
        'sa'      : 'Depth-rate series A (Lagrange scalar)',
        'sb'      : 'Depth-rate series B (Lagrange scalar)',
    },
    'COT-01': {
        'adapter' : from_cot,
        'params'  : COT_PARAMS,
        'sa'      : 'Layer 4 residual norm at reasoning checkpoint',
        'sb'      : 'Layer 11 residual norm at reasoning checkpoint',
    },
    'CG-01': {
        'adapter' : from_conversation,
        'params'  : RRGParams(W=5),
        'sa'      : 'Turn-level scalar per AI->Human exchange',
        'sb'      : 'Turn-level scalar per Human->AI exchange',
    },
    'PROXY-01': {
        'adapter' : from_api_stream,
        'params'  : PROXY_PARAMS,
        'sa'      : 'cosine_sim(embed(user_turn_t), embed(claude_turn_t)) — within-turn alignment',
        'sb'      : 'cosine_sim(embed(claude_turn_t), embed(claude_turn_{t-1})) — consecutive coherence; sb(0)=sa(0)',
    },
}


def describe(domain_id: str) -> None:
    """Print the observable mapping and parameters for a domain."""
    if domain_id not in DOMAIN_REGISTRY:
        print(f"Unknown domain '{domain_id}'. Available: {list(DOMAIN_REGISTRY.keys())}")
        return

    d = DOMAIN_REGISTRY[domain_id]
    p = d['params']
    print(f"Domain   : {domain_id}")
    print(f"sa       : {d['sa']}")
    print(f"sb       : {d['sb']}")
    print(f"W        : {p.W}")
    print(f"τ_lock_ρ : {p.tau_lock_rho}")
    print(f"τ_lock_dρ: {p.tau_lock_dr}")
    if 'windows' in d:
        print(f"windows  : {d['windows']} (multi-scale)")
