# Elara — RRG Memory Compression Proxy

Elara is a local HTTP proxy that sits between your application (or Claude Code) and the Anthropic API. It applies **RRG (Resonance–Regime–Gate) geometric compression** to conversation context: when the conversation geometry is stable ("Sufficient" regime), older turns are replaced with a compact seed summary, cutting input tokens without losing coherence.

```
Your app / Claude Code  →  ElaraClient / proxy:8877  →  Anthropic API
```

## How it works

Each user/assistant turn pair produces two signals via a local sentence embedding model (`all-MiniLM-L6-v2`):

- `sa` — cosine similarity between the user turn and the assistant reply (within-turn alignment)
- `sb` — cosine similarity between consecutive assistant replies (turn-to-turn coherence)

The engine tracks the rolling correlation `ρ(sa, sb)` over a window `W` and computes:

- **d_rho** — variance of the last `W` correlation values (local stability)
- **Fisher bound** — `(1 − ρ*²)² / W` (Eq 16)
- **delta** — `d_rho − fisher_bound` (Eq 22): `δ ≤ 0` → safe to compress, `δ > 0` → block
- **Lyapunov potential** — `V = d_rho + (1 − |ρ*|)`, ≈ 0 at the Sufficient attractor

Compression fires only when **both** gates agree:

- **Gate 1** (per instance): `d_rho < tau_lock_dr AND |ρ*| > tau_lock_rho`
- **Gate 2** (resonance-lock guard): if `max(d_rho_series) ≥ 0.05` the lock is flagged as resonance (not genuine) and the effective `tau_lock_dr` is tightened by 30 % until a juncture resets it

A **juncture** (all three required: prior regime was Sufficient, `δ > 0`, `d_rho_meta > tau_meta`) invalidates stale seeds and passes raw context through.

## Results

End-to-end against the real API (n = 6 turns per category):

| Category | lock_frac | Tokens saved |
|----------|-----------|--------------|
| Coherent | 0.750     | 43.9 %       |
| Mixed    | 0.250     | 18.7 %       |
| Varied   | 0.500     | 46.1 %       |

Phase D (n = 20 real sessions): Gate 2 threshold 0.05 achieves perfect separation between coherent sessions (max d_rho 0.036) and sessions with topic breaks (min 0.073).

## Quick start

### Requirements

- Python 3.10+
- `anthropic`, `numpy`, `sentence-transformers`, `pystray`, `Pillow`
- An Anthropic API key in the `ANTHROPIC_API_KEY` environment variable (never hardcoded)

The first run downloads the `all-MiniLM-L6-v2` embedding model; after that, set `HF_HUB_OFFLINE=1` to run fully offline.

### As a Python wrapper

```python
from elara import ElaraClient

with ElaraClient() as client:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": "..."}],
    )
```

### As a standalone proxy

```cmd
set ANTHROPIC_BASE_URL=http://localhost:8877
set HF_HUB_OFFLINE=1
python elara/proxy.py
```

Any Anthropic SDK client pointed at `ANTHROPIC_BASE_URL=http://localhost:8877` is transparently compressed and forwarded.

### System tray app (Windows)

```cmd
python elara_tray.py
```

Runs the proxy in-process with a tray icon (green = running, red = stopped, yellow dot = resonance lock active). The menu offers config switching, session reset, and stop/start.

## Configuration

Calibration profiles live in `calibration_registry.json` (per-domain empirical thresholds, e.g. `PROXY-01` for prose, `CODING-01` for coding Q&A). The active profile is selected in `proxy_active_config.json`:

```json
{"active_id": "CODING-01"}
```

The proxy hot-reloads this file on every request — no restart needed. A live metrics widget (`widget_server.py`, port 8878) shows cycle counts, lock fraction, and lets you switch profiles from the browser.

To calibrate a new domain:

```cmd
python calibrate_proxy.py --domain coding --collect --W 3 --n-per-cat 6
```

## Repository layout

| Path | Purpose |
|------|---------|
| `elara/engine.py` | Core RRG math (correlation, d_rho, delta, Lyapunov) — authoritative |
| `elara/regime.py` | Regime classification, AND gate, juncture detection |
| `elara/gatekeeper.py` | Gate 1 + Gate 2 with resonance-lock action |
| `elara/proxy.py` | HTTP proxy server with hot config reload |
| `elara/adapters.py` | Signal extraction from API streams; calibration loading |
| `elara/seed.py` / `elara/store.py` | Seed dataclass + hot/warm/cold tier persistence |
| `elara/metrics.py` | In-memory + SQLite metrics logging |
| `elara/hooks/` | Claude Code hooks (block native compaction, log prompts, launch tray) |
| `elara_tray.py` | System tray entry point |
| `widget_server.py` / `widget.html` | Live floating metrics widget |
| `calibrate_proxy.py` | Per-domain threshold calibration against the real API |
| `test_rrg.py` | Test suite (31 tests) |

## Testing

```cmd
python test_rrg.py
```

All 31 tests should pass.

## Status

Experimental research pilot. Known open items:

- Gate 2 threshold (0.05) was calibrated on unwindowed `d_rho_series`; re-validation post-windowing is pending.
- `CODING-01` does not yet cover code-paste-heavy sessions (`CODING-02` planned).
