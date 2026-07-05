# Elara — Memory Compression Proxy for Claude

> ⚠️ **Experimental research pilot.** Not production-ready — APIs, thresholds, and behavior may change without notice. Use at your own risk, and don't point it at anything you can't afford to have miscompressed.

## What it is

Elara is a local proxy that sits between your application (or Claude Code) and the Anthropic API. It watches a conversation as it grows and, when it's confident the topic has stayed stable, swaps out older turns for a compact summary before forwarding the request — cutting how many tokens get sent and billed, without changing what the model effectively sees as "the conversation so far."

```
Your app / Claude Code  →  Elara (proxy on :8877)  →  Anthropic API
```

You don't need to change how you call the API. Point your existing client at Elara's local address (or use the included wrapper) and it works transparently.

## What it does

Long conversations get expensive: every request resends the full message history. Common workarounds either truncate history (losing context permanently) or summarize on a fixed schedule, regardless of whether that's actually a safe moment to do so — e.g. right when the topic changes. Elara instead measures *how coherent the conversation currently is*, turn by turn, and only compresses when it's safe: when doing so is unlikely to drop anything the model would have actually used.

For each turn, Elara compares:

- how well the assistant's reply matches what the user just said, and
- how similar the assistant's replies are to each other from one turn to the next.

If both of these have been holding steady over the last few turns, the conversation is judged stable, and Elara replaces the older turns with a short summary instead of sending them verbatim. If the topic jumps around, or a stability metric shifts significantly, Elara holds off and sends the full history through untouched — no data is ever lost, it's just sent in full instead of compressed.

Two independent checks (think of them as a lock with two keys) both have to agree before anything gets compressed, plus a safety valve that detects "this only looked stable because we got lucky" and temporarily makes the check stricter. If the conversation genuinely changes direction after a stable stretch, Elara notices and immediately reverts to sending full context — nothing stays compressed past its useful life.

<details>
<summary>Technical details</summary>

Each user/assistant turn pair produces two signals via a local sentence embedding model (`all-MiniLM-L6-v2`):

- `sa` — cosine similarity between the user turn and the assistant reply (within-turn alignment)
- `sb` — cosine similarity between consecutive assistant replies (turn-to-turn coherence)

The engine tracks the rolling correlation `ρ(sa, sb)` over a window `W` and computes:

- **d_rho** — variance of the last `W` correlation values (local stability)
- **Fisher bound** — `(1 − ρ*²)² / W`
- **delta** — `d_rho − fisher_bound`: `δ ≤ 0` → safe to compress, `δ > 0` → block
- **Stability score** — `V = d_rho + (1 − |ρ*|)`, close to 0 when the conversation is at its most stable

Compression fires only when both gates agree:

- **Gate 1** (per instance): `d_rho < tau_lock_dr AND |ρ*| > tau_lock_rho`
- **Gate 2** (false-positive guard): if `max(d_rho_series) ≥ 0.05`, the "stable" reading is treated as a fluke rather than genuine stability, and the threshold for Gate 1 is tightened by 30% until the conversation is confirmed stable again

A topic-change is only declared once all three are true: the conversation was previously judged stable, `δ > 0`, and a secondary drift metric crosses its threshold. When that happens, old summaries are discarded and full context is sent again.

</details>

## What's been demonstrated

End-to-end against the real API (6 turns per category):

| Conversation type | How often it compressed | Tokens saved |
|--------------------|--------------------------|--------------|
| Stayed on topic     | 75%                     | 43.9%        |
| Mixed topics        | 25%                     | 18.7%        |
| Topic varied a lot  | 50%                     | 46.1%        |

Across 20 longer real sessions, the stability check cleanly separated conversations that stayed on topic (never got close to the "unstable" threshold) from ones with an actual topic break (all comfortably over it) — no overlap between the two groups.

These results are from controlled test sessions, not production traffic — see [Current status](#current-status) for what's still open.

## Quick start

### Requirements

- Python 3.10+
- An Anthropic API key in the `ANTHROPIC_API_KEY` environment variable

```cmd
pip install -r requirements.txt
```

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

Runs the proxy in the background with a tray icon (green = running, red = stopped, yellow dot = a compression got flagged as a false positive and is temporarily paused). The menu offers switching between calibration profiles, resetting the session, and stopping/starting the proxy.

## Configuration

Elara's compression thresholds are tuned differently depending on the kind of conversation (casual prose vs. coding Q&A). These tuning profiles live in `calibration_registry.json`; the active one is selected in `proxy_active_config.json`:

```json
{"active_id": "CODING-01"}
```

The proxy picks up changes to this file on the next request — no restart needed. A live metrics widget (`widget_server.py`, port 8878) shows how often compression is firing and lets you switch profiles from the browser.

To calibrate a new profile from real conversation samples:

```cmd
python calibrate_proxy.py --domain coding --collect --W 3 --n-per-cat 6
```

## Repository layout

| Path | Purpose |
|------|---------|
| `elara/engine.py` | Core math (stability signals, thresholds) — authoritative |
| `elara/regime.py` | Stability classification and the "topic changed" detector |
| `elara/gatekeeper.py` | The two-gate compression decision |
| `elara/proxy.py` | HTTP proxy server with hot config reload |
| `elara/adapters.py` | Signal extraction from API streams; calibration loading |
| `elara/seed.py` / `elara/store.py` | Summary format + hot/warm/cold tier persistence |
| `elara/metrics.py` | In-memory + SQLite metrics logging |
| `elara/hooks/` | Claude Code hooks (block native compaction, log prompts, launch tray) |
| `elara_tray.py` | System tray entry point |
| `widget_server.py` / `widget.html` | Live floating metrics widget |
| `calibrate_proxy.py` | Per-domain threshold calibration against the real API |
| `calibration_registry.json` | Per-domain calibration profiles (e.g. `PROXY-01`, `CODING-01`) |
| `proxy_active_config.json` | Currently selected calibration profile |
| `run_baseline.py` / `run_phase_b.py` / `phase_b.py` | Experiment scripts used to produce the results above |
| `calibrate.py` / `extract_signals.py` / `connect_llm.py` | Early standalone pipeline (local transformer model) predating the proxy-based calibration in `calibrate_proxy.py`; kept for reference |
| `outputdb.py` | Ad hoc script for inspecting `elara_proxy_metrics.db` |
| `daydreaming.md` | Not project documentation — an internal test of running this pipeline against Fable 5, kept in the repo as a record of that experiment |
| `test_rrg.py` | Test suite (31 tests) |

## Testing

```cmd
python test_rrg.py
```

All 31 tests should pass.

## Current status

This is a research pilot validated on controlled test sessions and one 20-session real-world batch, not on production traffic. Known open items:

- The false-positive threshold (0.05) needs re-validation against the current windowed calculation — it was originally tuned on an older version of the metric.
- The coding-conversation profile (`CODING-01`) doesn't yet cover sessions that are mostly pasted code (planned as `CODING-02`).

Issues and pull requests are welcome — this is an early-stage project and feedback on the approach is as useful as code contributions.

## License

MIT — see [LICENSE](LICENSE).
