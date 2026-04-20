# Turn Taking

> **Experience Metric**: poor timing — interrupting the user or leaving awkward silences — makes the conversation feel unnatural even if the content is correct.

## Overview

Code-based metric (no LLM) that scores each user→assistant transition on a continuous `[0, 1]` scale derived from the ElevenLabs audio timestamps already stored on `MetricContext`. The main score is the plain mean of the per-turn scores. A flat set of sub-metrics surfaces supporting headline numbers (latency percentiles, interruption rates, recovery/yield rates) that show up as their own columns in analysis tools.

## Scope

- **Greeting (turn 0) is excluded.**
- A turn is **evaluable** only when both `audio_timestamps_user_turns[t]` and `audio_timestamps_assistant_turns[t]` are non-empty. Turns without both sides are silently excluded from the set that feeds the score and sub-metrics.
- If no evaluable turns exist, the metric returns `normalized_score = None` with an error `"No turns with both user and assistant audio timestamps"`.

## Inputs (from `MetricContext`)

- `audio_timestamps_user_turns` / `audio_timestamps_assistant_turns` — per-turn audio segment lists `[(start_s, end_s), ...]`. Drive the evaluable set, overlap, and yield computations.
- `latency_assistant_turns` — per-turn latency (`first_asst_start - last_user_end`) in seconds. Drives the latency curve.
- `assistant_interrupted_turns` / `user_interrupted_turns` — turn-level interruption flags set by the processor.

## Per-turn Score

For each evaluable turn, one of four signals is used depending on the turn's flags:

| Turn flag state | Signal → Score |
|---|---|
| `turn ∈ assistant_interrupted_turns` only | **Overlap score** (capped) |
| `turn ∈ user_interrupted_turns` only | **Yield score** |
| `turn` in **both** sets | `min(overlap_score, yield_score)` |
| neither flag | **Latency score** |

Each signal has its own continuous curve (below).

### Latency curve (piecewise linear)

Ramp up from `LATENCY_HARD_EARLY_MS` (-500ms) to `LATENCY_SWEET_SPOT_LOW_MS` (500ms), plateau at 1.0 through `LATENCY_SWEET_SPOT_HIGH_MS` (2000ms), ramp down to `LATENCY_HARD_LATE_MS` (5000ms). Outside the outer bounds the score is clamped at 0.

| Latency (ms) | Score |
|---|---|
| ≤ -500 (hard early) | 0.00 |
| 0 | 0.50 |
| 200 | 0.70 |
| 500–2000 (sweet spot) | 1.00 |
| 3000 | 0.67 |
| 4000 | 0.33 |
| ≥ 5000 (hard late) | 0.00 |

### Overlap curve (agent-interrupt turns)

`overlap_ms` is the **total** simultaneous-speech duration between user and assistant in the turn, computed as the sum of pairwise segment intersections (streamed turns with interleaved silence would wildly over-count under a naive full-range intersection).

```
raw = max(0, 1 - overlap_ms / OVERLAP_HARD_MS)
score = AGENT_INTERRUPT_MAX_SCORE * raw
```

The cap (default `0.5`) ensures agent interruptions are always penalized — a clean recovery is still worse than not interrupting at all.

| Overlap (ms) | Score |
|---|---|
| 0 | 0.50 (capped) |
| 500 | 0.375 |
| 1000 | 0.25 |
| ≥ 2000 | 0.00 |

### Yield curve (user-interrupt turns)

`yield_ms` is how long the agent kept speaking after the user barged in: `assistant_audio_end[t-1] - user_audio_start[t]`.

`score = max(0, 1 - yield_ms / YIELD_HARD_MS)`

User interruptions are not the agent's fault, so the score is uncapped — a fast-yielding agent gets the full `1.0`.

| Yield (ms) | Score |
|---|---|
| 0 | 1.00 |
| 600 | 0.70 |
| 1000 | 0.50 |
| ≥ 2000 | 0.00 |

## Main Score

`turn_taking.score = turn_taking.normalized_score = mean(per_turn_score)` over evaluable turns. No weighting.

## Per-turn Evidence

Every evaluated turn contributes one entry to `details.per_turn_evidence[turn_id]`. The fields depend on which signal fired:

**Latency turns**
- `latency_ms`
- `latency_score`

**Agent-interrupt turns** (either agent-only or dual)
- `overlap_ms`
- `overlap_score`
- `n_interrupt_segments` — diagnostic: how many distinct agent audio segments overlap the user's speech in this turn.
- `post_interrupt_latency_ms` — gap between user's last audio end and the agent's first *settled* segment (the first one starting **after** the user finished). Present only when such a segment exists.
- `post_interrupt_latency_score` — `_latency_score(post_interrupt_latency_ms)`. The agent's turn score folds this in: `turn_score = min(overlap_score, post_interrupt_latency_score)`, so "brief barge-in then 10s wait" is penalized even when the barge-in itself was short.

**User-interrupt turns** (either user-only or dual)
- `yield_ms`
- `yield_score`

## Details Fields

`details` on the main `MetricScore` contains:

| Field | Description |
|---|---|
| `per_turn_score` | `{turn_id: float}` — the final 0–1 score per turn. |
| `per_turn_reason` | `{turn_id: "latency" / "agent_interrupt" / "user_interrupt" / "dual_interrupt"}` — which signal fired. |
| `per_turn_evidence` | `{turn_id: {...}}` — see previous section. |
| `num_turns` | Highest turn_id present in either user or assistant audio timestamps (greeting excluded). |
| `num_evaluated` | Number of turns actually scored (both timestamp sides present). |

## Sub-metrics (flat)

Emitted as `sub_metrics` on the main `MetricScore`, in this order. The runner aggregates each generically into `metrics_summary.json` as its own column, preserving insertion order.

**Latency (always present when at least one latency measurement exists)**

| Key | Normalized? | Meaning |
|---|---|---|
| `mean_latency_ms` | no | Arithmetic mean of per-turn latencies in ms. |
| `p50_latency_ms` | no | Median latency. |
| `p90_latency_ms` | no | 90th-percentile latency. |
| `on_time_rate` | yes | Fraction with `EARLY_THRESHOLD_MS ≤ latency < LATE_THRESHOLD_MS`. |
| `early_rate` | yes | Fraction with `latency < EARLY_THRESHOLD_MS` (default 200 ms). |
| `late_rate` | yes | Fraction with `latency ≥ LATE_THRESHOLD_MS` (default 4000 ms). |

**Agent interruptions** (dotted prefix so tables group them visibly)

| Key | Normalized? | When present | Meaning |
| --- | --- | --- | --- |
| `agent_interruption.rate` | yes | always | Fraction of evaluable turns in `assistant_interrupted_turns`. |
| `agent_interruption.mean_overlap_ms` | no | rate > 0 | Arithmetic mean of `overlap_ms` across agent-interrupt turns. |
| `agent_interruption.mean_overlap_score` | yes | rate > 0 | Mean of the per-turn overlap scores (capped at `AGENT_INTERRUPT_MAX_SCORE`). |
| `agent_interruption.mean_post_interrupt_latency_ms` | no | ≥ 1 interrupt turn has a settled response | Mean `post_interrupt_latency_ms` across agent-interrupt turns that emit a settled segment after the user finishes. |
| `agent_interruption.mean_post_interrupt_latency_score` | yes | same as above | Mean of the post-interrupt latency scores that feed the main score. |

**User interruptions**

| Key | Normalized? | When present | Meaning |
| --- | --- | --- | --- |
| `user_interruption.rate` | yes | always | Fraction of evaluable turns in `user_interrupted_turns`. |
| `user_interruption.mean_yield_ms` | no | rate > 0 | Arithmetic mean of `yield_ms` across user-interrupt turns. |
| `user_interruption.mean_yield_score` | yes | rate > 0 | Mean of the per-turn yield scores that feed the main score. |

Rate sub-metrics are emitted as `normalized_score` (they already live on `[0, 1]`). Raw-ms sub-metrics have `normalized_score = None` so they don't corrupt cross-metric averages.

## Tunable Constants

All thresholds live as class-level attributes on `TurnTakingMetric`. Override by subclassing or editing in place.

| Constant | Default | Purpose |
|---|---|---|
| `LATENCY_HARD_EARLY_MS` | -500 | Left edge of the latency ramp (score = 0 at or below). |
| `LATENCY_SWEET_SPOT_LOW_MS` | 500 | Left edge of the latency plateau (score reaches 1). |
| `LATENCY_SWEET_SPOT_HIGH_MS` | 2000 | Right edge of the plateau (score starts descending). |
| `LATENCY_HARD_LATE_MS` | 5000 | Right edge of the ramp (score = 0 at or above). |
| `OVERLAP_HARD_MS` | 2000 | Overlap at which the agent-interrupt raw score hits 0 (pre-cap). |
| `AGENT_INTERRUPT_MAX_SCORE` | 0.5 | Cap on the agent-interrupt score — never > this, even with 0ms overlap. |
| `YIELD_HARD_MS` | 2000 | Yield time at which the user-interrupt score hits 0. |
| `EARLY_THRESHOLD_MS` | 200 | Latency classification cutoff — below ⇒ "early". |
| `LATE_THRESHOLD_MS` | 4000 | Latency classification cutoff — at or above ⇒ "late". |

Note: the latency *curve* and the latency *classification* use independent thresholds. The curve is continuous (`LATENCY_HARD_EARLY_MS` / `LATENCY_HARD_LATE_MS`), while `EARLY_THRESHOLD_MS` / `LATE_THRESHOLD_MS` bucket turns for the `early_rate` / `on_time_rate` / `late_rate` sub-metrics only.

## Example Output

```json
{
  "name": "turn_taking",
  "score": 0.85,
  "normalized_score": 0.85,
  "details": {
    "per_turn_score": {"1": 1.0, "2": 0.67, "3": 0.45, "4": 0.95},
    "per_turn_reason": {"1": "latency", "2": "latency", "3": "agent_interrupt", "4": "user_interrupt"},
    "per_turn_evidence": {
      "1": {"latency_ms": 1000, "latency_score": 1.0},
      "2": {"latency_ms": 3000, "latency_score": 0.667},
      "3": {"overlap_ms": 200, "overlap_score": 0.45, "n_interrupt_segments": 1,
            "post_interrupt_latency_ms": 1000, "post_interrupt_latency_score": 1.0},
      "4": {"yield_ms": 100, "yield_score": 0.95}
    },
    "num_turns": 4,
    "num_evaluated": 4
  },
  "sub_metrics": {
    "mean_latency_ms":                        {"score": 2000.0, "normalized_score": null},
    "p50_latency_ms":                         {"score": 2000.0, "normalized_score": null},
    "p90_latency_ms":                         {"score": 3000.0, "normalized_score": null},
    "on_time_rate":                           {"score": 1.0,    "normalized_score": 1.0},
    "early_rate":                             {"score": 0.0,    "normalized_score": 0.0},
    "late_rate":                              {"score": 0.0,    "normalized_score": 0.0},
    "agent_interruption.rate":                                {"score": 0.25,   "normalized_score": 0.25},
    "agent_interruption.mean_overlap_ms":                     {"score": 200.0,  "normalized_score": null},
    "agent_interruption.mean_overlap_score":                  {"score": 0.45,   "normalized_score": 0.45},
    "agent_interruption.mean_post_interrupt_latency_ms":      {"score": 1000.0, "normalized_score": null},
    "agent_interruption.mean_post_interrupt_latency_score":   {"score": 1.0,    "normalized_score": 1.0},
    "user_interruption.rate":                                 {"score": 0.25,   "normalized_score": 0.25},
    "user_interruption.mean_yield_ms":        {"score": 100.0,  "normalized_score": null},
    "user_interruption.mean_yield_score":     {"score": 0.95,   "normalized_score": 0.95}
  }
}
```

## Related Metrics

- [`response_speed`](response_speed.md) — raw per-turn latency values, no curve or bucketing.

## Implementation Details

- **File**: `src/eva/metrics/experience/turn_taking.py`
- **Class**: `TurnTakingMetric`
- **Base class**: `CodeMetric`
