"""Tests for TurnTakingMetric.

Scoring model (continuous, 0–1):
  - Latency → piecewise linear (ramp 0 → 1 over [-500ms, 500ms], flat 1 to 2000ms, ramp 1 → 0 to 5000ms).
  - Agent-interrupt turns → overlap-based, capped at AGENT_INTERRUPT_MAX_SCORE = 0.5.
  - User-interrupt turns → agent yield-latency-based (ramp 1 → 0 over [0, 2000ms]).
  - Both flags on one turn → min of the two.

Sub-metrics (flat, one number each):
  Latency:              mean_latency_ms, p50_latency_ms, p90_latency_ms,
                        on_time_rate, early_rate, late_rate
  Agent interruptions:  agent_interruption.rate (always),
                        agent_interruption.mean_overlap_ms,
                        agent_interruption.mean_overlap_score (only when rate > 0)
  User interruptions:   user_interruption.rate (always),
                        user_interruption.mean_yield_ms,
                        user_interruption.mean_yield_score (only when rate > 0)
"""

import logging

import pytest

from eva.metrics.experience.turn_taking import TurnTakingMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = TurnTakingMetric()
    m.logger = logging.getLogger("test_turn_taking")
    return m


# ---------- Curve unit tests ----------


class TestLatencyScore:
    @pytest.mark.parametrize(
        "latency_ms, expected",
        [
            (-1000, 0.00),
            (-500, 0.00),
            (-200, 0.30),
            (0, 0.50),
            (200, 0.70),
            (500, 1.00),
            (1000, 1.00),
            (2000, 1.00),
            (2500, 0.8333),
            (3500, 0.5),
            (4500, 0.1667),
            (5000, 0.00),
            (8000, 0.00),
        ],
    )
    def test_latency_score_points(self, metric, latency_ms, expected):
        assert metric._latency_score(latency_ms) == pytest.approx(expected, abs=1e-3)


class TestOverlapScore:
    @pytest.mark.parametrize(
        "overlap_ms, expected",
        [
            (0, 0.50),
            (200, 0.45),
            (500, 0.375),
            (1000, 0.25),
            (2000, 0.00),
        ],
    )
    def test_overlap_score(self, metric, overlap_ms, expected):
        assert metric._overlap_score(overlap_ms) == pytest.approx(expected, abs=1e-3)


class TestYieldScore:
    @pytest.mark.parametrize(
        "yield_ms, expected",
        [
            (0, 1.00),
            (200, 0.90),
            (600, 0.70),
            (1000, 0.50),
            (2000, 0.00),
        ],
    )
    def test_yield_score(self, metric, yield_ms, expected):
        assert metric._yield_score(yield_ms) == pytest.approx(expected, abs=1e-3)


# ---------- End-to-end scenarios ----------


class TestComputeScenarios:
    @pytest.mark.asyncio
    async def test_all_on_time_ideal(self, metric):
        """3 turns all in sweet spot (1s latency) → mean = 1.0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)], 3: [(10.0, 11.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.5)], 2: [(7.0, 8.5)], 3: [(12.0, 13.5)]},
        )
        result = await metric.compute(context)
        assert result.error is None
        assert result.normalized_score == pytest.approx(1.0, abs=1e-3)
        assert all(r == "latency" for r in result.details["per_turn_reason"].values())

    @pytest.mark.asyncio
    async def test_agent_interrupt_post_interrupt_latency_penalizes_slow_recovery(self, metric):
        """Brief interrupt (overlap_score would be 0.45) but 8s wait for the real response → score 0."""
        context = make_metric_context(
            # User speaks 0–1s. Agent barges in at 0.8 for 200ms overlap, then goes silent until
            # 9s (8s AFTER user end) for its "settled" response — way beyond the latency curve.
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (9.0, 10.0)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["overlap_ms"] == pytest.approx(200, abs=1)
        assert ev["overlap_score"] == pytest.approx(0.45, abs=1e-3)
        # Post-interrupt latency is 9.0 - 1.0 = 8s, outside the latency curve → score 0.
        assert ev["post_interrupt_latency_ms"] == pytest.approx(8000, abs=1)
        assert ev["post_interrupt_latency_score"] == pytest.approx(0.0, abs=1e-3)
        # Turn score is min of the two signals — even a clean overlap can't save a slow recovery.
        assert result.details["per_turn_score"][1] == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_settled_response_on_time(self, metric):
        """Brief interrupt + 1s follow-up → overlap dominates (capped 0.45), post latency score is 1."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (2.0, 3.0)],  # overlap 200ms, settled at 2.0 → post = 1000ms → score 1.0
                2: [(7.0, 8.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["post_interrupt_latency_ms"] == pytest.approx(1000, abs=1)
        assert ev["post_interrupt_latency_score"] == pytest.approx(1.0, abs=1e-3)
        # min(0.45, 1.0) = 0.45
        assert result.details["per_turn_score"][1] == pytest.approx(0.45, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_no_settled_response_omits_post_latency(self, metric):
        """Agent only overlaps user and never emits a later segment → no post_interrupt_latency."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)]},
            audio_timestamps_assistant_turns={1: [(0.5, 1.5)]},  # fully within user speech
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert "post_interrupt_latency_ms" not in ev
        assert "post_interrupt_latency_score" not in ev

    @pytest.mark.asyncio
    async def test_agent_interrupt_continuous_speech_skips_post_latency(self, metric):
        """Agent's interrupt segment spans user_last_end, then keeps streaming — no silent gap.

        Real-world shape: short overlap (170ms) then agent speaks continuously for many seconds,
        split into multiple contiguous streaming chunks. The "post-interrupt latency" should be
        treated as N/A — the agent was already responding, there's no wait to measure.
        """
        context = make_metric_context(
            # User ends at 1.0. Agent segment 1 starts at 0.83 (170ms overlap) and runs to 5.0.
            # Agent segments 2 and 3 are later contiguous chunks of the same ongoing response.
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(10.0, 11.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.83, 5.0), (5.0, 7.5), (7.5, 9.0)],
                2: [(12.0, 13.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        # Overlap detected and scored (still penalized for the barge-in) — no bogus 4s-latency signal.
        assert ev["overlap_ms"] == pytest.approx(170, abs=1)
        assert "post_interrupt_latency_ms" not in ev
        assert "post_interrupt_latency_score" not in ev
        # Turn score reflects only overlap, not a spurious 0 from a fake latency.
        assert result.details["per_turn_score"][1] == pytest.approx(ev["overlap_score"], abs=1e-4)

    @pytest.mark.asyncio
    async def test_agent_interrupt_evidence(self, metric):
        """One agent interrupt segment with 200ms overlap → continuous overlap_score, single barge-in."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 2.0)], 2: [(7.0, 8.0)]},  # 200ms overlap at turn 1
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["overlap_ms"] == pytest.approx(200, abs=1)
        assert ev["overlap_score"] == pytest.approx(0.45, abs=1e-3)
        assert ev["n_interrupt_segments"] == 1

    @pytest.mark.asyncio
    async def test_agent_reinterrupts_within_same_turn(self, metric):
        """Multiple agent segments overlap the user's single turn — n_interrupt_segments reflects that."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 10.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(2.0, 2.5), (5.0, 5.5), (8.0, 8.5)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["n_interrupt_segments"] == 3

    @pytest.mark.asyncio
    async def test_user_interrupt_evidence(self, metric):
        """User barges in and agent yields within 100ms → continuous yield_score ≈ 0.95."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 4.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 3.1)], 2: [(5.0, 6.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][2]
        assert ev["yield_ms"] == pytest.approx(100, abs=1)
        assert ev["yield_score"] == pytest.approx(0.95, abs=1e-3)

    @pytest.mark.asyncio
    async def test_no_timestamps_returns_skipped(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
        )
        result = await metric.compute(context)
        assert result.normalized_score is None
        assert result.error


# ---------- Sub-metric structure ----------


class TestFlatSubMetrics:
    @pytest.mark.asyncio
    async def test_latency_headlines_populated(self, metric):
        """5 turns, mix of latencies → 6 latency sub-metrics present with expected values."""
        # Latencies: 100ms (early), 300ms, 1000ms, 3000ms, 5000ms (late)
        context = make_metric_context(
            audio_timestamps_user_turns={i: [(i * 10.0, i * 10.0 + 1.0)] for i in range(1, 6)},
            audio_timestamps_assistant_turns={
                1: [(11.1, 12.0)],  # 100ms
                2: [(21.3, 22.0)],  # 300ms
                3: [(32.0, 33.0)],  # 1000ms
                4: [(44.0, 45.0)],  # 3000ms
                5: [(56.0, 57.0)],  # 5000ms
            },
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        for k in ("mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "on_time_rate", "early_rate", "late_rate"):
            assert k in sub
        assert sub["on_time_rate"].score == pytest.approx(0.6)
        assert sub["early_rate"].score == pytest.approx(0.2)
        assert sub["late_rate"].score == pytest.approx(0.2)
        assert sub["p50_latency_ms"].score == pytest.approx(1000, abs=1)
        # Raw-ms sub-metrics are not normalized
        assert sub["mean_latency_ms"].normalized_score is None
        assert sub["p50_latency_ms"].normalized_score is None
        # Rate sub-metrics are normalized
        assert sub["on_time_rate"].normalized_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_no_agent_interruptions_omits_conditional_subs(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.0)]},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.rate"].score == 0.0
        assert "agent_interruption.mean_overlap_ms" not in sub
        assert "agent_interruption.mean_overlap_score" not in sub
        assert sub["user_interruption.rate"].score == 0.0
        assert "user_interruption.mean_yield_ms" not in sub
        assert "user_interruption.mean_yield_score" not in sub

    @pytest.mark.asyncio
    async def test_agent_interrupt_populates_overlap_sub_metrics(self, metric):
        """Agent-interrupt turn with 200ms overlap → mean_overlap_ms=200, mean_overlap_score=0.45."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 2.0)], 2: [(7.0, 8.0)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.rate"].score == pytest.approx(0.5)  # 1 of 2 turns
        assert sub["agent_interruption.mean_overlap_ms"].score == pytest.approx(200, abs=1)
        assert sub["agent_interruption.mean_overlap_score"].score == pytest.approx(0.45, abs=1e-3)
        # mean_overlap_score is normalized (in [0, 1]); raw ms is not.
        assert sub["agent_interruption.mean_overlap_score"].normalized_score == pytest.approx(0.45, abs=1e-3)
        assert sub["agent_interruption.mean_overlap_ms"].normalized_score is None

    @pytest.mark.asyncio
    async def test_post_interrupt_sub_metrics_populated(self, metric):
        """Settled response 1s after user → mean_post_interrupt_latency_* populated."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (2.0, 3.0)],  # overlap + settled 1000ms later
                2: [(7.0, 8.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.mean_post_interrupt_latency_ms"].score == pytest.approx(1000, abs=1)
        assert sub["agent_interruption.mean_post_interrupt_latency_score"].score == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_post_interrupt_sub_metrics_omitted_when_none(self, metric):
        """No settled response after interrupt → post_interrupt_* sub-metrics omitted."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)]},
            audio_timestamps_assistant_turns={1: [(0.5, 1.5)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert "agent_interruption.mean_post_interrupt_latency_ms" not in sub
        assert "agent_interruption.mean_post_interrupt_latency_score" not in sub

    @pytest.mark.asyncio
    async def test_user_interrupt_populates_yield_sub_metrics(self, metric):
        """User-interrupt turn with 100ms yield → mean_yield_ms=100, mean_yield_score=0.95."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 4.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 3.1)], 2: [(5.0, 6.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["user_interruption.rate"].score == pytest.approx(0.5)
        assert sub["user_interruption.mean_yield_ms"].score == pytest.approx(100, abs=1)
        assert sub["user_interruption.mean_yield_score"].score == pytest.approx(0.95, abs=1e-3)

    @pytest.mark.asyncio
    async def test_user_interrupt_slow_yield_low_score(self, metric):
        """Agent keeps talking 1500ms after barge-in → mean_yield_score ≈ 0.25."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 4.5)], 2: [(6.0, 7.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["user_interruption.mean_yield_ms"].score == pytest.approx(1500, abs=1)
        assert sub["user_interruption.mean_yield_score"].score == pytest.approx(0.25, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_overlap_uses_pairwise_intersection(self, metric):
        """Multi-segment streamed turn: overlap = sum of pairwise segment intersections.

        Guards against the prior bug where full-range intersection inflated overlap to 20+s.
        """
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 4.0), (18.0, 27.0)]},
            audio_timestamps_assistant_turns={1: [(3.8, 14.0), (30.0, 40.0)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        # Only real simultaneous speech is 3.8–4.0 = 200ms.
        assert result.details["per_turn_evidence"][1]["overlap_ms"] == pytest.approx(200, abs=1)
