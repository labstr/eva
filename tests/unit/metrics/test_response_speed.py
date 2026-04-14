"""Tests for the ResponseSpeedMetric."""

import json

import pytest

from eva.metrics.diagnostic.response_speed import (
    ResponseSpeedMetric,
    ResponseSpeedNoToolCallsMetric,
    ResponseSpeedWithToolCallsMetric,
)

from .conftest import make_metric_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics_json(tmp_path, per_turn_latency: dict) -> None:
    """Write a minimal metrics.json with turn_taking per_turn_latency data."""
    data = {
        "metrics": {
            "turn_taking": {
                "details": {
                    "per_turn_latency": per_turn_latency,
                }
            }
        }
    }
    (tmp_path / "metrics.json").write_text(json.dumps(data))


def _make_trace(tool_call_turn_ids: set[int], all_turn_ids: set[int]) -> list[dict]:
    """Build a minimal conversation_trace with the given turn structure."""
    trace = []
    for tid in sorted(all_turn_ids):
        trace.append({"turn_id": tid, "type": "transcribed", "content": "user utterance"})
        if tid in tool_call_turn_ids:
            trace.append({"turn_id": tid, "type": "tool_call", "tool_name": "some_tool"})
            trace.append({"turn_id": tid, "type": "tool_response", "tool_name": "some_tool"})
    return trace


# ---------------------------------------------------------------------------
# ResponseSpeedMetric
# ---------------------------------------------------------------------------


class TestResponseSpeedMetric:
    @pytest.mark.asyncio
    async def test_no_latencies_none(self):
        """None latencies returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=None)

        result = await metric.compute(ctx)

        assert result.name == "response_speed"
        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No response latencies" in result.error

    @pytest.mark.asyncio
    async def test_no_latencies_empty(self):
        """Empty list returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[])

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_valid_latencies(self):
        """Valid latencies produce correct mean, max, and per-turn details."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[1.0, 2.0, 3.0])

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(2.0)
        assert result.normalized_score is None
        assert result.error is None
        assert result.details["mean_speed_seconds"] == pytest.approx(2.0)
        assert result.details["max_speed_seconds"] == pytest.approx(3.0)
        assert result.details["num_turns"] == 3
        assert result.details["per_turn_speeds"] == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_filters_invalid_values(self):
        """Negative and >1000s values are filtered out."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[-1.0, 0.5, 1500.0, 2.5, 0.0])

        result = await metric.compute(ctx)

        # Only 0.5 and 2.5 are valid (0 < x < 1000); 0.0 is excluded (not > 0)
        assert result.error is None
        assert result.details["num_turns"] == 2
        expected_mean = (0.5 + 2.5) / 2
        assert result.score == pytest.approx(expected_mean)
        assert result.details["max_speed_seconds"] == pytest.approx(2.5)
        assert result.details["per_turn_speeds"] == [0.5, 2.5]

    @pytest.mark.asyncio
    async def test_all_latencies_filtered_out(self):
        """When all values are invalid, returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[-5.0, 0.0, 2000.0])

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No valid response speeds" in result.error

    @pytest.mark.asyncio
    async def test_single_latency_value(self):
        """Single valid latency works correctly."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[0.75])

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(0.75)
        assert result.details["mean_speed_seconds"] == pytest.approx(0.75)
        assert result.details["max_speed_seconds"] == pytest.approx(0.75)
        assert result.details["num_turns"] == 1
        assert result.details["per_turn_speeds"] == [0.75]


# ---------------------------------------------------------------------------
# ResponseSpeedWithToolCallsMetric
# ---------------------------------------------------------------------------


class TestResponseSpeedWithToolCallsMetric:
    @pytest.mark.asyncio
    async def test_no_output_dir(self):
        """Missing output_dir returns error."""
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context()

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_metrics_json(self, tmp_path):
        """output_dir exists but has no metrics.json — returns error."""
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_turn_taking_data(self, tmp_path):
        """metrics.json exists but has no turn_taking entry — returns error."""
        (tmp_path / "metrics.json").write_text(json.dumps({"metrics": {}}))
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_no_turns_with_tool_calls(self, tmp_path):
        """Record has no tool-call turns — returns 'not found' error."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 2.0, "3": 3.0})
        trace = _make_trace(tool_call_turn_ids=set(), all_turn_ids={1, 2, 3})
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None
        assert "No turns with tool calls" in result.error

    @pytest.mark.asyncio
    async def test_mixed_turns(self, tmp_path):
        """Correctly includes only tool-call turn latencies."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 5.0, "3": 3.0, "4": 7.0})
        # Turns 2 and 4 have tool calls
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx((5.0 + 7.0) / 2)
        assert result.details["max_speed_seconds"] == pytest.approx(7.0)
        assert result.details["per_turn_speeds"] == [5.0, 7.0]

    @pytest.mark.asyncio
    async def test_all_turns_have_tool_calls(self, tmp_path):
        """When every turn has a tool call, all latencies are included."""
        _write_metrics_json(tmp_path, {"1": 2.0, "2": 4.0})
        trace = _make_trace(tool_call_turn_ids={1, 2}, all_turn_ids={1, 2})
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_filters_invalid_latency_values(self, tmp_path):
        """Sanity filter (0 < x < 1000) applies to per_turn_latency values."""
        _write_metrics_json(tmp_path, {"1": -1.0, "2": 5.0, "3": 2000.0, "4": 3.0})
        trace = _make_trace(tool_call_turn_ids={1, 2, 3, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedWithToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["num_turns"] == 2  # only 5.0 and 3.0 pass
        assert result.score == pytest.approx((5.0 + 3.0) / 2)


# ---------------------------------------------------------------------------
# ResponseSpeedNoToolCallsMetric
# ---------------------------------------------------------------------------


class TestResponseSpeedNoToolCallsMetric:
    @pytest.mark.asyncio
    async def test_no_output_dir(self):
        """Missing output_dir returns error."""
        metric = ResponseSpeedNoToolCallsMetric()
        ctx = make_metric_context()

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_metrics_json(self, tmp_path):
        """output_dir exists but has no metrics.json — returns error."""
        metric = ResponseSpeedNoToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_all_turns_have_tool_calls(self, tmp_path):
        """Every turn has a tool call — no-tool bucket is empty."""
        _write_metrics_json(tmp_path, {"1": 2.0, "2": 4.0})
        trace = _make_trace(tool_call_turn_ids={1, 2}, all_turn_ids={1, 2})
        metric = ResponseSpeedNoToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None
        assert "No turns without tool calls" in result.error

    @pytest.mark.asyncio
    async def test_mixed_turns(self, tmp_path):
        """Correctly includes only non-tool-call turn latencies."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 5.0, "3": 3.0, "4": 7.0})
        # Turns 2 and 4 have tool calls; turns 1 and 3 do not
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedNoToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx((1.0 + 3.0) / 2)
        assert result.details["max_speed_seconds"] == pytest.approx(3.0)
        assert result.details["per_turn_speeds"] == [1.0, 3.0]

    @pytest.mark.asyncio
    async def test_no_turns_with_tool_calls(self, tmp_path):
        """Record with no tool-call turns — all latencies included."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 2.0, "3": 3.0})
        trace = _make_trace(tool_call_turn_ids=set(), all_turn_ids={1, 2, 3})
        metric = ResponseSpeedNoToolCallsMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["num_turns"] == 3
        assert result.score == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_with_and_no_tool_split_is_exhaustive(self, tmp_path):
        """with_tool + no_tool latencies together cover all per_turn_latency values."""
        per_turn = {"1": 1.0, "2": 5.0, "3": 3.0, "4": 7.0, "5": 2.0}
        _write_metrics_json(tmp_path, per_turn)
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4, 5})

        ctx_with = make_metric_context(output_dir=tmp_path, conversation_trace=trace)
        ctx_no = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result_with = await ResponseSpeedWithToolCallsMetric().compute(ctx_with)
        result_no = await ResponseSpeedNoToolCallsMetric().compute(ctx_no)

        combined = result_with.details["per_turn_speeds"] + result_no.details["per_turn_speeds"]
        assert sorted(combined) == sorted(per_turn.values())
