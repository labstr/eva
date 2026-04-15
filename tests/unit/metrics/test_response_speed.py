"""Tests for the ResponseSpeedMetric."""

import json

import pytest

from eva.metrics.diagnostic.response_speed import ResponseSpeedMetric

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
    async def test_no_output_dir(self):
        """Missing output_dir returns error — no per_turn_latency data."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context()

        result = await metric.compute(ctx)

        assert result.name == "response_speed"
        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "turn_taking" in result.error

    @pytest.mark.asyncio
    async def test_missing_metrics_json(self, tmp_path):
        """output_dir exists but has no metrics.json — returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_turn_taking_data(self, tmp_path):
        """metrics.json exists but has no turn_taking entry — returns error."""
        (tmp_path / "metrics.json").write_text(json.dumps({"metrics": {}}))
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_valid_latencies(self, tmp_path):
        """Valid per_turn_latency produces correct mean, max, and per-turn details."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 2.0, "3": 3.0})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(2.0)
        assert result.normalized_score is None
        assert result.error is None
        assert result.details["mean_speed_seconds"] == pytest.approx(2.0)
        assert result.details["max_speed_seconds"] == pytest.approx(3.0)
        assert result.details["num_turns"] == 3

    @pytest.mark.asyncio
    async def test_filters_invalid_values(self, tmp_path):
        """Negative and >1000s values are filtered out."""
        _write_metrics_json(tmp_path, {"1": -1.0, "2": 0.5, "3": 1500.0, "4": 2.5, "5": 0.0})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        # Only 0.5 and 2.5 are valid (0 < x < 1000); 0.0 is excluded (not > 0)
        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx((0.5 + 2.5) / 2)
        assert result.details["max_speed_seconds"] == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_all_latencies_filtered_out(self, tmp_path):
        """When all values are invalid, returns error."""
        _write_metrics_json(tmp_path, {"1": -5.0, "2": 0.0, "3": 2000.0})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No valid response speeds" in result.error

    @pytest.mark.asyncio
    async def test_single_latency_value(self, tmp_path):
        """Single valid latency works correctly."""
        _write_metrics_json(tmp_path, {"1": 0.75})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(0.75)
        assert result.details["mean_speed_seconds"] == pytest.approx(0.75)
        assert result.details["max_speed_seconds"] == pytest.approx(0.75)
        assert result.details["num_turns"] == 1
        assert result.details["per_turn_speeds"] == [0.75]

    @pytest.mark.asyncio
    async def test_no_tool_call_breakdown_without_trace(self, tmp_path):
        """with_tool_calls is None and no_tool_calls covers all turns when trace is absent."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 2.0})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path)

        result = await metric.compute(ctx)

        assert result.error is None
        # No trace → no tool call turn ids → all turns go into no_tool bucket
        assert result.details["with_tool_calls"] is None
        assert result.details["no_tool_calls"] is not None
        assert result.details["no_tool_calls"]["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_mixed_turns(self, tmp_path):
        """with_tool_calls and no_tool_calls sub-fields reflect the correct split."""
        _write_metrics_json(tmp_path, {"1": 1.0, "2": 5.0, "3": 3.0, "4": 7.0})
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        with_tc = result.details["with_tool_calls"]
        no_tc = result.details["no_tool_calls"]
        assert with_tc is not None
        assert no_tc is not None
        assert with_tc["num_turns"] == 2
        assert with_tc["mean_speed_seconds"] == pytest.approx((5.0 + 7.0) / 2)
        assert with_tc["max_speed_seconds"] == pytest.approx(7.0)
        assert no_tc["num_turns"] == 2
        assert no_tc["mean_speed_seconds"] == pytest.approx((1.0 + 3.0) / 2)
        assert no_tc["max_speed_seconds"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_all_tool_turns(self, tmp_path):
        """no_tool_calls is None when every turn has a tool call."""
        _write_metrics_json(tmp_path, {"1": 2.0, "2": 4.0})
        trace = _make_trace(tool_call_turn_ids={1, 2}, all_turn_ids={1, 2})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.details["with_tool_calls"] is not None
        assert result.details["with_tool_calls"]["num_turns"] == 2
        assert result.details["no_tool_calls"] is None

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_filters_invalid_latencies(self, tmp_path):
        """Sanity filter (0 < x < 1000) applies within the breakdown sub-fields."""
        _write_metrics_json(tmp_path, {"1": -1.0, "2": 5.0, "3": 2000.0, "4": 3.0})
        trace = _make_trace(tool_call_turn_ids={1, 2, 3, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        with_tc = result.details["with_tool_calls"]
        assert with_tc is not None
        assert with_tc["num_turns"] == 2  # only 5.0 and 3.0 pass the filter

    @pytest.mark.asyncio
    async def test_with_and_no_tool_split_is_exhaustive(self, tmp_path):
        """with_tool + no_tool latencies together cover all per_turn_latency values."""
        per_turn = {"1": 1.0, "2": 5.0, "3": 3.0, "4": 7.0, "5": 2.0}
        _write_metrics_json(tmp_path, per_turn)
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4, 5})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(output_dir=tmp_path, conversation_trace=trace)

        result = await metric.compute(ctx)

        assert result.error is None
        combined = (
            result.details["with_tool_calls"]["per_turn_speeds"] + result.details["no_tool_calls"]["per_turn_speeds"]
        )
        assert sorted(combined) == sorted(per_turn.values())
