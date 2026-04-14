"""Response speed metric measuring latency between user and assistant.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import json
from abc import abstractmethod
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def _split_turn_taking_latencies_by_tool_calls(
    context: MetricContext,
) -> tuple[list[float], list[float]]:
    """Partition turn_taking per_turn_latency values into (with_tool_calls, no_tool_calls).

    Reads metrics/turn_taking/details/per_turn_latency from the record's
    metrics.json, then checks conversation_trace to determine which turn_ids
    had at least one tool call.

    Returns:
        (with_tool_latencies, no_tool_latencies)
    """
    if not context.output_dir:
        return [], []

    metrics_path = Path(context.output_dir) / "metrics.json"
    if not metrics_path.exists():
        return [], []

    with open(metrics_path) as f:
        data = json.load(f)

    per_turn_latency: dict[str, float] = (
        data.get("metrics", {}).get("turn_taking", {}).get("details", {}).get("per_turn_latency", {})
    )
    if not per_turn_latency:
        return [], []

    tool_call_turn_ids = {
        entry["turn_id"] for entry in (context.conversation_trace or []) if entry.get("type") == "tool_call"
    }

    with_tool: list[float] = []
    no_tool: list[float] = []
    for turn_id_str, latency in per_turn_latency.items():
        if int(turn_id_str) in tool_call_turn_ids:
            with_tool.append(latency)
        else:
            no_tool.append(latency)

    return with_tool, no_tool


class _ResponseSpeedBase(CodeMetric):
    """Base class for response-speed metrics.

    Subclasses implement `_get_latencies` to return the subset of latencies
    to compute over; everything else is shared.
    """

    category = "diagnostic"
    exclude_from_pass_at_k = True

    @abstractmethod
    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        """Return (latencies, error_if_empty) for this metric variant."""

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            latencies, empty_error = self._get_latencies(context)

            if not latencies:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error=empty_error,
                )

            speeds = []
            per_turn_speeds = []
            for latency in latencies:
                if 0 < latency < 1000:
                    speeds.append(latency)
                    per_turn_speeds.append(round(latency, 3))
                else:
                    self.logger.warning(
                        f"[{context.record_id}] Unusual response speed detected and dropped: {latency} seconds"
                    )

            if not speeds:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error="No valid response speeds computed",
                )

            mean_speed = sum(speeds) / len(speeds)

            return MetricScore(
                name=self.name,
                score=round(mean_speed, 3),
                normalized_score=None,
                details={
                    "mean_speed_seconds": round(mean_speed, 3),
                    "max_speed_seconds": round(max(speeds), 3),
                    "num_turns": len(speeds),
                    "per_turn_speeds": per_turn_speeds,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)


@register_metric
class ResponseSpeedMetric(_ResponseSpeedBase):
    """Response speed metric.

    Measures the elapsed time between the end of the user's utterance
    and the beginning of the assistant's response, using Pipecat's
    UserBotLatencyObserver measurements.

    Reports raw latency values in seconds — no normalization applied.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "response_speed"
    description = "Debug metric: latency between user utterance end and assistant response start"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        return (
            context.response_speed_latencies,
            "No response latencies available (UserBotLatencyObserver data missing)",
        )


@register_metric
class ResponseSpeedWithToolCallsMetric(_ResponseSpeedBase):
    """Response speed restricted to turns where the assistant made at least one tool call.

    Uses per_turn_latency from the turn_taking metric and filters to turns
    that contain a tool_call entry in the conversation trace.
    This is a diagnostic metric not used in final evaluation scores.
    """

    name = "response_speed_with_tool_calls"
    description = "Debug metric: response latency for turns that included a tool call"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        with_tool, _ = _split_turn_taking_latencies_by_tool_calls(context)
        return with_tool, "No turns with tool calls found (or turn_taking latency data unavailable)"


@register_metric
class ResponseSpeedNoToolCallsMetric(_ResponseSpeedBase):
    """Response speed restricted to turns where the assistant made no tool calls.

    Uses per_turn_latency from the turn_taking metric and filters to turns
    that contain no tool_call entry in the conversation trace.
    This is a diagnostic metric not used in final evaluation scores.
    """

    name = "response_speed_no_tool_calls"
    description = "Debug metric: response latency for turns that did not include a tool call"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        _, no_tool = _split_turn_taking_latencies_by_tool_calls(context)
        return no_tool, "No turns without tool calls found (or turn_taking latency data unavailable)"
