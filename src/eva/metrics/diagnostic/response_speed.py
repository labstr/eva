"""Response speed metric measuring latency between user and assistant.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from abc import abstractmethod

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def _split_latencies_by_tool_calls(
    context: MetricContext,
) -> tuple[list[float], list[float]]:
    """Partition response_speed_latencies into (with_tool_calls, no_tool_calls).

    The i-th latency corresponds to the i-th user turn in chronological order.
    We look at the conversation_trace to find which turn_ids contain at least
    one tool_call entry.

    Returns:
        (with_tool_latencies, no_tool_latencies)
    """
    trace = context.conversation_trace or []

    user_turn_ids = sorted({entry["turn_id"] for entry in trace if entry.get("type") == "transcribed"})
    tool_call_turn_ids = {entry["turn_id"] for entry in trace if entry.get("type") == "tool_call"}

    with_tool: list[float] = []
    no_tool: list[float] = []

    for i, latency in enumerate(context.response_speed_latencies):
        if i >= len(user_turn_ids):
            break
        if user_turn_ids[i] in tool_call_turn_ids:
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
            if not context.response_speed_latencies:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error="No response latencies available (UserBotLatencyObserver data missing)",
                )

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
    and the beginning of the assistant's response.

    Reports raw latency values in seconds — no normalization applied.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "response_speed"
    description = "Debug metric: latency between user utterance end and assistant response start"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        return context.response_speed_latencies, "No valid response speeds computed"


@register_metric
class ResponseSpeedWithToolCallsMetric(_ResponseSpeedBase):
    """Response speed restricted to turns where the assistant made at least one tool call.

    Computed the same way as response_speed but only over tool-call turns.
    This is a diagnostic metric not used in final evaluation scores.
    """

    name = "response_speed_with_tool_calls"
    description = "Debug metric: response latency for turns that included a tool call"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        with_tool, _ = _split_latencies_by_tool_calls(context)
        return with_tool, "No turns with tool calls found"


@register_metric
class ResponseSpeedNoToolCallsMetric(_ResponseSpeedBase):
    """Response speed restricted to turns where the assistant made no tool calls.

    Computed the same way as response_speed but only over non-tool-call turns.
    This is a diagnostic metric not used in final evaluation scores.
    """

    name = "response_speed_no_tool_calls"
    description = "Debug metric: response latency for turns that did not include a tool call"

    def _get_latencies(self, context: MetricContext) -> tuple[list[float], str]:
        _, no_tool = _split_latencies_by_tool_calls(context)
        return no_tool, "No turns without tool calls found"
