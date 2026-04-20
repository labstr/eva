"""Shared utilities for metrics computation."""

import base64
from io import BytesIO
from itertools import groupby
from pathlib import Path
from statistics import harmonic_mean
from typing import Any

from pydub import AudioSegment

from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json
from eva.utils.logging import get_logger

logger = get_logger(__name__)


def parse_judge_response(response_text: str, record_id: str, metric_logger) -> dict | None:
    """Parse LLM judge response using robust JSON extraction.

    Args:
        response_text: Raw response from LLM
        record_id: Record ID for logging
        metric_logger: Logger instance from the metric

    Returns:
        Parsed response dict or None if parsing fails
    """
    response = extract_and_load_json(response_text)

    if response is None:
        metric_logger.error(f"Failed to extract JSON from judge response for {record_id}")
        metric_logger.error(f"Response text: {response_text}")
        return None

    return response


def parse_judge_response_list(response_text: str | None) -> list[dict] | None:
    """Parse LLM judge response expecting a JSON array of per-turn evaluations.

    Handles common LLM response patterns:
    - JSON array directly → returned as-is
    - Single dict → wrapped in a list
    - None or unparseable → returns None

    Args:
        response_text: Raw response from LLM, or None

    Returns:
        List of per-turn evaluation dicts, or None if parsing fails
    """
    if response_text is None:
        return None
    parsed = extract_and_load_json(response_text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return None


def format_transcript(turns: list[dict]) -> str:
    """Format transcript as simple User/Assistant dialogue.

    Args:
        turns: List of turn dictionaries with 'role' and 'content'

    Returns:
        Formatted transcript string
    """
    if not turns:
        return "No transcript available"

    lines = []
    for turn in turns:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def format_transcript_with_tools(turns: list[dict]) -> str:
    """Format transcript grouped by turn_id, including tool calls and responses.

    Args:
        turns: List of turn dictionaries including tool calls

    Returns:
        Formatted transcript string grouped by turn_id
    """
    blocks = []
    for turn_id, group in groupby(turns, key=lambda e: e.get("turn_id", 0)):
        lines = [f"Turn {turn_id}:"]
        for entry in group:
            role = entry.get("role")
            if role in ("user", "assistant"):
                lines.append(f"  {role}: {entry.get('content', '')}")
            elif entry.get("type") == "tool_call":
                lines.append(f"  tool_call: {entry.get('tool_name', '')}({entry.get('parameters', {})})")
            elif entry.get("type") == "tool_response":
                lines.append(f"  tool_response: {entry.get('tool_name', '')}: {entry.get('tool_response', '')}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def audio_to_base64(audio_segment: AudioSegment) -> str:
    """Convert audio segment to base64-encoded WAV.

    Args:
        audio_segment: PyDub AudioSegment

    Returns:
        Base64-encoded WAV string
    """
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    audio_bytes = buffer.getvalue()
    return base64.b64encode(audio_bytes).decode("utf-8")


def load_audio_file(audio_path: Path) -> AudioSegment | None:
    """Load audio file from path.

    Args:
        audio_path: Path to audio file

    Returns:
        AudioSegment or None if file doesn't exist
    """
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        return AudioSegment.from_file(str(audio_path))
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        return None


def resolve_turn_id(
    response_item: dict,
    expected_turn_ids: list[int],
    metric_name: str = "",
) -> int | None:
    """Extract and validate turn_id from an LLM judge response item.

    Accepts the ``turn_id`` only when it is present in the response and matches
    one of the *expected_turn_ids*.  Returns ``None`` otherwise — no positional
    fallback, so callers skip items with missing or invalid turn_ids.

    Args:
        response_item: A single item from the judge's response array.
        expected_turn_ids: Ordered list of turn IDs that were sent to the judge.
        metric_name: Name of the calling metric for logging context.

    Returns:
        Validated turn_id, or ``None`` when missing or not in expected list.
    """
    if not isinstance(response_item, dict) or "turn_id" not in response_item:
        return None
    turn_id = response_item["turn_id"]
    if turn_id not in expected_turn_ids:
        prefix = f"[{metric_name}] " if metric_name else ""
        logger.warning(
            f"{prefix}Judge returned turn_id={turn_id} not in expected {expected_turn_ids}, skipping this turn"
        )
        return None
    return turn_id


def validate_rating(rating: Any, valid_range: list[int], default: int, record_id: str, metric_logger) -> int:
    """Validate and clamp rating to valid range.

    Args:
        rating: Raw rating value from LLM
        valid_range: List of valid rating values
        default: Default value if invalid
        record_id: Record ID for logging
        metric_logger: Logger instance from the metric

    Returns:
        Valid rating integer
    """
    if rating not in valid_range:
        metric_logger.warning(f"Invalid rating {rating} for {record_id}, using default {default}")
        return default
    return int(rating)


def normalize_rating(rating: int, min_val: int, max_val: int) -> float:
    """Normalize rating to 0.0-1.0 range.

    Args:
        rating: Rating value
        min_val: Minimum possible rating
        max_val: Maximum possible rating

    Returns:
        Normalized score between 0.0 and 1.0
    """
    if max_val == min_val:
        return 1.0
    return (rating - min_val) / (max_val - min_val)


def extract_wer_errors(output) -> dict[str, list]:
    """Extract specific error examples from jiwer WordOutput.

    Args:
        output: jiwer.WordOutput from jiwer.process_words()

    Returns:
        Dict with 'substitutions', 'deletions', 'insertions' lists
    """
    errors: dict[str, list] = {
        "substitutions": [],
        "deletions": [],
        "insertions": [],
    }

    # Get reference and hypothesis words (handle both string and list formats)
    ref_words = output.references[0]
    hyp_words = output.hypotheses[0]

    if not isinstance(ref_words, list):
        ref_words = ref_words.split()
    if not isinstance(hyp_words, list):
        hyp_words = hyp_words.split()

    # Iterate through alignment chunks
    for chunk in output.alignments[0]:
        if chunk.type == "substitute":
            ref_word = " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx])
            hyp_word = " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx])
            errors["substitutions"].append(
                {
                    "expected": ref_word,
                    "actual": hyp_word,
                }
            )
        elif chunk.type == "delete":
            ref_word = " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx])
            errors["deletions"].append(ref_word)
        elif chunk.type == "insert":
            hyp_word = " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx])
            errors["insertions"].append(hyp_word)

    return errors


def aggregate_wer_errors(output) -> dict[str, Any]:
    """Aggregate error statistics across all turns from jiwer WordOutput.

    Args:
        output: jiwer.WordOutput from jiwer.process_words()

    Returns:
        Dict with top substitutions, deletions, insertions by frequency
    """
    all_errors = extract_wer_errors(output)

    # Count most common substitutions
    sub_counts: dict[str, int] = {}
    for sub in all_errors["substitutions"]:
        key = f"{sub['expected']} → {sub['actual']}"
        sub_counts[key] = sub_counts.get(key, 0) + 1

    top_substitutions = sorted(sub_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Count most common deletions
    del_counts: dict[str, int] = {}
    for word in all_errors["deletions"]:
        del_counts[word] = del_counts.get(word, 0) + 1

    top_deletions = sorted(del_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Count most common insertions
    ins_counts: dict[str, int] = {}
    for word in all_errors["insertions"]:
        ins_counts[word] = ins_counts.get(word, 0) + 1

    top_insertions = sorted(ins_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "top_substitutions": [{"error": error, "count": count} for error, count in top_substitutions],
        "top_deletions": [{"word": word, "count": count} for word, count in top_deletions],
        "top_insertions": [{"word": word, "count": count} for word, count in top_insertions],
    }


def smart_harmonic_mean(scores: list[float]) -> float | None:
    """Calculate the harmonic mean of a list of scores, ignoring None values. Round to 3 decimal places."""
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return None
    return round(harmonic_mean(valid_scores), 3)


def compute_aggregation(aggregation: str, scores: list[int | float | None]) -> float | None:
    """Compute the aggregation of the scores."""
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    if aggregation == "hmean":
        return smart_harmonic_mean(scores)
    elif aggregation == "mean":
        return round(sum(scores) / len(scores), 3)
    elif aggregation == "abs_mean":
        scores = [abs(score) for score in scores]
        return round(sum(scores) / len(scores), 3)
    elif aggregation == "min":
        return min(scores)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def reverse_word_error_rate(wer: float) -> float:
    """Convert WER to accuracy."""
    return 1 - min(1.0, wer)


def make_rate_sub_metric(
    parent_name: str,
    key: str,
    numerator: int,
    denominator: int,
    details: dict[str, Any],
    precision: int = 3,
) -> MetricScore:
    """Build a rate-style sub-metric where ``score == normalized_score == numerator/denominator``.

    Returns a zero-rate sub-metric when ``denominator <= 0`` so callers never
    divide by zero. Callers are responsible for choosing the ``details`` shape
    (counts, turn IDs, reference counts, etc.) that makes sense for their metric.

    Args:
        parent_name: Parent metric's name (used as prefix in sub-metric name).
        key: Sub-metric key suffix (final name is ``f"{parent_name}.{key}"``).
        numerator: Count (e.g., flagged turns, component errors, correct entities).
        denominator: Denominator (e.g., rated turns, reference words, total calls).
        details: Details dict attached to the sub-metric.
        precision: Number of decimal places to round the rate to.

    Returns:
        A MetricScore with equal ``score`` and ``normalized_score`` fields.
    """
    rate = numerator / denominator if denominator > 0 else 0.0
    rounded = round(rate, precision)
    return MetricScore(
        name=f"{parent_name}.{key}",
        score=rounded,
        normalized_score=rounded,
        details=details,
    )


def build_dimension_sub_metrics(
    parent_name: str,
    dimensions: dict[str, Any],
    dimension_keys: tuple[str, ...],
    rating_scale: tuple[int, int],
) -> dict[str, MetricScore]:
    """Build sub-metrics for judge dimensions rated on a scale (e.g., 1-3).

    Each dimension entry is expected to have at minimum a ``rating`` (int) and
    optionally ``flagged`` (bool) and ``evidence`` (str). Only dimensions with
    valid integer ratings in-range are surfaced.

    Args:
        parent_name: The parent metric's name (used as prefix in sub-metric name).
        dimensions: Mapping of dimension key to its judge-response entry.
        dimension_keys: Ordered tuple of expected dimension keys.
        rating_scale: Tuple of (min, max) valid rating values.

    Returns:
        Dict keyed by dimension name with a MetricScore per surfaceable dimension.
    """
    min_r, max_r = rating_scale
    valid_range = set(range(min_r, max_r + 1))
    sub_metrics: dict[str, MetricScore] = {}

    for dim_key in dimension_keys:
        entry = dimensions.get(dim_key)
        if not isinstance(entry, dict):
            continue
        rating = entry.get("rating")
        if not isinstance(rating, int) or rating not in valid_range:
            continue
        sub_metrics[dim_key] = MetricScore(
            name=f"{parent_name}.{dim_key}",
            score=float(rating),
            normalized_score=normalize_rating(rating, min_r, max_r),
            details={
                "rating": rating,
                "flagged": entry.get("flagged"),
                "evidence": entry.get("evidence", ""),
            },
        )

    return sub_metrics


def aggregate_per_turn_scores(scores: list[float | None], aggregation: str) -> float | None:
    """Aggregate per-turn scores using specified method.

    Args:
        scores: List of per-turn scores (may contain None)
        aggregation: Aggregation method (mean, min, max, hmean, abs_mean)

    Returns:
        Aggregated score or None if all scores are None
    """
    return compute_aggregation(aggregation, scores)
