"""Pronunciation quality metric using audio + LLM judge (Gemini).

Diagnostic metric for evaluating how naturally and correctly an agent's
speech is pronounced, using the same error taxonomy as human annotators.
"""

from typing import Any

from eva.metrics.base import MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import aggregate_per_turn_scores, normalize_rating, resolve_turn_id
from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json

ERROR_DIMENSIONS = [
    "non_standard_accent",
    "bad_stress",
    "bad_sound",
    "bad_date",
    "bad_currency",
    "bad_acronym",
    "other_special_words",
]


@register_metric
class PronunciationMetric(SpeechFidelityBaseMetric):
    """Audio-based pronunciation quality metric for agent speech using Gemini.

    Evaluates pronunciation against the same seven error categories used by
    human annotators:
        non_standard_accent, bad_stress, bad_sound, bad_date,
        bad_currency, bad_acronym, other_special_words

    Each category is captured per turn with a flagged bool and evidence string,
    mirroring the faithfulness/conversation_progression dimension pattern so
    AI judgements can be directly compared to human annotations.

    Rating scale per turn (binary pass/fail — high bar):
        1 — Great: native-quality General American English (human rubric: "Great")
        0 — Not great: one or more noticeable errors in sounds or stress
            (human rubric: "Acceptable" + "Unacceptable" collapsed)
    Normalized: same as raw score (already 0–1).
    """

    name = "pronunciation"
    description = "Diagnostic metric: audio judge evaluation of agent pronunciation quality per turn"
    category = "diagnostic"
    role = "assistant"
    rating_scale = (0, 1)
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute pronunciation score, capturing per-turn error dimensions."""
        try:
            audio_segment = self.load_role_audio(context, self.role)
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant audio file available",
                )

            intended_turns = self._get_intended_turns(context)
            if not intended_turns:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No intended assistant turns available",
                )

            num_turns = len(intended_turns)
            audio_b64 = self.encode_audio_segment(audio_segment)
            intended_turns_formatted = self._format_intended_turns(intended_turns)

            prompt = self.get_judge_prompt(
                prompt_key="user_prompt",
                intended_turns_formatted=intended_turns_formatted,
            )

            messages = self.create_audio_message(audio_b64, prompt)
            response_text, turns = await self._call_and_parse(
                messages, context, audio_segment, prompt
            )

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            if len(turns) != num_turns:
                self.logger.warning(
                    f"[{context.record_id}] Expected {num_turns} pronunciation ratings, got {len(turns)}"
                )

            tts_turn_ids = sorted(intended_turns.keys())
            min_rating, max_rating = self.rating_scale
            valid_ratings_range = list(range(min_rating, max_rating + 1))

            per_turn_ratings: dict[int, int | None] = {}
            per_turn_explanations: dict[int, str] = {}
            per_turn_transcripts: dict[int, str] = {}
            per_turn_normalized: dict[int, float] = {}
            # Per-dimension data: {dim: {turn_id: {"flagged": bool, "evidence": str}}}
            per_turn_errors: dict[int, dict[str, Any]] = {}

            for item in turns:
                turn_id = resolve_turn_id(item, tts_turn_ids, self.name)
                if turn_id is None:
                    continue

                rating = item.get("rating")
                if rating not in valid_ratings_range:
                    self.logger.warning(
                        f"[{context.record_id}] Invalid rating {rating} for turn {turn_id}"
                    )
                    per_turn_ratings[turn_id] = None
                    per_turn_explanations[turn_id] = f"Invalid rating: {rating}"
                    continue

                per_turn_ratings[turn_id] = rating
                per_turn_explanations[turn_id] = item.get("explanation", "")
                per_turn_transcripts[turn_id] = item.get("transcript", "")
                per_turn_normalized[turn_id] = normalize_rating(rating, min_rating, max_rating)

                # Extract structured error dimensions
                errors_raw = item.get("errors") or {}
                per_turn_errors[turn_id] = {
                    dim: {
                        "flagged": bool(errors_raw.get(dim, {}).get("flagged", False)),
                        "evidence": errors_raw.get(dim, {}).get("evidence", ""),
                    }
                    for dim in ERROR_DIMENSIONS
                }

            valid_ratings = [r for r in per_turn_ratings.values() if r is not None]
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0.0
            aggregated_score = aggregate_per_turn_scores(
                list(per_turn_normalized.values()), self.aggregation
            )

            # Build per-dimension flag summaries across all turns
            per_dimension_summary: dict[str, Any] = {}
            for dim in ERROR_DIMENSIONS:
                flagged_turns = [
                    tid
                    for tid, errs in per_turn_errors.items()
                    if errs.get(dim, {}).get("flagged", False)
                ]
                per_dimension_summary[dim] = {
                    "flagged_turn_count": len(flagged_turns),
                    "flagged_turn_ids": flagged_turns,
                }

            details: dict[str, Any] = {
                "aggregation": self.aggregation,
                "num_turns": num_turns,
                "num_evaluated": len(valid_ratings),
                "per_turn_ratings": per_turn_ratings,
                "per_turn_normalized": per_turn_normalized,
                "per_turn_explanations": per_turn_explanations,
                "per_turn_transcripts": per_turn_transcripts,
                "per_turn_errors": per_turn_errors,
                "per_dimension_summary": per_dimension_summary,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3),
                normalized_score=round(aggregated_score, 3) if aggregated_score is not None else 0.0,
                details=details,
                error="Aggregation failed" if aggregated_score is None else None,
            )

        except Exception as e:
            return self._handle_error(e, context)
