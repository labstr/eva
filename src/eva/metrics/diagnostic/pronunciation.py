"""Pronunciation quality metric using audio + LLM judge (Gemini).

Diagnostic metric for evaluating how naturally and correctly an agent's
speech is pronounced — phoneme quality, stress contrast, connected speech.
"""

from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric


@register_metric
class PronunciationMetric(SpeechFidelityBaseMetric):
    """Audio-based pronunciation quality metric for agent speech using Gemini.

    Evaluates the phonetic naturalness of the agent's spoken audio:
    - Vowel quality (are vowels hitting their phonemic targets?)
    - Stress contrast (are stressed syllables noticeably longer/louder than unstressed?)
    - Consonant articulation (stops, fricatives clearly produced?)
    - Connected speech (natural coarticulation vs clipped, isolated phonemes)
    - Overall delivery (fluid human-like speech vs flat/robotic TTS)

    This is a diagnostic metric — it measures HOW words are spoken, not whether
    the correct words were spoken (that is agent_speech_fidelity's job).

    Rating scale per turn:
        3 — Natural, fluent pronunciation throughout
        2 — Mostly natural with noticeable but minor issues
        1 — Clear pronunciation problems that impair naturalness
    Normalized: (rating - 1) / 2  →  0.0–1.0
    """

    name = "pronunciation"
    description = "Diagnostic metric: audio judge evaluation of agent pronunciation quality per turn"
    category = "diagnostic"
    role = "assistant"
    rating_scale = (1, 3)
    exclude_from_pass_at_k = True
