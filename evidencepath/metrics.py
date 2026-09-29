"""Question-answering, support-coverage, and statistical evaluation metrics."""

from __future__ import annotations

from collections import Counter
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .types import QuestionExample, SentenceUnit


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(value: str) -> str:
    """Apply the normalization used for EM and token-level F1."""

    value = str(value).casefold()
    value = _PUNCT_RE.sub(" ", value)
    value = _ARTICLE_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def answer_exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return max((float(normalized == normalize_answer(reference)) for reference in references), default=0.0)


def answer_f1(prediction: str, references: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not references:
        return 0.0
    best = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not prediction_tokens or not reference_tokens:
            score = float(prediction_tokens == reference_tokens)
            best = max(best, score)
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def support_recall(support_units: Iterable[str], retained_units: Iterable[str]) -> float:
    support = set(support_units)
    retained = set(retained_units)
    if not support:
        return 0.0
    return len(support & retained) / len(support)


def complete_coverage(support_units: Iterable[str], retained_units: Iterable[str]) -> float:
    support = set(support_units)
    return float(support <= set(retained_units)) if support else 0.0


def retained_units_for_level(
    sentence_ids: Iterable[str],
    paragraph_ids: Iterable[str],
    *,
    support_level: str,
) -> tuple[str, ...]:
    """Select sentence- or paragraph-level retained IDs for a question."""

    if support_level.casefold().startswith("paragraph"):
        return tuple(dict.fromkeys(paragraph_ids))
    return tuple(dict.fromkeys(sentence_ids))


def coverage_for_question(
    example: QuestionExample,
    *,
    retained_sentence_ids: Iterable[str],
    retained_paragraph_ids: Iterable[str] = (),
) -> tuple[float, float]:
    retained = retained_units_for_level(
        retained_sentence_ids,
        retained_paragraph_ids,
        support_level=example.support_level,
    )
    return (
        support_recall(example.support_units, retained),
        complete_coverage(example.support_units, retained),
    )


def preassembly_units(
    source_ids: Iterable[str],
    source_units: Mapping[str, SentenceUnit],
    *,
    support_level: str,
) -> tuple[str, ...]:
    """Map selected source sentences to the coverage unit required by a dataset."""

    if support_level.casefold().startswith("paragraph"):
        return tuple(
            dict.fromkeys(
                source_units[source_id].paragraph_unit_id
                for source_id in source_ids
                if source_id in source_units
            )
        )
    return tuple(dict.fromkeys(source_ids))


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Compute the paired bootstrap CI and two-sided sign p-value."""

    if len(scores_a) != len(scores_b):
        raise ValueError("Paired bootstrap inputs must have the same length")
    if not scores_a:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    differences = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    bootstrap_means_parts: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch_size = min(256, remaining)
        indices = rng.integers(0, len(differences), size=(batch_size, len(differences)))
        bootstrap_means_parts.append(differences[indices].mean(axis=1))
        remaining -= batch_size
    bootstrap_means = np.concatenate(bootstrap_means_parts)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    non_positive = float(np.mean(bootstrap_means <= 0.0))
    non_negative = float(np.mean(bootstrap_means >= 0.0))
    p_value = min(1.0, 2.0 * min(non_positive, non_negative))
    return {
        "n": len(differences),
        "delta": float(differences.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Holm's step-down family-wise correction."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def aggregate_prediction_rows(rows: Sequence[Mapping]) -> dict[str, float | int]:
    """Aggregate per-question result records into a table row."""

    if not rows:
        return {"n": 0, "em": 0.0, "f1": 0.0, "tokens": 0.0, "support_recall": 0.0, "complete_coverage": 0.0}
    keys = {
        "em": "em",
        "f1": "f1",
        "context_tokens": "tokens",
        "support_recall": "support_recall",
        "pre_support_recall": "pre_support_recall",
        "complete_coverage": "complete_coverage",
        "terminal_coverage": "terminal_coverage",
        "selected_intermediates": "intermediates",
        "retrieval_seconds": "retrieval_seconds",
        "generation_seconds": "generation_seconds",
        "end_to_end_seconds": "end_to_end_seconds",
        "peak_gpu_memory_gb": "peak_gpu_memory_gb",
    }
    result: dict[str, float | int] = {"n": len(rows)}
    for source_key, output_key in keys.items():
        result[output_key] = float(np.mean([float(row.get(source_key, 0.0)) for row in rows]))
    return result
