"""Baseline Deviation Scorer — detects behavioral shifts within a conversation.

Instead of comparing against population averages, this builds a baseline
from the target's OWN early messages and flags deviations in later messages.

This catches what static word-category analysis misses:
- A normally verbose person who suddenly gives short answers
- A casual texter who suddenly becomes formal
- Someone who uses "I" frequently but drops pronouns on a specific topic
- Consistent sentence length that spikes when rationalizing

The baseline is adaptive — it updates as more turns are processed,
using a rolling window with exponential decay.

Tunable parameters (optimizable by Optuna):
- Calibration window size (how many turns to establish baseline)
- Deviation threshold (how many std devs before flagging)
- Per-metric weights (which deviations matter most for deception)
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Import the text analysis function
from btom_engine.osint.behavioral_baseline import _analyze_text


@dataclass
class TurnMetrics:
    """Raw metrics for a single turn."""
    word_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    unique_ratio: float = 0.0
    exclamations: int = 0
    questions: int = 0
    self_ref_rate: float = 0.0
    cognitive_ratio: float = 0.0
    certainty_ratio: float = 0.0
    concrete_ratio: float = 0.0
    filler_ratio: float = 0.0
    hedging: int = 0


@dataclass
class DeviationResult:
    """Result of comparing a turn against the running baseline."""
    has_baseline: bool = False
    turns_in_baseline: int = 0
    total_deviation_score: float = 0.0    # composite anomaly (0-1)
    bluff_delta: float = 0.0              # suggested hypothesis adjustment
    significant_deviations: list[str] = field(default_factory=list)
    rationale: str = ""


class ConversationBaseline:
    """Maintains a rolling baseline of the target's behavior within a conversation.

    Builds up during early turns, then flags deviations in later turns.
    """

    def __init__(
        self,
        calibration_window: int = 5,
        deviation_threshold: float = 1.5,
    ):
        self.calibration_window = calibration_window
        self.deviation_threshold = deviation_threshold
        self._turn_history: list[TurnMetrics] = []
        self._seed_count: int = 0  # tracks how many entries are from seeding
        self._metric_names = [
            "word_count", "avg_word_length", "avg_sentence_length",
            "unique_ratio", "self_ref_rate", "cognitive_ratio",
            "certainty_ratio", "concrete_ratio", "filler_ratio",
        ]

    def _extract_turn_metrics(self, text: str) -> TurnMetrics:
        """Extract metrics from a single turn's text."""
        raw = _analyze_text(text)
        if not raw:
            return TurnMetrics()

        return TurnMetrics(
            word_count=raw.get("word_count", 0),
            avg_word_length=raw.get("avg_word_length", 0),
            avg_sentence_length=raw.get("avg_sentence_length", 0),
            unique_ratio=raw.get("unique_ratio", 0),
            exclamations=raw.get("exclamations", 0),
            questions=raw.get("questions", 0),
            self_ref_rate=raw.get("self_ref_rate", 0),
            cognitive_ratio=raw.get("cognitive_ratio", 0),
            certainty_ratio=raw.get("certainty_ratio", 0),
            concrete_ratio=raw.get("concrete_ratio", 0),
            filler_ratio=raw.get("filler_ratio", 0),
            hedging=raw.get("hedging", 0),
        )

    def _get_baseline_stats(self) -> dict[str, tuple[float, float]]:
        """Compute mean and std for each metric from the baseline window."""
        if len(self._turn_history) < 2:
            return {}

        window = self._turn_history[:self.calibration_window]
        if len(window) < 2:
            window = self._turn_history

        stats = {}
        for metric in self._metric_names:
            values = [getattr(t, metric, 0) for t in window]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.1  # min std to avoid div/0
            stats[metric] = (mean, std)

        return stats

    def process_turn(self, text: str, weights: "EngineWeights | None" = None) -> DeviationResult:
        """Process a new turn and return deviation analysis.

        During calibration phase (first N turns): builds baseline, returns no deviations.
        After calibration: compares new turn against baseline, returns deviations.
        """
        metrics = self._extract_turn_metrics(text)
        self._turn_history.append(metrics)

        result = DeviationResult(
            turns_in_baseline=min(len(self._turn_history), self.calibration_window),
        )

        # Still in calibration phase
        if len(self._turn_history) <= self.calibration_window:
            result.has_baseline = False
            result.rationale = f"Calibrating ({len(self._turn_history)}/{self.calibration_window})"
            return result

        result.has_baseline = True

        # Compute deviations
        stats = self._get_baseline_stats()
        if not stats:
            return result

        # Load deviation weights
        try:
            from btom_engine.weights import WEIGHTS
            w = weights or WEIGHTS
            dev_weights = {
                "word_count": getattr(w, 'dev_word_count_weight', 0.08),
                "avg_word_length": getattr(w, 'dev_word_length_weight', 0.05),
                "avg_sentence_length": getattr(w, 'dev_sentence_length_weight', 0.10),
                "unique_ratio": getattr(w, 'dev_unique_ratio_weight', 0.05),
                "self_ref_rate": getattr(w, 'dev_self_ref_weight', 0.15),
                "cognitive_ratio": getattr(w, 'dev_cognitive_weight', 0.15),
                "certainty_ratio": getattr(w, 'dev_certainty_weight', 0.12),
                "concrete_ratio": getattr(w, 'dev_concrete_weight', 0.12),
                "filler_ratio": getattr(w, 'dev_filler_weight', 0.08),
            }
        except Exception:
            dev_weights = {m: 0.10 for m in self._metric_names}

        total_z = 0.0
        n_significant = 0
        rationale_parts = []

        for metric in self._metric_names:
            if metric not in stats:
                continue

            mean, std = stats[metric]
            current = getattr(metrics, metric, 0)

            # Z-score: how many standard deviations from baseline
            z = (current - mean) / std if std > 0 else 0.0

            weight = dev_weights.get(metric, 0.10)

            if abs(z) > self.deviation_threshold:
                n_significant += 1
                direction = "spike" if z > 0 else "drop"
                result.significant_deviations.append(
                    f"{metric}: {direction} (z={z:.1f}, baseline={mean:.2f}, current={current:.2f})"
                )

                # Weighted contribution to bluff delta
                # Positive z in cognitive/certainty/filler = suspicious
                # Negative z in self_ref/concrete = suspicious (distancing, abstraction)
                if metric in ("cognitive_ratio", "certainty_ratio", "filler_ratio"):
                    total_z += max(0, z) * weight  # only positive deviations matter
                elif metric in ("self_ref_rate", "concrete_ratio"):
                    total_z += max(0, -z) * weight  # only negative deviations (drops) matter
                elif metric in ("avg_sentence_length", "word_count"):
                    total_z += abs(z) * weight * 0.5  # any direction matters, but less
                else:
                    total_z += abs(z) * weight * 0.3

        # Normalize to 0-1 range
        result.total_deviation_score = min(1.0, total_z / 3.0)

        # Convert to bluff delta (scaled by integration weight)
        try:
            from btom_engine.weights import WEIGHTS
            dev_integration = getattr(WEIGHTS, 'dev_integration_weight', 0.5)
        except Exception:
            dev_integration = 0.5

        result.bluff_delta = result.total_deviation_score * 0.3 * dev_integration

        if result.significant_deviations:
            result.rationale = (
                f"{n_significant} significant deviation(s): "
                + "; ".join(result.significant_deviations[:3])
            )
        else:
            result.rationale = "Within baseline range"

        return result

    def seed_from_texts(self, texts: list[str]) -> int:
        """Pre-seed the baseline with known truthful text samples.

        REPLACES any existing seed data (conversation turns are preserved).
        Sets the calibration window to match the seed count so the baseline
        is immediately ready after seeding.

        Returns the number of samples successfully added.
        """
        seed_metrics = []
        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            metrics = self._extract_turn_metrics(text)
            if metrics.word_count > 0:
                seed_metrics.append(metrics)

        seeded = len(seed_metrics)
        if seeded == 0:
            return 0

        # Separate any existing conversation turns (non-seed) from history
        # Conversation turns are anything beyond the current calibration window
        conv_turns = self._turn_history[self._seed_count:]

        # Replace: seed metrics + existing conversation turns
        self._turn_history = seed_metrics + conv_turns
        self._seed_count = seeded
        self.calibration_window = max(seeded, 3)  # at least 3 for stats

        logger.info("Baseline seeded with %d samples (window=%d)", seeded, self.calibration_window)
        return seeded

    def add_baseline_turn(self, text: str) -> bool:
        """Add a single turn to the baseline calibration pool.

        Used by "mark as baseline" — inserts the turn's metrics into the
        seed region and expands the window by 1.
        Returns True if successfully added.
        """
        if not text or len(text.strip()) < 10:
            return False
        metrics = self._extract_turn_metrics(text)
        if metrics.word_count == 0:
            return False

        # Insert at end of seed region
        self._turn_history.insert(self._seed_count, metrics)
        self._seed_count += 1
        self.calibration_window = max(self._seed_count, self.calibration_window)
        return True

    def recompute_all(self, texts: list[str], weights: "EngineWeights | None" = None) -> list[DeviationResult]:
        """Retroactively recompute deviation scores for all conversation turns.

        Called after baseline is modified (seed or mark-as-baseline) to
        refresh historical scores. Only recomputes turns AFTER the baseline
        window — baseline turns always return has_baseline=False.

        Args:
            texts: all conversation turn texts in order (excluding seed texts)
            weights: optional engine weights

        Returns:
            list of DeviationResult, one per turn
        """
        # Preserve seed metrics (front of history up to original calibration window)
        seed_count = self._seed_count
        seed_metrics = self._turn_history[:seed_count]

        # Rebuild from scratch with seeds intact
        self._turn_history = list(seed_metrics)

        results = []
        for text in texts:
            result = self.process_turn(text, weights)
            results.append(result)

        return results

    @property
    def baseline_ready(self) -> bool:
        """Whether the baseline has enough data to start deviation detection."""
        # Ready if: seeded with 3+ samples, OR conversation has passed calibration window
        if self._seed_count >= 3:
            return True
        return len(self._turn_history) > self.calibration_window

    @property
    def baseline_sample_count(self) -> int:
        """Number of samples in the baseline calibration pool."""
        return min(len(self._turn_history), self.calibration_window)

    def reset(self) -> None:
        """Reset the baseline (e.g., new conversation)."""
        self._turn_history.clear()
        self._seed_count = 0
