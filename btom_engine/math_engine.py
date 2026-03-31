"""Bayesian Math Engine (Layer 3).

Implements the core update function, temporal decay, covariance penalty,
EMA smoothing, neutral band, diminishing returns, and pre-turn decay.

All parameters are read from the weights system (EngineWeights).
The update() function accepts an optional weights parameter for per-trial
optimization via Optuna.
"""

from __future__ import annotations

from btom_engine.weights import WEIGHTS as _DEFAULT_WEIGHTS, EngineWeights
from btom_engine.schema import ExtractedSignals, Hypothesis, SignalReading, StateLedger


# --- Signal-to-hypothesis mapping (loaded from weights) ---
SIGNAL_HYPOTHESIS_MAP = _DEFAULT_WEIGHTS.get_signal_hypothesis_map()


def _effective_learning_rate(signals: ExtractedSignals, w: EngineWeights) -> float:
    """Apply covariance penalty if fragmentation and emotional_intensity both > threshold."""
    frag = signals.syntactic_fragmentation.value
    emo = signals.emotional_intensity.value
    lr = w.learning_rate
    if frag > w.covariance_threshold and emo > w.covariance_threshold:
        lr *= w.covariance_penalty
    return lr


def _bayesian_update(
    p_old: float,
    effective_signal: float,
    learning_rate: float,
    direction: int,
) -> float:
    """Single-step Bayesian update per spec."""
    if direction > 0:  # supporting
        p_new = p_old + learning_rate * effective_signal * (1 - p_old)
    else:  # contradicting
        p_new = p_old - learning_rate * effective_signal * p_old
    return max(0.0, min(1.0, p_new))


def _temporal_decay(
    current_p: float,
    baseline_p: float,
    turns_since: int,
    decay_factor: float,
) -> float:
    """Decay probability toward baseline over idle turns."""
    return baseline_p + (current_p - baseline_p) * (decay_factor ** turns_since)


def _smooth_signals(
    current: ExtractedSignals,
    previous: ExtractedSignals,
    alpha: float,
) -> ExtractedSignals:
    """EMA-smooth current signals with previous turn's values."""
    def _blend(cur: SignalReading, prev: SignalReading) -> SignalReading:
        return SignalReading(
            value=alpha * cur.value + (1 - alpha) * prev.value,
            signal_reliability=max(cur.signal_reliability, prev.signal_reliability),
        )

    return ExtractedSignals(
        syntactic_fragmentation=_blend(
            current.syntactic_fragmentation, previous.syntactic_fragmentation
        ),
        defensive_justification=_blend(
            current.defensive_justification, previous.defensive_justification
        ),
        emotional_intensity=_blend(
            current.emotional_intensity, previous.emotional_intensity
        ),
        evasive_deflection=_blend(
            current.evasive_deflection, previous.evasive_deflection
        ),
        direct_answer_compliance=_blend(
            current.direct_answer_compliance, previous.direct_answer_compliance
        ),
    )


def update(
    state: StateLedger,
    baselines: dict[str, float],
    weights: EngineWeights | None = None,
) -> StateLedger:
    """Run one full Bayesian update cycle on the state ledger.

    Args:
        state: current state ledger (mutated in place)
        baselines: hypothesis baseline probabilities
        weights: optional per-trial weights (defaults to global WEIGHTS singleton)

    Mutates and returns *state* for convenience.
    """
    w = weights or _DEFAULT_WEIGHTS

    raw_signals = state.extracted_signals_current_turn

    # --- EMA smoothing ---
    if state.current_turn > 1 and hasattr(state, "_prev_signals"):
        signals = _smooth_signals(raw_signals, state._prev_signals, w.ema_smoothing_alpha)
        state.extracted_signals_current_turn = signals
    else:
        signals = raw_signals
    state._prev_signals = raw_signals  # type: ignore[attr-defined]

    # --- Pre-turn decay: drift all hypotheses toward baseline BEFORE evidence ---
    for hyp_name, hyp in state.active_hypotheses.items():
        baseline = baselines.get(hyp_name, 0.5)
        hyp.probability = baseline + (hyp.probability - baseline) * w.pre_turn_decay

    lr = _effective_learning_rate(signals, w)

    # Use per-trial signal map if weights differ from default
    if weights is not None:
        sig_hyp_map = weights.get_signal_hypothesis_map()
    else:
        sig_hyp_map = SIGNAL_HYPOTHESIS_MAP

    signal_fields = {
        "syntactic_fragmentation": signals.syntactic_fragmentation,
        "defensive_justification": signals.defensive_justification,
        "emotional_intensity": signals.emotional_intensity,
        "evasive_deflection": signals.evasive_deflection,
        "direct_answer_compliance": signals.direct_answer_compliance,
    }

    # Track which hypotheses got meaningful evidence this turn
    hypotheses_with_evidence: set[str] = set()

    for signal_name, reading in signal_fields.items():
        effective_signal = reading.value * reading.signal_reliability

        # Dead-zone: ignore noise-level signals
        if effective_signal < w.signal_dead_zone:
            continue

        # Neutral band: signals below threshold are treated as noise
        if reading.value < w.neutral_band:
            continue

        mapping = sig_hyp_map.get(signal_name, {})
        for hyp_name, direction in mapping.items():
            hyp = state.active_hypotheses.get(hyp_name)
            if hyp is None:
                continue

            # Diminishing returns: dampen weak supporting signals at high P
            adjusted_signal = effective_signal
            if (
                direction > 0
                and hyp.probability > w.diminishing_returns_threshold
                and effective_signal < w.diminishing_weak_signal_cap
            ):
                dampening = 1.0 - (hyp.probability - w.diminishing_returns_threshold) / (1.0 - w.diminishing_returns_threshold)
                adjusted_signal = effective_signal * max(0.05, dampening)

            old_p = hyp.probability
            new_p = _bayesian_update(old_p, adjusted_signal, lr, direction)
            hyp.probability = new_p
            hyp.momentum = new_p - old_p
            hypotheses_with_evidence.add(hyp_name)

            tag = "+" if direction > 0 else "-"
            hyp.evidence_trace.append(
                f"T{state.current_turn}: {signal_name}={reading.value:.2f} "
                f"[{tag}] -> P={new_p:.4f}"
            )

    # --- Temporal decay (idle hypotheses only) ---
    for hyp_name, hyp in state.active_hypotheses.items():
        if hyp_name in hypotheses_with_evidence:
            state.turns_since_last_signal[hyp_name] = 0
        else:
            turns = state.turns_since_last_signal.get(hyp_name, 0) + 1
            state.turns_since_last_signal[hyp_name] = turns
            baseline = baselines.get(hyp_name, 0.5)
            old_p = hyp.probability
            hyp.probability = _temporal_decay(old_p, baseline, turns, w.temporal_decay)
            hyp.momentum = 0.0

    state.check_system_status()
    return state
