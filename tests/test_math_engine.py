"""Tests for the Bayesian Math Engine.

These test the math primitives and the full update cycle using
the current 5-signal schema. They do NOT test LLM sensor behavior
or interaction context — those are covered by the fixture harness.
"""

from btom_engine.math_engine import _bayesian_update, _temporal_decay, update
from btom_engine.schema import ExtractedSignals, SignalReading, StateLedger


def test_supporting_update_increases_probability():
    p = _bayesian_update(p_old=0.3, effective_signal=0.5, learning_rate=0.25, direction=+1)
    assert p > 0.3


def test_contradicting_update_decreases_probability():
    p = _bayesian_update(p_old=0.6, effective_signal=0.5, learning_rate=0.25, direction=-1)
    assert p < 0.6


def test_probability_stays_in_bounds():
    p_high = _bayesian_update(p_old=0.99, effective_signal=1.0, learning_rate=1.0, direction=+1)
    assert 0.0 <= p_high <= 1.0
    p_low = _bayesian_update(p_old=0.01, effective_signal=1.0, learning_rate=1.0, direction=-1)
    assert 0.0 <= p_low <= 1.0


def test_temporal_decay_toward_baseline():
    decayed = _temporal_decay(current_p=0.9, baseline_p=0.5, turns_since=3, decay_factor=0.78)
    assert 0.5 < decayed < 0.9


def test_full_update_cycle():
    """Exercise the full update with the current 5-signal schema.

    Signal mapping:
      fragmentation:  +bluff, +withhold
      defense:        +bluff, -withhold
      emotion:        +bluff
      evasion:        +bluff, +withhold
      compliance:     -bluff, -withhold
    """
    state = StateLedger.new_session()
    state.extracted_signals_current_turn = ExtractedSignals(
        syntactic_fragmentation=SignalReading(value=0.7, signal_reliability=0.85),
        defensive_justification=SignalReading(value=0.6, signal_reliability=0.75),
        emotional_intensity=SignalReading(value=0.4, signal_reliability=0.80),
        evasive_deflection=SignalReading(value=0.5, signal_reliability=0.80),
        direct_answer_compliance=SignalReading(value=0.1, signal_reliability=0.70),
    )
    baselines = {"target_is_bluffing": 0.10, "target_is_withholding_info": 0.40}
    state.current_turn = 1
    update(state, baselines)

    bluff = state.active_hypotheses["target_is_bluffing"].probability
    withhold = state.active_hypotheses["target_is_withholding_info"].probability

    # Bluffing should increase: frag(+), defense(+), emotion(+), evasion(+) all support it
    assert bluff > 0.10, f"bluffing should increase from baseline, got {bluff:.4f}"

    # Withholding: frag(+), evasion(+) support, defense(-) contradicts. Net should be mild.
    # Pre-turn decay pulls toward 0.40. Result should be near or slightly above baseline.
    assert 0.30 < withhold < 0.60, f"withholding should stay moderate, got {withhold:.4f}"


def test_covariance_penalty_triggers_on_frag_plus_emotion():
    """Covariance penalty fires when fragmentation AND emotional_intensity both > 0.7.

    This reduces the learning rate, dampening the update. With penalty active,
    the same signals should produce a smaller probability increase.
    """
    # Case 1: penalty active (frag=0.9, emotion=0.9 — both > 0.7)
    state_penalty = StateLedger.new_session()
    state_penalty.extracted_signals_current_turn = ExtractedSignals(
        syntactic_fragmentation=SignalReading(value=0.9, signal_reliability=0.9),
        emotional_intensity=SignalReading(value=0.9, signal_reliability=0.9),
        defensive_justification=SignalReading(value=0.5, signal_reliability=0.8),
    )
    baselines = {"target_is_bluffing": 0.10, "target_is_withholding_info": 0.40}
    state_penalty.current_turn = 1
    update(state_penalty, baselines)
    bluff_with_penalty = state_penalty.active_hypotheses["target_is_bluffing"].probability

    # Case 2: no penalty (frag=0.9, emotion=0.3 — emotion below threshold)
    state_no_penalty = StateLedger.new_session()
    state_no_penalty.extracted_signals_current_turn = ExtractedSignals(
        syntactic_fragmentation=SignalReading(value=0.9, signal_reliability=0.9),
        emotional_intensity=SignalReading(value=0.3, signal_reliability=0.9),
        defensive_justification=SignalReading(value=0.5, signal_reliability=0.8),
    )
    state_no_penalty.current_turn = 1
    update(state_no_penalty, baselines)
    bluff_no_penalty = state_no_penalty.active_hypotheses["target_is_bluffing"].probability

    # Both should increase from baseline
    assert bluff_with_penalty > 0.10
    assert bluff_no_penalty > 0.10

    # Penalty case should increase LESS (dampened learning rate)
    assert bluff_with_penalty < bluff_no_penalty, (
        f"with penalty ({bluff_with_penalty:.4f}) should be less than "
        f"without ({bluff_no_penalty:.4f})"
    )


def test_direct_answer_compliance_deescalation():
    """High direct_answer_compliance should decrease both hypotheses.

    Compliance maps as -1 for both bluffing and withholding.
    Uses a pure compliance signal (no EMA contamination) to verify
    the contradicting update direction works correctly.
    """
    # Start with elevated bluffing (simulate prior suspicion)
    state = StateLedger.new_session()
    baselines = {"target_is_bluffing": 0.10, "target_is_withholding_info": 0.40}

    # Manually set elevated probability to test de-escalation
    state.active_hypotheses["target_is_bluffing"].probability = 0.35

    # Apply strong compliance signal
    state.extracted_signals_current_turn = ExtractedSignals(
        syntactic_fragmentation=SignalReading(value=0.0, signal_reliability=0.0),
        defensive_justification=SignalReading(value=0.0, signal_reliability=0.0),
        emotional_intensity=SignalReading(value=0.0, signal_reliability=0.0),
        evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        direct_answer_compliance=SignalReading(value=0.9, signal_reliability=0.90),
    )
    state.current_turn = 1
    update(state, baselines)

    bluff = state.active_hypotheses["target_is_bluffing"].probability

    # Pre-turn decay: 0.35 -> 0.10 + (0.35-0.10)*0.95 = 0.3375
    # Then compliance (-1 direction) should pull it down further
    assert bluff < 0.35, f"compliance should reduce bluffing from 0.35, got {bluff:.4f}"
    assert bluff < 0.33, f"compliance effect should be meaningful, got {bluff:.4f}"


def test_neutral_band_filters_weak_signals():
    """Signals with raw value below NEUTRAL_BAND (0.12) should not update hypotheses."""
    state = StateLedger.new_session()
    state.extracted_signals_current_turn = ExtractedSignals(
        syntactic_fragmentation=SignalReading(value=0.08, signal_reliability=0.9),
        defensive_justification=SignalReading(value=0.05, signal_reliability=0.9),
        emotional_intensity=SignalReading(value=0.10, signal_reliability=0.9),
        evasive_deflection=SignalReading(value=0.03, signal_reliability=0.9),
        direct_answer_compliance=SignalReading(value=0.06, signal_reliability=0.9),
    )
    baselines = {"target_is_bluffing": 0.10, "target_is_withholding_info": 0.40}
    state.current_turn = 1
    update(state, baselines)

    # Pre-turn decay pulls slightly toward baseline, but no signal updates should fire
    # Bluffing: 0.10 * 0.95 + 0.10 * 0.05 = 0.10 (no change after decay toward own baseline)
    bluff = state.active_hypotheses["target_is_bluffing"].probability
    withhold = state.active_hypotheses["target_is_withholding_info"].probability

    # Should stay very close to baseline (only pre-turn decay, no signal updates)
    assert abs(bluff - 0.10) < 0.02, f"bluffing should be near baseline, got {bluff:.4f}"
    assert abs(withhold - 0.40) < 0.02, f"withholding should be near baseline, got {withhold:.4f}"
