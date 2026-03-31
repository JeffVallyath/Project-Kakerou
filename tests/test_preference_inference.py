"""Tests for the Preference Inference module (ToM Level 2).

Tests divergence math, regex extraction, entity normalization,
multi-item trade handling, and the full tracker pipeline.
"""

from btom_engine.preference_inference import (
    PreferenceInferenceTracker,
    PreferenceEntry,
    _compute_divergence,
    _compute_signal,
    _normalize_item,
    _extract_claims_regex,
    _extract_actions_regex,
)


# --- Divergence math ---

def test_divergence_aligned():
    """Aligned stated and revealed → divergence near 0."""
    assert abs(_compute_divergence(-1.0, -0.8) - 0.2) < 1e-9
    assert abs(_compute_divergence(1.0, 0.8) - 0.2) < 1e-9


def test_divergence_opposite():
    """Opposite stated and revealed → max divergence."""
    assert _compute_divergence(-1.0, 1.0) == 2.0
    assert _compute_divergence(1.0, -1.0) == 2.0


def test_divergence_partial():
    """Partial mismatch → moderate divergence."""
    div = _compute_divergence(-1.0, 0.3)
    assert 1.0 < div < 2.0


# --- Signal computation ---

def test_signal_no_divergence():
    prefs = {
        "water": PreferenceEntry(
            item="water", stated_value=-1.0, revealed_value=-0.8,
            has_stated=True, has_revealed=True,
        ),
    }
    max_div, signal, rationale = _compute_signal(prefs)
    assert max_div == 0.0  # below 0.5 threshold
    assert signal == 0.0
    assert "no divergence" in rationale.lower()


def test_signal_with_divergence():
    prefs = {
        "water": PreferenceEntry(
            item="water", stated_value=-1.0, revealed_value=0.8,
            stated_turn=1, revealed_turn=3,
            has_stated=True, has_revealed=True,
        ),
    }
    max_div, signal, rationale = _compute_signal(prefs)
    assert max_div == 1.8
    assert 0.0 < signal <= 1.0
    assert "PREFERENCE DIVERGENCE" in rationale


def test_signal_only_stated_no_divergence():
    """Items with only stated (no revealed) should not trigger divergence."""
    prefs = {
        "water": PreferenceEntry(
            item="water", stated_value=-1.0, has_stated=True, has_revealed=False,
        ),
    }
    max_div, signal, _ = _compute_signal(prefs)
    assert max_div == 0.0


# --- Entity normalization ---

def test_normalize_exact_match():
    assert _normalize_item("Water", ["food", "water", "firewood"]) == "water"


def test_normalize_substring_match():
    assert _normalize_item("some firewood", ["food", "water", "firewood"]) == "firewood"


def test_normalize_no_match():
    assert _normalize_item("gold", ["food", "water", "firewood"]) is None


def test_normalize_freeform():
    assert _normalize_item("  Water  ", None) == "water"


def test_normalize_too_short():
    assert _normalize_item("a", ["a", "water"]) is None


# --- Regex extraction ---

def test_regex_claims_positive():
    claims = _extract_claims_regex("I really need water for my camp.")
    items = [c["item"] for c in claims]
    assert any("water" in i for i in items)
    assert all(c["valence"] > 0 for c in claims)


def test_regex_claims_negative():
    claims = _extract_claims_regex("I don't need firewood at all.")
    items = [c["item"] for c in claims]
    assert any("firewood" in i for i in items)
    assert all(c["valence"] < 0 for c in claims)


def test_regex_actions_offer():
    actions = _extract_actions_regex("I'll give you my water.")
    assert len(actions) > 0
    assert actions[0]["action_type"] == "offer"
    assert actions[0]["item_impacts"][0]["implied_value"] < 0


def test_regex_actions_request():
    actions = _extract_actions_regex("Give me the food please.")
    assert len(actions) > 0
    assert actions[0]["action_type"] == "request"
    assert actions[0]["item_impacts"][0]["implied_value"] > 0


# --- Full tracker pipeline (no LLM) ---

def test_tracker_catches_strategic_liar():
    """The core CaSiNo scenario: says don't want X, then demands X."""
    tracker = PreferenceInferenceTracker(use_llm=False, valid_items=["food", "water", "firewood"])

    # Turn 1: claims they don't need water
    tracker.process_turn("I don't need water, you can have it.", turn_number=1)

    # Turn 2: demands water in a trade
    result = tracker.process_turn("Give me the water please.", turn_number=2)

    water_pref = result.preferences.get("water")
    assert water_pref is not None, "Should track 'water'"
    assert water_pref.has_stated, "Should have stated preference"
    assert water_pref.has_revealed, "Should have revealed preference"
    assert water_pref.stated_value < 0, "Stated: doesn't want water"
    assert water_pref.revealed_value > 0, "Revealed: demanding water"
    assert result.max_divergence > 0.5, f"Should detect divergence, got {result.max_divergence}"
    assert result.divergence_signal.value > 0.0, "Signal should be nonzero"


def test_tracker_no_false_positive_on_honest():
    """Honest player: says don't want X, gives away X → aligned."""
    tracker = PreferenceInferenceTracker(use_llm=False, valid_items=["food", "water", "firewood"])

    # Turn 1: claims they don't need water
    tracker.process_turn("I don't need water.", turn_number=1)

    # Turn 2: gives away water (consistent!)
    result = tracker.process_turn("I'll give you my water.", turn_number=2)

    water_pref = result.preferences.get("water")
    if water_pref and water_pref.has_stated and water_pref.has_revealed:
        # Both negative → aligned, divergence should be low
        div = abs(water_pref.stated_value - water_pref.revealed_value)
        assert div < 0.5, f"Honest behavior should not trigger divergence, got {div}"


def test_tracker_latest_overwrites():
    """Later claims overwrite earlier ones."""
    tracker = PreferenceInferenceTracker(use_llm=False)

    tracker.process_turn("I really need water.", turn_number=1)
    tracker.process_turn("I don't need water.", turn_number=3)

    result = tracker._mental_state
    water_pref = result.preferences.get("water")
    assert water_pref is not None
    # The latest claim (T3, negative) should overwrite T1 (positive)
    assert water_pref.stated_value < 0, f"Latest should overwrite: got {water_pref.stated_value}"
    assert water_pref.stated_turn == 3


def test_tracker_reset():
    tracker = PreferenceInferenceTracker(use_llm=False)
    tracker.process_turn("I need water.", turn_number=1)
    assert len(tracker._mental_state.preferences) > 0
    tracker.reset()
    assert len(tracker._mental_state.preferences) == 0
    assert tracker._turn_count == 0


def test_signal_reading_bounds():
    """Output SignalReading must be in [0, 1]."""
    tracker = PreferenceInferenceTracker(use_llm=False)
    # Create extreme divergence
    tracker.process_turn("I don't want food.", turn_number=1)
    result = tracker.process_turn("Give me the food now!", turn_number=2)
    assert 0.0 <= result.divergence_signal.value <= 1.0
    assert 0.0 <= result.divergence_signal.signal_reliability <= 1.0
