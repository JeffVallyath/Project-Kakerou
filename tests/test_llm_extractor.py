"""Tests for LLM transcript extractor — validation, conversion, replay queue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from llm_extractor import (
    validate_extraction, extraction_to_replay_queue,
    ExtractedTurn, ExtractionResult, _is_junk_speaker,
)


# --- Validation: valid output ---

def test_valid_extraction_passes():
    parsed = {
        "turns": [
            {"speaker_raw": "Alice", "text_full": "Hello there", "timestamp": "2:30 PM", "confidence": 0.95},
            {"speaker_raw": "Bob", "text_full": "Hey!", "timestamp": "2:31 PM", "confidence": 0.90},
        ],
        "unparsed_blocks": [],
        "warnings": [],
    }
    turns, unparsed, warnings = validate_extraction(parsed)
    assert len(turns) == 2
    assert turns[0].speaker_raw == "Alice"
    assert turns[0].text_full == "Hello there"
    assert turns[0].timestamp == "2:30 PM"
    assert turns[1].speaker_raw == "Bob"


def test_valid_long_message_preserved():
    """Full text must survive validation without truncation."""
    long_text = "This is important context. " * 30  # ~810 chars
    parsed = {
        "turns": [{"speaker_raw": "Alice", "text_full": long_text, "confidence": 0.9}],
        "unparsed_blocks": [],
    }
    turns, _, _ = validate_extraction(parsed)
    assert len(turns) == 1
    assert len(turns[0].text_full) > 500
    assert "important context" in turns[0].text_full
    # Preview should be truncated
    assert len(turns[0].text_preview) <= 80


# --- Validation: invalid output ---

def test_invalid_json_format():
    """Non-list turns field should fail."""
    turns, _, warnings = validate_extraction({"turns": "not_a_list"})
    assert len(turns) == 0
    assert any("non-list" in w for w in warnings)


def test_empty_speaker_skipped():
    parsed = {"turns": [{"speaker_raw": "", "text_full": "hello"}]}
    turns, _, warnings = validate_extraction(parsed)
    assert len(turns) == 0
    assert any("empty speaker" in w for w in warnings)


def test_empty_text_skipped():
    parsed = {"turns": [{"speaker_raw": "Alice", "text_full": ""}]}
    turns, _, warnings = validate_extraction(parsed)
    assert len(turns) == 0
    assert any("empty text" in w for w in warnings)


# --- Junk speaker rejection ---

def test_junk_speaker_url():
    assert _is_junk_speaker("https://example.com")


def test_junk_speaker_timestamp():
    assert _is_junk_speaker("2:30 PM")


def test_junk_speaker_ui_label():
    assert _is_junk_speaker("Reply")
    assert _is_junk_speaker("Share")
    assert _is_junk_speaker("View")


def test_junk_speaker_metrics():
    assert _is_junk_speaker("15 likes")


def test_normal_speaker_not_junk():
    assert not _is_junk_speaker("Alice")
    assert not _is_junk_speaker("Jeff Vallyath")
    assert not _is_junk_speaker("Tinyoranges6 [BS]")


def test_junk_speakers_filtered_in_validation():
    parsed = {
        "turns": [
            {"speaker_raw": "Alice", "text_full": "Hello", "confidence": 0.9},
            {"speaker_raw": "https://linkedin.com", "text_full": "profile page", "confidence": 0.5},
            {"speaker_raw": "Reply", "text_full": "some junk", "confidence": 0.3},
            {"speaker_raw": "Bob", "text_full": "Hi", "confidence": 0.9},
        ],
    }
    turns, _, warnings = validate_extraction(parsed)
    assert len(turns) == 2  # Alice and Bob only
    assert turns[0].speaker_raw == "Alice"
    assert turns[1].speaker_raw == "Bob"
    assert any("junk speaker" in w for w in warnings)


# --- Sanity checks ---

def test_too_few_turns_warning():
    parsed = {"turns": []}
    _, _, warnings = validate_extraction(parsed, raw_text_len=500)
    assert any("no valid turns" in w for w in warnings)


def test_single_turn_from_long_input():
    parsed = {"turns": [{"speaker_raw": "Alice", "text_full": "only turn", "confidence": 0.5}]}
    _, _, warnings = validate_extraction(parsed, raw_text_len=1000)
    assert any("only 1 turn" in w for w in warnings)


# --- Replay queue conversion ---

def test_replay_queue_uses_full_text():
    """Replay must use text_full, never truncated preview."""
    long_text = "Important message content. " * 20
    turns = [ExtractedTurn(speaker_raw="Alice", text_full=long_text, text_preview=long_text[:80])]
    role_map = {"Alice": "Target"}
    queue = extraction_to_replay_queue(turns, role_map)

    assert len(queue) == 1
    assert queue[0]["speaker"] == "Target"
    assert len(queue[0]["text"]) > 100  # full text, not preview
    assert queue[0]["text"] == long_text
    assert queue[0]["speaker_raw"] == "Alice"


def test_replay_queue_role_routing():
    turns = [
        ExtractedTurn(speaker_raw="Alice", text_full="hi"),
        ExtractedTurn(speaker_raw="Bob", text_full="hey"),
        ExtractedTurn(speaker_raw="Charlie", text_full="yo"),
    ]
    role_map = {"Alice": "User", "Bob": "Target", "Charlie": "Other"}
    queue = extraction_to_replay_queue(turns, role_map)

    assert queue[0]["speaker"] == "User"
    assert queue[1]["speaker"] == "Target"
    assert queue[2]["speaker"] == "Other"

    # No raw name leaks into speaker field
    valid_roles = {"User", "Target", "Other"}
    for q in queue:
        assert q["speaker"] in valid_roles


def test_replay_queue_default_other():
    """Unmapped speakers should default to Other."""
    turns = [ExtractedTurn(speaker_raw="Unknown", text_full="hello")]
    queue = extraction_to_replay_queue(turns, {})  # empty role map
    assert queue[0]["speaker"] == "Other"


def test_replay_preserves_speaker_raw():
    turns = [ExtractedTurn(speaker_raw="Tinyoranges6 [BS]", text_full="what's up")]
    queue = extraction_to_replay_queue(turns, {"Tinyoranges6 [BS]": "Target"})
    assert queue[0]["speaker_raw"] == "Tinyoranges6 [BS]"
    assert queue[0]["speaker"] == "Target"


# --- ExtractionResult ---

def test_extraction_result_empty():
    result = ExtractionResult()
    assert not result.success
    assert len(result.turns) == 0


# --- Unparsed blocks ---

def test_unparsed_blocks_preserved():
    parsed = {
        "turns": [{"speaker_raw": "Alice", "text_full": "Hello", "confidence": 0.9}],
        "unparsed_blocks": ["Some ambiguous text", "Another unclear block"],
        "warnings": ["could not determine speaker for one block"],
    }
    turns, unparsed, warnings = validate_extraction(parsed)
    assert len(unparsed) == 2
    assert "ambiguous" in unparsed[0]
    assert any("could not determine" in w for w in warnings)


# --- Speaker canonicalization ---

from llm_extractor import canonicalize_speaker


def test_canonicalize_strips_pronouns():
    assert canonicalize_speaker("Amir Fischer (He/Him)") == "Amir Fischer"
    assert canonicalize_speaker("Jane Smith (She/Her)") == "Jane Smith"
    assert canonicalize_speaker("Alex (They/Them)") == "Alex"


def test_canonicalize_strips_timestamp():
    assert canonicalize_speaker("Amir Fischer (He/Him) 9:27 AM") == "Amir Fischer"
    assert canonicalize_speaker("Jeff Vallyath (He/Him) 9") == "Jeff Vallyath"
    assert canonicalize_speaker("Alice 11") == "Alice"


def test_canonicalize_strips_profile_chrome():
    assert canonicalize_speaker("View Amir's profileAmir Fischer") == "Amir Fischer"


def test_canonicalize_strips_sent_message():
    assert canonicalize_speaker("Amir Fischer sent the following message at 11:58 AM") == "Amir Fischer"


def test_canonicalize_preserves_clean_name():
    assert canonicalize_speaker("Jeff Vallyath") == "Jeff Vallyath"
    assert canonicalize_speaker("Alice") == "Alice"


def test_canonicalize_collapses_whitespace():
    assert canonicalize_speaker("  Jeff   Vallyath  ") == "Jeff Vallyath"


def test_junk_speaker_call_with():
    assert _is_junk_speaker("Call with Amir")


def test_junk_speaker_view_profile():
    assert _is_junk_speaker("View Jeff's profile")


def test_junk_speaker_sent_msg():
    assert _is_junk_speaker("sent the following message at 12")


def test_junk_speaker_bare_number():
    assert _is_junk_speaker("42")


def test_canonical_used_for_replay():
    """Replay queue should use speaker_canonical for role mapping."""
    turns = [
        ExtractedTurn(
            speaker_raw="Amir Fischer (He/Him) 9:27 AM",
            speaker_canonical="Amir Fischer",
            text_full="Hello",
        ),
    ]
    role_map = {"Amir Fischer": "Target"}
    queue = extraction_to_replay_queue(turns, role_map)
    assert queue[0]["speaker"] == "Target"
    assert queue[0]["speaker_canonical"] == "Amir Fischer"


def test_canonical_deduplication():
    """Variants of the same speaker should canonicalize to the same name."""
    names = [
        "Amir Fischer (He/Him) 9:27 AM",
        "Amir Fischer (He/Him) 11",
        "Amir Fischer sent the following message at 11:58 AM",
        "View Amir's profileAmir Fischer",
    ]
    canonical = set(canonicalize_speaker(n) for n in names)
    assert len(canonical) == 1
    assert "Amir Fischer" in canonical
