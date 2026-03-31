"""Tests for transcript parser — parsing, role mapping, quality scoring, and replay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from transcript_parser import (
    parse_transcript, to_replay_queue, ParsedTurn, ParseQuality,
    _normalize_alias, _normalize_for_matching, _assess_quality, ParseResult,
    _llm_output_to_result,
)


# --- Basic inline parsing ---

def test_simple_colon_format():
    raw = "Alice: Hello there\nBob: Hi Alice"
    result = parse_transcript(raw)
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Alice"
    assert result.turns[0].text == "Hello there"
    assert result.turns[1].speaker_raw == "Bob"


def test_dash_format():
    raw = "Alice - Hello\nBob - Hi"
    result = parse_transcript(raw)
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Alice"


def test_emdash_format():
    raw = "Alice \u2014 Hello\nBob \u2014 Hi"
    result = parse_transcript(raw)
    assert len(result.turns) == 2


def test_timestamp_inline_format():
    raw = "[10:30] Alice: Hello\n[10:31] Bob: Hi"
    result = parse_transcript(raw)
    assert len(result.turns) == 2
    assert result.turns[0].timestamp == "10:30"
    assert result.turns[0].speaker_raw == "Alice"


def test_blank_lines_ignored():
    raw = "Alice: Hello\n\n\nBob: Hi\n\n"
    result = parse_transcript(raw)
    assert len(result.turns) == 2


def test_multiline_continuation():
    raw = "Alice: This is a long message\nthat continues on the next line\nBob: Short reply"
    result = parse_transcript(raw)
    assert len(result.turns) == 2
    assert "continues" in result.turns[0].text
    assert result.turns[1].speaker_raw == "Bob"


# --- Header/body format ---

def test_header_body_format():
    raw = "Alice \u2014 2:53 PM\nHey how are you?\nBob \u2014 2:56 PM\nI'm good thanks"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Alice"
    assert result.turns[0].speaker_role == "User"
    assert "how are you" in result.turns[0].text
    assert result.turns[1].speaker_raw == "Bob"
    assert result.turns[1].speaker_role == "Target"


def test_header_body_with_dash():
    raw = "Alice - 10:30 AM\nHello\nBob - 10:31 AM\nHi"
    result = parse_transcript(raw)
    assert len(result.turns) == 2
    assert result.turns[0].timestamp is not None


# --- Alias normalization ---

def test_alias_mapping_user_target():
    raw = "Alice: Hello\nBob: Hi\nAlice: How are you?"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    assert result.turns[0].speaker_role == "User"
    assert result.turns[1].speaker_role == "Target"
    assert result.turns[2].speaker_role == "User"
    assert result.user_count == 2
    assert result.target_count == 1


def test_unknown_speaker_is_other():
    raw = "Alice: Hello\nBob: Hi\nCharlie: Hey guys"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    assert result.turns[2].speaker_role == "Other"
    assert result.other_count == 1


def test_case_insensitive_alias():
    raw = "ALICE: Hello\nalice: Hi again"
    result = parse_transcript(raw, user_aliases=["Alice"])
    assert result.turns[0].speaker_role == "User"
    assert result.turns[1].speaker_role == "User"


def test_multiple_aliases():
    raw = "jeff: Hello\nJeff V: Hi\nJeffrey: Hey"
    result = parse_transcript(raw, target_aliases=["jeff", "Jeff V", "Jeffrey"])
    assert all(t.speaker_role == "Target" for t in result.turns)


def test_bracket_tag_alias_matching():
    """Tinyoranges6[BS] should match alias 'Tinyoranges6'."""
    raw = "Tinyoranges6[BS]: when you tryna hop"
    result = parse_transcript(raw, target_aliases=["Tinyoranges6"])
    assert result.turns[0].speaker_role == "Target"


def test_bracket_with_space():
    raw = "Tinyoranges6 [BS]: when you tryna hop"
    result = parse_transcript(raw, target_aliases=["Tinyoranges6"])
    assert result.turns[0].speaker_role == "Target"


def test_no_aliases_all_other():
    raw = "Alice: Hello\nBob: Hi"
    result = parse_transcript(raw)
    assert all(t.speaker_role == "Other" for t in result.turns)


# --- Parse quality scoring ---

def test_quality_good_parse():
    result = ParseResult()
    result.turns = [ParsedTurn(speaker_raw="A"), ParsedTurn(speaker_raw="B")]
    result.raw_speakers = ["A", "B"]
    q = _assess_quality(result, 4)
    assert q.score >= 0.8
    assert not q.needs_repair


def test_quality_high_drop_rate():
    result = ParseResult()
    result.turns = [ParsedTurn(speaker_raw="A")]
    result.dropped_lines = ["L1: x", "L2: y", "L3: z"]
    q = _assess_quality(result, 5)
    assert q.score < 0.7
    assert any("drop_rate" in w for w in q.warnings)


def test_quality_single_speaker_long():
    result = ParseResult()
    result.turns = [ParsedTurn(speaker_raw="A")] * 5
    result.raw_speakers = ["A"]
    q = _assess_quality(result, 10)
    assert any("single_speaker" in w for w in q.warnings)


def test_quality_no_turns():
    result = ParseResult()
    q = _assess_quality(result, 5)
    assert q.score < 0.5
    assert q.needs_repair


def test_quality_long_message():
    result = ParseResult()
    result.turns = [ParsedTurn(speaker_raw="A", text="x" * 600)]
    result.raw_speakers = ["A"]
    q = _assess_quality(result, 2)
    assert any("long_message" in w for w in q.warnings)


# --- LLM output validation ---

def test_llm_output_validation_good():
    llm_output = {
        "turns": [
            {"speaker_raw": "Alice", "text": "Hello", "confidence": 0.9},
            {"speaker_raw": "Bob", "text": "Hi", "confidence": 0.95},
        ],
        "unparsed_lines": [],
    }
    result = _llm_output_to_result(llm_output, {"alice"}, {"bob"}, {"alice"}, {"bob"})
    assert len(result.turns) == 2
    assert result.turns[0].speaker_role == "User"
    assert result.turns[1].speaker_role == "Target"
    assert result.parse_source == "llm_repair"


def test_llm_output_validation_bad_format():
    result = _llm_output_to_result({"turns": "not_a_list"}, set(), set(), set(), set())
    assert len(result.turns) == 0
    assert result.quality.score == 0.0


def test_llm_preserves_exact_text():
    llm_output = {
        "turns": [{"speaker_raw": "A", "text": "lol bruh whattt", "confidence": 0.9}],
        "unparsed_lines": [],
    }
    result = _llm_output_to_result(llm_output, set(), set(), set(), set())
    assert result.turns[0].text == "lol bruh whattt"


# --- Replay queue ---

def test_replay_queue_format():
    raw = "Alice: Hello\nBob: Hi"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    queue = to_replay_queue(result)
    assert len(queue) == 2
    assert queue[0]["speaker"] == "User"
    assert queue[0]["text"] == "Hello"
    assert queue[0]["speaker_raw"] == "Alice"
    assert queue[1]["speaker"] == "Target"


def test_other_turns_in_replay():
    raw = "Alice: Hello\nCharlie: Hey\nBob: Hi"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    queue = to_replay_queue(result)
    assert queue[1]["speaker"] == "Other"
    assert queue[1]["speaker_raw"] == "Charlie"


def test_turn_order_preserved():
    raw = "A: 1\nB: 2\nA: 3\nC: 4\nB: 5"
    result = parse_transcript(raw, user_aliases=["A"], target_aliases=["B"])
    texts = [t.text for t in result.turns]
    assert texts == ["1", "2", "3", "4", "5"]


# --- Edge cases ---

def test_dropped_lines_surfaced():
    raw = "This is not a valid turn\nAlice: Hello"
    result = parse_transcript(raw, user_aliases=["Alice"])
    assert len(result.turns) == 1
    assert len(result.dropped_lines) == 1
    assert "L1" in result.dropped_lines[0]


def test_empty_input():
    result = parse_transcript("")
    assert len(result.turns) == 0
    assert result.user_count == 0


def test_raw_speakers_list():
    raw = "Alice: Hi\nBob: Hey\nAlice: Yo"
    result = parse_transcript(raw)
    assert "Alice" in result.raw_speakers
    assert "Bob" in result.raw_speakers


# --- Alias normalization unit tests ---

def test_normalize_alias_basic():
    assert _normalize_alias("  Alice  ") == "alice"
    assert _normalize_alias("ALICE") == "alice"
    assert _normalize_alias("Alice,") == "alice"


def test_normalize_for_matching_brackets():
    assert _normalize_for_matching("Tinyoranges6[BS]") == "tinyoranges6"
    assert _normalize_for_matching("Tinyoranges6 [BS]") == "tinyoranges6"
    assert _normalize_for_matching("Admin (mod)") == "admin"


# --- Replay routing contract tests ---

def test_replay_queue_target_has_normalized_role():
    """Target turns in replay queue must have speaker='Target', not raw name."""
    raw = "Tinyoranges6 [BS]: when you tryna hop\nEl Jefe: Like 9"
    result = parse_transcript(raw, user_aliases=["El Jefe"], target_aliases=["Tinyoranges6"])
    queue = to_replay_queue(result)

    target_entries = [q for q in queue if q["speaker"] == "Target"]
    user_entries = [q for q in queue if q["speaker"] == "User"]

    assert len(target_entries) == 1, f"Expected 1 Target turn, got {len(target_entries)}"
    assert len(user_entries) == 1, f"Expected 1 User turn, got {len(user_entries)}"

    # speaker must be the ROLE, not the raw name
    assert target_entries[0]["speaker"] == "Target"
    assert target_entries[0]["speaker_raw"] == "Tinyoranges6 [BS]"
    assert user_entries[0]["speaker"] == "User"
    assert user_entries[0]["speaker_raw"] == "El Jefe"


def test_replay_queue_other_has_normalized_role():
    """Other turns must have speaker='Other', not raw name."""
    raw = "Alice: Hi\nBystander: Hey\nBob: Hello"
    result = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    queue = to_replay_queue(result)

    other_entries = [q for q in queue if q["speaker"] == "Other"]
    assert len(other_entries) == 1
    assert other_entries[0]["speaker"] == "Other"
    assert other_entries[0]["speaker_raw"] == "Bystander"


def test_replay_queue_no_raw_name_leaks_into_speaker():
    """The speaker field must never contain a raw name — only User/Target/Other."""
    raw = "Tinyoranges6 [BS]: msg1\nEl Jefe: msg2\nRandom: msg3"
    result = parse_transcript(raw, user_aliases=["El Jefe"], target_aliases=["Tinyoranges6"])
    queue = to_replay_queue(result)

    valid_roles = {"User", "Target", "Other"}
    for q in queue:
        assert q["speaker"] in valid_roles, (
            f"speaker field '{q['speaker']}' is not a valid role. "
            f"Raw name '{q['speaker_raw']}' may have leaked."
        )


def test_header_body_replay_routing():
    """Header/body format should also produce correct replay roles."""
    raw = "Tinyoranges6 [BS] \u2014 2:53 PM\nwhen you tryna hop\nEl Jefe \u2014 2:56 PM\nLike 9"
    result = parse_transcript(raw, user_aliases=["El Jefe"], target_aliases=["Tinyoranges6"])
    queue = to_replay_queue(result)

    assert len(queue) == 2
    assert queue[0]["speaker"] == "Target"
    assert queue[1]["speaker"] == "User"
