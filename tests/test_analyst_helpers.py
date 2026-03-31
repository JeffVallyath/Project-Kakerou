"""Tests for analyst helpers — speaker remap, ledger, OSINT trace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from analyst_helpers import (
    build_speaker_mappings, apply_remap,
    build_ledger_entry, build_osint_trace,
    LedgerEntry, OsintTrace,
)


# --- Speaker remap ---

def test_build_speaker_mappings():
    raw = ["Alice", "Bob", "Charlie"]
    auto = {"Alice": "User", "Bob": "Target"}
    mappings = build_speaker_mappings(raw, auto)
    assert len(mappings) == 3
    assert mappings[0].assigned_role == "User"
    assert mappings[0].auto_matched
    assert mappings[1].assigned_role == "Target"
    assert mappings[2].assigned_role == "Other"
    assert not mappings[2].auto_matched


def test_apply_remap_overrides():
    queue = [
        {"speaker": "Other", "text": "hi", "speaker_raw": "Charlie"},
        {"speaker": "User", "text": "hey", "speaker_raw": "Alice"},
    ]
    remap = {"Charlie": "Target", "Alice": "User"}
    updated = apply_remap(queue, remap)
    assert updated[0]["speaker"] == "Target"
    assert updated[1]["speaker"] == "User"


def test_apply_remap_preserves_raw():
    queue = [{"speaker": "Other", "text": "x", "speaker_raw": "Messy Name 123"}]
    updated = apply_remap(queue, {"Messy Name 123": "Target"})
    assert updated[0]["speaker"] == "Target"
    assert updated[0]["speaker_raw"] == "Messy Name 123"


def test_apply_remap_no_leak():
    """Remap should never put raw name into speaker field."""
    queue = [{"speaker": "Other", "text": "x", "speaker_raw": "Tinyoranges6 [BS]"}]
    updated = apply_remap(queue, {"Tinyoranges6 [BS]": "Target"})
    assert updated[0]["speaker"] in ("User", "Target", "Other")


# --- Ledger ---

def test_ledger_entry_basic():
    entry = {"speaker": "Target", "text": "Hello world", "speaker_raw": "Bob"}
    le = build_ledger_entry(1, entry)
    assert le.turn_index == 1
    assert le.speaker_role == "Target"
    assert le.speaker_raw == "Bob"
    assert le.processed == True
    assert "Hello" in le.text_excerpt


def test_ledger_entry_other_not_processed():
    entry = {"speaker": "Other", "text": "side comment", "speaker_raw": "Charlie"}
    le = build_ledger_entry(3, entry)
    assert le.processed == False
    assert le.speaker_role == "Other"


def test_ledger_entry_with_result():
    """Mock a result-like object to test ledger extraction."""
    class MockHyp:
        probability = 0.35
        momentum = 0.02

    class MockState:
        active_hypotheses = {
            "target_is_bluffing": MockHyp(),
            "target_is_withholding_info": MockHyp(),
        }

    class MockPressure:
        aggregate = 0.25

    class MockReview:
        ran = True
        primary_class = "hostile_imperative"

    class MockResult:
        state = MockState()
        user_pressure = MockPressure()
        semantic_review = MockReview()
        prior_context = {"claims": [{"type": "test"}], "retrieval_path": "prior_statements", "effect": None}
        target_context_summary = {"behavioral_patterns": [{"id": "p1"}]}

    entry = {"speaker": "Target", "text": "test", "speaker_raw": "Bob"}
    le = build_ledger_entry(1, entry, MockResult())

    assert le.bluffing == 0.35
    assert le.bluffing_delta == 0.02
    assert le.pressure_agg == 0.25
    assert le.review_ran == True
    assert le.motif_primary == "hostile_imperative"
    assert le.claims_count == 1
    assert le.patterns_active == 1


# --- OSINT trace ---

def test_osint_trace_no_claims():
    trace = build_osint_trace({})
    assert not trace.fired
    assert "no prior context" in trace.reason


def test_osint_trace_no_retrieval_worthy():
    trace = build_osint_trace({"claims": []})
    assert not trace.fired
    assert "no retrieval-worthy" in trace.reason


def test_osint_trace_no_retrieval():
    trace = build_osint_trace({"claims": [{"type": "test"}], "retrieval_path": "none"})
    assert not trace.fired
    assert "no_retrieval" in trace.reason


def test_osint_trace_fired():
    ctx = {
        "claims": [{"type": "factual_assertion", "text": "test"}],
        "retrieval_path": "web_search_then_page_read",
        "selected_urls": [{"url": "https://example.com"}],
        "retrieval_records": [{"snippet": "Acme Corp closed", "source": "web_search"}],
        "comparisons": [{"outcome": "supported_by_prior", "confidence": 0.7, "rationale": "match"}],
    }
    trace = build_osint_trace(ctx)
    assert trace.fired
    assert "web_search" in trace.retrieval_path
    assert len(trace.selected_urls) == 1
    assert len(trace.evidence_snippets) == 1
    assert len(trace.comparison_outcomes) == 1


# --- Remap persistence / replay queue contract ---

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))
from transcript_parser import parse_transcript, to_replay_queue


def test_manual_remap_overrides_auto():
    """Manual remap must override alias auto-detection."""
    raw = "Alice: Hi\nBob: Hey\nCharlie: Yo"
    parsed = parse_transcript(raw, user_aliases=["Alice"], target_aliases=["Bob"])
    queue = to_replay_queue(parsed)

    # Alice=User, Bob=Target, Charlie=Other by auto
    assert queue[2]["speaker"] == "Other"

    # Manual remap: Charlie -> Target
    remapped = apply_remap(queue, {"Alice": "User", "Bob": "Other", "Charlie": "Target"})
    assert remapped[0]["speaker"] == "User"
    assert remapped[1]["speaker"] == "Other"  # Bob overridden to Other
    assert remapped[2]["speaker"] == "Target"  # Charlie overridden to Target


def test_preview_counts_reflect_current_remap():
    """Counts should reflect the current manual mapping, not original auto-mapping."""
    raw = "Alice: Hi\nBob: Hey\nCharlie: Yo"
    parsed = parse_transcript(raw)

    # Original: all Other (no aliases)
    assert parsed.user_count == 0
    assert parsed.target_count == 0
    assert parsed.other_count == 3

    # Simulate manual remap
    role_map = {"Alice": "User", "Bob": "Target", "Charlie": "Other"}
    user_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "User")
    target_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "Target")
    other_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "Other")
    assert user_c == 1
    assert target_c == 1
    assert other_c == 1


def test_replay_queue_uses_current_remap():
    """Replay queue must use the current manual role map, not original auto-mapping."""
    raw = "Alice: Hi\nBob: Hey"
    parsed = parse_transcript(raw)  # no aliases -> all Other
    queue = to_replay_queue(parsed)

    # All Other initially
    assert all(q["speaker"] == "Other" for q in queue)

    # Apply manual remap
    remapped = apply_remap(queue, {"Alice": "User", "Bob": "Target"})
    assert remapped[0]["speaker"] == "User"
    assert remapped[1]["speaker"] == "Target"

    # Raw names preserved
    assert remapped[0]["speaker_raw"] == "Alice"
    assert remapped[1]["speaker_raw"] == "Bob"


def test_reparse_resets_mapping():
    """Reparsing a new transcript should produce a fresh role map."""
    raw1 = "Alice: Hi\nBob: Hey"
    parsed1 = parse_transcript(raw1, user_aliases=["Alice"])
    map1 = {t.speaker_raw: t.speaker_role for t in parsed1.turns}
    assert map1["Alice"] == "User"

    raw2 = "Charlie: Hello\nDave: Hi"
    parsed2 = parse_transcript(raw2, target_aliases=["Dave"])
    map2 = {t.speaker_raw: t.speaker_role for t in parsed2.turns}
    assert "Alice" not in map2  # old speakers gone
    assert map2["Dave"] == "Target"
    assert map2["Charlie"] == "Other"
