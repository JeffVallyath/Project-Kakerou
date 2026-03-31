"""Tests for block-based transcript normalizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from block_normalizer import (
    segment_blocks, classify_block, assemble_turns,
    normalize_blocks, BlockCandidate, ClassifiedBlock, NormalizedTurn,
)
from transcript_parser import parse_transcript_blocks, to_replay_queue


# --- Block segmentation ---

def test_segment_blank_separated():
    text = "Alice - 2:30 PM\nHello there\n\nBob - 2:31 PM\nHey"
    blocks = segment_blocks(text)
    assert len(blocks) == 2
    assert "Alice" in blocks[0].raw_text
    assert "Bob" in blocks[1].raw_text


def test_segment_no_blanks_per_line():
    text = "Alice: Hello\nBob: Hi\nAlice: How are you?"
    blocks = segment_blocks(text)
    assert len(blocks) == 3


def test_segment_empty():
    blocks = segment_blocks("")
    assert len(blocks) == 0


# --- Block classification ---

def test_classify_url():
    block = BlockCandidate(raw_text="https://example.com/article")
    cb = classify_block(block)
    assert cb.block_type == "attachment_or_link_card"


def test_classify_ui_junk():
    block = BlockCandidate(raw_text="15 Likes")
    cb = classify_block(block)
    assert cb.block_type == "reaction_or_ui_junk"


def test_classify_ui_junk_reply():
    block = BlockCandidate(raw_text="Reply")
    cb = classify_block(block)
    assert cb.block_type == "reaction_or_ui_junk"


def test_classify_profile_chrome():
    block = BlockCandidate(raw_text="View Jeff's profile")
    cb = classify_block(block)
    assert cb.block_type == "profile_chrome"


def test_classify_timestamp():
    block = BlockCandidate(raw_text="2:30 PM")
    cb = classify_block(block)
    assert cb.block_type == "timestamp"


def test_classify_date_separator():
    block = BlockCandidate(raw_text="Monday, January 15")
    cb = classify_block(block)
    assert cb.block_type == "date_separator"


def test_classify_inline_message():
    block = BlockCandidate(raw_text="Alice: Hello there")
    cb = classify_block(block)
    assert cb.block_type == "speaker_header"
    assert cb.speaker_raw == "Alice"


def test_classify_header_with_timestamp():
    block = BlockCandidate(raw_text="Jeff Vallyath - 11:30 AM")
    cb = classify_block(block)
    assert cb.block_type == "speaker_header"
    assert "Jeff" in cb.speaker_raw


def test_classify_message_body():
    block = BlockCandidate(raw_text="This is a longer message that should be treated as message body content for the current turn")
    cb = classify_block(block)
    assert cb.block_type == "message_body"


def test_url_not_speaker():
    """URLs must NOT be classified as speaker headers."""
    block = BlockCandidate(raw_text="https://linkedin.com/in/jvallyath")
    cb = classify_block(block)
    assert cb.block_type != "speaker_header"


# --- Turn assembly ---

def test_assemble_simple_conversation():
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="Alice: Hello", block_type="speaker_header", speaker_raw="Alice"),
        ClassifiedBlock(block_id=1, raw_text="Bob: Hi there", block_type="speaker_header", speaker_raw="Bob"),
    ]
    turns, dropped = assemble_turns(blocks)
    assert len(turns) == 2
    assert turns[0].speaker_raw == "Alice"
    assert "Hello" in turns[0].text_full
    assert turns[1].speaker_raw == "Bob"


def test_assemble_header_then_body():
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="Jeff - 2:30 PM", block_type="speaker_header", speaker_raw="Jeff", timestamp="2:30 PM"),
        ClassifiedBlock(block_id=1, raw_text="Hey are you free tomorrow?", block_type="message_body"),
    ]
    turns, dropped = assemble_turns(blocks)
    assert len(turns) == 1
    assert turns[0].speaker_raw == "Jeff"
    assert "free tomorrow" in turns[0].text_full
    assert turns[0].timestamp == "2:30 PM"


def test_assemble_drops_chrome():
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="View Jeff's profile", block_type="profile_chrome"),
        ClassifiedBlock(block_id=1, raw_text="Jeff: Hey", block_type="speaker_header", speaker_raw="Jeff"),
        ClassifiedBlock(block_id=2, raw_text="15 Likes", block_type="reaction_or_ui_junk"),
    ]
    turns, dropped = assemble_turns(blocks)
    assert len(turns) == 1
    assert turns[0].speaker_raw == "Jeff"
    assert len(dropped) == 2


def test_assemble_url_attaches():
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="Alice: Check this out", block_type="speaker_header", speaker_raw="Alice"),
        ClassifiedBlock(block_id=1, raw_text="https://example.com", block_type="attachment_or_link_card", url="https://example.com"),
    ]
    turns, _ = assemble_turns(blocks)
    assert len(turns) == 1
    assert "https://example.com" in turns[0].attachments


def test_timestamp_not_in_text():
    """Timestamps must NOT leak into engine-facing text."""
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="Alice - 2:30 PM", block_type="speaker_header", speaker_raw="Alice", timestamp="2:30 PM"),
        ClassifiedBlock(block_id=1, raw_text="Hello how are you", block_type="message_body"),
    ]
    turns, _ = assemble_turns(blocks)
    assert "2:30" not in turns[0].text_full
    assert "Hello how are you" in turns[0].text_full


# --- Full text preservation ---

def test_long_message_preserved():
    """Long messages must be fully preserved, not truncated."""
    long_msg = "This is a very important message. " * 20  # ~680 chars
    blocks = [
        ClassifiedBlock(block_id=0, raw_text="Alice: start", block_type="speaker_header", speaker_raw="Alice"),
        ClassifiedBlock(block_id=1, raw_text=long_msg, block_type="message_body"),
    ]
    turns, _ = assemble_turns(blocks)
    assert len(turns[0].text_full) > 500
    assert "very important message" in turns[0].text_full
    # Preview should be truncated
    assert len(turns[0].text_preview) <= 80


def test_replay_uses_full_text():
    """Replay queue must use full text, not truncated preview."""
    raw = "Alice: " + "This is important content. " * 15
    result = parse_transcript_blocks(raw, user_aliases=["Alice"])
    queue = to_replay_queue(result)
    assert len(queue) >= 1
    # The replay text should be the full text
    assert len(queue[0]["text"]) > 100


# --- Full pipeline ---

def test_normalize_discord_like():
    text = "Alice - Today at 2:30 PM\nHello how are you?\n\nBob - Today at 2:31 PM\nI'm good thanks!\n\n15 Likes"
    result = normalize_blocks(text)
    assert len(result.turns) >= 2
    assert any("Alice" in t.speaker_raw for t in result.turns)
    assert any("Bob" in t.speaker_raw for t in result.turns)


def test_normalize_linkedin_like():
    text = """Jeff Vallyath sent the following message at 12:00 PM

Jeff Vallyath - 12:00 PM
Hey are you available for a call?

View Jeff's profile
2 hr ago"""
    result = normalize_blocks(text)
    # Should have at least one real turn
    assert len(result.turns) >= 1
    assert any("Jeff" in t.speaker_raw for t in result.turns)
    assert any("available for a call" in t.text_full for t in result.turns)


def test_normalize_reddit_like():
    text = """u/username123

I think this is a really important point about the topic.

u/other_user

I agree with your take."""
    result = normalize_blocks(text)
    # Should extract the actual comments
    assert any("important point" in t.text_full for t in result.turns)
    assert len(result.turns) >= 2


def test_normalize_simple_colon():
    text = "Alice: Hello\nBob: Hi\nAlice: How are you?"
    result = normalize_blocks(text)
    assert len(result.turns) == 3


def test_normalize_empty():
    result = normalize_blocks("")
    assert len(result.turns) == 0


# --- Integration with parser ---

def test_block_parse_with_aliases():
    text = "Jeff: Hey how's it going?\nBob: Good thanks"
    result = parse_transcript_blocks(text, user_aliases=["Jeff"], target_aliases=["Bob"])
    assert len(result.turns) == 2
    assert result.turns[0].speaker_role == "User"
    assert result.turns[1].speaker_role == "Target"
    assert result.parse_source == "block"


def test_block_parse_falls_back_on_empty():
    """If block normalizer produces nothing, should fall back to line parser."""
    text = ""
    result = parse_transcript_blocks(text)
    assert len(result.turns) == 0


def test_block_parse_replay_queue():
    text = "Alice: Hello world\nBob: Hey there"
    result = parse_transcript_blocks(text, user_aliases=["Alice"], target_aliases=["Bob"])
    queue = to_replay_queue(result)
    assert len(queue) == 2
    assert queue[0]["speaker"] == "User"
    assert queue[1]["speaker"] == "Target"
    assert queue[0]["speaker_raw"] == "Alice"
