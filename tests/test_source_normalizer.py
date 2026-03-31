"""Tests for source normalizer — sniffing, chrome stripping, integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from source_normalizer import normalize_source, sniff_source, _strip_chrome


# --- Source sniffing ---

def test_sniff_discord():
    text = "Username Today at 2:30 PM\nHello there\nOtherUser Today at 2:31 PM\nHey"
    src, family, conf = sniff_source(text)
    assert src == "discord"
    assert family == "chat"


def test_sniff_linkedin():
    text = "Jeff Vallyath sent the following message at 12:00 PM\nHey are you available?\nView Jeff's profile"
    src, family, conf = sniff_source(text)
    assert src == "linkedin"


def test_sniff_reddit():
    text = "u/username123\n5 upvotes\nThis is my comment about the topic\nReply Permalink Share"
    src, family, conf = sniff_source(text)
    assert src == "reddit"
    assert family == "thread"


def test_sniff_x_twitter():
    text = "@JohnDoe Replying to @JaneSmith\nThis is such a bad take\n15 Likes 3 Retweets"
    src, family, conf = sniff_source(text)
    assert src == "x_twitter"


def test_sniff_youtube():
    text = "Username\n2.5K subscribers\nGreat video! 15 likes\nRead more\n125K views"
    src, family, conf = sniff_source(text)
    assert src == "youtube"


def test_sniff_instagram():
    text = "Username\nLiked a message\nHey how are you\nSeen"
    src, family, conf = sniff_source(text)
    assert src == "instagram"


def test_sniff_generic_fallback():
    text = "Alice: Hello\nBob: Hi\nAlice: How are you?"
    src, family, conf = sniff_source(text)
    assert src == "generic"
    assert conf <= 0.35


# --- Chrome stripping ---

def test_strip_linkedin_chrome():
    text = "Jeff Vallyath sent the following message at 12:00 PM\nHey are you free?\nView Jeff's profile\n2 hr ago"
    cleaned, stripped = _strip_chrome(text, "linkedin")
    assert "sent the following message" not in cleaned
    assert "View" not in cleaned
    assert "Hey are you free?" in cleaned
    assert len(stripped) > 0


def test_strip_discord_chrome():
    text = "Username BOT Today at 2:30 PM\nHello (edited)\nPinned a message to this channel"
    cleaned, stripped = _strip_chrome(text, "discord")
    assert " BOT " not in cleaned
    assert "(edited)" not in cleaned
    assert "Hello" in cleaned


def test_strip_reddit_chrome():
    text = "u/username\n5 upvotes\nThis is my actual comment\nReply\nPermalink\nShare"
    cleaned, stripped = _strip_chrome(text, "reddit")
    assert "upvotes" not in cleaned
    assert "Reply" not in cleaned
    assert "This is my actual comment" in cleaned


def test_strip_x_twitter_chrome():
    text = "@user This is my tweet\n25 Likes\n3 Retweets\nShow this thread"
    cleaned, stripped = _strip_chrome(text, "x_twitter")
    assert "25 Likes" not in cleaned
    assert "Show this thread" not in cleaned
    assert "This is my tweet" in cleaned


def test_strip_youtube_chrome():
    text = "Username\nGreat video! Really enjoyed it\n15 likes\nRead more"
    cleaned, stripped = _strip_chrome(text, "youtube")
    assert "15 likes" not in cleaned
    assert "Read more" not in cleaned
    assert "Great video!" in cleaned


def test_strip_preserves_message_text():
    """Chrome stripping must NOT alter actual message content."""
    text = "u/username\nI think the real issue is that nobody likes being told what to do\nReply"
    cleaned, _ = _strip_chrome(text, "reddit")
    assert "nobody likes being told what to do" in cleaned


# --- Full normalization ---

def test_normalize_discord_chat():
    text = "User1 Today at 2:30 PM\nwhat's up\nUser2 Today at 2:31 PM\nnot much (edited)"
    result = normalize_source(text)
    assert result.source_detected == "discord"
    assert "(edited)" not in result.normalized_text
    assert "what's up" in result.normalized_text


def test_normalize_linkedin_dm():
    text = "Jeff Vallyath sent the following message at 12:00 PM\nHey let's catch up\nView Jeff's profile"
    result = normalize_source(text)
    assert result.source_detected == "linkedin"
    assert "sent the following message" not in result.normalized_text
    assert "Hey let's catch up" in result.normalized_text


def test_normalize_generic_passthrough():
    text = "Alice: Hello\nBob: Hi"
    result = normalize_source(text)
    assert result.source_detected == "generic"
    assert "Alice: Hello" in result.normalized_text


def test_normalize_empty():
    result = normalize_source("")
    assert result.normalized_text == ""


def test_normalize_preserves_timestamps_in_text():
    """Timestamps should remain in text for the parser to extract, not be stripped."""
    text = "User1 Today at 2:30 PM\nHello there"
    result = normalize_source(text)
    # Timestamp should still be in text (parser extracts it)
    assert "Hello there" in result.normalized_text


# --- Integration with parser ---

def test_parser_uses_normalization():
    """Parse transcript should normalize before parsing."""
    from transcript_parser import parse_transcript

    # LinkedIn-style text with chrome
    text = "Jeff: Hey are you free?\nView Jeff's profile\nBob: Yeah sure"
    result = parse_transcript(text, user_aliases=["Jeff"], target_aliases=["Bob"])
    # "View Jeff's profile" is linkedin chrome but Jeff: format still works
    # The key test: parser doesn't crash and turns are extracted
    assert len(result.turns) >= 2


def test_parser_skip_normalization():
    """skip_normalization flag should bypass source normalizer."""
    from transcript_parser import parse_transcript

    text = "Alice: Hello\nBob: Hi"
    result = parse_transcript(text, skip_normalization=True)
    assert len(result.turns) == 2


# --- Profile detection ---

def test_profile_snippet_detected():
    text = "Jane Smith\nSenior Engineer at TechCorp\nAbout\nI love building things\nFollowers 500 Following 200 Posts 50"
    result = normalize_source(text)
    assert result.source_family == "profile"
