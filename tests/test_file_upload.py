"""Tests for structured file upload — JSON/CSV transcript import."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cockpit"))

from file_upload import import_file, upload_to_replay_queue, UploadedTurn


# --- JSON import ---

def test_simple_json_array():
    data = json.dumps([
        {"speaker": "Alice", "text": "Hello", "timestamp": "2:30 PM"},
        {"speaker": "Bob", "text": "Hi there", "timestamp": "2:31 PM"},
    ])
    result = import_file(data.encode(), "chat.json")
    assert result.success
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Alice"
    assert result.turns[0].text_full == "Hello"
    assert result.turns[0].timestamp == "2:30 PM"


def test_json_with_content_field():
    data = json.dumps([
        {"author": "Alice", "content": "Hello world"},
        {"author": "Bob", "content": "Hey!"},
    ])
    result = import_file(data.encode(), "chat.json")
    assert result.success
    assert len(result.turns) == 2


def test_json_with_message_field():
    data = json.dumps([
        {"username": "user1", "message": "what's up"},
    ])
    result = import_file(data.encode(), "chat.json")
    assert result.success
    assert result.turns[0].text_full == "what's up"


def test_discord_export_format():
    data = json.dumps({
        "messages": [
            {"author": {"name": "Zoink", "nickname": "Zoink"}, "content": "yo what's up", "timestamp": "2024-01-15T14:30:00"},
            {"author": {"name": "Julie", "nickname": "Julie"}, "content": "not much lol", "timestamp": "2024-01-15T14:31:00"},
        ]
    })
    result = import_file(data.encode(), "export.json")
    assert result.success
    assert result.format_detected == "discord_export"
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Zoink"
    assert result.turns[0].text_full == "yo what's up"


def test_json_with_turns_key():
    data = json.dumps({
        "turns": [
            {"speaker": "Alice", "text": "Hello"},
        ]
    })
    result = import_file(data.encode(), "chat.json")
    assert result.success


def test_json_empty_array():
    result = import_file(b"[]", "chat.json")
    assert not result.success
    assert any("empty" in w for w in result.warnings)


def test_json_invalid():
    result = import_file(b"not json at all", "chat.json")
    assert not result.success


def test_json_missing_speaker():
    data = json.dumps([{"text": "no speaker here"}])
    result = import_file(data.encode(), "chat.json")
    assert not result.success


# --- CSV import ---

def test_simple_csv():
    csv_text = "speaker,text,timestamp\nAlice,Hello,2:30 PM\nBob,Hi there,2:31 PM\n"
    result = import_file(csv_text.encode(), "chat.csv")
    assert result.success
    assert len(result.turns) == 2
    assert result.turns[0].speaker_raw == "Alice"
    assert result.turns[0].text_full == "Hello"


def test_csv_author_content():
    csv_text = "author,content\nAlice,Hello world\nBob,Hey!\n"
    result = import_file(csv_text.encode(), "chat.csv")
    assert result.success
    assert len(result.turns) == 2


def test_csv_missing_columns():
    csv_text = "foo,bar\n1,2\n"
    result = import_file(csv_text.encode(), "chat.csv")
    assert not result.success


# --- Auto-detect format ---

def test_auto_detect_json():
    data = json.dumps([{"speaker": "A", "text": "hi"}])
    result = import_file(data.encode(), "chat.txt")
    assert result.success
    assert result.format_detected == "json"


def test_unsupported_format():
    result = import_file(b"random binary data \x00\x01", "chat.xyz")
    assert not result.success


# --- Full text preservation ---

def test_long_message_preserved():
    long_msg = "This is very important context. " * 30
    data = json.dumps([{"speaker": "Alice", "text": long_msg}])
    result = import_file(data.encode(), "chat.json")
    assert result.success
    assert len(result.turns[0].text_full) > 500


# --- Replay queue ---

def test_replay_queue_roles():
    turns = [
        UploadedTurn(speaker_raw="Alice", text_full="hi"),
        UploadedTurn(speaker_raw="Bob", text_full="hey"),
        UploadedTurn(speaker_raw="Charlie", text_full="yo"),
    ]
    role_map = {"Alice": "User", "Bob": "Target", "Charlie": "Other"}
    queue = upload_to_replay_queue(turns, role_map)

    assert queue[0]["speaker"] == "User"
    assert queue[1]["speaker"] == "Target"
    assert queue[2]["speaker"] == "Other"
    assert all(q["speaker"] in ("User", "Target", "Other") for q in queue)


def test_replay_queue_uses_full_text():
    long_msg = "Important message. " * 20
    turns = [UploadedTurn(speaker_raw="A", text_full=long_msg)]
    queue = upload_to_replay_queue(turns, {"A": "Target"})
    assert len(queue[0]["text"]) > 100
    assert queue[0]["text"] == long_msg


def test_replay_queue_preserves_raw():
    turns = [UploadedTurn(speaker_raw="Zoink#1234", text_full="hey")]
    queue = upload_to_replay_queue(turns, {"Zoink#1234": "Target"})
    assert queue[0]["speaker_raw"] == "Zoink#1234"
    assert queue[0]["speaker"] == "Target"


def test_replay_default_other():
    turns = [UploadedTurn(speaker_raw="Unknown", text_full="hi")]
    queue = upload_to_replay_queue(turns, {})
    assert queue[0]["speaker"] == "Other"


# --- Speaker detection ---

def test_raw_speakers_deduplicated():
    data = json.dumps([
        {"speaker": "Alice", "text": "hi"},
        {"speaker": "Bob", "text": "hey"},
        {"speaker": "Alice", "text": "yo"},
    ])
    result = import_file(data.encode(), "chat.json")
    assert result.raw_speakers == ["Alice", "Bob"]
