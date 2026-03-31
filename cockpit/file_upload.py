"""File Upload — structured transcript import from JSON/CSV exports.

Supports:
  - JSON: array of {speaker/author/username, text/content/message, timestamp?}
  - CSV: columns for speaker + text, optional timestamp
  - DiscordChatExporter JSON format
  - Generic chat export formats

No parsing ambiguity — structured data goes straight to replay.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UploadedTurn:
    """One turn from an uploaded file."""
    speaker_raw: str = ""
    text_full: str = ""
    timestamp: Optional[str] = None


@dataclass
class UploadResult:
    """Result of file upload processing."""
    turns: list[UploadedTurn] = field(default_factory=list)
    raw_speakers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    format_detected: str = "unknown"
    success: bool = False


# ---------------------------------------------------------------------------
# Speaker field name variants across platforms
# ---------------------------------------------------------------------------

_SPEAKER_KEYS = ["speaker", "author", "username", "user", "name", "sender", "from",
                 "Author", "Speaker", "Username", "User", "Name", "Sender", "From"]
_TEXT_KEYS = ["text", "content", "message", "body", "msg",
             "Text", "Content", "Message", "Body"]
_TIMESTAMP_KEYS = ["timestamp", "time", "date", "datetime", "created_at", "sent_at",
                   "Timestamp", "Time", "Date", "DateTime"]


def _find_key(obj: dict, candidates: list[str]) -> str | None:
    """Find the first matching key from candidates in a dict."""
    for k in candidates:
        if k in obj:
            return k
    return None


# ---------------------------------------------------------------------------
# JSON import
# ---------------------------------------------------------------------------

def _import_json(data: Any) -> UploadResult:
    """Import from parsed JSON data."""
    result = UploadResult(format_detected="json")

    # Handle DiscordChatExporter format: {messages: [...]}
    if isinstance(data, dict):
        if "messages" in data:
            data = data["messages"]
            result.format_detected = "discord_export"
        elif "turns" in data:
            data = data["turns"]
        else:
            result.warnings.append("JSON object has no 'messages' or 'turns' key")
            return result

    if not isinstance(data, list):
        result.warnings.append("JSON root is not a list")
        return result

    if len(data) == 0:
        result.warnings.append("empty message list")
        return result

    # Detect field names from first entry
    sample = data[0] if isinstance(data[0], dict) else {}
    speaker_key = _find_key(sample, _SPEAKER_KEYS)
    text_key = _find_key(sample, _TEXT_KEYS)
    ts_key = _find_key(sample, _TIMESTAMP_KEYS)

    # DiscordChatExporter has nested author: {name: "..."}
    discord_nested = False
    if speaker_key == "author" and isinstance(sample.get("author"), dict):
        discord_nested = True
        result.format_detected = "discord_export"

    if not speaker_key and not discord_nested:
        result.warnings.append(f"cannot find speaker field in keys: {list(sample.keys())[:10]}")
        return result
    if not text_key:
        result.warnings.append(f"cannot find text field in keys: {list(sample.keys())[:10]}")
        return result

    seen: set[str] = set()

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            result.warnings.append(f"entry {i}: not a dict, skipped")
            continue

        # Extract speaker
        if discord_nested:
            author = entry.get("author", {})
            speaker = str(author.get("name", author.get("nickname", ""))).strip()
        else:
            speaker = str(entry.get(speaker_key, "")).strip()

        # Extract text
        text = str(entry.get(text_key, "")).strip()

        # Extract timestamp
        ts = None
        if ts_key:
            raw_ts = entry.get(ts_key)
            if raw_ts:
                ts = str(raw_ts)

        if not speaker or not text:
            continue

        result.turns.append(UploadedTurn(
            speaker_raw=speaker,
            text_full=text,
            timestamp=ts,
        ))
        seen.add(speaker)

    result.raw_speakers = sorted(seen)
    result.success = len(result.turns) > 0

    if not result.success:
        result.warnings.append("no valid turns extracted from JSON")

    return result


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

def _import_csv(text: str) -> UploadResult:
    """Import from CSV text."""
    result = UploadResult(format_detected="csv")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        result.warnings.append("CSV has no header row")
        return result

    # Detect field names
    fields = {f.lower(): f for f in reader.fieldnames}
    speaker_col = None
    text_col = None
    ts_col = None

    for candidate in _SPEAKER_KEYS:
        if candidate.lower() in fields:
            speaker_col = fields[candidate.lower()]
            break
    for candidate in _TEXT_KEYS:
        if candidate.lower() in fields:
            text_col = fields[candidate.lower()]
            break
    for candidate in _TIMESTAMP_KEYS:
        if candidate.lower() in fields:
            ts_col = fields[candidate.lower()]
            break

    if not speaker_col:
        result.warnings.append(f"cannot find speaker column in: {list(reader.fieldnames)}")
        return result
    if not text_col:
        result.warnings.append(f"cannot find text column in: {list(reader.fieldnames)}")
        return result

    seen: set[str] = set()

    for row in reader:
        speaker = str(row.get(speaker_col, "")).strip()
        text = str(row.get(text_col, "")).strip()
        ts = str(row.get(ts_col, "")).strip() if ts_col else None

        if not speaker or not text:
            continue

        result.turns.append(UploadedTurn(
            speaker_raw=speaker,
            text_full=text,
            timestamp=ts if ts else None,
        ))
        seen.add(speaker)

    result.raw_speakers = sorted(seen)
    result.success = len(result.turns) > 0

    if not result.success:
        result.warnings.append("no valid turns extracted from CSV")

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def import_file(file_content: bytes, filename: str) -> UploadResult:
    """Import a structured transcript file (JSON or CSV).

    Returns UploadResult with clean turns ready for role mapping and replay.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_content.decode("utf-8-sig")
        except Exception:
            return UploadResult(warnings=["cannot decode file — not UTF-8"])

    if ext == "json":
        try:
            data = json.loads(text)
            return _import_json(data)
        except json.JSONDecodeError as e:
            return UploadResult(warnings=[f"invalid JSON: {e}"])

    elif ext == "csv":
        return _import_csv(text)

    else:
        # Try JSON first, then CSV
        try:
            data = json.loads(text)
            return _import_json(data)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return _import_csv(text)
        except Exception:
            pass
        return UploadResult(warnings=[f"unsupported file format: .{ext}"])


def upload_to_replay_queue(
    turns: list[UploadedTurn],
    role_map: dict[str, str],
) -> list[dict]:
    """Convert uploaded turns to replay queue format.

    role_map: {speaker_raw -> "User"/"Target"/"Other"}
    """
    queue = []
    for turn in turns:
        role = role_map.get(turn.speaker_raw, "Other")
        queue.append({
            "speaker": role,
            "text": turn.text_full,
            "speaker_raw": turn.speaker_raw,
        })
    return queue
