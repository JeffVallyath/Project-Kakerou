"""Fact Registry — shared working memory for multi-agent ToM analysis.

Stores structured facts (claims, accusations, defenses, votes, role claims)
extracted from all players' text. Enables cross-player reasoning by providing
a central source of truth that all player engine instances can query.

Extraction uses fuzzy regex + heuristic rules, not strict pattern matching.
Expected ~60-70% recall on messy forum text.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Accusation:
    accuser: str
    target: str
    turn: int
    quote: str = ""


@dataclass
class Defense:
    defender: str
    defended: str
    turn: int
    quote: str = ""


@dataclass
class Vote:
    voter: str
    target: str
    turn: int
    quote: str = ""


@dataclass
class RoleClaim:
    player: str
    claimed_role: str
    turn: int
    quote: str = ""


@dataclass
class Elimination:
    player: str
    turn: int
    method: str = ""  # "lynch", "night_kill", etc.


# ---------------------------------------------------------------------------
# Extraction patterns (fuzzy, not strict)
# ---------------------------------------------------------------------------

_VOTE_PATTERNS = re.compile(
    r"\b(?:vote|lynch|eliminate|hammer|unvote)\b",
    re.IGNORECASE,
)

_ACCUSATION_PATTERNS = re.compile(
    r"\b(?:scum|mafia|wolf|suspicious|suspect|lying|liar|"
    r"fake\s*claim|don'?t\s*trust|don'?t\s*believe|"
    r"is\s+the\s+(?:mafia|wolf|scum|killer)|"
    r"guilty|sketchy|shady|fishy|sus)\b",
    re.IGNORECASE,
)

_DEFENSE_PATTERNS = re.compile(
    r"\b(?:town(?:ie)?|trust|believe|confirmed|innocent|clear|"
    r"vouch|legit|genuine|not\s+(?:mafia|scum|wolf)|"
    r"is\s+(?:town|clear|confirmed|innocent)|"
    r"leave\s+\S+\s+alone|defend)\b",
    re.IGNORECASE,
)

_ROLE_CLAIM_PATTERNS = re.compile(
    r"\b(?:i\s+am\s+(?:the\s+)?|i'?m\s+(?:the\s+)?|i\s+claim\s+|my\s+role\s+is\s+)"
    r"(seer|cop|detective|investigator|doctor|medic|guard|"
    r"witch|hunter|vigilante|oracle|tracker|watcher|"
    r"villager|townie|vanilla|citizen|mason|"
    r"werewolf|wolf|mafia|godfather|goon)\b",
    re.IGNORECASE,
)

_SELF_DEFENSE_PATTERNS = re.compile(
    r"\b(?:i'?m\s+not\s+(?:mafia|scum|wolf|the\s+wolf|the\s+mafia)|"
    r"i\s+didn'?t\s+do\s+it|"
    r"i\s+swear|"
    r"why\s+would\s+i|"
    r"i'?m\s+(?:town|innocent|clear))\b",
    re.IGNORECASE,
)


def _build_name_index(player_names: list[str]) -> dict[str, str]:
    """Build a lookup index mapping nicknames/abbreviations to full player names.

    For "Brian Skies" generates: {"brian skies", "brian", "skies"}
    For "F-16_Fighting_Falcon" generates: {"f-16_fighting_falcon", "f-16", "fighting_falcon", "falcon"}
    """
    index: dict[str, str] = {}
    for name in player_names:
        # Full name (lowered)
        index[name.lower()] = name
        # Individual parts (split on space, underscore, hyphen-if-long)
        parts = re.split(r"[\s_]+", name)
        for part in parts:
            part_lower = part.lower().strip(".,!?:;\"'()[]")
            if len(part_lower) >= 3:  # skip very short parts
                index[part_lower] = name
        # First word as nickname
        if parts and len(parts[0]) >= 3:
            index[parts[0].lower()] = name
    return index


def _extract_target_near_keyword(text: str, keyword_pos: int, player_names: list[str]) -> str | None:
    """Find the most likely player name near a keyword position.

    ONLY returns matches against known player names. No capitalized word fallback.
    Uses a nickname index for partial matching (e.g., "Brian" → "Brian Skies").
    """
    if not player_names:
        return None

    name_index = _build_name_index(player_names)

    words = text.split()
    # Find word index closest to character position
    char_count = 0
    keyword_word_idx = 0
    for i, w in enumerate(words):
        char_count += len(w) + 1
        if char_count >= keyword_pos:
            keyword_word_idx = i
            break

    # Search window: 8 words before, 8 words after
    start = max(0, keyword_word_idx - 8)
    end = min(len(words), keyword_word_idx + 9)
    window = words[start:end]
    window_text = " ".join(window).lower()

    # 1. Try full name substring match (handles multi-word names)
    for name in player_names:
        if name.lower() in window_text:
            return name

    # 2. Try each window word against the nickname index
    for w in window:
        clean = w.strip(".,!?:;\"'()[]").lower()
        if clean in name_index:
            return name_index[clean]

    # 3. Fuzzy match (higher cutoff = fewer false positives)
    for w in window:
        clean = w.strip(".,!?:;\"'()[]").lower()
        if len(clean) < 3:
            continue
        matches = get_close_matches(clean, list(name_index.keys()), n=1, cutoff=0.75)
        if matches:
            return name_index[matches[0]]

    # No fallback to random capitalized words — if we can't match a player, return None
    return None


# ---------------------------------------------------------------------------
# Fact Registry
# ---------------------------------------------------------------------------

class FactRegistry:
    """Shared working memory for multi-agent analysis.

    Stores structured facts extracted from all players' text.
    Supports cross-player queries for relational feature extraction.
    """

    def __init__(self, player_names: list[str] | None = None):
        self.player_names: list[str] = player_names or []
        self.accusations: list[Accusation] = []
        self.defenses: list[Defense] = []
        self.votes: list[Vote] = []
        self.role_claims: dict[str, list[RoleClaim]] = defaultdict(list)
        self.eliminations: list[Elimination] = []
        self._turn_count = 0

    def process_post(self, player_id: str, text: str, turn: int) -> dict[str, int]:
        """Extract all facts from a single post and store them.

        Returns counts of extracted facts for diagnostics.
        """
        self._turn_count = max(self._turn_count, turn)
        counts = {"accusations": 0, "defenses": 0, "votes": 0, "role_claims": 0}

        # --- Votes ---
        # First try the explicit "VOTE: PlayerName" format (Mafiascum convention)
        vote_explicit = re.finditer(
            r"(?:^|\n)\s*(?:vote|VOTE)\s*:\s*(.+?)(?:\n|$)", text, re.MULTILINE)
        explicit_vote_found = False
        name_index = _build_name_index(self.player_names)
        for match in vote_explicit:
            raw_target = match.group(1).strip().strip(".,!?:;\"'()[]")
            # Resolve against player names
            resolved = None
            if raw_target.lower() in name_index:
                resolved = name_index[raw_target.lower()]
            else:
                # Try fuzzy
                close = get_close_matches(raw_target.lower(), list(name_index.keys()), n=1, cutoff=0.6)
                if close:
                    resolved = name_index[close[0]]
            if resolved and resolved.lower() != player_id.lower():
                self.votes.append(Vote(voter=player_id, target=resolved, turn=turn,
                                       quote=text[max(0, match.start()):match.end()][:80]))
                counts["votes"] += 1
                explicit_vote_found = True

        # Fall back to keyword-based vote extraction if no explicit format found
        if not explicit_vote_found:
            for match in _VOTE_PATTERNS.finditer(text):
                if "unvote" in match.group().lower():
                    continue
                target = _extract_target_near_keyword(text, match.start(), self.player_names)
                if target and target.lower() != player_id.lower():
                    self.votes.append(Vote(voter=player_id, target=target, turn=turn,
                                           quote=text[max(0, match.start()-20):match.end()+30][:80]))
                    counts["votes"] += 1

        # --- Accusations ---
        for match in _ACCUSATION_PATTERNS.finditer(text):
            target = _extract_target_near_keyword(text, match.start(), self.player_names)
            if target and target.lower() != player_id.lower():
                self.accusations.append(Accusation(accuser=player_id, target=target, turn=turn,
                                                    quote=text[max(0, match.start()-20):match.end()+30][:80]))
                counts["accusations"] += 1

        # --- Self-defense ---
        if _SELF_DEFENSE_PATTERNS.search(text):
            self.defenses.append(Defense(defender=player_id, defended=player_id, turn=turn,
                                         quote=text[:80]))
            counts["defenses"] += 1

        # --- Defenses of others ---
        for match in _DEFENSE_PATTERNS.finditer(text):
            target = _extract_target_near_keyword(text, match.start(), self.player_names)
            if target and target.lower() != player_id.lower():
                self.defenses.append(Defense(defender=player_id, defended=target, turn=turn,
                                              quote=text[max(0, match.start()-20):match.end()+30][:80]))
                counts["defenses"] += 1

        # --- Role claims ---
        for match in _ROLE_CLAIM_PATTERNS.finditer(text):
            role = match.group(1).lower()
            self.role_claims[player_id].append(RoleClaim(player=player_id, claimed_role=role, turn=turn,
                                                          quote=text[max(0, match.start()-10):match.end()+20][:80]))
            counts["role_claims"] += 1

        return counts

    # --- Query methods ---

    def get_accusations_by(self, player_id: str) -> list[Accusation]:
        return [a for a in self.accusations if a.accuser.lower() == player_id.lower()]

    def get_accusations_of(self, player_id: str) -> list[Accusation]:
        return [a for a in self.accusations if a.target.lower() == player_id.lower()]

    def get_defenses_by(self, player_id: str) -> list[Defense]:
        return [d for d in self.defenses if d.defender.lower() == player_id.lower()]

    def get_defenses_of(self, player_id: str) -> list[Defense]:
        return [d for d in self.defenses if d.defended.lower() == player_id.lower()]

    def get_votes_by(self, player_id: str) -> list[Vote]:
        return [v for v in self.votes if v.voter.lower() == player_id.lower()]

    def get_role_claims(self, player_id: str) -> list[RoleClaim]:
        return self.role_claims.get(player_id, [])

    def get_all_players_claiming_role(self, role: str) -> list[str]:
        """Find all players who claimed a specific role."""
        players = []
        for pid, claims in self.role_claims.items():
            if any(c.claimed_role.lower() == role.lower() for c in claims):
                players.append(pid)
        return players

    def get_unique_accusation_targets(self, player_id: str) -> set[str]:
        return {a.target.lower() for a in self.get_accusations_by(player_id)}

    def get_unique_defense_targets(self, player_id: str) -> set[str]:
        return {d.defended.lower() for d in self.get_defenses_by(player_id)
                if d.defended.lower() != player_id.lower()}

    def summary(self) -> dict[str, int]:
        return {
            "accusations": len(self.accusations),
            "defenses": len(self.defenses),
            "votes": len(self.votes),
            "role_claims": sum(len(v) for v in self.role_claims.values()),
            "players": len(self.player_names),
            "turns": self._turn_count,
        }
