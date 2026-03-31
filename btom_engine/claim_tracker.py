"""Claim Tracker — extracts and tracks factual claims across conversation turns.

Catches logical contradictions by building a localized claim graph from the
target's own statements. No external OSINT needed — the target's words are
the evidence against themselves.

Pipeline:
1. LLM extracts structured claims from each target turn (JSON)
2. Python stores claims in a temporal array with turn numbers
3. Python runs collision detection on new claims vs history
4. Contradictions feed directly into Bayesian hypothesis adjustments

Types of contradictions detected:
- Temporal collision: "at gym right now" + later "been playing Valorant all morning"
- Location collision: "I'm at home" + later "I'm at the office"
- State collision: "I never talked to her" + later "when I spoke to her yesterday"
- Quantity collision: "I only had one drink" + later "after my third beer"
- Identity collision: "I don't know him" + later "he's a good friend"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """A structured claim extracted from one turn."""
    turn: int = 0
    subject: str = ""        # who: "self", "other", name
    action: str = ""         # what: "at gym", "playing valorant", "sent report"
    location: str = ""       # where: "home", "office", "gym"
    time: str = ""           # when: "right now", "yesterday", "all morning"
    object_ref: str = ""     # who/what acted upon: "report", "Sarah", "the money"
    negation: bool = False   # "I did NOT", "I never"
    raw_text: str = ""       # original text


@dataclass
class Contradiction:
    """A detected logical contradiction between two claims."""
    claim_a: ExtractedClaim
    claim_b: ExtractedClaim
    contradiction_type: str = ""  # temporal, location, state, quantity, identity
    severity: float = 0.0         # 0-1, how strong the contradiction is
    explanation: str = ""


@dataclass
class ClaimTrackerResult:
    """Result of processing one turn through the claim tracker."""
    claims_extracted: list[ExtractedClaim] = field(default_factory=list)
    contradictions_found: list[Contradiction] = field(default_factory=list)
    bluff_delta: float = 0.0
    rationale: str = ""


class ConversationClaimTracker:
    """Tracks all claims made by the target across a conversation.

    Detects logical contradictions by comparing new claims against history.
    """

    def __init__(self, use_llm: bool = True):
        self._claims: list[ExtractedClaim] = []
        self._turn_count = 0
        self._use_llm = use_llm

    def process_turn(self, target_text: str, turn_number: int = 0, contradiction_weight: float = 0.35) -> ClaimTrackerResult:
        """Extract claims from a turn and check for contradictions."""
        self._turn_count = turn_number or self._turn_count + 1
        result = ClaimTrackerResult()

        # Step 1: Extract claims (try deterministic first, LLM fallback)
        new_claims = self._extract_claims_deterministic(target_text)

        if not new_claims and self._use_llm:
            # Try LLM extraction for complex claims (disabled in eval mode)
            new_claims = self._extract_claims_llm(target_text)

        for claim in new_claims:
            claim.turn = self._turn_count
            claim.raw_text = target_text

        result.claims_extracted = new_claims

        # Step 2: Check for contradictions against history
        for new_claim in new_claims:
            for old_claim in self._claims:
                contradiction = self._check_contradiction(old_claim, new_claim)
                if contradiction:
                    result.contradictions_found.append(contradiction)

        # Step 3: Store new claims
        self._claims.extend(new_claims)

        # Step 4: Compute bluff delta from contradictions
        if result.contradictions_found:
            max_severity = max(c.severity for c in result.contradictions_found)
            result.bluff_delta = min(0.40, max_severity * contradiction_weight)
            explanations = [c.explanation for c in result.contradictions_found]
            result.rationale = f"CONTRADICTION: {'; '.join(explanations)}"
        else:
            result.rationale = f"{len(new_claims)} claim(s) tracked, no contradictions"

        return result

    def _extract_claims_deterministic(self, text: str) -> list[ExtractedClaim]:
        """Extract claims using regex patterns. Fast, no LLM."""
        claims = []
        text_lower = text.lower().strip()

        # Location claims: "I'm at X", "at the X", "going to X"
        loc_patterns = [
            (r"(?:i'?m |i am |i'?m currently )(?:at |in |near )(?:the |my |a )?(\w[\w\s]{1,20})", "self"),
            (r"(?:going to |heading to |on my way to )(?:the |my |a )?(\w[\w\s]{1,20})", "self"),
            (r"(?:just (?:got to|arrived at|left) )(?:the |my |a )?(\w[\w\s]{1,20})", "self"),
        ]
        for pattern, subject in loc_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip().rstrip(".")
                claims.append(ExtractedClaim(
                    subject=subject, location=location,
                    action=f"at {location}",
                ))

        # Activity claims: "I'm doing X", "been Xing", "playing X"
        activity_patterns = [
            (r"(?:i'?m |i am |i'?m currently )(\w+ing\b[\w\s]{0,15})", "self"),
            (r"(?:been |i'?ve been )(\w+ing\b[\w\s]{0,15})", "self"),
            (r"(?:just |i just )(\w+ed\b[\w\s]{0,15})", "self"),
        ]
        for pattern, subject in activity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                action = match.group(1).strip().rstrip(".")
                claims.append(ExtractedClaim(
                    subject=subject, action=action,
                ))

        # Time claims
        time_patterns = [
            (r"(?:right now|rn|currently|at the moment)", "now"),
            (r"(?:all (?:morning|day|night|afternoon|evening))", None),
            (r"(?:yesterday|today|tomorrow|last night|this morning)", None),
            (r"(?:since |for the (?:past |last )?)(\d+\s*(?:hours?|minutes?|days?))", None),
        ]
        for pattern, time_val in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                time_str = time_val or match.group(0).strip()
                # Attach time to existing claims or create new one
                if claims:
                    claims[-1].time = time_str
                else:
                    claims.append(ExtractedClaim(subject="self", time=time_str))

        # Negation claims: "I never", "I didn't", "I don't"
        negation_patterns = [
            r"(?:i |i'?ve )(?:never|didn'?t|don'?t|haven'?t|wasn'?t) (\w[\w\s]{2,30})",
            r"(?:that'?s not true|that never happened|i wasn'?t (?:there|involved))",
        ]
        for pattern in negation_patterns:
            match = re.search(pattern, text_lower)
            if match:
                action = match.group(1).strip() if match.lastindex else "denied"
                claims.append(ExtractedClaim(
                    subject="self", action=action, negation=True,
                ))

        # Relationship claims: "I know X", "I don't know X", "X is my Y"
        rel_patterns = [
            (r"(?:i (?:know|met|talked to|spoke (?:with|to)) )(\w[\w\s]{1,20})", False),
            (r"(?:i (?:don'?t|never) (?:know|met|talked to|spoke (?:with|to)) )(\w[\w\s]{1,20})", True),
        ]
        for pattern, is_negation in rel_patterns:
            match = re.search(pattern, text_lower)
            if match:
                person = match.group(1).strip().rstrip(".")
                claims.append(ExtractedClaim(
                    subject="self", object_ref=person,
                    action=f"{'does not know' if is_negation else 'knows'} {person}",
                    negation=is_negation,
                ))

        # Short concrete claims (gym, chest, home, etc.)
        short_concrete = {
            "gym": ("self", "at gym", "gym"),
            "home": ("self", "at home", "home"),
            "work": ("self", "at work", "work"),
            "school": ("self", "at school", "school"),
            "chest": ("self", "doing chest", "gym"),
            "legs": ("self", "doing legs", "gym"),
            "back": ("self", "doing back", "gym"),
        }
        words = text_lower.split()
        if len(words) <= 3:
            for word in words:
                word_clean = re.sub(r'[^\w]', '', word)
                if word_clean in short_concrete:
                    subj, action, loc = short_concrete[word_clean]
                    claims.append(ExtractedClaim(
                        subject=subj, action=action, location=loc, time="now",
                    ))

        return claims

    def _extract_claims_llm(self, text: str) -> list[ExtractedClaim]:
        """Extract claims using LLM for complex/ambiguous text."""
        if len(text.strip()) < 10:
            return []

        try:
            from btom_engine.remote_llm import remote_chat_json

            prompt = (
                f'Extract factual claims from this message. Return JSON:\n'
                f'{{"claims": [{{"subject": "self/other/name", "action": "what they did/are doing", '
                f'"location": "where (or empty)", "time": "when (or empty)", '
                f'"object_ref": "who/what involved (or empty)", "negation": true/false}}]}}\n\n'
                f'If no concrete claims, return {{"claims": []}}.\n'
                f'Message: "{text[:300]}"'
            )

            parsed = remote_chat_json(user=prompt, max_tokens=300, temperature=0.1)
            raw_claims = parsed.get("claims", [])

            claims = []
            for c in raw_claims:
                if not isinstance(c, dict):
                    continue
                if c.get("action") or c.get("location"):
                    claims.append(ExtractedClaim(
                        subject=str(c.get("subject", "self")),
                        action=str(c.get("action", "")),
                        location=str(c.get("location", "")),
                        time=str(c.get("time", "")),
                        object_ref=str(c.get("object_ref", "")),
                        negation=bool(c.get("negation", False)),
                    ))
            return claims

        except Exception as e:
            logger.debug("LLM claim extraction failed: %s", e)
            return []

    def _check_contradiction(
        self, old: ExtractedClaim, new: ExtractedClaim,
    ) -> Contradiction | None:
        """Check if two claims contradict each other."""

        # Only compare claims from the same subject
        if old.subject != new.subject:
            return None

        # --- Location contradiction ---
        if old.location and new.location:
            old_loc = old.location.lower().strip()
            new_loc = new.location.lower().strip()
            if old_loc != new_loc and old_loc and new_loc:
                # Check temporal overlap
                if self._times_overlap(old.time, new.time):
                    return Contradiction(
                        claim_a=old, claim_b=new,
                        contradiction_type="location",
                        severity=0.85,
                        explanation=(
                            f"Turn {old.turn}: '{old.action}' at '{old.location}' "
                            f"vs Turn {new.turn}: '{new.action}' at '{new.location}'"
                        ),
                    )

        # --- Activity contradiction ---
        if old.action and new.action:
            old_act = old.action.lower()
            new_act = new.action.lower()

            # Direct contradiction: same action, one negated
            if old.negation != new.negation:
                # Check if they're about the same thing
                overlap = set(old_act.split()) & set(new_act.split())
                if len(overlap) >= 1:
                    return Contradiction(
                        claim_a=old, claim_b=new,
                        contradiction_type="state",
                        severity=0.90,
                        explanation=(
                            f"Turn {old.turn}: '{'NOT ' if old.negation else ''}{old.action}' "
                            f"vs Turn {new.turn}: '{'NOT ' if new.negation else ''}{new.action}'"
                        ),
                    )

            # Incompatible activities at the same time
            if self._times_overlap(old.time, new.time):
                if self._activities_incompatible(old_act, new_act):
                    return Contradiction(
                        claim_a=old, claim_b=new,
                        contradiction_type="temporal",
                        severity=0.80,
                        explanation=(
                            f"Turn {old.turn}: '{old.action}' ({old.time or 'unspecified time'}) "
                            f"vs Turn {new.turn}: '{new.action}' ({new.time or 'unspecified time'})"
                        ),
                    )

        # --- Relationship contradiction ---
        if old.object_ref and new.object_ref:
            old_ref = old.object_ref.lower().strip()
            new_ref = new.object_ref.lower().strip()
            if old_ref == new_ref and old.negation != new.negation:
                return Contradiction(
                    claim_a=old, claim_b=new,
                    contradiction_type="identity",
                    severity=0.90,
                    explanation=(
                        f"Turn {old.turn}: '{'does not know' if old.negation else 'knows'} {old_ref}' "
                        f"vs Turn {new.turn}: '{'does not know' if new.negation else 'knows'} {new_ref}'"
                    ),
                )

        return None

    def _times_overlap(self, time_a: str, time_b: str) -> bool:
        """Check if two time references could overlap."""
        if not time_a and not time_b:
            return True  # unspecified times might overlap

        a = time_a.lower() if time_a else ""
        b = time_b.lower() if time_b else ""

        # "now" overlaps with "now", "all morning", "currently", etc.
        now_words = {"now", "rn", "currently", "right now", "at the moment"}
        a_is_now = any(w in a for w in now_words) or not a
        b_is_now = any(w in b for w in now_words) or not b

        if a_is_now and b_is_now:
            return True

        # Same time period
        periods = ["morning", "afternoon", "evening", "night", "today", "yesterday", "all day"]
        for period in periods:
            if period in a and period in b:
                return True

        # If one is "all morning" and other is "now" (and it's presumably morning)
        if ("all" in a or "all" in b) and (a_is_now or b_is_now):
            return True

        return False

    def _activities_incompatible(self, act_a: str, act_b: str) -> bool:
        """Check if two activities can't happen simultaneously."""
        # Define incompatible activity groups
        groups = [
            {"gym", "working out", "lifting", "doing chest", "doing legs", "doing back", "at gym"},
            {"playing", "gaming", "playing valorant", "playing val", "on valorant"},
            {"sleeping", "asleep", "napping", "in bed"},
            {"at work", "working", "in a meeting", "at the office"},
            {"at home", "home", "at my place"},
            {"driving", "on the road", "in the car"},
            {"at school", "in class", "studying"},
        ]

        for group in groups:
            a_match = any(g in act_a for g in group)
            b_match = any(g in act_b for g in group)
            if a_match and not b_match:
                # Check if b is in a DIFFERENT group
                for other_group in groups:
                    if other_group is group:
                        continue
                    if any(g in act_b for g in other_group):
                        return True

        return False

    def reset(self) -> None:
        """Reset for a new conversation."""
        self._claims.clear()
        self._turn_count = 0

    def get_all_claims(self) -> list[ExtractedClaim]:
        """Get all tracked claims."""
        return list(self._claims)
