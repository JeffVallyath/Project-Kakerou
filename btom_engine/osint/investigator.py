"""Investigation Agent — autonomous ReAct profiler for target intelligence.

Given a target identity, proactively investigates using search + page-read
to build a comprehensive public profile before claims are even made.

Architecture:
  target identity established
  -> agent dispatched with tools (search, read_page, extract_profile)
  -> ReAct loop: think -> act -> observe (bounded steps)
  -> structured findings saved to dossier

The agent uses the existing LM Studio / Qwen endpoint and existing providers.
It does NOT use a separate framework — just a simple ReAct loop.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from btom_engine.config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
from btom_engine.osint.evidence_schema import EvidenceRecord
from btom_engine.osint.providers import SearchProvider, PageProvider, SearchResult, PageContent
from btom_engine.osint.profile_extraction import extract_profile
from btom_engine.osint.target_context import (
    TargetContext, SupportedClaim, EvidenceLink,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_SEARCH_PER_INVESTIGATION = 5
MAX_PAGE_READS = 3


# ---------------------------------------------------------------------------
# Investigation result
# ---------------------------------------------------------------------------

@dataclass
class InvestigationStep:
    """One step in the agent's investigation."""
    step_num: int = 0
    thought: str = ""
    action: str = ""        # search, read_page, save_finding, done
    action_input: str = ""
    observation: str = ""
    duration_ms: float = 0.0


@dataclass
class InvestigationResult:
    """Complete result of an autonomous investigation."""
    target_name: str = ""
    steps: list[InvestigationStep] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    # Each finding: {fact_type, fact_text, source_url, confidence}
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    total_searches: int = 0
    total_page_reads: int = 0
    total_duration_ms: float = 0.0
    success: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# ReAct prompt
# ---------------------------------------------------------------------------

_REACT_SYSTEM = """You are an OSINT investigation agent. Your job is to build a public profile of a target person using available tools.

You have these tools:
- search(query): Search the web. Returns titles, URLs, and snippets.
- read_page(url): Read a webpage and extract text content.
- save_finding(fact_type, fact_text, source_url, confidence): Save a verified finding to the dossier.
- done(summary): End the investigation with a summary.

Valid fact_types for save_finding: employment, role, education, funding, investment, affiliation, credential, publication, social_presence, company_info, contradiction, other

RULES:
1. Start by searching for the person's name.
2. Read the most promising result pages (LinkedIn, Crunchbase, news articles).
3. Extract concrete facts and save them with save_finding.
4. Cross-reference claims when possible — if they claim to work at X, check X's team page.
5. Note contradictions or missing evidence explicitly.
6. Be efficient — you have a limited number of steps.
7. Do NOT fabricate findings. Only save what you actually found on real pages.
8. End with done() summarizing what you found and what you couldn't verify.

Respond with EXACTLY one action per step in this JSON format:
{"thought": "<your reasoning>", "action": "<tool_name>", "action_input": "<input>"}

For save_finding, use:
{"thought": "...", "action": "save_finding", "action_input": {"fact_type": "...", "fact_text": "...", "source_url": "...", "confidence": 0.8}}

For done, use:
{"thought": "...", "action": "done", "action_input": "<summary>"}"""


def _build_react_prompt(target_name: str, context: str = "", history: str = "") -> str:
    """Build the full prompt for one ReAct step."""
    user_msg = f"Investigate this person: {target_name}"
    if context:
        user_msg += f"\n\nAdditional context from conversation:\n{context}"
    if history:
        user_msg += f"\n\nPrevious steps:\n{history}"
    user_msg += "\n\nWhat is your next step?"
    return user_msg


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(
    action: str,
    action_input: Any,
    search_provider: SearchProvider | None,
    page_provider: PageProvider | None,
    search_count: int,
    page_count: int,
) -> tuple[str, int, int]:
    """Execute one tool call. Returns (observation, new_search_count, new_page_count)."""

    if action == "search":
        if search_provider is None:
            return "Error: no search provider available", search_count, page_count
        if search_count >= MAX_SEARCH_PER_INVESTIGATION:
            return "Error: search limit reached", search_count, page_count

        query = str(action_input).strip()
        try:
            results = search_provider.search(query, top_k=5)
            if not results:
                return "No results found.", search_count + 1, page_count
            obs_parts = []
            for i, r in enumerate(results):
                obs_parts.append(f"{i+1}. [{r.title}]({r.url})\n   {r.snippet[:150]}")
            return "\n".join(obs_parts), search_count + 1, page_count
        except Exception as e:
            return f"Search error: {e}", search_count + 1, page_count

    elif action == "read_page":
        if page_provider is None:
            return "Error: no page provider available", search_count, page_count
        if page_count >= MAX_PAGE_READS:
            return "Error: page read limit reached", search_count, page_count

        url = str(action_input).strip()
        try:
            page = page_provider.fetch(url, max_chars=3000)
            if not page.fetch_success:
                return f"Failed to read page: {page.error}", search_count, page_count + 1
            # Extract profile if it looks like a profile page
            profile = extract_profile(page.text, page.title)
            if profile.extraction_confidence > 0.4:
                parts = []
                if profile.display_name:
                    parts.append(f"Name: {profile.display_name}")
                if profile.role_or_title:
                    parts.append(f"Role: {profile.role_or_title}")
                if profile.organization:
                    parts.append(f"Org: {profile.organization}")
                if profile.bio_excerpt:
                    parts.append(f"Bio: {profile.bio_excerpt[:200]}")
                profile_str = "\n".join(parts)
                return f"Title: {page.title}\nProfile extracted:\n{profile_str}\n\nPage text (first 500 chars):\n{page.text[:500]}", search_count, page_count + 1
            else:
                return f"Title: {page.title}\n\nPage text (first 800 chars):\n{page.text[:800]}", search_count, page_count + 1
        except Exception as e:
            return f"Page read error: {e}", search_count, page_count + 1

    elif action == "save_finding":
        # Findings are collected by the caller, not executed as a tool
        return "Finding saved.", search_count, page_count

    elif action == "done":
        return "Investigation complete.", search_count, page_count

    else:
        return f"Unknown action: {action}", search_count, page_count


# ---------------------------------------------------------------------------
# LLM call for agent reasoning
# ---------------------------------------------------------------------------

def _agent_think(system: str, user_msg: str) -> dict:
    """One LLM call for agent reasoning. Returns parsed action dict."""
    import httpx

    resp = httpx.post(
        f"{LLM_BASE_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False,
        },
        timeout=LLM_TIMEOUT_SECONDS + 15,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    # Extract JSON
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main investigation loop
# ---------------------------------------------------------------------------

def investigate_target(
    target_name: str,
    search_provider: SearchProvider | None = None,
    page_provider: PageProvider | None = None,
    context: str = "",
) -> InvestigationResult:
    """Run an autonomous investigation on a target.

    ReAct loop: think -> act -> observe, bounded to MAX_STEPS.
    Returns structured findings for dossier integration.
    """
    result = InvestigationResult(target_name=target_name)
    t0 = time.time()

    if search_provider is None:
        result.summary = "No search provider available — investigation skipped."
        return result

    history_parts: list[str] = []
    search_count = 0
    page_count = 0

    for step_num in range(1, MAX_STEPS + 1):
        step = InvestigationStep(step_num=step_num)
        step_t0 = time.time()

        try:
            history = "\n".join(history_parts)
            user_msg = _build_react_prompt(target_name, context, history)
            parsed = _agent_think(_REACT_SYSTEM, user_msg)

            step.thought = str(parsed.get("thought", ""))
            step.action = str(parsed.get("action", "done"))
            step.action_input = parsed.get("action_input", "")

            print(f"INVESTIGATOR Step {step_num}: {step.action}({str(step.action_input)[:60]})")

            # Handle save_finding
            if step.action == "save_finding":
                finding = step.action_input if isinstance(step.action_input, dict) else {}
                if finding.get("fact_text"):
                    result.findings.append({
                        "fact_type": str(finding.get("fact_type", "other")),
                        "fact_text": str(finding.get("fact_text", "")),
                        "source_url": str(finding.get("source_url", "")),
                        "confidence": float(finding.get("confidence", 0.5)),
                    })
                step.observation = "Finding saved."
                history_parts.append(
                    f"Step {step_num}: Thought: {step.thought}\n"
                    f"Action: save_finding({finding.get('fact_type', '')})\n"
                    f"Observation: Finding saved."
                )

            elif step.action == "done":
                step.observation = str(step.action_input)
                result.summary = step.observation
                step.duration_ms = (time.time() - step_t0) * 1000
                result.steps.append(step)
                break

            else:
                # Execute tool
                observation, search_count, page_count = _execute_tool(
                    step.action, step.action_input,
                    search_provider, page_provider,
                    search_count, page_count,
                )
                step.observation = observation[:500]  # cap observation length for history

                history_parts.append(
                    f"Step {step_num}: Thought: {step.thought}\n"
                    f"Action: {step.action}({str(step.action_input)[:80]})\n"
                    f"Observation: {observation[:300]}"
                )

        except Exception as e:
            step.thought = f"Error: {e}"
            step.action = "error"
            step.observation = str(e)
            logger.warning("Investigation step %d failed: %s", step_num, e)

        step.duration_ms = (time.time() - step_t0) * 1000
        result.steps.append(step)

    result.total_searches = search_count
    result.total_page_reads = page_count
    result.total_duration_ms = (time.time() - t0) * 1000
    result.success = len(result.findings) > 0

    # Convert findings to evidence records
    for f in result.findings:
        result.evidence_records.append(EvidenceRecord(
            source_id="investigator_agent",
            source_family="investigation",
            target_type="person",
            target_value=target_name,
            content_type="finding",
            title=f.get("fact_type", ""),
            snippet=f.get("fact_text", ""),
            url_or_citation=f.get("source_url", ""),
            confidence=f.get("confidence", 0.5),
            reliability_tier="medium",
            relevance_score=0.8,
        ))

    return result


# ---------------------------------------------------------------------------
# Dossier integration
# ---------------------------------------------------------------------------

def integrate_investigation(ctx: TargetContext, result: InvestigationResult) -> TargetContext:
    """Merge investigation findings into the target dossier."""
    for finding in result.findings:
        fact_type = finding.get("fact_type", "other")
        fact_text = finding.get("fact_text", "")
        source_url = finding.get("source_url", "")
        confidence = finding.get("confidence", 0.5)

        ev = EvidenceLink(
            source_family="investigation",
            snippet=fact_text[:100],
            url_or_citation=source_url,
            confidence=confidence,
        )

        # Map to dossier fields
        if fact_type in ("employment", "role", "affiliation"):
            short = fact_text[:60]
            if fact_type in ("employment", "affiliation"):
                if short not in ctx.known_affiliations and len(ctx.known_affiliations) < 5:
                    ctx.known_affiliations.append(short)
            if fact_type == "role":
                if short not in ctx.known_roles and len(ctx.known_roles) < 5:
                    ctx.known_roles.append(short)

        # Add as supported claim
        existing = next((s for s in ctx.supported_claims if s.claim_text[:40] == fact_text[:40]), None)
        if existing:
            existing.support_count += 1
            existing.evidence.append(ev)
        elif len(ctx.supported_claims) < 10:
            ctx.supported_claims.append(SupportedClaim(
                claim_text=fact_text,
                claim_type=f"investigated_{fact_type}",
                evidence=[ev],
                confidence=confidence,
            ))

        ctx.evidence_count += 1

    if result.summary:
        if len(ctx.statement_patterns) < 5:
            ctx.statement_patterns.append(f"investigation: {result.summary[:80]}")

    return ctx
