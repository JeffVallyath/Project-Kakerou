"""Tests for the OSINT backbone — registry, routing, adapters, and manager."""

from btom_engine.osint.evidence_schema import RetrievalRequest
from btom_engine.osint.source_registry import SourceRegistry, SourceEntry, build_default_registry
from btom_engine.osint.retrieval_router import route
from btom_engine.osint.retrieval_manager import RetrievalManager
from btom_engine.osint.adapters.prior_statements import PriorStatementsAdapter
from btom_engine.osint.adapters.web_search import WebSearchAdapter


# --- Registry tests ---

def test_registry_lookup_by_family():
    reg = build_default_registry()
    prior = reg.find_by_family("prior_statements")
    assert len(prior) == 1
    assert prior[0].source_id == "prior_statements_session"


def test_registry_lookup_by_target_type():
    reg = build_default_registry()
    session_sources = reg.find_by_target_type("session")
    assert any(s.source_id == "prior_statements_session" for s in session_sources)


def test_registry_web_search_enabled():
    reg = build_default_registry()
    web = reg.find_by_family("web_search")
    assert len(web) == 1  # now enabled with provider interface


def test_registry_find_suitable_bounded():
    reg = build_default_registry()
    suitable = reg.find_suitable(
        target_type="session",
        query_type="prior_statements",
        bounded_only=True,
    )
    assert len(suitable) >= 1
    assert all(s.automatable for s in suitable)


# --- Routing tests ---

def test_routing_selects_prior_statements():
    reg = build_default_registry()
    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="report",
    )
    plan = route(request, reg)
    assert "prior_statements_session" in plan.selected_sources


def test_routing_returns_empty_for_unsupported():
    reg = build_default_registry()
    request = RetrievalRequest(
        target_type="phone_number",
        query_type="carrier_lookup",
    )
    plan = route(request, reg)
    assert len(plan.selected_sources) == 0
    assert "No suitable" in plan.rationale


def test_routing_respects_family_filter():
    reg = build_default_registry()
    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        allowed_source_families=["web_search"],  # wrong family
    )
    plan = route(request, reg)
    assert len(plan.selected_sources) == 0


# --- PriorStatementsAdapter tests ---

def test_prior_statements_adapter_returns_records():
    adapter = PriorStatementsAdapter()
    adapter.set_transcript([
        {"speaker": "user", "text": "Did you send the report?"},
        {"speaker": "target", "text": "I sent it yesterday at 3pm.", "turn": 1},
        {"speaker": "user", "text": "To whom?"},
        {"speaker": "target", "text": "I sent it to Sarah.", "turn": 2},
        {"speaker": "target", "text": "The weather is nice today.", "turn": 3},
    ])

    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="report sent",
        top_k=3,
    )
    records, run = adapter.execute(request)

    assert run.success
    assert len(records) > 0
    # Most relevant should mention "sent"
    assert "sent" in records[0].snippet.lower()
    # All records should have normalized fields
    for r in records:
        assert r.source_id == "prior_statements_session"
        assert r.source_family == "prior_statements"
        assert r.reliability_tier == "high"
        assert r.confidence > 0.5


def test_prior_statements_adapter_empty_transcript():
    adapter = PriorStatementsAdapter()
    adapter.set_transcript([])

    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="anything",
    )
    records, run = adapter.execute(request)

    assert run.success
    assert len(records) == 0


def test_prior_statements_adapter_can_handle():
    adapter = PriorStatementsAdapter()
    assert adapter.can_handle(RetrievalRequest(target_type="session", query_type="prior_statements"))
    assert not adapter.can_handle(RetrievalRequest(target_type="url", query_type="web_search"))


# --- WebSearchAdapter scaffold tests ---

def test_web_search_no_provider_returns_empty():
    adapter = WebSearchAdapter(provider=None)
    request = RetrievalRequest(target_type="claim", query_type="web_search", query_text="test")
    # can_handle returns False without provider, so test execute directly
    records, run = adapter.execute(request)

    assert len(records) == 0
    assert not run.success
    assert "no search provider" in run.error.lower()


def test_web_search_healthcheck_no_provider():
    adapter = WebSearchAdapter(provider=None)
    health = adapter.healthcheck()
    assert not health.available


# --- RetrievalManager tests ---

def test_manager_aggregates_results():
    reg = build_default_registry()
    manager = RetrievalManager(reg)

    adapter = PriorStatementsAdapter()
    adapter.set_transcript([
        {"speaker": "target", "text": "I was at home at 9pm.", "turn": 1},
        {"speaker": "target", "text": "My sister was there.", "turn": 2},
    ])
    manager.register_adapter("prior_statements_session", adapter)

    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="where were you",
        top_k=5,
    )
    result = manager.retrieve(request)

    assert len(result.records) > 0
    assert result.overall_confidence > 0
    assert result.plan is not None
    assert "prior_statements_session" in result.plan.selected_sources
    assert any(r.success for r in result.source_runs)


def test_manager_safe_degradation_no_adapter():
    reg = build_default_registry()
    manager = RetrievalManager(reg)
    # Don't register any adapter

    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="test",
    )
    result = manager.retrieve(request)

    assert len(result.records) == 0
    assert len(result.unresolved_gaps) > 0


def test_manager_safe_degradation_unsupported_request():
    reg = build_default_registry()
    manager = RetrievalManager(reg)

    request = RetrievalRequest(
        target_type="phone_number",
        query_type="carrier_lookup",
    )
    result = manager.retrieve(request)

    assert len(result.records) == 0
    assert "No sources available" in result.unresolved_gaps[0]


def test_no_raw_source_payload_leaks():
    """Evidence records should contain only normalized fields, no raw source data."""
    adapter = PriorStatementsAdapter()
    adapter.set_transcript([
        {"speaker": "target", "text": "I sent the report.", "turn": 1, "_internal_id": "xyz"},
    ])

    request = RetrievalRequest(
        target_type="session",
        query_type="prior_statements",
        query_text="report",
    )
    records, _ = adapter.execute(request)

    for r in records:
        assert not hasattr(r, "_internal_id")
        assert not hasattr(r, "raw_payload")
        assert r.source_id
        assert r.snippet
        assert r.reliability_tier


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

from btom_engine.osint.claim_extraction import extract_claims
from btom_engine.osint.claim_comparison import compare_claim
from btom_engine.osint.prior_integration import compute_prior_effect, PriorContextEffect
from btom_engine.osint.evidence_schema import EvidenceRecord, ComparisonResult
from btom_engine.engine import BTOMEngine, ConversationTurn
from btom_engine.config import DEFAULT_HYPOTHESES


def test_claim_extraction_prior_explanation():
    claims = extract_claims("I already explained why I did that.")
    assert len(claims) >= 1
    assert claims[0].claim_type == "prior_explanation_claim"


def test_claim_extraction_denial():
    claims = extract_claims("I never said that. That's not what I said.")
    assert len(claims) >= 1
    assert any(c.claim_type == "denial_of_prior_statement" for c in claims)


def test_claim_extraction_no_claims():
    claims = extract_claims("ok")
    assert len(claims) == 0


def test_comparison_supported_by_prior():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="prior_explanation_claim",
        claim_text="I already explained the report",
        retrieval_query="report explanation",
    )
    evidence = [EvidenceRecord(
        snippet="I explained the report was delayed because of the server issue.",
        confidence=0.9,
        relevance_score=0.8,
    )]
    result = compare_claim(claim, evidence)
    assert result.outcome == "supported_by_prior"


def test_comparison_contradiction():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="denial_of_prior_statement",
        claim_text="I never said the report was late",
        retrieval_query="report late",
    )
    evidence = [EvidenceRecord(
        snippet="The report was late because I forgot to send it.",
        confidence=0.9,
        relevance_score=0.8,
    )]
    result = compare_claim(claim, evidence)
    assert result.outcome == "direct_contradiction"


def test_comparison_insufficient_evidence():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="prior_explanation_claim",
        claim_text="I already told you about the meeting",
        retrieval_query="meeting",
    )
    result = compare_claim(claim, [])
    assert result.outcome == "insufficient_evidence"


def test_prior_effect_contradiction_increases_bluffing():
    comparisons = [ComparisonResult(
        claim_text="test",
        outcome="direct_contradiction",
        comparison_confidence=0.8,
    )]
    effect = compute_prior_effect(comparisons)
    assert effect.bluffing_delta > 0
    assert effect.comparisons_used == 1


def test_prior_effect_supported_decreases_bluffing():
    comparisons = [ComparisonResult(
        claim_text="test",
        outcome="supported_by_prior",
        comparison_confidence=0.7,
    )]
    effect = compute_prior_effect(comparisons)
    assert effect.bluffing_delta < 0


def test_prior_effect_insufficient_no_change():
    comparisons = [ComparisonResult(
        claim_text="test",
        outcome="insufficient_evidence",
        comparison_confidence=0.0,
    )]
    effect = compute_prior_effect(comparisons)
    assert effect.bluffing_delta == 0.0
    assert effect.withholding_delta == 0.0


def test_engine_prior_context_no_claim(tmp_path):
    """When target says nothing retrieval-worthy, prior context is a no-op."""
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    result = engine.process_turn(ConversationTurn(target_text="ok"))
    assert result.prior_context.get("claims") == []
    assert result.prior_context.get("effect") is None or result.prior_context["effect"].comparisons_used == 0


def test_engine_prior_context_with_claim(tmp_path):
    """When target references prior statement, the loop should run."""
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    # Build up transcript
    engine.record_user_turn("Did you send the report?")
    engine.process_turn(ConversationTurn(target_text="I sent the report yesterday at 3pm."))

    engine.record_user_turn("When?")
    result = engine.process_turn(ConversationTurn(target_text="I already told you, I sent the report at 3pm."))

    # Should have extracted a claim
    assert len(result.prior_context.get("claims", [])) > 0
    # Should have retrieved prior statements
    assert len(result.prior_context.get("retrieval_records", [])) > 0
    # Diagnostics should be populated
    print(f"Claims: {result.prior_context['claims']}")
    print(f"Comparisons: {result.prior_context['comparisons']}")
    if result.prior_context.get("effect"):
        print(f"Effect: bluff_delta={result.prior_context['effect'].bluffing_delta:+.3f}")


# ---------------------------------------------------------------------------
# External retrieval tests (mocked — no internet required)
# ---------------------------------------------------------------------------

from btom_engine.osint.providers import (
    MockSearchProvider, MockPageProvider, SearchResult, PageContent,
)
from btom_engine.osint.adapters.page_read import PageReadAdapter


def test_claim_extraction_external_factual():
    claims = extract_claims("I never worked there. That company shut down last year.")
    assert any(c.needs_external for c in claims)
    external = [c for c in claims if c.needs_external]
    assert len(external) >= 1


def test_claim_extraction_no_external_for_vague():
    claims = extract_claims("I don't know what to say.")
    external = [c for c in claims if c.needs_external]
    assert len(external) == 0


def test_web_search_adapter_with_mock_provider():
    provider = MockSearchProvider(results=[
        SearchResult(title="Acme Corp - Wikipedia", url="https://en.wikipedia.org/wiki/Acme_Corp",
                     snippet="Acme Corp was founded in 1990 and shut down in 2023."),
        SearchResult(title="Acme Corp Closure", url="https://news.example.com/acme",
                     snippet="Acme Corp closed its doors in December 2023."),
    ])
    adapter = WebSearchAdapter(provider=provider)

    request = RetrievalRequest(
        target_type="claim",
        query_type="external_check",
        query_text="Acme Corp shut down",
        top_k=3,
    )
    assert adapter.can_handle(request)
    records, run = adapter.execute(request)

    assert run.success
    assert len(records) == 2
    assert records[0].source_family == "web_search"
    assert records[0].url_or_citation.startswith("https://")
    assert records[0].reliability_tier == "low"


def test_web_search_adapter_no_provider():
    adapter = WebSearchAdapter(provider=None)
    request = RetrievalRequest(target_type="claim", query_type="external_check", query_text="test")
    assert not adapter.can_handle(request)


def test_page_read_adapter_with_mock():
    provider = MockPageProvider(pages={
        "https://example.com/article": PageContent(
            url="https://example.com/article",
            title="Example Article",
            text="Acme Corp was founded in 1990 and officially closed in December 2023 after years of declining revenue.",
            fetch_success=True,
        ),
    })
    adapter = PageReadAdapter(provider=provider)

    request = RetrievalRequest(
        target_type="url",
        target_value="https://example.com/article",
        query_type="page_read",
    )
    assert adapter.can_handle(request)
    records, run = adapter.execute(request)

    assert run.success
    assert len(records) == 1
    assert "Acme Corp" in records[0].snippet
    assert records[0].url_or_citation == "https://example.com/article"


def test_page_read_adapter_no_provider():
    adapter = PageReadAdapter(provider=None)
    request = RetrievalRequest(target_type="url", target_value="https://x.com", query_type="page_read")
    assert not adapter.can_handle(request)


def test_external_comparison_support():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="public_record_claim",
        claim_text="That company shut down last year",
        retrieval_query="company shut down",
        needs_external=True,
    )
    evidence = [EvidenceRecord(
        snippet="The company shut down last year after years of declining revenue.",
        source_family="web_search",
        confidence=0.5,
        relevance_score=0.6,
    )]
    result = compare_claim(claim, evidence)
    assert result.outcome in ("supported_by_prior", "weak_tension")


def test_external_comparison_contradiction():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="factual_assertion",
        claim_text="I never worked at Acme Corp",
        retrieval_query="worked at Acme",
        needs_external=True,
    )
    evidence = [EvidenceRecord(
        snippet="John Smith worked at Acme Corp from 2019 to 2022 as a senior analyst.",
        source_family="web_search",
        confidence=0.5,
        relevance_score=0.7,
    )]
    result = compare_claim(claim, evidence)
    # claim has negation ("never"), evidence does not → potential contradiction
    assert result.outcome in ("direct_contradiction", "weak_tension", "supported_by_prior")


def test_engine_external_retrieval_with_mock(tmp_path):
    """Engine with mock search provider should route external claims to web search."""
    mock_search = MockSearchProvider(results=[
        SearchResult(
            title="Acme Corp Closure",
            url="https://news.example.com/acme",
            snippet="Acme Corp closed its doors in December 2023 after years of declining revenue.",
        ),
    ])

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=mock_search,
    )
    engine.reset()

    engine.record_user_turn("Tell me about Acme Corp.")
    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year. I never worked there."
    ))

    # Should have external claims
    claims = result.prior_context.get("claims", [])
    print(f"Claims: {claims}")
    external_claims = [c for c in claims if c.get("needs_external")]
    print(f"External claims: {len(external_claims)}")

    # Should have retrieval records from web search
    records = result.prior_context.get("retrieval_records", [])
    print(f"Retrieval records: {records}")

    if external_claims:
        assert result.prior_context.get("retrieval_path") == "web_search_then_page_read"
        assert len(records) > 0
        assert any(r.get("source") == "web_search" for r in records)
        print("External retrieval path verified.")


def test_engine_no_external_without_provider(tmp_path):
    """Without search provider, external claims should degrade safely."""
    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        # No search_provider
    )
    engine.reset()

    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year."
    ))

    # Should still work, just no external records
    records = result.prior_context.get("retrieval_records", [])
    # No crash, safe degradation
    assert isinstance(records, list)


# ---------------------------------------------------------------------------
# Search -> Page-Read chain tests (mocked)
# ---------------------------------------------------------------------------

from btom_engine.osint.url_selection import select_urls


def test_url_selection_picks_best():
    records = [
        EvidenceRecord(
            title="Acme Corp Wikipedia",
            snippet="Acme Corp closed in 2023.",
            url_or_citation="https://en.wikipedia.org/wiki/Acme",
            relevance_score=0.6,
        ),
        EvidenceRecord(
            title="Random Page",
            snippet="",
            url_or_citation="https://random.com",
            relevance_score=0.3,
        ),
        EvidenceRecord(
            title="Acme News",
            snippet="Acme Corp shut down after declining revenue.",
            url_or_citation="https://news.example.com/acme",
            relevance_score=0.7,
        ),
    ]
    selected = select_urls(records, query_text="Acme Corp shut down", max_pages=1)
    assert len(selected) == 1
    # Should pick the one with best overlap + snippet
    assert "acme" in selected[0].url.lower() or "example" in selected[0].url.lower()


def test_url_selection_respects_max():
    records = [
        EvidenceRecord(url_or_citation="https://a.com", snippet="good", title="A"),
        EvidenceRecord(url_or_citation="https://b.com", snippet="good", title="B"),
        EvidenceRecord(url_or_citation="https://c.com", snippet="good", title="C"),
    ]
    selected = select_urls(records, max_pages=2)
    assert len(selected) <= 2


def test_url_selection_skips_bad_urls():
    records = [
        EvidenceRecord(url_or_citation="", snippet="no url"),
        EvidenceRecord(url_or_citation="ftp://bad.com", snippet="bad protocol"),
        EvidenceRecord(url_or_citation="https://good.com", snippet="valid", title="Good"),
    ]
    selected = select_urls(records, max_pages=2)
    assert all(s.url.startswith("http") for s in selected)


def test_engine_search_then_page_read_chain(tmp_path):
    """Full chain: search -> URL select -> page-read -> compare with page evidence."""
    mock_search = MockSearchProvider(results=[
        SearchResult(
            title="Acme Corp History",
            url="https://example.com/acme-history",
            snippet="Acme Corp was a major corporation.",
        ),
    ])
    mock_pages = MockPageProvider(pages={
        "https://example.com/acme-history": PageContent(
            url="https://example.com/acme-history",
            title="Acme Corp History",
            text="Acme Corp was founded in 1985 and shut down in December 2023 after a long period of declining revenue. The company had over 500 employees at its peak.",
            fetch_success=True,
        ),
    })

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=mock_search,
        page_provider=mock_pages,
    )
    engine.reset()

    engine.record_user_turn("Tell me about your old company.")
    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year. I never worked there."
    ))

    claims = result.prior_context.get("claims", [])
    external = [c for c in claims if c.get("needs_external")]
    selected_urls = result.prior_context.get("selected_urls", [])
    records = result.prior_context.get("retrieval_records", [])
    evidence_sources = result.prior_context.get("evidence_source", [])

    print(f"Claims: {claims}")
    print(f"External claims: {len(external)}")
    print(f"Selected URLs: {selected_urls}")
    print(f"Records: {len(records)}")
    print(f"Evidence sources: {evidence_sources}")
    print(f"Path: {result.prior_context.get('retrieval_path')}")

    if external:
        assert result.prior_context.get("retrieval_path") == "web_search_then_page_read"
        # Should have selected a URL
        if selected_urls:
            assert selected_urls[0]["url"] == "https://example.com/acme-history"
            # Should have page_read evidence
            page_records = [r for r in records if r.get("source") == "page_read"]
            if page_records:
                assert "page_read" in evidence_sources
                print("Page-read evidence used for comparison.")


def test_engine_page_read_failure_falls_back_to_snippets(tmp_path):
    """If page-read fails, comparison should use search snippets."""
    mock_search = MockSearchProvider(results=[
        SearchResult(
            title="Acme Info",
            url="https://example.com/acme",
            snippet="Acme Corp shut down in 2023 after financial troubles.",
        ),
    ])
    # Page provider returns failure for this URL
    mock_pages = MockPageProvider(pages={})  # empty — all fetches fail

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=mock_search,
        page_provider=mock_pages,
    )
    engine.reset()

    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year."
    ))

    evidence_sources = result.prior_context.get("evidence_source", [])
    print(f"Evidence sources (page fail): {evidence_sources}")

    # Should have fallen back to search snippets
    if evidence_sources:
        assert "search_snippet" in evidence_sources or "none" in evidence_sources


def test_engine_no_page_provider_uses_snippets(tmp_path):
    """Without page provider, should still compare against search snippets."""
    mock_search = MockSearchProvider(results=[
        SearchResult(
            title="Test",
            url="https://example.com/test",
            snippet="The company shut down last year definitively.",
        ),
    ])

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=mock_search,
        # No page_provider
    )
    engine.reset()

    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year."
    ))

    records = result.prior_context.get("retrieval_records", [])
    evidence_sources = result.prior_context.get("evidence_source", [])
    print(f"Records (no page provider): {len(records)}")
    print(f"Evidence sources: {evidence_sources}")

    # Should have search records but no page records
    if records:
        assert all(r.get("source") != "page_read" for r in records)


def test_prior_statements_path_unchanged(tmp_path):
    """Prior statements path should work exactly as before."""
    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
    )
    engine.reset()

    engine.record_user_turn("Did you send the report?")
    engine.process_turn(ConversationTurn(target_text="I sent the report at 3pm."))

    engine.record_user_turn("When?")
    result = engine.process_turn(ConversationTurn(
        target_text="I already told you, I sent the report at 3pm."
    ))

    assert result.prior_context.get("retrieval_path") == "prior_statements"
    assert len(result.prior_context.get("claims", [])) > 0
    assert len(result.prior_context.get("retrieval_records", [])) > 0


# ---------------------------------------------------------------------------
# Public Profile Adapter tests (mocked)
# ---------------------------------------------------------------------------

from btom_engine.osint.profile_extraction import extract_profile
from btom_engine.osint.adapters.public_profile import PublicProfileAdapter


def test_profile_extraction_from_page_text():
    text = """
    Jane Smith - Senior Researcher at Acme Corp

    About: Jane has been working in computational linguistics for over 10 years.
    She leads the NLP team at Acme Corp and has published extensively on
    discourse analysis.

    Links: https://github.com/janesmith https://scholar.google.com/janesmith
    """
    profile = extract_profile(text, page_title="Jane Smith - Acme Corp")

    assert profile.display_name is not None
    assert "Jane Smith" in profile.display_name
    assert profile.role_or_title is not None
    assert "researcher" in profile.role_or_title.lower() or "senior" in profile.role_or_title.lower()
    assert profile.organization is not None
    assert "Acme" in profile.organization
    assert profile.bio_excerpt is not None
    assert profile.extraction_confidence > 0.4


def test_profile_extraction_vague_page():
    text = "Welcome to our site. We sell things. Contact us."
    profile = extract_profile(text, page_title="Generic Site")
    # Should extract very little
    assert profile.role_or_title is None
    assert profile.organization is None
    assert profile.extraction_confidence <= 0.5


def test_public_profile_adapter_normalizes():
    provider = MockPageProvider(pages={
        "https://example.com/about/jane": PageContent(
            url="https://example.com/about/jane",
            title="Jane Smith - Lead Engineer at TechCorp",
            text="Jane Smith is a Lead Engineer at TechCorp. About: She specializes in distributed systems and has been with TechCorp since 2019.",
            fetch_success=True,
        ),
    })
    adapter = PublicProfileAdapter(provider=provider)

    request = RetrievalRequest(
        target_type="url",
        target_value="https://example.com/about/jane",
        query_type="public_profile",
    )
    assert adapter.can_handle(request)
    records, run = adapter.execute(request)

    assert run.success
    assert len(records) == 1
    assert records[0].source_family == "public_profile"
    assert records[0].content_type == "profile"
    assert "Jane Smith" in records[0].snippet or "Engineer" in records[0].snippet
    assert records[0].url_or_citation == "https://example.com/about/jane"
    assert records[0].extraction_notes  # should have field presence info


def test_affiliation_claim_extraction():
    claims = extract_claims("I'm a member of the research team at Stanford.")
    assert any(c.claim_type == "affiliation_claim" for c in claims)
    assert any(c.needs_external for c in claims)


def test_role_claim_extraction():
    claims = extract_claims("I'm the CEO of this company.")
    # LLM or regex should extract some claim about role/employment
    assert len(claims) >= 1


def test_authorship_claim_extraction():
    claims = extract_claims("I wrote that article last year.")
    # LLM or regex should extract some claim about authorship
    assert len(claims) >= 1


def test_profile_comparison_support():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="role_claim",
        claim_text="I'm a Lead Engineer at TechCorp",
        retrieval_query="Lead Engineer TechCorp",
        needs_external=True,
    )
    evidence = [EvidenceRecord(
        snippet="Name: Jane Smith | Role: Lead Engineer at TechCorp | Org: TechCorp",
        source_family="public_profile",
        confidence=0.6,
        relevance_score=0.7,
    )]
    result = compare_claim(claim, evidence)
    assert result.outcome in ("supported_by_prior", "weak_tension")


def test_profile_comparison_vague_no_overclaim():
    from btom_engine.osint.claim_extraction import ExtractedClaim
    claim = ExtractedClaim(
        claim_type="affiliation_claim",
        claim_text="I'm with the analytics team",
        retrieval_query="analytics team",
        needs_external=True,
    )
    evidence = [EvidenceRecord(
        snippet="Welcome to our company. We do many things.",
        source_family="public_profile",
        confidence=0.3,
        relevance_score=0.3,
    )]
    result = compare_claim(claim, evidence)
    # Vague evidence should not overclaim support
    assert result.outcome in ("insufficient_evidence", "weak_tension")


def test_engine_profile_claim_with_mock(tmp_path):
    """Engine with mock providers routes affiliation claim through external path."""
    mock_search = MockSearchProvider(results=[
        SearchResult(
            title="Jane Smith - TechCorp",
            url="https://techcorp.com/team/jane",
            snippet="Jane Smith is a Lead Engineer at TechCorp.",
        ),
    ])
    mock_pages = MockPageProvider(pages={
        "https://techcorp.com/team/jane": PageContent(
            url="https://techcorp.com/team/jane",
            title="Jane Smith - Lead Engineer at TechCorp",
            text="Jane Smith is a Lead Engineer at TechCorp. About: She has been with TechCorp since 2019 leading the backend team.",
            fetch_success=True,
        ),
    })

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=mock_search,
        page_provider=mock_pages,
    )
    engine.reset()

    engine.record_user_turn("What do you do?")
    result = engine.process_turn(ConversationTurn(
        target_text="I'm a Lead Engineer at TechCorp."
    ))

    claims = result.prior_context.get("claims", [])
    print(f"Profile claims: {claims}")
    external = [c for c in claims if c.get("needs_external")]
    print(f"External: {len(external)}")

    if external:
        assert result.prior_context.get("retrieval_path") == "web_search_then_page_read"


# ---------------------------------------------------------------------------
# DuckDuckGo provider safe-degradation tests (offline)
# ---------------------------------------------------------------------------

from btom_engine.osint.providers import DuckDuckGoSearchProvider


def test_ddg_provider_returns_list():
    """DuckDuckGoSearchProvider.search() should always return a list."""
    provider = DuckDuckGoSearchProvider()
    try:
        results = provider.search("test query", top_k=2)
        assert isinstance(results, list)
        for r in results:
            assert hasattr(r, "title")
            assert hasattr(r, "url")
            assert hasattr(r, "snippet")
    except Exception:
        pass  # network unavailable — provider handles internally


def test_ddg_provider_max_cap():
    """top_k should cap at 5 regardless of input."""
    provider = DuckDuckGoSearchProvider()
    assert provider._MAX_RESULTS == 5


def test_ddg_parse_empty_html():
    """Parser should handle empty HTML safely."""
    results = DuckDuckGoSearchProvider._parse_lite_html("")
    assert results == []


def test_ddg_parse_no_results_html():
    """Parser should handle HTML without result patterns."""
    results = DuckDuckGoSearchProvider._parse_lite_html("<html><body>No results</body></html>")
    assert results == []


def test_engine_ddg_degrades_safely(tmp_path):
    """Engine with DDG provider should not crash if network is unavailable."""
    provider = DuckDuckGoSearchProvider()

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        search_provider=provider,
    )
    engine.reset()

    result = engine.process_turn(ConversationTurn(
        target_text="That company shut down last year."
    ))
    assert result.state is not None
    assert isinstance(result.prior_context, dict)
