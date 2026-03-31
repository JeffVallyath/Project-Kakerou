"""Tests for DossierStore — persistence, identity, and engine integration."""

from btom_engine.osint.dossier_store import DossierStore
from btom_engine.osint.target_context import (
    TargetContext, SupportedClaim, ContradictedClaim, EvidenceLink,
)
from btom_engine.engine import BTOMEngine, ConversationTurn
from btom_engine.config import DEFAULT_HYPOTHESES


# --- DossierStore unit tests ---

def test_store_save_and_load(tmp_path):
    store = DossierStore(tmp_path / "dossiers")

    ctx = TargetContext(target_id="user_123", evidence_count=3, overall_consistency=0.8)
    ctx.known_affiliations = ["TechCorp"]
    ctx.known_roles = ["Engineer"]
    ctx.supported_claims.append(SupportedClaim(
        claim_text="I work at TechCorp", claim_type="affiliation_claim",
        support_count=2, confidence=0.7,
        evidence=[EvidenceLink(source_family="public_profile", snippet="Engineer at TechCorp", confidence=0.7)],
    ))
    ctx.created_session = "sess_001"
    ctx.last_updated_session = "sess_001"

    assert store.save(ctx)
    assert store.exists("user_123")

    loaded = store.load("user_123")
    assert loaded is not None
    assert loaded.target_id == "user_123"
    assert loaded.evidence_count == 3
    assert loaded.known_affiliations == ["TechCorp"]
    assert len(loaded.supported_claims) == 1
    assert loaded.supported_claims[0].support_count == 2
    assert len(loaded.supported_claims[0].evidence) == 1


def test_store_load_nonexistent(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    assert store.load("nonexistent") is None


def test_store_delete(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    ctx = TargetContext(target_id="to_delete", evidence_count=1)
    store.save(ctx)
    assert store.exists("to_delete")
    store.delete("to_delete")
    assert not store.exists("to_delete")


def test_store_list_targets(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    store.save(TargetContext(target_id="alice"))
    store.save(TargetContext(target_id="bob"))
    targets = store.list_targets()
    assert "alice" in targets
    assert "bob" in targets


def test_store_resolve_exact_identity(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    store.save(TargetContext(target_id="user_123"))
    assert store.resolve_identity("user_123") == "user_123"
    assert store.resolve_identity("unknown") is None


def test_store_resolve_alias_identity(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    ctx = TargetContext(target_id="user_123", aliases=["@jane_smith", "jsmith"])
    store.save(ctx)

    # Exact alias match should resolve
    assert store.resolve_identity("new_id", aliases=["@jane_smith"]) == "user_123"
    # Non-matching alias should not resolve
    assert store.resolve_identity("new_id", aliases=["@unknown"]) is None


def test_store_no_fuzzy_merge(tmp_path):
    """Ambiguous names alone should NOT auto-merge."""
    store = DossierStore(tmp_path / "dossiers")
    store.save(TargetContext(target_id="jane_smith_1"))
    store.save(TargetContext(target_id="jane_smith_2"))

    # No aliases, no resolution — even though names are similar
    assert store.resolve_identity("jane_smith_3") is None


def test_store_empty_target_id_not_saved(tmp_path):
    store = DossierStore(tmp_path / "dossiers")
    ctx = TargetContext(target_id="")
    assert not store.save(ctx)


# --- Engine integration tests ---

def test_engine_no_store_works_normally(tmp_path):
    """Engine without dossier store should work exactly as before."""
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()
    result = engine.process_turn(ConversationTurn(target_text="Hello."))
    assert result.state is not None
    assert isinstance(result.target_context_summary, dict)


def test_engine_with_store_creates_dossier(tmp_path):
    """Engine with store should create and persist a dossier."""
    store = DossierStore(tmp_path / "dossiers")
    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        dossier_store=store,
        target_id="test_target",
    )
    engine.reset()

    engine.record_user_turn("What happened?")
    engine.process_turn(ConversationTurn(target_text="I already explained this before."))

    # Dossier should be persisted
    assert store.exists("test_target")
    loaded = store.load("test_target")
    assert loaded is not None
    assert loaded.target_id == "test_target"


def test_engine_reloads_dossier_across_sessions(tmp_path):
    """Session 2 should reload dossier from session 1."""
    store = DossierStore(tmp_path / "dossiers")

    # Session 1: build context
    engine1 = BTOMEngine(
        state_path=tmp_path / "state1.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        dossier_store=store,
        target_id="persistent_target",
    )
    engine1.reset()
    engine1.record_user_turn("Did you send it?")
    engine1.process_turn(ConversationTurn(target_text="I already explained, I sent the report at 3pm."))

    # Verify dossier was saved
    saved = store.load("persistent_target")
    evidence_after_s1 = saved.evidence_count if saved else 0
    print(f"Session 1: evidence_count={evidence_after_s1}")

    # Session 2: new engine, same target_id → should reload
    engine2 = BTOMEngine(
        state_path=tmp_path / "state2.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        dossier_store=store,
        target_id="persistent_target",
    )

    # Context should be loaded from store
    summary = engine2._target_context.summary()
    print(f"Session 2 loaded: {summary}")
    assert engine2._target_context.target_id == "persistent_target"

    # If session 1 built any context, session 2 should see it
    if evidence_after_s1 > 0:
        assert engine2._target_context.evidence_count >= evidence_after_s1


def test_engine_reset_preserves_persistent_dossier(tmp_path):
    """Reset should reload dossier from store, not destroy it."""
    store = DossierStore(tmp_path / "dossiers")

    engine = BTOMEngine(
        state_path=tmp_path / "state.json",
        hypotheses=dict(DEFAULT_HYPOTHESES),
        dossier_store=store,
        target_id="preserved_target",
    )
    engine.reset()
    engine.record_user_turn("What happened?")
    engine.process_turn(ConversationTurn(target_text="I already told you about this."))

    # Save should have happened
    assert store.exists("preserved_target")

    # Reset
    engine.reset()

    # Dossier should still exist in store
    assert store.exists("preserved_target")
    # Engine should have reloaded it
    assert engine._target_context.target_id == "preserved_target"


def test_low_confidence_does_not_overwrite_strong(tmp_path):
    """Weak evidence should not degrade strong stored fields."""
    store = DossierStore(tmp_path / "dossiers")

    # Pre-populate a strong dossier
    strong_ctx = TargetContext(
        target_id="strong_target", evidence_count=5, overall_consistency=0.9,
    )
    strong_ctx.supported_claims.append(SupportedClaim(
        claim_text="I work at BigCorp", claim_type="affiliation_claim",
        support_count=3, confidence=0.85,
        evidence=[EvidenceLink(source_family="public_profile", snippet="Senior at BigCorp", confidence=0.85)],
    ))
    store.save(strong_ctx)

    # Load and add weak evidence
    from btom_engine.osint.target_context import synthesize_context
    loaded = store.load("strong_target")

    weak_comp = [{"claim": "I work at BigCorp", "outcome": "supported_by_prior",
                   "confidence": 0.2, "rationale": "weak match"}]
    loaded = synthesize_context(loaded, weak_comp, [], [], current_turn=10)

    # Strong claim should still have high confidence (incremented slightly, not overwritten)
    claim = loaded.supported_claims[0]
    assert claim.confidence >= 0.85  # should not decrease
    assert claim.support_count == 4  # incremented
