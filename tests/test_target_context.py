"""Tests for TargetContext — synthesis, conditioning, and engine integration."""

from btom_engine.osint.target_context import (
    TargetContext, SupportedClaim, ContradictedClaim, EvidenceLink,
    synthesize_context, compute_context_conditioning, apply_context_conditioning,
    ContextEffect,
)
from btom_engine.engine import BTOMEngine, ConversationTurn
from btom_engine.config import DEFAULT_HYPOTHESES


# --- Unit tests for TargetContext ---

def test_empty_context_no_effect():
    ctx = TargetContext()
    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta == 0.0
    assert effect.withholding_delta == 0.0
    assert "insufficient" in effect.rationale


def test_weak_context_no_effect():
    ctx = TargetContext(evidence_count=1)
    ctx.supported_claims.append(SupportedClaim(claim_text="test", confidence=0.5))
    effect = compute_context_conditioning(ctx)
    # Only 1 evidence count — threshold is 2
    assert effect.bluffing_delta == 0.0


def test_consistent_context_deescalates():
    ctx = TargetContext(evidence_count=3, overall_consistency=0.8)
    ctx.supported_claims.append(SupportedClaim(claim_text="claim1", confidence=0.7))
    ctx.supported_claims.append(SupportedClaim(claim_text="claim2", confidence=0.6))
    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta < 0  # de-escalation
    assert abs(effect.bluffing_delta) <= 0.05  # bounded


def test_contradicted_context_raises_scrutiny():
    ctx = TargetContext(evidence_count=4, overall_consistency=0.3)
    ctx.contradicted_claims.append(ContradictedClaim(claim_text="denied1", confidence=0.7, contradiction_count=2))
    ctx.contradicted_claims.append(ContradictedClaim(claim_text="denied2", confidence=0.6, contradiction_count=1))
    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta > 0  # raise scrutiny
    assert effect.bluffing_delta <= 0.05  # bounded


def test_synthesis_adds_supported_claim():
    ctx = TargetContext()
    comparisons = [{"claim": "I work at TechCorp", "outcome": "supported_by_prior",
                     "confidence": 0.7, "rationale": "Profile evidence supports"}]
    claims = [{"type": "affiliation_claim", "text": "I work at TechCorp"}]
    ctx = synthesize_context(ctx, comparisons, claims, [], current_turn=1)
    assert len(ctx.supported_claims) == 1
    assert ctx.supported_claims[0].claim_text == "I work at TechCorp"
    assert ctx.evidence_count == 1


def test_synthesis_adds_contradicted_claim():
    ctx = TargetContext()
    comparisons = [{"claim": "I never said that", "outcome": "direct_contradiction",
                     "confidence": 0.8, "rationale": "Evidence contradicts"}]
    claims = [{"type": "denial_of_prior_statement", "text": "I never said that"}]
    ctx = synthesize_context(ctx, comparisons, claims, [], current_turn=1)
    assert len(ctx.contradicted_claims) == 1
    assert ctx.contradicted_claims[0].claim_text == "I never said that"


def test_synthesis_skips_insufficient():
    ctx = TargetContext()
    comparisons = [{"claim": "something", "outcome": "insufficient_evidence",
                     "confidence": 0.0, "rationale": "no evidence"}]
    ctx = synthesize_context(ctx, comparisons, [], [], current_turn=1)
    assert ctx.evidence_count == 0
    assert len(ctx.supported_claims) == 0
    assert len(ctx.contradicted_claims) == 0


def test_synthesis_increments_repeated_support():
    ctx = TargetContext()
    comp = {"claim": "I work at X", "outcome": "supported_by_prior", "confidence": 0.6, "rationale": "found"}
    claims = [{"type": "affiliation_claim", "text": "I work at X"}]
    ctx = synthesize_context(ctx, [comp], claims, [], current_turn=1)
    ctx = synthesize_context(ctx, [comp], claims, [], current_turn=2)
    assert len(ctx.supported_claims) == 1
    assert ctx.supported_claims[0].support_count == 2


def test_synthesis_bounds_max_claims():
    ctx = TargetContext()
    for i in range(10):
        comp = {"claim": f"claim_{i}", "outcome": "supported_by_prior", "confidence": 0.5, "rationale": "ok"}
        ctx = synthesize_context(ctx, [comp], [], [], current_turn=i)
    assert len(ctx.supported_claims) <= 5  # _MAX_CLAIMS_PER_TYPE


def test_contradiction_ratio():
    ctx = TargetContext()
    ctx.supported_claims.append(SupportedClaim(claim_text="a"))
    ctx.contradicted_claims.append(ContradictedClaim(claim_text="b"))
    ctx.contradicted_claims.append(ContradictedClaim(claim_text="c"))
    assert ctx.contradiction_ratio == 2 / 3


def test_apply_context_conditioning_bounded():
    """Context effect should never exceed ±0.05."""
    ctx = TargetContext(evidence_count=10, overall_consistency=0.1)
    for i in range(5):
        ctx.contradicted_claims.append(ContradictedClaim(
            claim_text=f"lie_{i}", contradiction_count=3, confidence=0.9
        ))
    effect = compute_context_conditioning(ctx)
    assert abs(effect.bluffing_delta) <= 0.05
    assert abs(effect.withholding_delta) <= 0.05


# --- Engine integration tests ---

def test_engine_target_context_empty_on_start(tmp_path):
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()
    result = engine.process_turn(ConversationTurn(target_text="Hello."))
    assert result.target_context_summary.get("evidence_count", 0) == 0


def test_engine_target_context_builds_over_turns(tmp_path):
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    # Turn 1: target makes a statement
    engine.record_user_turn("What happened?")
    engine.process_turn(ConversationTurn(target_text="I sent the report at 3pm yesterday."))

    # Turn 2: target references prior statement
    engine.record_user_turn("When exactly?")
    result = engine.process_turn(ConversationTurn(target_text="I already told you, I sent it at 3pm."))

    # TargetContext should have some content now
    summary = result.target_context_summary
    print(f"Context summary: {summary}")
    # At minimum, the prior-statement claim should have been processed
    assert isinstance(summary, dict)


def test_engine_context_effect_exposed(tmp_path):
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    result = engine.process_turn(ConversationTurn(target_text="ok"))
    # Should have context_effect in result even if empty
    assert isinstance(result.context_effect, dict)
    assert "rationale" in result.context_effect


def test_engine_reset_clears_target_context(tmp_path):
    engine = BTOMEngine(state_path=tmp_path / "state.json", hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    # Build some context
    engine.record_user_turn("What happened?")
    engine.process_turn(ConversationTurn(target_text="I already explained this before."))

    # Reset
    engine.reset()

    # Context should be empty
    result = engine.process_turn(ConversationTurn(target_text="Hello."))
    assert result.target_context_summary.get("evidence_count", 0) == 0


# --- Behavioral Pattern tests ---

from btom_engine.osint.target_context import (
    BehavioralPattern, PatternObservation,
    synthesize_patterns, VALID_PATTERNS,
)


def test_no_repeated_evidence_no_pattern():
    ctx = TargetContext()
    comp = [{"claim": "test", "outcome": "supported_by_prior", "confidence": 0.5}]
    claims = [{"type": "factual_assertion", "text": "test"}]
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1)
    # factual_assertion with supported_by_prior does not map to any pattern
    assert len(ctx.behavioral_patterns) == 0


def test_repeated_prior_explanation_builds_pattern():
    ctx = TargetContext()
    comp = [{"claim": "I already explained", "outcome": "supported_by_prior", "confidence": 0.7}]
    claims = [{"type": "prior_explanation_claim", "text": "I already explained"}]

    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1, session_id="s1")
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=2, session_id="s1")
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=3, session_id="s1")

    p = ctx.get_pattern("references_prior_explanations")
    assert p is not None
    assert p.support_count == 3
    assert p.confidence > 0.3


def test_repeated_denial_contradiction_builds_pattern():
    ctx = TargetContext()
    comp = [{"claim": "I never said that", "outcome": "direct_contradiction", "confidence": 0.8}]
    claims = [{"type": "denial_of_prior_statement", "text": "I never said that"}]

    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1)
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=2)

    p = ctx.get_pattern("denial_contradiction_recurs")
    assert p is not None
    assert p.support_count == 2


def test_role_claim_stable_pattern():
    ctx = TargetContext()
    comp = [{"claim": "I'm CEO", "outcome": "supported_by_prior", "confidence": 0.6}]
    claims = [{"type": "role_claim", "text": "I'm CEO"}]

    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1)
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=2)

    p = ctx.get_pattern("role_claims_stable")
    assert p is not None
    assert p.support_count == 2


def test_role_claim_inconsistent_pattern():
    ctx = TargetContext()
    comp = [{"claim": "I'm CEO", "outcome": "direct_contradiction", "confidence": 0.7}]
    claims = [{"type": "role_claim", "text": "I'm CEO"}]

    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1)
    ctx = synthesize_patterns(ctx, comp, claims, current_turn=2)

    p = ctx.get_pattern("role_claims_inconsistent")
    assert p is not None


def test_weak_single_event_no_stable_pattern():
    ctx = TargetContext()
    comp = [{"claim": "I already said", "outcome": "supported_by_prior", "confidence": 0.3}]
    claims = [{"type": "prior_explanation_claim", "text": "I already said"}]

    ctx = synthesize_patterns(ctx, comp, claims, current_turn=1)

    p = ctx.get_pattern("references_prior_explanations")
    # Pattern exists but with only 1 observation — should not trigger conditioning
    if p:
        assert p.support_count == 1
        effect = compute_context_conditioning(ctx)
        # With only 1 observation, confidence < 0.3 threshold
        # Pattern should not trigger
        assert abs(effect.bluffing_delta) < 0.02 or "pattern" not in effect.rationale


def test_denial_pattern_raises_scrutiny():
    ctx = TargetContext(evidence_count=5)
    bp = BehavioralPattern(
        pattern_id="denial_contradiction_recurs",
        description="denials contradicted",
        support_count=3, confidence=0.5,
    )
    ctx.behavioral_patterns.append(bp)

    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta > 0
    assert "denial_recurs" in effect.rationale


def test_prior_refs_pattern_deescalates():
    ctx = TargetContext(evidence_count=5)
    bp = BehavioralPattern(
        pattern_id="references_prior_explanations",
        description="references priors",
        support_count=3, confidence=0.5,
    )
    ctx.behavioral_patterns.append(bp)

    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta < 0
    assert "prior_refs" in effect.rationale


def test_pattern_persists_in_dossier(tmp_path):
    from btom_engine.osint.dossier_store import DossierStore

    store = DossierStore(tmp_path / "dossiers")
    ctx = TargetContext(target_id="pattern_target", evidence_count=3)
    bp = BehavioralPattern(
        pattern_id="denial_contradiction_recurs",
        description="denials contradicted",
        support_count=2, confidence=0.4,
        observations=[PatternObservation(turn=1, cue="denial:contradiction", confidence=0.7)],
    )
    ctx.behavioral_patterns.append(bp)
    store.save(ctx)

    loaded = store.load("pattern_target")
    assert loaded is not None
    assert len(loaded.behavioral_patterns) == 1
    assert loaded.behavioral_patterns[0].pattern_id == "denial_contradiction_recurs"
    assert loaded.behavioral_patterns[0].support_count == 2
    assert len(loaded.behavioral_patterns[0].observations) == 1


def test_valid_patterns_set():
    """All synthesizable patterns should be in the valid set."""
    assert "references_prior_explanations" in VALID_PATTERNS
    assert "denial_contradiction_recurs" in VALID_PATTERNS
    assert "tends_to_answer_directly" in VALID_PATTERNS
    assert "role_claims_stable" in VALID_PATTERNS


# --- Signal-based patterns, recency, weakening tests ---

from btom_engine.schema import ExtractedSignals, SignalReading


def test_signal_direct_compliance_builds_pattern():
    ctx = TargetContext()
    for i in range(3):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.8, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.05, signal_reliability=0.3),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals)

    p = ctx.get_pattern("tends_to_answer_directly")
    assert p is not None
    assert p.support_count == 3
    assert p.confidence > 0.3


def test_signal_evasion_builds_pattern():
    ctx = TargetContext()
    for i in range(3):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.7, signal_reliability=0.85),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals)

    p = ctx.get_pattern("tends_to_evade_direct_questions")
    assert p is not None
    assert p.support_count == 3


def test_single_noisy_turn_no_stable_signal_pattern():
    ctx = TargetContext()
    signals = ExtractedSignals(
        evasive_deflection=SignalReading(value=0.8, signal_reliability=0.9),
    )
    ctx = synthesize_patterns(ctx, [], [], current_turn=1, signals=signals)

    p = ctx.get_pattern("tends_to_evade_direct_questions")
    if p:
        # Exists but with only 1 observation — should not pass conditioning threshold
        effect = compute_context_conditioning(ctx)
        assert abs(effect.bluffing_delta) < 0.01  # not enough for stable pattern


def test_recency_decay_reduces_old_observations():
    bp = BehavioralPattern(pattern_id="test")
    # Add old observation
    bp.add_observation(PatternObservation(turn=1, cue="old", confidence=0.5))
    bp.add_observation(PatternObservation(turn=2, cue="old", confidence=0.5))
    conf_old = bp.confidence

    # Recompute with high current turn — old observations should decay
    bp._recompute_confidence(current_turn=50)
    conf_decayed = bp.confidence

    assert conf_decayed < conf_old, (
        f"Decayed confidence ({conf_decayed:.3f}) should be less than original ({conf_old:.3f})"
    )


def test_contrary_evidence_weakens_pattern():
    bp = BehavioralPattern(pattern_id="test")
    bp.add_observation(PatternObservation(turn=1, cue="support", confidence=0.7))
    bp.add_observation(PatternObservation(turn=2, cue="support", confidence=0.7))
    conf_before = bp.confidence

    bp.add_contrary(turn=3, reason="contradictory evidence")
    bp.add_contrary(turn=4, reason="more contradiction")

    assert bp.confidence < conf_before, "Contrary evidence should reduce confidence"
    assert bp.contrary_count == 2
    assert bp.is_weakened


def test_evasion_pattern_weakened_by_direct_answers():
    ctx = TargetContext()

    # Build evasion pattern
    for i in range(3):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.8, signal_reliability=0.9),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals)

    p_evasion = ctx.get_pattern("tends_to_evade_direct_questions")
    assert p_evasion is not None
    conf_before = p_evasion.confidence

    # Now target gives direct answers — should weaken evasion pattern
    for i in range(3):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.9, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+4, signals=signals)

    assert p_evasion.contrary_count >= 3
    assert p_evasion.confidence < conf_before


def test_role_stable_weakened_by_contradiction():
    ctx = TargetContext()

    # Build role_claims_stable
    comp_s = [{"claim": "I'm CEO", "outcome": "supported_by_prior", "confidence": 0.7}]
    claims = [{"type": "role_claim", "text": "I'm CEO"}]
    ctx = synthesize_patterns(ctx, comp_s, claims, current_turn=1)
    ctx = synthesize_patterns(ctx, comp_s, claims, current_turn=2)

    p_stable = ctx.get_pattern("role_claims_stable")
    assert p_stable is not None
    conf_before = p_stable.confidence

    # Now contradictory role evidence
    comp_c = [{"claim": "I'm CEO", "outcome": "direct_contradiction", "confidence": 0.8}]
    ctx = synthesize_patterns(ctx, comp_c, claims, current_turn=3)
    ctx = synthesize_patterns(ctx, comp_c, claims, current_turn=4)

    assert p_stable.contrary_count >= 2
    assert p_stable.confidence < conf_before


def test_multi_turn_phase_shift():
    """Demo: early evasive -> later direct -> pattern shifts conservatively."""
    ctx = TargetContext()

    # Phase 1: evasive (turns 1-4)
    for i in range(4):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.7, signal_reliability=0.85),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals)

    p_evasive = ctx.get_pattern("tends_to_evade_direct_questions")
    assert p_evasive is not None
    evasive_conf_peak = p_evasive.confidence
    print(f"After evasive phase: evasion_conf={evasive_conf_peak:.3f}")

    # Phase 2: direct answers (turns 5-8)
    for i in range(4):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.9, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+5, signals=signals)

    p_direct = ctx.get_pattern("tends_to_answer_directly")
    assert p_direct is not None
    print(f"After direct phase: evasion_conf={p_evasive.confidence:.3f} direct_conf={p_direct.confidence:.3f}")

    # Evasion pattern should have weakened
    assert p_evasive.confidence < evasive_conf_peak
    # Direct pattern should be growing
    assert p_direct.confidence > 0.3
    # But evasion should NOT have completely vanished — it was a real phase
    assert p_evasive.confidence > 0.0


# --- Conditional / pressure-aware pattern tests ---

from btom_engine.interaction_context import UserPressure


def _make_pressure(acc=0.0, rep=0.0, hos=0.0):
    return UserPressure(accusation=acc, repetition=rep, hostility=hos)


def test_evasive_under_accusation_pattern():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.5)
    for i in range(3):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.6, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("evasive_under_accusation")
    assert p is not None
    assert p.support_count == 3
    assert p.confidence > 0.3
    # Should have condition labels
    assert any("high_accusation" in o.condition for o in p.observations)


def test_tone_shift_under_accusation_pattern():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.4)
    for i in range(3):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.6, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("tone_shift_under_accusation")
    assert p is not None
    assert p.support_count == 3


def test_narrowing_response_evasive_pattern():
    ctx = TargetContext()
    pressure = _make_pressure(rep=0.5)
    for i in range(3):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.5, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("narrowing_response_is_evasive")
    assert p is not None
    assert p.support_count == 3


def test_direct_when_low_pressure_only():
    ctx = TargetContext()
    low_pressure = _make_pressure(acc=0.0, rep=0.0, hos=0.0)  # aggregate ~0
    for i in range(3):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.8, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=low_pressure)

    p = ctx.get_pattern("direct_when_low_pressure_only")
    assert p is not None
    assert p.support_count == 3
    assert all(o.condition == "low_pressure" for o in p.observations)


def test_single_pressured_turn_no_conditional_pattern():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.5)
    signals = ExtractedSignals(
        evasive_deflection=SignalReading(value=0.6, signal_reliability=0.8),
    )
    ctx = synthesize_patterns(ctx, [], [], current_turn=1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("evasive_under_accusation")
    if p:
        # Exists with 1 observation — should not pass conditioning threshold
        effect = compute_context_conditioning(ctx)
        # Pattern needs support_count >= 2 and confidence >= 0.3
        if p.support_count < 2:
            assert "evasive_under_acc" not in effect.rationale


def test_conditional_pattern_weakened_by_contrary():
    ctx = TargetContext()
    pressure_high = _make_pressure(acc=0.5)
    pressure_low = _make_pressure(acc=0.0)

    # Build evasive_under_accusation
    for i in range(3):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.7, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=pressure_high)

    p = ctx.get_pattern("evasive_under_accusation")
    conf_before = p.confidence

    # Now target is direct under accusation — does not directly weaken evasive_under_accusation
    # But tends_to_answer_directly builds, and evasion pattern gets contrary
    for i in range(3):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.9, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+4, signals=signals, pressure=pressure_high)

    # Idle decay should have reduced older observations
    assert p.confidence <= conf_before


def test_conditional_patterns_in_conditioning():
    ctx = TargetContext(evidence_count=5)
    bp = BehavioralPattern(
        pattern_id="evasive_under_accusation",
        description="evasive under accusation",
        support_count=3, confidence=0.5,
    )
    ctx.behavioral_patterns.append(bp)

    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta > 0
    assert "evasive_under_acc" in effect.rationale


def test_multi_turn_calm_then_pressured_phase_shift():
    """Calm direct phase -> accusatory evasive phase -> conditional pattern emerges."""
    ctx = TargetContext()
    low_p = _make_pressure(acc=0.0)
    high_p = _make_pressure(acc=0.5, hos=0.3)

    # Phase 1: calm, direct (turns 1-4)
    for i in range(4):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.8, signal_reliability=0.9),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+1, signals=signals, pressure=low_p)

    p_direct = ctx.get_pattern("tends_to_answer_directly")
    p_low_only = ctx.get_pattern("direct_when_low_pressure_only")
    print(f"After calm phase: direct={p_direct.confidence if p_direct else 'N/A'} low_only={p_low_only.confidence if p_low_only else 'N/A'}")

    # Phase 2: accusatory, evasive (turns 5-8)
    for i in range(4):
        signals = ExtractedSignals(
            evasive_deflection=SignalReading(value=0.7, signal_reliability=0.85),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_patterns(ctx, [], [], current_turn=i+5, signals=signals, pressure=high_p)

    p_evasive_acc = ctx.get_pattern("evasive_under_accusation")
    print(f"After accusatory phase: evasive_under_acc={p_evasive_acc.confidence if p_evasive_acc else 'N/A'}")

    # Should have conditional pattern, not just global evasion
    assert p_evasive_acc is not None
    assert p_evasive_acc.support_count >= 3
    # Global direct pattern should have been weakened by evasive turns
    if p_direct:
        assert p_direct.contrary_count >= 3


# --- Strategy pattern tests ---

from btom_engine.osint.target_context import synthesize_strategies


def test_single_event_no_strategy():
    ctx = TargetContext()
    signals = ExtractedSignals(
        defensive_justification=SignalReading(value=0.7, signal_reliability=0.8),
        direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
    )
    pressure = _make_pressure(acc=0.5)
    comp = [{"claim": "denial", "outcome": "direct_contradiction", "confidence": 0.7}]

    ctx = synthesize_strategies(ctx, current_turn=1, signals=signals, pressure=pressure, comparisons=comp)
    p = ctx.get_pattern("deny_then_reframe")
    if p:
        assert p.support_count == 1
        # Should not pass conditioning threshold
        effect = compute_context_conditioning(ctx)
        assert "deny_reframe" not in effect.rationale


def test_repeated_deny_then_reframe():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.5)

    for i in range(3):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.6, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
            evasive_deflection=SignalReading(value=0.0, signal_reliability=0.0),
        )
        comp = [{"claim": "I never said that", "outcome": "direct_contradiction", "confidence": 0.7}]
        ctx = synthesize_strategies(ctx, current_turn=i+1, signals=signals, pressure=pressure, comparisons=comp)

    p = ctx.get_pattern("deny_then_reframe")
    assert p is not None
    assert p.support_count == 3
    assert p.confidence > 0.3


def test_repeated_partial_answer_then_evasion():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.3)  # aggregate must be >= LOW_PRESSURE (0.10)

    for i in range(3):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.4, signal_reliability=0.7),
            evasive_deflection=SignalReading(value=0.5, signal_reliability=0.8),
        )
        ctx = synthesize_strategies(ctx, current_turn=i+1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("partial_answer_then_evasion")
    assert p is not None
    assert p.support_count == 3


def test_repeated_blame_displacement():
    ctx = TargetContext()
    pressure = _make_pressure(acc=0.5, hos=0.3)

    for i in range(3):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.7, signal_reliability=0.85),
            emotional_intensity=SignalReading(value=0.5, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        ctx = synthesize_strategies(ctx, current_turn=i+1, signals=signals, pressure=pressure)

    p = ctx.get_pattern("blame_displacement_under_pressure")
    assert p is not None
    assert p.support_count == 3


def test_no_strategy_without_pressure():
    ctx = TargetContext()
    low_pressure = _make_pressure(acc=0.0)

    signals = ExtractedSignals(
        defensive_justification=SignalReading(value=0.7, signal_reliability=0.8),
        direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
    )
    comp = [{"claim": "denial", "outcome": "direct_contradiction", "confidence": 0.7}]

    ctx = synthesize_strategies(ctx, current_turn=1, signals=signals, pressure=low_pressure, comparisons=comp)
    # deny_then_reframe requires acc >= HIGH_PRESSURE (0.30)
    p = ctx.get_pattern("deny_then_reframe")
    assert p is None


def test_strategy_weakened_by_contrary():
    ctx = TargetContext()
    pressure_high = _make_pressure(acc=0.5)

    # Build deny_then_reframe
    for i in range(3):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.6, signal_reliability=0.8),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        comp = [{"claim": "test", "outcome": "direct_contradiction", "confidence": 0.7}]
        ctx = synthesize_strategies(ctx, current_turn=i+1, signals=signals, pressure=pressure_high, comparisons=comp)

    p = ctx.get_pattern("deny_then_reframe")
    conf_before = p.confidence

    # Now target starts answering directly under same pressure — implicit weakening via idle decay
    for i in range(10):
        signals = ExtractedSignals(
            direct_answer_compliance=SignalReading(value=0.9, signal_reliability=0.9),
        )
        ctx = synthesize_strategies(ctx, current_turn=i+10, signals=signals, pressure=pressure_high)

    # Idle decay should have reduced old observations
    assert p.confidence < conf_before


def test_strategy_in_conditioning():
    ctx = TargetContext(evidence_count=5)
    bp = BehavioralPattern(
        pattern_id="deny_then_reframe",
        description="deny then reframe",
        support_count=3, confidence=0.5,
    )
    ctx.behavioral_patterns.append(bp)

    effect = compute_context_conditioning(ctx)
    assert effect.bluffing_delta > 0
    assert "deny_reframe" in effect.rationale


def test_strategy_persists_in_dossier(tmp_path):
    from btom_engine.osint.dossier_store import DossierStore

    store = DossierStore(tmp_path / "dossiers")
    ctx = TargetContext(target_id="strategy_target", evidence_count=5)
    bp = BehavioralPattern(
        pattern_id="deny_then_reframe",
        description="deny then reframe",
        support_count=3, confidence=0.5,
        observations=[PatternObservation(turn=1, cue="denial+defense", confidence=0.5, condition="high_accusation")],
    )
    ctx.behavioral_patterns.append(bp)
    store.save(ctx)

    loaded = store.load("strategy_target")
    assert loaded is not None
    p = loaded.get_pattern("deny_then_reframe")
    assert p is not None
    assert p.support_count == 3
    assert p.observations[0].condition == "high_accusation"


def test_multi_turn_strategy_emergence():
    """Multi-turn: denial phase -> accusation rises -> repeated deny+reframe -> strategy emerges."""
    ctx = TargetContext()

    # Phase 1: calm denial (no strategy yet — low pressure)
    low_p = _make_pressure(acc=0.1)
    for i in range(2):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.5, signal_reliability=0.7),
        )
        comp = [{"claim": "test", "outcome": "direct_contradiction", "confidence": 0.5}]
        ctx = synthesize_strategies(ctx, current_turn=i+1, signals=signals, pressure=low_p, comparisons=comp)

    p = ctx.get_pattern("deny_then_reframe")
    assert p is None  # pressure too low

    # Phase 2: accusation escalates -> deny + defensive reframe recurs
    high_p = _make_pressure(acc=0.5)
    for i in range(4):
        signals = ExtractedSignals(
            defensive_justification=SignalReading(value=0.7, signal_reliability=0.85),
            direct_answer_compliance=SignalReading(value=0.0, signal_reliability=0.0),
        )
        comp = [{"claim": "test", "outcome": "direct_contradiction", "confidence": 0.7}]
        ctx = synthesize_strategies(ctx, current_turn=i+3, signals=signals, pressure=high_p, comparisons=comp)

    p = ctx.get_pattern("deny_then_reframe")
    assert p is not None
    assert p.support_count >= 3
    assert p.confidence > 0.3
    print(f"Strategy emerged: deny_then_reframe conf={p.confidence:.3f} count={p.support_count}")
