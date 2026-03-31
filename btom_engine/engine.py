"""Main orchestrator — chains the 4 layers together per turn.

All pipeline logic (sensor -> normalization -> math -> diagnostics -> persistence)
lives here. The UI layer should call process_turn() and render the result.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btom_engine.config import (
    DEFAULT_HYPOTHESES,
    STATE_FILE,
    TEXT_MODE,
    TEXT_MODE_COSIGNAL_THRESHOLD,
    TEXT_MODE_FRAG_REINFORCED_CAP,
    TEXT_MODE_FRAG_RELIABILITY_CAP,
)
from btom_engine.interaction_context import (
    UserPressure,
    apply_contextual_discounting,
    compute_pressure,
)
from btom_engine.novelty import NoveltyResult, compute_novelty
from btom_engine.math_engine import update as bayesian_update
from btom_engine.semantic_review import SemanticReviewResult, run_semantic_review
from btom_engine.schema import ExtractedSignals, StateLedger
from btom_engine.sensor import extract_signals_sync, last_debug_info
from btom_engine.sensor_calibration import apply_calibration
from btom_engine.speech_acts import analyze_turn
from btom_engine.liwc_signals import extract_liwc_signals, compute_liwc_bluff_delta
from btom_engine.baseline_scorer import ConversationBaseline
from btom_engine.claim_tracker import ConversationClaimTracker
from btom_engine.preference_inference import PreferenceInferenceTracker
from btom_engine.osint.claim_extraction import extract_claims
from btom_engine.osint.claim_comparison import compare_claim
from btom_engine.osint.evidence_schema import RetrievalRequest
from btom_engine.osint.prior_integration import apply_prior_effect, compute_prior_effect
from btom_engine.osint.retrieval_manager import RetrievalManager
from btom_engine.osint.adapters.prior_statements import PriorStatementsAdapter
from btom_engine.osint.adapters.web_search import WebSearchAdapter
from btom_engine.osint.adapters.page_read import PageReadAdapter
from btom_engine.osint.adapters.public_profile import PublicProfileAdapter
from btom_engine.osint.source_registry import build_default_registry
from btom_engine.osint.url_selection import select_urls
from btom_engine.osint.target_context import (
    TargetContext,
    synthesize_context, synthesize_patterns, synthesize_strategies,
    compute_context_conditioning, apply_context_conditioning,
)
from btom_engine.osint.dossier_store import DossierStore
# Investigation disabled on presentation branch — imports kept for master branch compatibility
from btom_engine.osint.investigator import investigate_target, integrate_investigation, InvestigationResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single conversational turn fed into the engine."""

    target_text: str
    user_text: str = ""
    timestamp: str = ""


@dataclass
class TurnResult:
    """Everything the UI needs to render after one turn. No logic required."""

    state: StateLedger
    signals: ExtractedSignals
    warnings: list[str] = field(default_factory=list)
    plot_rows: list[dict[str, Any]] = field(default_factory=list)
    process_time: float = 0.0
    sensor_debug: dict[str, Any] = field(default_factory=dict)
    user_pressure: UserPressure = field(default_factory=UserPressure)
    semantic_review: SemanticReviewResult = field(default_factory=SemanticReviewResult)
    novelty: NoveltyResult = field(default_factory=NoveltyResult)
    prior_context: dict[str, Any] = field(default_factory=dict)
    target_context_summary: dict[str, Any] = field(default_factory=dict)
    context_effect: dict[str, Any] = field(default_factory=dict)
    investigation: dict[str, Any] = field(default_factory=dict)
    speech_act: dict[str, Any] = field(default_factory=dict)
    claim_tracker: dict[str, Any] = field(default_factory=dict)
    preference_inference: dict[str, Any] = field(default_factory=dict)


# --- Neutral-confirmation normalization (post-sensor policy) ---
_ESCALATION_MARKERS = re.compile(r"\.\.\.|[?]{2,}|[!]{2,}|\buh\b|\bum\b", re.IGNORECASE)
_NEUTRAL_CONFIRMS = {"ok", "ok.", "okay", "yes", "sure", "fine", "got it"}

SIGNAL_NAMES = [
    "syntactic_fragmentation",
    "defensive_justification",
    "emotional_intensity",
    "evasive_deflection",
    "direct_answer_compliance",
]


def _apply_neutral_override(text: str, signals: ExtractedSignals) -> ExtractedSignals:
    """Cap signals for short neutral confirmations. Post-sensor normalization."""
    stripped = re.sub(r"[^\w\s.]", "", text).strip().lower()
    words = stripped.split()
    if (
        len(words) <= 3
        and stripped in _NEUTRAL_CONFIRMS
        and not _ESCALATION_MARKERS.search(text)
    ):
        signals.syntactic_fragmentation.value = min(signals.syntactic_fragmentation.value, 0.05)
        signals.syntactic_fragmentation.signal_reliability = 0.10
        signals.defensive_justification.value = 0.0
        signals.defensive_justification.signal_reliability = 0.10
        signals.evasive_deflection.value = 0.0
        signals.evasive_deflection.signal_reliability = 0.10
        signals.direct_answer_compliance.value = 0.0
        signals.direct_answer_compliance.signal_reliability = 0.10
    return signals


def _apply_text_mode_calibration(signals: ExtractedSignals) -> ExtractedSignals:
    """Demote fragmentation reliability in text-only mode.

    In online text, fragmentation is noisy (style, memes, casual shorthand).
    This caps fragmentation reliability unless corroborated by co-signals.

    Standalone: reliability capped to TEXT_MODE_FRAG_RELIABILITY_CAP (0.25)
    Reinforced: if any co-signal (emotion, defense, evasion) has effective
                signal > threshold, cap raised to TEXT_MODE_FRAG_REINFORCED_CAP (0.50)
    """
    if not TEXT_MODE:
        return signals

    frag = signals.syntactic_fragmentation
    if frag.signal_reliability <= TEXT_MODE_FRAG_RELIABILITY_CAP:
        return signals  # already low enough

    # Check for reinforcing co-signals
    co_signals = [
        signals.emotional_intensity,
        signals.defensive_justification,
        signals.evasive_deflection,
    ]
    reinforced = any(
        s.value * s.signal_reliability > TEXT_MODE_COSIGNAL_THRESHOLD
        for s in co_signals
    )

    cap = TEXT_MODE_FRAG_REINFORCED_CAP if reinforced else TEXT_MODE_FRAG_RELIABILITY_CAP
    frag.signal_reliability = min(frag.signal_reliability, cap)

    return signals


def _apply_novelty_discount(
    signals: ExtractedSignals,
    novelty: NoveltyResult,
) -> ExtractedSignals:
    """Reduce signal reliability based on novelty factor.

    Repeated content still signals persistence, but later copies
    contribute less new information. The novelty factor multiplies
    all signal reliabilities.
    """
    if novelty.novelty_factor >= 0.99:
        return signals  # fully novel, no discount

    factor = novelty.novelty_factor
    for name in SIGNAL_NAMES:
        reading = getattr(signals, name)
        reading.signal_reliability *= factor

    return signals


def _generate_diagnostics(state: StateLedger) -> list[str]:
    """Generate warnings from the current state. Backend diagnostic logic."""
    warnings: list[str] = []
    sigs = state.extracted_signals_current_turn
    reliabilities = [getattr(sigs, name).signal_reliability for name in SIGNAL_NAMES]
    avg_rel = sum(reliabilities) / len(reliabilities)

    if avg_rel < 0.5:
        warnings.append(f"Low average signal reliability: {avg_rel:.2f}")
    if state.system_status == "insufficient_evidence":
        warnings.append("System status: INSUFFICIENT EVIDENCE")

    for name in SIGNAL_NAMES:
        val = getattr(sigs, name).value
        if val > 0.85:
            warnings.append(f"Abnormal spike: {name.replace('_', ' ')} = {val:.2f}")

    # Text-mode fragmentation diagnostic
    if TEXT_MODE:
        frag = sigs.syntactic_fragmentation
        if frag.value > 0.1 and frag.signal_reliability <= TEXT_MODE_FRAG_RELIABILITY_CAP:
            warnings.append(f"Frag demoted (text-mode): val={frag.value:.2f} rel={frag.signal_reliability:.2f} [standalone]")
        elif frag.value > 0.1 and frag.signal_reliability <= TEXT_MODE_FRAG_REINFORCED_CAP:
            warnings.append(f"Frag reduced (text-mode): val={frag.value:.2f} rel={frag.signal_reliability:.2f} [reinforced]")

    return warnings


def _build_plot_rows(state: StateLedger, baselines: dict[str, float]) -> list[dict]:
    """Build UI-ready plot history rows from state."""
    rows = []
    for hyp_name, hyp in state.active_hypotheses.items():
        rows.append({
            "turn": state.current_turn,
            "hypothesis": hyp_name.replace("target_is_", "").replace("_", " ").title(),
            "probability": hyp.probability,
            "baseline": baselines.get(hyp_name, 0.5),
            "momentum": hyp.momentum,
        })
    return rows


class BTOMEngine:
    """Bayesian Theory of Mind Engine.

    Orchestrates the full pipeline:
      Layer 1 — Input Pipeline  (this class)
      Layer 2 — LLM Sensor      (sensor.py)
      Normalization — neutral override, future calibration
      Layer 3 — Math Engine      (math_engine.py)
      Diagnostics — warnings, plot data
      Layer 4 — State Ledger     (schema.py / JSON file)
    """

    def __init__(
        self,
        state_path: Path = STATE_FILE,
        hypotheses: dict[str, float] | None = None,
        search_provider: Any = None,
        page_provider: Any = None,
        dossier_store: Any = None,
        target_id: str = "",
    ) -> None:
        self.state_path = state_path
        self.baselines = hypotheses or dict(DEFAULT_HYPOTHESES)
        self._recent_user_turns: deque[str] = deque(maxlen=3)
        self._recent_target_turns: deque[str] = deque(maxlen=3)
        self._last_target_text: str = ""
        self._transcript: list[dict] = []
        self._dossier_store: DossierStore | None = dossier_store
        self._target_id = target_id or f"session_{id(self)}"
        self._search_provider = search_provider
        self._page_provider = page_provider
        self._investigation_ran = False
        self._conversation_baseline = ConversationBaseline()
        self._claim_tracker = ConversationClaimTracker()
        self._preference_tracker = PreferenceInferenceTracker(use_llm=True)

        # Load or create TargetContext
        self._target_context = self._load_or_create_context()

        # OSINT retrieval manager
        self._osint_registry = build_default_registry()
        self._osint_manager = RetrievalManager(self._osint_registry)
        self._prior_adapter = PriorStatementsAdapter()
        self._osint_manager.register_adapter("prior_statements_session", self._prior_adapter)

        # External adapters (optional — require providers)
        if search_provider is not None:
            self._osint_manager.register_adapter(
                "web_search_generic", WebSearchAdapter(provider=search_provider)
            )
        if page_provider is not None:
            self._osint_manager.register_adapter(
                "page_read_generic", PageReadAdapter(provider=page_provider)
            )
            self._osint_manager.register_adapter(
                "public_profile_page", PublicProfileAdapter(provider=page_provider)
            )

        if state_path.exists():
            self.state = StateLedger.load(state_path)
            logger.info("Resumed session %s at turn %d", self.state.session_id, self.state.current_turn)
        else:
            self.state = StateLedger.new_session(self.baselines)
            logger.info("Started new session %s", self.state.session_id)

    def record_user_turn(self, text: str) -> None:
        """Store a user turn for interaction-context computation. No math triggered."""
        self._recent_user_turns.append(text)
        self._transcript.append({"speaker": "user", "text": text})

    def process_turn(self, turn: ConversationTurn) -> TurnResult:
        """Execute the full pipeline: sensor -> normalize -> context -> math -> diagnostics -> persist."""
        t0 = time.time()

        # Layer 2 — Sensor (pure perception)
        signals = extract_signals_sync(turn.target_text)
        sensor_dbg = dict(last_debug_info)

        # Sensor calibration — correct systematic LLM biases
        signals = apply_calibration(signals)

        # Normalization — post-sensor policy
        signals = _apply_neutral_override(turn.target_text, signals)

        # Text-mode calibration — demote fragmentation in text-only mode
        signals = _apply_text_mode_calibration(signals)

        # Novelty discount — reduce reliability for repeated target content
        novelty = compute_novelty(turn.target_text, list(self._recent_target_turns))
        signals = _apply_novelty_discount(signals, novelty)

        # Interaction context — rule-based prior
        pressure = compute_pressure(list(self._recent_user_turns))

        # Semantic review — bounded Qwen corrector (runs selectively)
        # Uses the most recent user turn for review, with last target as context
        most_recent_user = self._recent_user_turns[-1] if self._recent_user_turns else ""
        pressure, review = run_semantic_review(
            most_recent_user, pressure, prev_target_text=self._last_target_text
        )

        # Contextual discounting with final (possibly adjusted) pressure
        signals = apply_contextual_discounting(signals, pressure)

        # Layer 3 — Math update
        self.state.current_turn += 1
        self.state.extracted_signals_current_turn = signals
        bayesian_update(self.state, self.baselines)

        # Speech act analysis — structural conversation analysis (pure Python)
        last_user = self._recent_user_turns[-1] if self._recent_user_turns else ""
        speech_act_result = analyze_turn(
            target_text=turn.target_text,
            user_text=last_user,
            turn_number=self.state.current_turn,
        )

        # Apply speech act adjustments to hypotheses (Optuna-optimized weight)
        try:
            from btom_engine.weights import WEIGHTS
            sa_weight = WEIGHTS.speech_act_weight
        except Exception:
            sa_weight = 0.5
        bluff_hyp = self.state.active_hypotheses.get("target_is_bluffing")
        withhold_hyp = self.state.active_hypotheses.get("target_is_withholding_info")
        if bluff_hyp and abs(speech_act_result.bluffing_delta) > 0.01:
            bluff_hyp.probability = max(0.0, min(1.0,
                bluff_hyp.probability + speech_act_result.bluffing_delta * sa_weight
            ))
        if withhold_hyp and abs(speech_act_result.withholding_delta) > 0.01:
            withhold_hyp.probability = max(0.0, min(1.0,
                withhold_hyp.probability + speech_act_result.withholding_delta * sa_weight
            ))

        # LIWC psycholinguistic signals (pure Python, no LLM)
        liwc = extract_liwc_signals(turn.target_text)
        liwc_delta = compute_liwc_bluff_delta(liwc)
        if bluff_hyp and abs(liwc_delta) > 0.005:
            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + liwc_delta))

        # Baseline deviation (compare against target's own conversation baseline)
        dev_result = self._conversation_baseline.process_turn(turn.target_text)
        if dev_result.has_baseline and bluff_hyp and abs(dev_result.bluff_delta) > 0.005:
            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + dev_result.bluff_delta))

        # Claim tracking — detect logical contradictions in target's own statements
        try:
            from btom_engine.weights import WEIGHTS
            ct_weight = WEIGHTS.claim_contradiction_weight
        except Exception:
            ct_weight = 0.35
        claim_result = self._claim_tracker.process_turn(
            turn.target_text, self.state.current_turn, contradiction_weight=ct_weight,
        )
        if claim_result.contradictions_found and bluff_hyp:
            bluff_hyp.probability = max(0.0, min(1.0,
                bluff_hyp.probability + claim_result.bluff_delta
            ))

        # Preference inference — ToM Level 2 (action-claim divergence)
        try:
            from btom_engine.weights import WEIGHTS
            pref_weight = WEIGHTS.preference_divergence_weight
        except Exception:
            pref_weight = 0.5
        pref_result = self._preference_tracker.process_turn(
            turn.target_text, self.state.current_turn,
            opponent_text=last_user,
        )
        if pref_result.max_divergence > 0.5 and bluff_hyp:
            pref_delta = pref_result.divergence_signal.value * pref_weight
            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + pref_delta))

        # Investigation disabled in presentation build — conversation analysis only
        investigation_result = {}

        # Prior-context retrieval (AFTER math update, bounded adjustment)
        prior_ctx = self._run_prior_context(turn.target_text)
        if prior_ctx.get("effect"):
            apply_prior_effect(self.state.active_hypotheses, prior_ctx["effect"])

        # Synthesize TargetContext from this turn's evidence
        self._target_context = synthesize_context(
            self._target_context,
            prior_ctx.get("comparisons", []),
            prior_ctx.get("claims", []),
            prior_ctx.get("retrieval_records", []),
            self.state.current_turn,
        )

        # Synthesize behavioral patterns (claim-based + signal-based + conditional)
        self._target_context = synthesize_patterns(
            self._target_context,
            prior_ctx.get("comparisons", []),
            prior_ctx.get("claims", []),
            self.state.current_turn,
            session_id=self.state.session_id,
            signals=self.state.extracted_signals_current_turn,
            pressure=pressure,
        )

        # Synthesize strategies (higher-order patterns from signal+pressure combos)
        self._target_context = synthesize_strategies(
            self._target_context,
            self.state.current_turn,
            session_id=self.state.session_id,
            signals=self.state.extracted_signals_current_turn,
            pressure=pressure,
            comparisons=prior_ctx.get("comparisons", []),
        )

        # Soft conditioning from accumulated TargetContext
        ctx_effect = compute_context_conditioning(self._target_context)
        apply_context_conditioning(self.state.active_hypotheses, ctx_effect)

        # Persist dossier if store is configured
        self._save_context()

        # Track recent target turns for novelty + context
        self._recent_target_turns.append(turn.target_text)
        self._last_target_text = turn.target_text
        self._transcript.append({"speaker": "target", "text": turn.target_text, "turn": self.state.current_turn})

        # Layer 4 — Persist
        self.state.save(self.state_path)

        # Diagnostics
        warnings = _generate_diagnostics(self.state)
        if pressure.aggregate > 0.3:
            warnings.append(f"User pressure detected: agg={pressure.aggregate:.2f} "
                            f"(acc={pressure.accusation:.2f} rep={pressure.repetition:.2f} "
                            f"hos={pressure.hostility:.2f})")
        if review.ran and not review.fallback_used:
            warnings.append(f"Semantic review ran ({review.trigger_reason}): "
                            f"conf={review.confidence:.2f}")
        if review.fallback_used:
            warnings.append(f"Semantic review fallback: {review.fallback_reason[:60]}")
        if novelty.novelty_factor < 0.95:
            warnings.append(f"Novelty discount: factor={novelty.novelty_factor:.2f} "
                            f"tier={novelty.tier} sim={novelty.max_similarity:.2f}")
        effect = prior_ctx.get("effect")
        if effect and effect.comparisons_used > 0:
            warnings.append(f"Prior context: {effect.rationale} "
                            f"(bluff_delta={effect.bluffing_delta:+.3f} withhold_delta={effect.withholding_delta:+.3f})")
        if abs(ctx_effect.bluffing_delta) > 0.001:
            warnings.append(f"Target context conditioning: {ctx_effect.rationale} "
                            f"(bluff_delta={ctx_effect.bluffing_delta:+.3f})")
        plot_rows = _build_plot_rows(self.state, self.baselines)

        return TurnResult(
            state=self.state,
            signals=self.state.extracted_signals_current_turn,
            warnings=warnings,
            plot_rows=plot_rows,
            process_time=time.time() - t0,
            prior_context=prior_ctx,
            target_context_summary=self._target_context.summary(),
            context_effect={"bluffing_delta": ctx_effect.bluffing_delta,
                           "withholding_delta": ctx_effect.withholding_delta,
                           "rationale": ctx_effect.rationale},
            sensor_debug=sensor_dbg,
            user_pressure=pressure,
            semantic_review=review,
            novelty=novelty,
            investigation=investigation_result,
            speech_act={
                "target_act": speech_act_result.target_act,
                "user_act": speech_act_result.user_act,
                "violation": speech_act_result.structural_violation,
                "severity": speech_act_result.violation_severity,
                "bluffing_delta": speech_act_result.bluffing_delta,
                "withholding_delta": speech_act_result.withholding_delta,
                "rationale": speech_act_result.rationale,
            },
            claim_tracker={
                "claims_extracted": len(claim_result.claims_extracted),
                "contradictions": [
                    {
                        "type": c.contradiction_type,
                        "severity": c.severity,
                        "explanation": c.explanation,
                    }
                    for c in claim_result.contradictions_found
                ],
                "bluff_delta": claim_result.bluff_delta,
                "rationale": claim_result.rationale,
                "total_claims_tracked": len(self._claim_tracker.get_all_claims()),
            },
            preference_inference={
                "max_divergence": pref_result.max_divergence,
                "divergence_signal": pref_result.divergence_signal.value,
                "divergence_reliability": pref_result.divergence_signal.signal_reliability,
                "rationale": pref_result.rationale,
                "preferences": {
                    item: {
                        "stated_value": pref.stated_value,
                        "revealed_value": pref.revealed_value,
                        "has_stated": pref.has_stated,
                        "has_revealed": pref.has_revealed,
                        "stated_turn": pref.stated_turn,
                        "revealed_turn": pref.revealed_turn,
                    }
                    for item, pref in pref_result.preferences.items()
                },
            },
        )

    def process_turn_with_signals(
        self, turn: ConversationTurn, signals: ExtractedSignals
    ) -> TurnResult:
        """Process a turn with pre-computed signals (for testing/eval)."""
        t0 = time.time()

        signals = _apply_neutral_override(turn.target_text, signals)
        signals = _apply_text_mode_calibration(signals)
        novelty = compute_novelty(turn.target_text, list(self._recent_target_turns))
        signals = _apply_novelty_discount(signals, novelty)
        pressure = compute_pressure(list(self._recent_user_turns))

        most_recent_user = self._recent_user_turns[-1] if self._recent_user_turns else ""
        pressure, review = run_semantic_review(
            most_recent_user, pressure, prev_target_text=self._last_target_text
        )

        signals = apply_contextual_discounting(signals, pressure)

        self.state.current_turn += 1
        self.state.extracted_signals_current_turn = signals
        bayesian_update(self.state, self.baselines)
        self._recent_target_turns.append(turn.target_text)
        self._last_target_text = turn.target_text
        self.state.save(self.state_path)

        warnings = _generate_diagnostics(self.state)
        if pressure.aggregate > 0.3:
            warnings.append(f"User pressure detected: agg={pressure.aggregate:.2f}")
        if novelty.novelty_factor < 0.95:
            warnings.append(f"Novelty discount: factor={novelty.novelty_factor:.2f} tier={novelty.tier}")
        plot_rows = _build_plot_rows(self.state, self.baselines)

        return TurnResult(
            state=self.state,
            signals=self.state.extracted_signals_current_turn,
            warnings=warnings,
            plot_rows=plot_rows,
            process_time=time.time() - t0,
            sensor_debug={},
            user_pressure=pressure,
            semantic_review=review,
            novelty=novelty,
        )

    def run_investigation(self, target_name: str, context: str = "") -> InvestigationResult:
        """Run an autonomous investigation on the target.

        Fires once per session. Results are integrated into the dossier.
        """
        if self._investigation_ran:
            return InvestigationResult(
                target_name=target_name,
                summary="Investigation already ran this session.",
            )

        result = investigate_target(
            target_name=target_name,
            search_provider=self._search_provider,
            page_provider=self._page_provider,
            context=context,
        )

        if result.success:
            self._target_context = integrate_investigation(self._target_context, result)
            self._save_context()
            logger.info("Investigation complete: %d findings for '%s'", len(result.findings), target_name)

        self._investigation_ran = True
        return result

    def _run_prior_context(self, target_text: str) -> dict[str, Any]:
        """Extract claims, retrieve evidence, compare, compute effect.

        Routes between prior_statements and web_search based on claim type.
        Returns a diagnostic dict with claims, comparisons, effect, and records.
        Safe no-op if no claims are extracted or no evidence is found.
        """
        result: dict[str, Any] = {
            "claims": [], "comparisons": [], "effect": None,
            "retrieval_records": [], "retrieval_path": "none",
        }

        # Step 1: Extract claims
        claims = extract_claims(target_text)
        result["claims"] = [
            {"type": c.claim_type, "text": c.claim_text, "needs_external": c.needs_external}
            for c in claims
        ]

        if not claims:
            return result

        self._prior_adapter.set_transcript(self._transcript)
        all_comparisons = []

        for claim in claims:
            if claim.needs_external:
                # External retrieval path: multi-query search -> page-read -> compare
                result["retrieval_path"] = "web_search_then_page_read"

                # Step 2a: Multi-query bounded search
                queries = claim.search_queries if claim.search_queries else [claim.retrieval_query]
                result["search_queries"] = queries
                all_search_records = []
                seen_urls: set[str] = set()

                for query in queries:
                    search_request = RetrievalRequest(
                        target_type="claim",
                        query_type="external_check",
                        query_text=query,
                        session_id=self.state.session_id,
                        top_k=3,
                        allowed_source_families=["web_search"],
                    )
                    search_retrieval = self._osint_manager.retrieve(search_request)
                    for r in search_retrieval.records:
                        if r.url_or_citation not in seen_urls:
                            seen_urls.add(r.url_or_citation)
                            all_search_records.append(r)

                search_records = all_search_records

                result["retrieval_records"].extend(
                    {"snippet": r.snippet, "relevance": r.relevance_score,
                     "source": r.source_family, "url": r.url_or_citation}
                    for r in search_records
                )

                # Step 2b: Select top URL(s) for page reading (up to 2 now)
                selected = select_urls(search_records, claim.retrieval_query, max_pages=2)
                result["selected_urls"] = [{"url": s.url, "title": s.title, "score": s.score, "reason": s.reason} for s in selected]

                # Step 2c: Page-read if URLs selected and adapter available
                page_evidence = []
                for sel in selected:
                    page_request = RetrievalRequest(
                        target_type="url",
                        target_value=sel.url,
                        query_type="page_read",
                        query_text=claim.retrieval_query,
                        top_k=1,
                        allowed_source_families=["page_read"],
                    )
                    page_retrieval = self._osint_manager.retrieve(page_request)
                    page_evidence.extend(page_retrieval.records)
                    result["retrieval_records"].extend(
                        {"snippet": r.snippet[:100], "relevance": r.relevance_score,
                         "source": r.source_family, "url": r.url_or_citation}
                        for r in page_retrieval.records
                    )

                # Step 3: Compare — prefer page evidence, fall back to search snippets
                if page_evidence:
                    comp = compare_claim(claim, page_evidence)
                    result.setdefault("evidence_source", []).append("page_read")
                elif search_records:
                    comp = compare_claim(claim, search_records)
                    result.setdefault("evidence_source", []).append("search_snippet")
                else:
                    from btom_engine.osint.evidence_schema import ComparisonResult
                    comp = ComparisonResult(
                        claim_text=claim.claim_text,
                        outcome="insufficient_evidence",
                        rationale="No search results or page content available.",
                    )
                    result.setdefault("evidence_source", []).append("none")

            else:
                # Prior statements path (unchanged)
                result["retrieval_path"] = "prior_statements"
                request = RetrievalRequest(
                    target_type="session",
                    query_type="prior_statements",
                    query_text=claim.retrieval_query,
                    session_id=self.state.session_id,
                    top_k=3,
                    allowed_source_families=["prior_statements"],
                )
                retrieval = self._osint_manager.retrieve(request)
                result["retrieval_records"].extend(
                    {"snippet": r.snippet, "relevance": r.relevance_score,
                     "source": r.source_family, "url": r.url_or_citation}
                    for r in retrieval.records
                )
                comp = compare_claim(claim, retrieval.records)

            all_comparisons.append(comp)

        result["comparisons"] = [
            {"claim": c.claim_text, "outcome": c.outcome, "confidence": c.comparison_confidence, "rationale": c.rationale}
            for c in all_comparisons
        ]

        # Step 4: Compute effect
        effect = compute_prior_effect(all_comparisons)
        result["effect"] = effect

        return result

    def _load_or_create_context(self) -> TargetContext:
        """Load dossier from store if available, otherwise create new."""
        if self._dossier_store and self._target_id:
            loaded = self._dossier_store.load(self._target_id)
            if loaded:
                loaded.last_updated_session = self.state.session_id if hasattr(self, 'state') else ""
                logger.info("Loaded persistent dossier for '%s'", self._target_id)
                return loaded
        ctx = TargetContext(target_id=self._target_id)
        ctx.created_session = self.state.session_id if hasattr(self, 'state') else ""
        ctx.last_updated_session = ctx.created_session
        return ctx

    def _save_context(self) -> None:
        """Persist the current dossier if a store is configured."""
        if self._dossier_store and self._target_context.target_id:
            self._target_context.last_updated_session = self.state.session_id
            self._dossier_store.save(self._target_context)

    def seed_baseline(self, texts: list[str]) -> int:
        """Pre-seed the conversation baseline with known truthful text samples.

        Call before processing any turns. The baseline will be immediately
        ready for deviation detection without a calibration phase.

        Returns the number of samples successfully added.
        """
        return self._conversation_baseline.seed_from_texts(texts)

    def mark_turn_as_baseline(self, text: str) -> bool:
        """Mark a turn as baseline (normal behavior, not deceptive).

        Adds the turn to the calibration pool and expands the baseline window.
        Returns True if successfully added.
        """
        return self._conversation_baseline.add_baseline_turn(text)

    def recompute_after_baseline_change(self) -> list[dict]:
        """Retroactively recompute all turn scores after baseline modification.

        Called after seed_baseline() or mark_turn_as_baseline() to refresh
        historical deviation scores. Returns updated plot rows.
        """
        # Get all target turns from transcript
        target_texts = [
            entry["text"] for entry in self._transcript
            if entry.get("speaker") == "target"
        ]
        if not target_texts:
            return []

        # Recompute deviations
        results = self._conversation_baseline.recompute_all(target_texts)

        # Rebuild state from scratch
        self.state = StateLedger.new_session(self.baselines)
        self._claim_tracker.reset()
        self._preference_tracker.reset()

        # Re-process all turns through the full pipeline (minus LLM sensor — use cached)
        new_plot_rows = []
        for i, text in enumerate(target_texts):
            turn = ConversationTurn(target_text=text)
            result = self.process_turn(turn)
            new_plot_rows.extend(result.plot_rows)

        return new_plot_rows

    def reset(self) -> None:
        """Reset the session to a fresh state. Does NOT delete persistent dossier."""
        self.state = StateLedger.new_session(self.baselines)
        self.state.save(self.state_path)
        self._recent_user_turns.clear()
        self._recent_target_turns.clear()
        self._last_target_text = ""
        self._transcript.clear()
        self._conversation_baseline.reset()
        self._claim_tracker.reset()
        self._preference_tracker.reset()
        # Reload dossier from store (preserves cross-session context)
        self._target_context = self._load_or_create_context()

    def summary(self) -> dict:
        """Return a concise dict summary of current hypotheses."""
        return {
            "session": self.state.session_id,
            "turn": self.state.current_turn,
            "status": self.state.system_status,
            "hypotheses": {
                name: {
                    "probability": round(h.probability, 4),
                    "momentum": round(h.momentum, 4),
                }
                for name, h in self.state.active_hypotheses.items()
            },
        }
