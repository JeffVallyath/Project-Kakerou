"""Project Kakerou — God's Eye Intelligence Dashboard.

Streamlit dashboard for the B-ToM Engine.
Run with: streamlit run cockpit/app.py
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
import sys

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from btom_engine.config import DEFAULT_HYPOTHESES
from btom_engine.engine import BTOMEngine, ConversationTurn, TurnResult, SIGNAL_NAMES
from btom_engine.schema import ExtractedSignals, SignalReading, StateLedger
from btom_engine.strategic_classifier import (
    extract_features, predict as classifier_predict, detect_domain, FEATURE_NAMES,
)
from btom_engine.speech_acts import analyze_turn
from btom_engine.liwc_signals import extract_liwc_signals, compute_liwc_bluff_delta

from plotting import hypothesis_history_chart, signal_bar_chart
from transcript_parser import parse_transcript, parse_transcript_blocks, to_replay_queue, repair_with_llm
from source_normalizer import normalize_source
from llm_extractor import extract_transcript, extraction_to_replay_queue, ExtractionResult
from file_upload import import_file, upload_to_replay_queue, UploadResult
from analyst_helpers import (
    build_speaker_mappings, apply_remap,
    build_ledger_entry, build_osint_trace,
)
from ui_components import (
    chat_message,
    hypothesis_card,
    system_status_card,
    warning_banner,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kakerou",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Refined dark aesthetic — Apple spatial design meets intelligence tooling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

    /* === FOUNDATION — 3 tonal planes === */
    .stApp {
        background-color: #09090b;
        color: #a1a1aa;
    }
    section[data-testid="stSidebar"] {
        background-color: #0c0c0e;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* === TYPOGRAPHY — 4 tiers only ===
       T1: Hero numbers — 2.8em DM Mono 300, accent colors
       T2: Section labels — 0.75em DM Mono 400 uppercase, #a1a1aa
       T3: Body text — 0.85em DM Sans 400, #d4d4d8
       T4: Micro labels — 0.7em DM Mono 400, #71717a
    */
    h1, h2, h3, h4 {
        color: #e4e4e7 !important;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95em !important;
    }
    .stMarkdown { color: #a1a1aa; }
    div[data-testid="stMetricValue"] { color: #fafafa; }
    p, span, div { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    label { color: #a1a1aa !important; font-size: 0.85em !important; }

    /* === METRIC CARDS — hero elements === */
    .metric-card {
        background: #131316;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 32px 28px 28px;
        text-align: center;
        margin-bottom: 14px;
    }
    .metric-label {
        color: #a1a1aa;
        font-size: 0.7em;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'DM Mono', 'SF Mono', monospace;
        margin-bottom: 12px;
        font-weight: 400;
    }
    .metric-value {
        font-size: 2.8em;
        font-weight: 300;
        font-family: 'DM Mono', 'SF Mono', monospace;
        line-height: 1;
        letter-spacing: -2px;
    }
    .metric-sub {
        color: #71717a;
        font-size: 0.7em;
        font-family: 'DM Mono', 'SF Mono', monospace;
        margin-top: 8px;
        font-weight: 400;
    }

    /* === DOMAIN BADGE === */
    .domain-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.7em;
        font-weight: 500;
        font-family: 'DM Mono', 'SF Mono', monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* === FEATURE ROWS === */
    .feat-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-family: 'DM Mono', 'SF Mono', monospace;
        font-size: 0.75em;
    }
    .feat-name { color: #d4d4d8; flex: 1; font-weight: 400; }
    .feat-bar-bg {
        flex: 1.8; height: 4px;
        background: rgba(255,255,255,0.05);
        border-radius: 2px; margin: 0 14px; overflow: hidden;
    }
    .feat-bar { height: 100%; border-radius: 2px; }
    .feat-val { color: #d4d4d8; width: 55px; text-align: right; font-weight: 400; }

    /* === STATUS BAR === */
    .status-bar {
        background: #131316;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
        font-family: 'DM Mono', 'SF Mono', monospace;
        font-size: 0.7em;
        font-weight: 400;
    }

    /* === FORM OVERRIDES — match card surface #131316 === */
    .stTextArea textarea,
    .stTextInput input {
        background: #131316 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        color: #d4d4d8 !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        font-size: 0.85em !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: rgba(255,255,255,0.14) !important;
        box-shadow: none !important;
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #131316 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        color: #d4d4d8 !important;
        font-size: 0.85em !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background: #131316 !important;
    }
    .stButton > button {
        background: #131316 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #d4d4d8 !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85em !important;
        padding: 8px 16px !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #1a1a1e !important;
        border-color: rgba(255,255,255,0.14) !important;
    }
    .stButton > button[kind="primary"] {
        background: #1a1a1e !important;
        border-color: rgba(255,255,255,0.12) !important;
    }
    /* Radio buttons */
    .stRadio > div { gap: 0.3rem !important; }
    .stRadio label {
        color: #a1a1aa !important;
        font-size: 0.85em !important;
    }
    /* File uploader */
    .stFileUploader > div {
        background: #131316 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background: #131316 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        font-size: 0.85em !important;
        color: #a1a1aa !important;
    }
    .streamlit-expanderContent {
        background: #0e0e10 !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }
    /* Slider */
    .stSlider > div > div > div {
        color: #a1a1aa !important;
    }
    /* Checkbox */
    .stCheckbox label {
        color: #a1a1aa !important;
        font-size: 0.85em !important;
    }
    /* Caption override */
    .stCaption, small {
        color: #71717a !important;
        font-size: 0.75em !important;
    }

    /* === DIVIDERS === */
    hr { border-color: rgba(255,255,255,0.04) !important; }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header — minimal, typographic
st.markdown(
    "<div style='margin-bottom: 4px;'>"
    "<span style='color: #fafafa; font-size: 1.1em; font-weight: 500; letter-spacing: 4px; "
    "text-transform: uppercase; font-family: DM Mono, SF Mono, monospace;'>Kakerou</span>"
    "&nbsp;&nbsp;<span style='color: #27272a; font-size: 0.7em; font-family: DM Mono, SF Mono, monospace;'>"
    "v8.1</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _create_engine(state_path=None):
    """Create a BTOMEngine for conversational analysis."""
    if state_path is None:
        state_path = DATA_DIR / "cockpit_state.json"
    return BTOMEngine(
        state_path=state_path,
        hypotheses=dict(DEFAULT_HYPOTHESES),
    )


def _init_session() -> None:
    """Initialize session state on first load only."""
    if st.session_state.get("session_initialized"):
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    engine = _create_engine()
    engine.reset()

    st.session_state.engine = engine
    st.session_state.tracker_state = engine.state
    st.session_state.chat_log = []
    st.session_state.history = []  # list of dicts for plotting
    st.session_state.last_signals = None
    st.session_state.last_warnings = []
    st.session_state.last_process_time = 0.0
    st.session_state.backend_ready = True
    st.session_state.processing = False
    st.session_state.session_initialized = True
    st.session_state.replay_queue = []
    st.session_state.replay_active = False
    st.session_state.turn_probs = []
    st.session_state.classifier_result = None
    st.session_state.last_features = {}
    st.session_state.domain_override = "auto"
    st.session_state.baseline_texts = []  # persists across engine resets


_init_session()

engine: BTOMEngine = st.session_state.engine
state: StateLedger = st.session_state.tracker_state


# ---------------------------------------------------------------------------
# Helper: process a turn through the backend
# ---------------------------------------------------------------------------
def _process_target_turn(text: str, entry: dict | None = None) -> TurnResult | None:
    """Call the backend engine and unpack the result into session state."""
    turn = ConversationTurn(target_text=text)
    try:
        result: TurnResult = engine.process_turn(turn)
    except Exception as exc:
        st.session_state.last_warnings.append(f"LLM SENSOR ERROR: {type(exc).__name__}: {exc}")
        st.session_state.backend_ready = False
        st.error(f"LLM sensor failed: {exc}")
        return None

    st.session_state.tracker_state = result.state
    st.session_state.last_signals = result.signals
    st.session_state.last_process_time = result.process_time
    st.session_state.last_warnings = result.warnings
    st.session_state.sensor_debug = result.sensor_debug
    st.session_state.sensor_mode = "llm"
    st.session_state.backend_ready = True
    st.session_state.history.extend(result.plot_rows)
    st.session_state.last_user_pressure = result.user_pressure
    st.session_state.last_speech_act = result.speech_act
    st.session_state.last_claim_tracker = result.claim_tracker
    st.session_state.last_semantic_review = result.semantic_review
    st.session_state.last_pref_inference = getattr(result, "preference_inference", {})

    # Track bluff probability for classifier
    bluff = result.state.active_hypotheses.get("target_is_bluffing")
    if bluff:
        if "turn_probs" not in st.session_state:
            st.session_state.turn_probs = []
        st.session_state.turn_probs.append(bluff.probability)

    # Run domain-routed classifier
    _run_classifier()

    # Accumulate ledger entry
    if "replay_ledger" not in st.session_state:
        st.session_state.replay_ledger = []
    le = build_ledger_entry(
        len(st.session_state.replay_ledger) + 1,
        entry or {"speaker": "Target", "text": text, "speaker_raw": ""},
        result,
    )
    st.session_state.replay_ledger.append(le)

    return result


_CACHED_WEIGHTS = None

def _get_weights():
    global _CACHED_WEIGHTS
    if _CACHED_WEIGHTS is None:
        from btom_engine.weights import EngineWeights
        _CACHED_WEIGHTS = EngineWeights.load()
    return _CACHED_WEIGHTS


def _process_target_turn_fast(text: str, context: str = "", entry: dict | None = None) -> TurnResult | None:
    """Fast-mode: use synthetic signals (speech acts + LIWC) instead of LLM sensor.

    Same Bayesian pipeline, same claim tracking, same preference inference.
    Just skips the Gemini API call — runs in ~5ms per turn instead of ~2.5s.
    """
    import time as _time
    from eval_harness import _build_synthetic_signals
    from btom_engine.math_engine import update as bayesian_update

    t0 = _time.time()
    weights = _get_weights()

    # Build signals from pure Python
    signals = _build_synthetic_signals(text, context, weights)

    # Run Bayesian update
    engine.state.current_turn += 1
    engine.state.extracted_signals_current_turn = signals
    bayesian_update(engine.state, engine.baselines, weights=weights)

    # Speech act analysis (imports already at module level via eval_harness deps)
    last_user = list(engine._recent_user_turns)[-1] if engine._recent_user_turns else context
    speech_act_result = analyze_turn(target_text=text, user_text=last_user,
                                      turn_number=engine.state.current_turn)

    # Apply speech act adjustment
    sa_weight = weights.speech_act_weight
    bluff_hyp = engine.state.active_hypotheses.get("target_is_bluffing")
    if bluff_hyp and abs(speech_act_result.bluffing_delta) > 0.01:
        bluff_hyp.probability = max(0.0, min(1.0,
            bluff_hyp.probability + speech_act_result.bluffing_delta * sa_weight))

    liwc = extract_liwc_signals(text)
    liwc_delta = compute_liwc_bluff_delta(liwc, weights)
    if bluff_hyp and abs(liwc_delta) > 0.005:
        bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + liwc_delta))

    # Baseline deviation
    dev = engine._conversation_baseline.process_turn(text, weights)
    if dev.has_baseline and bluff_hyp and abs(dev.bluff_delta) > 0.005:
        bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + dev.bluff_delta))

    # Claim tracking
    ct_weight = getattr(weights, 'claim_contradiction_weight', 0.35)
    claim_result = engine._claim_tracker.process_turn(text, engine.state.current_turn,
                                                       contradiction_weight=ct_weight)
    if claim_result.contradictions_found and bluff_hyp:
        bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + claim_result.bluff_delta))

    # Preference inference
    pref_result = engine._preference_tracker.process_turn(text, engine.state.current_turn,
                                                           opponent_text=last_user)
    pref_weight = getattr(weights, 'preference_divergence_weight', 0.5)
    if pref_result.max_divergence > 0.5 and bluff_hyp:
        pref_delta = pref_result.divergence_signal.value * pref_weight
        bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + pref_delta))

    # Build plot rows
    plot_rows = []
    for hyp_name, hyp in engine.state.active_hypotheses.items():
        plot_rows.append({
            "turn": engine.state.current_turn,
            "hypothesis": hyp_name.replace("target_is_", "").replace("_", " ").title(),
            "probability": hyp.probability,
            "baseline": engine.baselines.get(hyp_name, 0.5),
            "momentum": hyp.momentum,
        })

    process_time = _time.time() - t0

    # Build a lightweight TurnResult
    result = TurnResult(
        state=engine.state,
        signals=signals,
        warnings=[],
        plot_rows=plot_rows,
        process_time=process_time,
        speech_act={
            "target_act": speech_act_result.target_act,
            "user_act": speech_act_result.user_act,
            "violation": speech_act_result.structural_violation,
            "severity": speech_act_result.violation_severity,
            "bluffing_delta": speech_act_result.bluffing_delta,
            "rationale": speech_act_result.rationale,
        },
        claim_tracker={
            "claims_extracted": len(claim_result.claims_extracted),
            "contradictions": [{"type": c.contradiction_type, "severity": c.severity,
                                "explanation": c.explanation} for c in claim_result.contradictions_found],
            "bluff_delta": claim_result.bluff_delta,
            "rationale": claim_result.rationale,
            "total_claims_tracked": len(engine._claim_tracker.get_all_claims()),
        },
        preference_inference={
            "max_divergence": pref_result.max_divergence,
            "divergence_signal": pref_result.divergence_signal.value,
            "rationale": pref_result.rationale,
        },
    )

    # Update session state (same as full mode)
    st.session_state.tracker_state = result.state
    st.session_state.last_signals = result.signals
    st.session_state.last_process_time = result.process_time
    st.session_state.last_warnings = result.warnings
    st.session_state.sensor_debug = {}
    st.session_state.sensor_mode = "fast (no LLM)"
    st.session_state.backend_ready = True
    st.session_state.history.extend(result.plot_rows)
    st.session_state.last_speech_act = result.speech_act
    st.session_state.last_claim_tracker = result.claim_tracker
    st.session_state.last_pref_inference = result.preference_inference

    if bluff_hyp:
        if "turn_probs" not in st.session_state:
            st.session_state.turn_probs = []
        st.session_state.turn_probs.append(bluff_hyp.probability)

    _run_classifier()

    # Ledger
    if "replay_ledger" not in st.session_state:
        st.session_state.replay_ledger = []
    le = build_ledger_entry(
        len(st.session_state.replay_ledger) + 1,
        entry or {"speaker": "Target", "text": text, "speaker_raw": ""},
        result,
        engine_turn=result.state.current_turn,
    )
    st.session_state.replay_ledger.append(le)

    # Track turns (skip disk save in fast mode — save at end of replay)
    engine._recent_target_turns.append(text)
    engine._last_target_text = text

    return result


def _seed_baseline_and_store(eng, texts: list[str]) -> int:
    """Seed the baseline and persist the texts so they survive engine resets."""
    n = eng.seed_baseline(texts)
    if n > 0:
        if "baseline_texts" not in st.session_state:
            st.session_state.baseline_texts = []
        st.session_state.baseline_texts.extend(texts)
    return n


def _restore_baseline_if_needed(eng):
    """Re-apply stored baseline texts to a fresh engine."""
    stored = st.session_state.get("baseline_texts", [])
    if stored:
        eng.seed_baseline(stored)


def _run_classifier():
    """Run domain-routed classifier on accumulated conversation."""
    probs = st.session_state.get("turn_probs", [])
    turns = [e for e in st.session_state.chat_log if e["speaker"] == "Target"]
    if not probs or not turns:
        return
    texts = [t["text"] for t in turns]
    override = st.session_state.get("domain_override", "auto")
    if override != "auto":
        domain, confidence = override, 1.0
    else:
        domain, confidence = detect_domain(texts)
    features = extract_features(probs, turns)
    result = classifier_predict(features, domain=domain, domain_confidence=confidence, texts=texts)
    st.session_state.classifier_result = result
    st.session_state.last_features = features


def _prob_color(p: float) -> str:
    if p > 0.7: return "#f87171"
    elif p > 0.5: return "#fbbf24"
    elif p > 0.3: return "#a1a1aa"
    else: return "#34d399"


# ---------------------------------------------------------------------------
# Layout: Center-commanding — left is a control strip, right is a diagnostic rail
# ---------------------------------------------------------------------------
left_col, center_col, right_col = st.columns([0.8, 2.6, 0.7], gap="small")


# ===== LEFT ZONE: Transcript / Input =====
with left_col:
    st.markdown(
        "<span style='color:#a1a1aa; font-size:0.75em; text-transform:uppercase; "
        "letter-spacing:2px; font-family: DM Mono, SF Mono, monospace;'>Input</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Domain selector
    domain_options = {
        "auto": "Auto-Detect",
        "negotiation": "Negotiation",
        "diplomacy": "Diplomacy",
        "interrogation": "Interrogation / Interview",
        "general": "General / Zero-Shot",
    }
    st.markdown("<span style='color:#5a5a7a; font-size:0.7em; text-transform:uppercase; "
                "letter-spacing:2px;'>Operating Domain</span>", unsafe_allow_html=True)
    selected_domain = st.selectbox(
        "Domain", options=list(domain_options.keys()),
        format_func=lambda x: domain_options[x],
        key="domain_selector", label_visibility="collapsed",
    )
    st.session_state.domain_override = selected_domain

    # --- Target Baseline ---
    with st.expander("Target Baseline", expanded=False):
        baseline_count = engine._conversation_baseline.baseline_sample_count
        baseline_ready = engine._conversation_baseline.baseline_ready
        status_color = "#34d399" if baseline_ready else "#fbbf24" if baseline_count > 0 else "#71717a"
        status_text = (f"Ready ({baseline_count} samples)" if baseline_ready
                       else f"Calibrating ({baseline_count}/{engine._conversation_baseline.calibration_window})"
                       if baseline_count > 0 else "No baseline")
        st.markdown(
            f"<span style='color:{status_color}; font-size:0.7em; font-family: DM Mono, SF Mono, monospace;'>"
            f"Baseline: {_html.escape(status_text)}</span>",
            unsafe_allow_html=True,
        )

        baseline_mode = st.radio(
            "Input mode", ["Reddit", "Paste transcript", "Upload file"],
            horizontal=True, key="baseline_mode", label_visibility="collapsed",
        )

        if baseline_mode == "Reddit":
            reddit_user = st.text_input(
                "Reddit username",
                placeholder="username (without u/)",
                key="baseline_reddit_user", label_visibility="collapsed",
            )
            reddit_limit = st.slider("Max posts", 10, 100, 50, key="bl_reddit_limit")
            if st.button("Fetch & Seed", use_container_width=True, key="seed_reddit_btn"):
                if reddit_user and reddit_user.strip():
                    username = reddit_user.strip().lstrip("u/").lstrip("/")
                    with st.spinner(f"Fetching u/{username}..."):
                        try:
                            import requests as _req
                            posts = []
                            after = None
                            while len(posts) < reddit_limit:
                                url = f"https://www.reddit.com/user/{username}/comments.json?limit=100"
                                if after:
                                    url += f"&after={after}"
                                resp = _req.get(url, headers={"User-Agent": "btom_engine/1.0"}, timeout=10)
                                if resp.status_code != 200:
                                    st.error(f"Reddit returned {resp.status_code} — user may not exist or is private")
                                    break
                                data = resp.json().get("data", {})
                                children = data.get("children", [])
                                if not children:
                                    break
                                for c in children:
                                    body = c["data"].get("body", "").strip()
                                    if len(body) >= 20 and not body.startswith("http"):
                                        posts.append(body)
                                after = data.get("after")
                                if not after:
                                    break
                            if posts:
                                n = _seed_baseline_and_store(engine,posts[:reddit_limit])
                                st.success(f"Seeded with {n} posts from u/{username}")
                                st.rerun()
                            else:
                                st.warning(f"No text posts found for u/{username}")
                        except Exception as e:
                            st.error(f"Failed: {e}")

        elif baseline_mode == "Paste transcript":
            # Reuse transcript parser — user pastes a conversation, picks who the target is
            bl_transcript = st.text_area(
                "Paste conversation",
                height=80, placeholder="Name: message\nor [timestamp] Name: message",
                key="baseline_transcript_input", label_visibility="collapsed",
            )
            bl_use_det = st.checkbox("Use deterministic parser", value=True, key="bl_det")

            if st.button("Parse Baseline", use_container_width=True, key="bl_parse_btn"):
                if bl_transcript and bl_transcript.strip():
                    try:
                        if bl_use_det:
                            parsed = parse_transcript_blocks(bl_transcript, [], [])
                        else:
                            extraction = extract_transcript(bl_transcript)
                            if extraction and extraction.turns:
                                from transcript_parser import ParseResult, ParsedTurn, ParseQuality
                                parsed = ParseResult(
                                    turns=[ParsedTurn(speaker=t.speaker, text=t.text)
                                           for t in extraction.turns],
                                    quality=ParseQuality(confidence=extraction.confidence),
                                )
                            else:
                                parsed = parse_transcript_blocks(bl_transcript, [], [])
                        st.session_state._bl_parsed = parsed
                        speakers = list({t.speaker for t in parsed.turns})
                        st.session_state._bl_speakers = speakers
                        # Default role map
                        if "_bl_role_map" not in st.session_state:
                            st.session_state._bl_role_map = {}
                        st.success(f"Parsed {len(parsed.turns)} turns, {len(speakers)} speakers")
                    except Exception as e:
                        st.error(f"Parse failed: {e}")

            # Speaker role assignment
            bl_parsed = st.session_state.get("_bl_parsed")
            if bl_parsed:
                bl_speakers = st.session_state.get("_bl_speakers", [])
                if bl_speakers:
                    st.markdown(
                        "<span style='color:#a1a1aa; font-size:0.75em;'>Select which speaker is the target:</span>",
                        unsafe_allow_html=True,
                    )
                    target_speaker = st.selectbox(
                        "Target speaker", bl_speakers,
                        key="bl_target_speaker", label_visibility="collapsed",
                    )
                    if st.button("Seed from Target's Messages", use_container_width=True, key="bl_seed_parsed"):
                        target_texts = [t.text for t in bl_parsed.turns
                                        if t.speaker == target_speaker and len(t.text.strip()) >= 10]
                        if target_texts:
                            n = _seed_baseline_and_store(engine,target_texts)
                            st.success(f"Seeded with {n} messages from {target_speaker}")
                            st.rerun()
                        else:
                            st.warning(f"No messages from {target_speaker}")

        elif baseline_mode == "Upload file":
            bl_file = st.file_uploader(
                "Upload transcript", type=["json", "csv"],
                key="baseline_file", label_visibility="collapsed",
            )
            if bl_file is not None:
                try:
                    upload_result = import_file(bl_file)
                    if upload_result and upload_result.turns:
                        bl_speakers_up = list({t.get("speaker", t.get("speaker_raw", "Unknown"))
                                               for t in upload_result.turns})
                        st.session_state._bl_upload_turns = upload_result.turns
                        st.session_state._bl_upload_speakers = bl_speakers_up
                        st.caption(f"{len(upload_result.turns)} turns, {len(bl_speakers_up)} speakers")
                except Exception as e:
                    st.error(f"Import failed: {e}")

            bl_up_turns = st.session_state.get("_bl_upload_turns")
            bl_up_speakers = st.session_state.get("_bl_upload_speakers")
            if bl_up_turns and bl_up_speakers:
                target_speaker_up = st.selectbox(
                    "Target speaker", bl_up_speakers,
                    key="bl_target_speaker_up", label_visibility="collapsed",
                )
                if st.button("Seed from File", use_container_width=True, key="bl_seed_file"):
                    target_texts = [
                        t.get("text", "") for t in bl_up_turns
                        if t.get("speaker", t.get("speaker_raw", "")) == target_speaker_up
                        and len(t.get("text", "").strip()) >= 10
                    ]
                    if target_texts:
                        n = _seed_baseline_and_store(engine,target_texts)
                        st.success(f"Seeded with {n} messages from {target_speaker_up}")
                        st.rerun()
                    else:
                        st.warning(f"No messages from {target_speaker_up}")

    st.markdown("---")

    # Speaker selector
    speaker = st.selectbox("Speaker", ["Target", "User", "System"], index=0)

    # Text input
    input_disabled = st.session_state.get("processing", False)
    turn_text = st.text_area(
        "Message",
        height=90,
        disabled=input_disabled,
        placeholder="Enter the speaker's message...",
        key="turn_input",
    )

    # Buttons row
    btn_cols = st.columns([1, 1])
    with btn_cols[0]:
        process_clicked = st.button(
            "Process Turn",
            use_container_width=True,
            disabled=input_disabled,
            type="primary",
        )
    with btn_cols[1]:
        reset_clicked = st.button("Reset Session", use_container_width=True)

    btn_cols2 = st.columns([1, 1])
    with btn_cols2[0]:
        demo_clicked = st.button("Load Demo", use_container_width=True)
    with btn_cols2[1]:
        export_clicked = st.button("Export Snapshot", use_container_width=True)

    # --- Paste Transcript ---
    with st.expander("Paste Raw Transcript"):
        raw_transcript = st.text_area(
            "Raw transcript text",
            height=150,
            placeholder="Paste raw transcript here...\nFormat: Name: message\nor [timestamp] Name: message",
            key="raw_transcript_input",
        )
        alias_cols = st.columns(2)
        with alias_cols[0]:
            user_aliases_input = st.text_input("User aliases (comma-separated)", key="user_aliases")
        with alias_cols[1]:
            target_aliases_input = st.text_input("Target aliases (comma-separated)", key="target_aliases")

        use_deterministic = st.checkbox("Use deterministic parser (fallback)", key="use_deterministic")

        paste_cols = st.columns([1, 1])
        with paste_cols[0]:
            parse_clicked = st.button("Parse & Preview", use_container_width=True)
        with paste_cols[1]:
            load_parsed = st.button("Load & Replay", use_container_width=True)

        # --- Parse action: LLM-first, deterministic fallback ---
        if parse_clicked and raw_transcript:
            user_aliases = [a.strip() for a in user_aliases_input.split(",") if a.strip()]
            target_aliases = [a.strip() for a in target_aliases_input.split(",") if a.strip()]

            if use_deterministic:
                # Deterministic fallback path
                parsed = parse_transcript_blocks(raw_transcript, user_aliases, target_aliases)
                st.session_state._parsed_transcript = parsed
                st.session_state._extraction_result = None
                st.session_state._source_detected = "deterministic"
                st.session_state._source_family = "deterministic"
                st.session_state._source_confidence = 1.0
            else:
                # LLM-first extraction path
                with st.spinner("Extracting transcript with LLM..."):
                    extraction = extract_transcript(raw_transcript)

                st.session_state._extraction_result = extraction
                st.session_state._source_detected = "llm_extractor"
                st.session_state._source_family = "llm"
                st.session_state._source_confidence = 0.0

                if extraction.success:
                    st.session_state._source_confidence = sum(t.confidence for t in extraction.turns) / max(len(extraction.turns), 1)
                    # Convert to ParseResult-like format for remap compatibility
                    from transcript_parser import ParseResult, ParsedTurn, ParseQuality
                    parsed = ParseResult(parse_source="llm_extractor")
                    for t in extraction.turns:
                        parsed.turns.append(ParsedTurn(
                            speaker_raw=t.speaker_canonical,  # use canonical for remap
                            speaker_role="Other",
                            text=t.text_full,       # FULL text — invariant
                            timestamp=t.timestamp,
                            parse_source="llm_extractor",
                            confidence=t.confidence,
                        ))
                    parsed.raw_speakers = extraction.raw_speakers  # already canonical
                    parsed.other_count = len(parsed.turns)
                    parsed.quality = ParseQuality(
                        score=0.8 if extraction.success else 0.3,
                        warnings=extraction.warnings,
                    )
                    st.session_state._parsed_transcript = parsed
                else:
                    # LLM failed — try deterministic fallback
                    st.warning("LLM extraction failed. Using deterministic parser.")
                    parsed = parse_transcript_blocks(raw_transcript, user_aliases, target_aliases)
                    st.session_state._parsed_transcript = parsed
                    st.session_state._extraction_result = None
                    st.session_state._source_detected = "deterministic_fallback"

            # Store parsed transcript and initialize role map
            parsed = st.session_state._parsed_transcript
            st.session_state._parsed_speakers = parsed.raw_speakers
            initial_map = {}
            for spk in parsed.raw_speakers:
                for t in parsed.turns:
                    if t.speaker_raw == spk:
                        initial_map[spk] = t.speaker_role
                        break
                if spk not in initial_map:
                    initial_map[spk] = "Other"
            st.session_state._speaker_role_map = initial_map

        # --- Persistent remap UI: renders whenever parsed transcript exists ---
        parsed = st.session_state.get("_parsed_transcript")
        speakers = st.session_state.get("_parsed_speakers", [])
        role_map = st.session_state.get("_speaker_role_map", {})

        if parsed and speakers:
            # Source detection info
            src_detected = st.session_state.get("_source_detected", "generic")
            src_family = st.session_state.get("_source_family", "generic")
            src_conf = st.session_state.get("_source_confidence", 0.3)
            if src_detected != "generic":
                st.markdown(
                    f"<span style='color:#3498db; font-size:0.85em;'>Source: {src_detected} ({src_family}) "
                    f"conf={src_conf:.0%}</span>",
                    unsafe_allow_html=True,
                )

            q = parsed.quality
            q_color = "#2ecc71" if q.score >= 0.8 else "#f39c12" if q.score >= 0.5 else "#e74c3c"
            st.markdown(
                f"<span style='color:{q_color}; font-weight:600;'>Parse quality: {q.score:.0%}</span>"
                f" <span style='color:#888;'>({parsed.parse_source})</span>",
                unsafe_allow_html=True,
            )
            if q.warnings:
                for w in q.warnings:
                    st.caption(f"! {w}")
            if parsed.dropped_lines:
                st.warning(f"Dropped {len(parsed.dropped_lines)} line(s): {', '.join(parsed.dropped_lines[:3])}")

            # Speaker remap dropdowns (persist in session state)
            st.markdown("**Speaker Role Mapping:**")
            role_options = ["User", "Target", "Other"]
            for spk in speakers:
                current = role_map.get(spk, "Other")
                idx = role_options.index(current) if current in role_options else 2
                chosen = st.selectbox(
                    spk, role_options, index=idx,
                    key=f"remap_{spk}",
                )
                role_map[spk] = chosen
            st.session_state._speaker_role_map = role_map

            # Show counts from CURRENT remap (not original auto-detection)
            user_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "User")
            target_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "Target")
            other_c = sum(1 for t in parsed.turns if role_map.get(t.speaker_raw, "Other") == "Other")
            st.markdown(f"**Turns:** {len(parsed.turns)} | "
                        f"**User:** {user_c} | **Target:** {target_c} | **Other:** {other_c}")
            st.markdown(f"**Speakers:** {', '.join(speakers)}")

            # Preview (first 8 turns with CURRENT remap roles)
            for t in parsed.turns[:8]:
                role = role_map.get(t.speaker_raw, "Other")
                role_color = {"User": "#3498db", "Target": "#e74c3c", "Other": "#95a5a6"}.get(role, "#888")
                st.markdown(
                    f"<span style='color:{role_color}; font-weight:600;'>[{role}]</span> "
                    f"<span style='color:#888;'>{_html.escape(t.speaker_raw)}:</span> "
                    f"<span style='color:#ccc;'>{_html.escape(t.text[:60])}</span>",
                    unsafe_allow_html=True,
                )
            if len(parsed.turns) > 8:
                st.caption(f"... and {len(parsed.turns) - 8} more turns")

        if load_parsed:
            parsed = st.session_state.get("_parsed_transcript")
            if parsed and parsed.turns:
                queue = to_replay_queue(parsed)
                # Apply current role map from session state
                role_map = st.session_state.get("_speaker_role_map", {})
                if role_map:
                    queue = apply_remap(queue, role_map)
                new_engine = _create_engine()
                new_engine.reset()
                _restore_baseline_if_needed(new_engine)
                st.session_state.engine = new_engine
                st.session_state.tracker_state = new_engine.state
                st.session_state.chat_log = []
                st.session_state.history = []
                st.session_state.last_signals = None
                st.session_state.last_warnings = []
                st.session_state.last_process_time = 0.0
                st.session_state.last_user_pressure = None
                st.session_state.last_semantic_review = None
                st.session_state.sensor_debug = {}
                st.session_state.sensor_mode = "not yet run"
                st.session_state.backend_ready = True
                st.session_state.processing = False
                st.session_state.replay_queue = queue
                st.session_state.replay_active = True
                st.session_state.replay_ledger = []
                st.rerun()
            else:
                st.error("No parsed transcript. Click 'Parse & Preview' first.")

    # --- Upload Structured File ---
    with st.expander("Upload Transcript File (JSON/CSV)"):
        st.caption("For clean imports: use platform export tools (DiscordChatExporter, LinkedIn data download, etc.)")
        uploaded_file = st.file_uploader(
            "Upload JSON or CSV transcript",
            type=["json", "csv"],
            key="transcript_file",
        )

        if uploaded_file is not None:
            upload_result = import_file(uploaded_file.getvalue(), uploaded_file.name)

            if upload_result.success:
                st.markdown(
                    f"<span style='color:#2ecc71; font-weight:600;'>Imported: {len(upload_result.turns)} turns</span>"
                    f" <span style='color:#888;'>({upload_result.format_detected})</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Speakers:** {', '.join(upload_result.raw_speakers)}")

                if upload_result.warnings:
                    for w in upload_result.warnings:
                        st.caption(f"! {w}")

                # Store and initialize role map
                st.session_state._upload_result = upload_result
                st.session_state._upload_speakers = upload_result.raw_speakers

                if "_upload_role_map" not in st.session_state or set(st.session_state.get("_upload_speakers_prev", [])) != set(upload_result.raw_speakers):
                    st.session_state._upload_role_map = {s: "Other" for s in upload_result.raw_speakers}
                    st.session_state._upload_speakers_prev = list(upload_result.raw_speakers)

                # Role mapping dropdowns
                st.markdown("**Speaker Role Mapping:**")
                role_options = ["User", "Target", "Other"]
                umap = st.session_state._upload_role_map
                for spk in upload_result.raw_speakers:
                    current = umap.get(spk, "Other")
                    idx = role_options.index(current) if current in role_options else 2
                    chosen = st.selectbox(spk, role_options, index=idx, key=f"upload_remap_{spk}")
                    umap[spk] = chosen
                st.session_state._upload_role_map = umap

                # Counts from current mapping
                user_c = sum(1 for t in upload_result.turns if umap.get(t.speaker_raw, "Other") == "User")
                target_c = sum(1 for t in upload_result.turns if umap.get(t.speaker_raw, "Other") == "Target")
                other_c = sum(1 for t in upload_result.turns if umap.get(t.speaker_raw, "Other") == "Other")
                st.markdown(f"**Turns:** {len(upload_result.turns)} | User: {user_c} | Target: {target_c} | Other: {other_c}")

                # Preview first 5 turns
                for t in upload_result.turns[:5]:
                    role = umap.get(t.speaker_raw, "Other")
                    role_color = {"User": "#3498db", "Target": "#e74c3c", "Other": "#95a5a6"}.get(role, "#888")
                    st.markdown(
                        f"<span style='color:{role_color}; font-weight:600;'>[{role}]</span> "
                        f"<span style='color:#888;'>{_html.escape(t.speaker_raw)}:</span> "
                        f"<span style='color:#ccc;'>{_html.escape(t.text_full[:60])}</span>",
                        unsafe_allow_html=True,
                    )
                if len(upload_result.turns) > 5:
                    st.caption(f"... and {len(upload_result.turns) - 5} more turns")

                # Load button
                if st.button("Load & Replay (from file)", use_container_width=True, key="load_upload"):
                    queue = upload_to_replay_queue(upload_result.turns, umap)
                    new_engine = _create_engine()
                    new_engine.reset()
                    _restore_baseline_if_needed(new_engine)
                    st.session_state.engine = new_engine
                    st.session_state.tracker_state = new_engine.state
                    st.session_state.chat_log = []
                    st.session_state.history = []
                    st.session_state.last_signals = None
                    st.session_state.last_warnings = []
                    st.session_state.last_process_time = 0.0
                    st.session_state.last_user_pressure = None
                    st.session_state.last_semantic_review = None
                    st.session_state.sensor_debug = {}
                    st.session_state.sensor_mode = "not yet run"
                    st.session_state.backend_ready = True
                    st.session_state.processing = False
                    st.session_state.replay_queue = queue
                    st.session_state.replay_active = True
                    st.session_state.replay_ledger = []
                    st.rerun()
            else:
                for w in upload_result.warnings:
                    st.error(w)

    # --- Handle Reset ---
    if reset_clicked:
        # Nuclear reset — destroy everything, no survivors
        new_engine = _create_engine()
        new_engine.reset()
        st.session_state.engine = new_engine
        st.session_state.tracker_state = new_engine.state
        # Conversation state
        st.session_state.chat_log = []
        st.session_state.history = []
        st.session_state.replay_queue = []
        st.session_state.replay_active = False
        st.session_state.replay_ledger = []
        # Analysis state
        st.session_state.last_signals = None
        st.session_state.last_warnings = []
        st.session_state.last_process_time = 0.0
        st.session_state.last_user_pressure = None
        st.session_state.last_semantic_review = None
        st.session_state.last_speech_act = {}
        st.session_state.last_claim_tracker = {}
        st.session_state.last_pref_inference = {}
        st.session_state.sensor_debug = {}
        st.session_state.sensor_mode = "not yet run"
        st.session_state.backend_ready = True
        st.session_state.processing = False
        # Classifier state
        st.session_state.turn_probs = []
        st.session_state.classifier_result = None
        st.session_state.last_features = {}
        # Baseline state — full clear
        st.session_state.baseline_texts = []
        # Transcript parser state
        for key in list(st.session_state.keys()):
            if key.startswith("_parsed") or key.startswith("_bl_") or key.startswith("_upload") or key.startswith("_speaker") or key.startswith("_extraction") or key.startswith("_source"):
                del st.session_state[key]
        st.rerun()

    # --- Handle Demo Load ---
    if demo_clicked:
        demo_path = FIXTURES_DIR / "demo_transcript.json"
        if demo_path.exists():
            transcript = json.loads(demo_path.read_text(encoding="utf-8"))
            # Full reinstantiation for demo — but preserve baseline
            new_engine = _create_engine()
            new_engine.reset()
            _restore_baseline_if_needed(new_engine)
            st.session_state.engine = new_engine
            st.session_state.tracker_state = new_engine.state
            st.session_state.chat_log = []
            st.session_state.history = []
            st.session_state.last_signals = None
            st.session_state.last_warnings = []
            st.session_state.last_process_time = 0.0
            st.session_state.last_user_pressure = None
            st.session_state.sensor_debug = {}
            st.session_state.sensor_mode = "not yet run"
            st.session_state.backend_ready = True
            st.session_state.processing = False
            st.session_state.replay_queue = list(transcript)
            st.session_state.replay_active = True
            st.rerun()
        else:
            st.error("Demo transcript not found.")

    # --- Handle Replay ---
    if st.session_state.get("replay_active") and st.session_state.get("replay_queue"):
        st.markdown("---")
        st.markdown("**Replay Mode**")
        fast_mode = st.checkbox("Fast analysis (no LLM sensor)", value=True, key="fast_mode_toggle")
        replay_cols = st.columns([1, 1])
        with replay_cols[0]:
            step_clicked = st.button("Step Next Turn", use_container_width=True)
        with replay_cols[1]:
            play_all = st.button("Play All", use_container_width=True)

        remaining = len(st.session_state.replay_queue)
        st.caption(f"{remaining} turn(s) remaining")

        if step_clicked and st.session_state.replay_queue:
            entry = st.session_state.replay_queue.pop(0)
            spk = entry.get("speaker", "Target")
            txt = entry.get("text", "")
            turn_num = st.session_state.tracker_state.current_turn + 1 if spk == "Target" else None
            st.session_state.chat_log.append({"speaker": spk, "text": txt, "turn": turn_num,
                                              "speaker_raw": entry.get("speaker_raw", "")})
            if spk == "Target":
                if fast_mode:
                    last_user = list(engine._recent_user_turns)[-1] if engine._recent_user_turns else ""
                    _process_target_turn_fast(txt, context=last_user, entry=entry)
                else:
                    _process_target_turn(txt, entry=entry)
            elif spk == "User":
                engine.record_user_turn(txt)
                # Ledger entry for user turns
                if "replay_ledger" not in st.session_state:
                    st.session_state.replay_ledger = []
                st.session_state.replay_ledger.append(build_ledger_entry(
                    len(st.session_state.replay_ledger) + 1, entry))
            else:
                # Ledger entry for Other turns
                if "replay_ledger" not in st.session_state:
                    st.session_state.replay_ledger = []
                st.session_state.replay_ledger.append(build_ledger_entry(
                    len(st.session_state.replay_ledger) + 1, entry))
            if not st.session_state.replay_queue:
                st.session_state.replay_active = False
            st.rerun()

        if play_all:
            import time as _time

            if fast_mode:
                # BATCH MODE: process everything in pure Python, update Streamlit ONCE
                total = len(st.session_state.replay_queue)
                t0 = _time.time()

                from eval_harness import _build_synthetic_signals
                from btom_engine.math_engine import update as bayesian_update

                weights = _get_weights()
                chat_entries = []
                plot_rows = []
                ledger_entries = []
                turn_probs_batch = []
                queue = list(st.session_state.replay_queue)
                st.session_state.replay_queue = []

                for entry in queue:
                    spk = entry.get("speaker", "Target")
                    txt = entry.get("text", "")

                    if spk == "Target":
                        turn_num = engine.state.current_turn + 1
                        chat_entries.append({"speaker": spk, "text": txt, "turn": turn_num,
                                             "speaker_raw": entry.get("speaker_raw", "")})

                        last_user = list(engine._recent_user_turns)[-1] if engine._recent_user_turns else ""
                        signals = _build_synthetic_signals(txt, last_user, weights)

                        engine.state.current_turn += 1
                        engine.state.extracted_signals_current_turn = signals
                        bayesian_update(engine.state, engine.baselines, weights=weights)

                        sa = analyze_turn(target_text=txt, user_text=last_user,
                                          turn_number=engine.state.current_turn)
                        bluff_hyp = engine.state.active_hypotheses.get("target_is_bluffing")
                        if bluff_hyp and abs(sa.bluffing_delta) > 0.01:
                            bluff_hyp.probability = max(0.0, min(1.0,
                                bluff_hyp.probability + sa.bluffing_delta * weights.speech_act_weight))

                        liwc = extract_liwc_signals(txt)
                        liwc_delta = compute_liwc_bluff_delta(liwc, weights)
                        if bluff_hyp and abs(liwc_delta) > 0.005:
                            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + liwc_delta))

                        dev = engine._conversation_baseline.process_turn(txt, weights)
                        if dev.has_baseline and bluff_hyp and abs(dev.bluff_delta) > 0.005:
                            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + dev.bluff_delta))

                            # Use regex-only claim tracking in fast mode (no LLM)
                        engine._claim_tracker._use_llm = False
                        ct_result = engine._claim_tracker.process_turn(
                            txt, engine.state.current_turn,
                            contradiction_weight=getattr(weights, 'claim_contradiction_weight', 0.35))
                        engine._claim_tracker._use_llm = True
                        if ct_result.contradictions_found and bluff_hyp:
                            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + ct_result.bluff_delta))

                        # Disable LLM for preference tracker in fast mode
                        engine._preference_tracker._use_llm = False
                        pref_result = engine._preference_tracker.process_turn(
                            txt, engine.state.current_turn, opponent_text=last_user)
                        engine._preference_tracker._use_llm = True
                        if pref_result.max_divergence > 0.5 and bluff_hyp:
                            pref_delta = pref_result.divergence_signal.value * getattr(weights, 'preference_divergence_weight', 0.5)
                            bluff_hyp.probability = max(0.0, min(1.0, bluff_hyp.probability + pref_delta))

                        if bluff_hyp:
                            turn_probs_batch.append(bluff_hyp.probability)

                        for hyp_name, hyp in engine.state.active_hypotheses.items():
                            plot_rows.append({
                                "turn": engine.state.current_turn,
                                "hypothesis": hyp_name.replace("target_is_", "").replace("_", " ").title(),
                                "probability": hyp.probability,
                                "baseline": engine.baselines.get(hyp_name, 0.5),
                                "momentum": hyp.momentum,
                            })

                        engine._recent_target_turns.append(txt)
                        engine._last_target_text = txt

                    elif spk == "User":
                        chat_entries.append({"speaker": spk, "text": txt, "turn": None,
                                             "speaker_raw": entry.get("speaker_raw", "")})
                        engine.record_user_turn(txt)
                        ledger_entries.append(build_ledger_entry(
                            len(ledger_entries) + 1, entry))
                    else:
                        chat_entries.append({"speaker": spk, "text": txt, "turn": None,
                                             "speaker_raw": entry.get("speaker_raw", "")})
                        ledger_entries.append(build_ledger_entry(
                            len(ledger_entries) + 1, entry))

                elapsed = _time.time() - t0

                # ONE session state update at the end
                st.session_state.chat_log.extend(chat_entries)
                st.session_state.history.extend(plot_rows)
                st.session_state.tracker_state = engine.state
                st.session_state.turn_probs.extend(turn_probs_batch)
                if "replay_ledger" not in st.session_state:
                    st.session_state.replay_ledger = []
                st.session_state.replay_ledger.extend(ledger_entries)
                st.session_state.last_process_time = elapsed
                st.session_state.sensor_mode = "fast (batch)"


                _run_classifier()

                st.session_state.replay_active = False
                st.session_state._last_batch_msg = (
                    f"Done — {total} turns in {elapsed:.1f}s "
                    f"({total/max(elapsed,0.01):.0f} turns/sec)"
                )
                st.rerun()

            else:
                # FULL MODE: per-turn LLM sensor calls with green progress bar
                total = len(st.session_state.replay_queue)
                bar_placeholder = st.empty()
                status_placeholder = st.empty()
                t0 = _time.time()
                processed = 0

                def _render_bar(pct, label):
                    bar_placeholder.markdown(
                        f"<div style='background:#1a1a1e; border-radius:8px; overflow:hidden; height:22px;'>"
                        f"<div style='background:#34d399; width:{pct*100:.1f}%; height:100%; "
                        f"border-radius:8px; transition:width 0.3s ease;'></div></div>",
                        unsafe_allow_html=True,
                    )
                    status_placeholder.markdown(
                        f"<span style='color:#a1a1aa; font-size:0.75em; "
                        f"font-family:DM Mono,SF Mono,monospace;'>{label}</span>",
                        unsafe_allow_html=True,
                    )

                _render_bar(0, f"Turn 0/{total}...")

                while st.session_state.replay_queue:
                    entry = st.session_state.replay_queue.pop(0)
                    spk = entry.get("speaker", "Target")
                    txt = entry.get("text", "")
                    turn_num = st.session_state.tracker_state.current_turn + 1 if spk == "Target" else None
                    st.session_state.chat_log.append({"speaker": spk, "text": txt, "turn": turn_num,
                                                      "speaker_raw": entry.get("speaker_raw", "")})
                    if spk == "Target":
                        _process_target_turn(txt, entry=entry)
                    elif spk == "User":
                        engine.record_user_turn(txt)
                        if "replay_ledger" not in st.session_state:
                            st.session_state.replay_ledger = []
                        st.session_state.replay_ledger.append(build_ledger_entry(
                            len(st.session_state.replay_ledger) + 1, entry))
                    else:
                        if "replay_ledger" not in st.session_state:
                            st.session_state.replay_ledger = []
                        st.session_state.replay_ledger.append(build_ledger_entry(
                            len(st.session_state.replay_ledger) + 1, entry))

                    processed += 1
                    elapsed = _time.time() - t0
                    rate = processed / max(elapsed, 0.01)
                    remaining_est = (total - processed) / max(rate, 0.01)
                    _render_bar(
                        processed / total,
                        f"Turn {processed}/{total} — {rate:.1f} turns/sec — ~{remaining_est:.0f}s left",
                    )

                _render_bar(1.0, f"Done — {total} turns in {_time.time() - t0:.1f}s")
                st.session_state.replay_active = False
                st.rerun()

    # Show batch completion message if present
    batch_msg = st.session_state.pop("_last_batch_msg", None)
    if batch_msg:
        st.success(batch_msg)

    # --- Handle Process Turn ---
    if process_clicked and turn_text and turn_text.strip():
        text = turn_text.strip()
        turn_num = None

        if speaker == "Target":
            turn_num = st.session_state.tracker_state.current_turn + 1

        st.session_state.chat_log.append({
            "speaker": speaker,
            "text": text,
            "turn": turn_num,
        })

        if speaker == "Target":
            with st.spinner("Kakerou Engine processing..."):
                _process_target_turn(text)
            st.rerun()
        else:
            # Record user/system turns so the engine has interaction context
            if speaker == "User":
                engine.record_user_turn(text)
            st.rerun()

    # --- Handle Export ---
    if export_clicked:
        snapshot = {
            "exported_at": datetime.now().isoformat(),
            "tracker_state": st.session_state.tracker_state.model_dump(),
            "chat_log": st.session_state.chat_log,
            "history": st.session_state.history,
        }
        filename = f"kakerou_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = DATA_DIR / filename
        export_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        st.success(f"Snapshot saved: {filename}")

    # --- Conversation History ---
    st.markdown("---")
    st.markdown(
        "<span style='color:#a1a1aa; font-size:0.75em; text-transform:uppercase; "
        "letter-spacing:2px; font-family: DM Mono, SF Mono, monospace;'>Conversation</span>",
        unsafe_allow_html=True,
    )
    if st.session_state.chat_log:
        for idx, entry in enumerate(st.session_state.chat_log):
            raw = entry.get("speaker_raw", "")
            role = entry["speaker"]
            if raw and raw != role and role in ("Target", "User"):
                display_speaker = f"{role} ({raw})"
            elif role == "Other" and raw:
                display_speaker = f"Other ({raw})"
            else:
                display_speaker = role

            is_baseline = entry.get("is_baseline", False)

            if is_baseline:
                st.markdown(
                    "<span style='color:#fbbf24; font-size:0.55em; font-family: DM Mono, SF Mono, monospace; "
                    "letter-spacing:1.5px;'>BASELINE</span>",
                    unsafe_allow_html=True,
                )
            chat_message(display_speaker, entry["text"], entry.get("turn"))

            if role == "Target" and not is_baseline:
                if st.button("Baseline", key=f"mark_baseline_{idx}"):
                    entry["is_baseline"] = True
                    engine.mark_turn_as_baseline(entry["text"])
                    target_texts = [e["text"] for e in st.session_state.chat_log
                                    if e["speaker"] == "Target" and not e.get("is_baseline")]
                    engine._conversation_baseline.recompute_all(target_texts)
                    st.rerun()
    else:
        st.markdown(
            "<div style='color:#52525b; font-size:0.78em; padding:24px 0; text-align:center; "
            "font-family: DM Mono, SF Mono, monospace;'>No messages yet</div>",
            unsafe_allow_html=True,
        )


# ===== CENTER ZONE: Belief-State Telemetry =====
with center_col:
    current_state: StateLedger = st.session_state.tracker_state

    # System status card
    system_status_card(
        turn=current_state.current_turn,
        status=current_state.system_status,
        session_id=current_state.session_id,
    )

    # --- Domain status bar ---
    cls_result = st.session_state.get("classifier_result")
    bluff_hyp = current_state.active_hypotheses.get("target_is_bluffing")
    bayesian_prob = bluff_hyp.probability if bluff_hyp else 0.0
    classifier_prob = cls_result.probability if cls_result and cls_result.classifier_used else None
    domain_detected = cls_result.domain if cls_result else "general"
    domain_conf = cls_result.domain_confidence if cls_result else 0.0
    cls_active = cls_result.classifier_used if cls_result else False

    # Baseline requirement warning for interrogation domain
    if domain_detected == "interrogation" and not engine._conversation_baseline.baseline_ready:
        st.markdown(
            "<div style='background:rgba(251,191,36,0.03); border:1px solid rgba(251,191,36,0.1); "
            "border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:0.7em; "
            "font-family: DM Mono, SF Mono, monospace;'>"
            "<span style='color:#fbbf24;'>Baseline required</span> "
            "<span style='color:#52525b;'>— Seed the target's known truthful text for accurate detection.</span></div>",
            unsafe_allow_html=True,
        )

    domain_colors = {
        "negotiation": "#fbbf24",
        "diplomacy": "#60a5fa",
        "interrogation": "#f87171",
        "general": "#52525b",
    }
    d_color = domain_colors.get(domain_detected, "#6b7280")

    st.markdown(
        f"""<div class='status-bar'>
            <div>
                <span style='color:#71717a;'>DOMAIN</span>&nbsp;&nbsp;
                <span class='domain-badge' style='background:{d_color}22; color:{d_color}; border:1px solid {d_color}44;'>
                    {domain_detected.upper().replace('_', ' ')}
                </span>
                <span style='color:#52525b; margin-left:6px;'>conf {domain_conf:.0%}</span>
            </div>
            <div>
                <span style='color:#71717a;'>CLASSIFIER</span>&nbsp;
                <span style='color:{"#22c55e" if cls_active else "#444"};'>{"ACTIVE" if cls_active else "STANDBY"}</span>
                &nbsp;&nbsp;
                <span style='color:#71717a;'>LATENCY</span>&nbsp;
                <span style='color:#888;'>{st.session_state.last_process_time:.2f}s</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- Dual metric readout ---
    bay_col, cls_col = st.columns(2)
    with bay_col:
        bay_color = _prob_color(bayesian_prob)
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Raw Bayesian Prior</div>
                <div class='metric-value' style='color:{bay_color};'>{bayesian_prob:.1%}</div>
                <div class='metric-sub'>Multi-signal inference</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cls_col:
        if classifier_prob is not None:
            cls_color = _prob_color(classifier_prob)
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Domain Classifier</div>
                    <div class='metric-value' style='color:{cls_color};'>{classifier_prob:.1%}</div>
                    <div class='metric-sub'>{domain_detected.replace('_', ' ').title()} model</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Domain Classifier</div>
                    <div class='metric-value' style='color:#52525b;'>---</div>
                    <div class='metric-sub'>Awaiting domain detection</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- Feature decomposition panel ---
    features = st.session_state.get("last_features", {})
    pi = st.session_state.get("last_pref_inference", {})
    ct_feats = st.session_state.get("last_claim_tracker", {})
    if features:
        st.markdown("<span style='color:#5a5a7a; font-size:0.75em; text-transform:uppercase; "
                    "letter-spacing:2px;'>Active Signal Decomposition</span>", unsafe_allow_html=True)
        display_feats = []
        for key in ["liwc_cognitive_rate", "liwc_certainty_rate", "liwc_tentative_rate",
                     "liwc_concrete_rate", "liwc_filler_rate", "liwc_self_ref_rate"]:
            v = features.get(key, 0)
            if v > 0.001:
                label = key.replace("liwc_", "").replace("_rate", "").replace("_", " ").title()
                display_feats.append((f"LIWC: {label}", v, "#a78bfa"))
        for key, label in [("std_prob", "Volatility"), ("slope", "Trend Slope"),
                           ("max_minus_min", "Range"), ("frac_above_50", "High-Risk Frac")]:
            v = features.get(key, 0)
            if abs(v) > 0.001:
                display_feats.append((label, abs(v), "#3498db"))
        for key in ["act_EVADE_rate", "act_DEFEND_rate", "act_DEFLECT_rate"]:
            v = features.get(key, 0)
            if v > 0.01:
                label = key.replace("act_", "").replace("_rate", "")
                display_feats.append((f"Act: {label}", v, "#fbbf24"))
        for key in ["bigram_evade_then_defend", "bigram_inform_then_evade"]:
            v = features.get(key, 0)
            if v > 0.01:
                label = key.replace("bigram_", "").replace("_then_", " -> ").upper()
                display_feats.append((f"Seq: {label}", v, "#ef4444"))
        for key, label in [("hedge_density", "Hedge Density"), ("first_person_density", "1st Person"),
                           ("question_rate", "Question Rate")]:
            v = features.get(key, 0)
            if v > 0.001:
                display_feats.append((label, v, "#60a5fa"))
        div = pi.get("max_divergence", 0)
        if div > 0.1:
            display_feats.append(("Pref Divergence", div, "#ef4444"))
        n_contra = len(ct_feats.get("contradictions", []))
        if n_contra > 0:
            display_feats.append(("Contradictions", n_contra * 0.3, "#ef4444"))

        display_feats.sort(key=lambda x: x[1], reverse=True)
        display_feats = display_feats[:10]
        if display_feats:
            max_val = max(f[1] for f in display_feats)
            max_val = max(max_val, 0.01)
            for name, val, color in display_feats:
                bar_pct = min(val / max_val * 100, 100)
                st.markdown(
                    f"""<div class='feat-row'>
                        <div class='feat-name'>{name}</div>
                        <div class='feat-bar-bg'>
                            <div class='feat-bar' style='width:{bar_pct:.0f}%; background:{color};'></div>
                        </div>
                        <div class='feat-val'>{val:.3f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # Contradiction alert
    ct = st.session_state.get("last_claim_tracker", {})
    if ct.get("contradictions"):
        for c in ct["contradictions"]:
            st.markdown(
                f"""<div style='background:#8b0000; color:#ffffff; padding:12px; border-radius:8px;
                border: 2px solid #ff4444; margin-bottom:12px; font-family:monospace;'>
                <span style='font-size:1.2em; font-weight:bold;'>LOGICAL CONTRADICTION DETECTED</span><br/><br/>
                <b>Type:</b> {c.get('type', '').upper()}<br/>
                <b>Severity:</b> {c.get('severity', 0):.0%}<br/>
                <b>Evidence:</b> {c.get('explanation', '')}<br/><br/>
                <span style='color:#ff6666;'>Bluff adjustment: +{ct.get('bluff_delta', 0):.3f}</span>
                </div>""",
                unsafe_allow_html=True,
            )
    elif ct.get("total_claims_tracked", 0) > 0:
        st.caption(f"Tracking {ct['total_claims_tracked']} claims — no contradictions detected")

    # Turn-by-turn scrubber
    history = st.session_state.history
    max_turn = current_state.current_turn
    if max_turn > 1 and history:
        # Build per-turn hypothesis snapshots from history data
        turn_snapshots = {}
        for row in history:
            t = row.get("turn", 0)
            hyp = row.get("hypothesis", "")
            prob = row.get("probability", 0)
            if t not in turn_snapshots:
                turn_snapshots[t] = {}
            turn_snapshots[t][hyp] = prob

        selected_turn = st.slider(
            "Turn", min_value=1, max_value=max_turn, value=max_turn,
            key="turn_scrubber", label_visibility="collapsed",
        )
        st.session_state._selected_turn = selected_turn

        # --- Context projection: show what was said at this turn ---
        # Find the Target message and the User message before it
        target_msg = None
        user_msg_before = None
        target_idx = -1
        for i, entry in enumerate(st.session_state.chat_log):
            if entry.get("turn") == selected_turn and entry.get("speaker") == "Target":
                target_msg = entry
                target_idx = i
                break
        if target_msg and target_idx > 0:
            # Find the last User message before this Target turn
            for i in range(target_idx - 1, -1, -1):
                if st.session_state.chat_log[i].get("speaker") == "User":
                    user_msg_before = st.session_state.chat_log[i]
                    break

        if target_msg:
            ctx_html = (
                f"<div style='background:#131316; border:1px solid rgba(255,255,255,0.06); "
                f"border-radius:10px; padding:16px 20px; margin-bottom:14px;'>"
                f"<div style='color:#52525b; font-size:0.58em; text-transform:uppercase; "
                f"letter-spacing:2px; font-family:DM Mono,SF Mono,monospace; margin-bottom:10px;'>"
                f"Turn {selected_turn}</div>"
            )
            if user_msg_before:
                ctx_html += (
                    f"<div style='color:#60a5fa; font-size:0.72em; font-weight:500; "
                    f"font-family:DM Sans,-apple-system,sans-serif; margin-bottom:4px;'>User</div>"
                    f"<div style='color:#71717a; font-size:0.82em; line-height:1.5; "
                    f"padding-left:12px; border-left:2px solid rgba(96,165,250,0.2); "
                    f"margin-bottom:12px;'>{_html.escape(user_msg_before['text'][:300])}</div>"
                )
            ctx_html += (
                f"<div style='color:#f87171; font-size:0.72em; font-weight:500; "
                f"font-family:DM Sans,-apple-system,sans-serif; margin-bottom:4px;'>Target</div>"
                f"<div style='color:#d4d4d8; font-size:0.82em; line-height:1.5; "
                f"padding-left:12px; border-left:2px solid rgba(248,113,113,0.3);'>"
                f"{_html.escape(target_msg['text'][:400])}</div>"
                f"</div>"
            )
            st.markdown(ctx_html, unsafe_allow_html=True)

        # Show hypothesis values at selected turn
        st.markdown(
            f"<span style='color:#71717a; font-size:0.62em; text-transform:uppercase; "
            f"letter-spacing:2.5px; font-family:DM Mono,SF Mono,monospace;'>Hypotheses</span>",
            unsafe_allow_html=True,
        )
        if selected_turn in turn_snapshots:
            snapshot = turn_snapshots[selected_turn]
            prev_snapshot = turn_snapshots.get(selected_turn - 1, {})
            for hyp_name in snapshot:
                prob = snapshot[hyp_name]
                baseline = DEFAULT_HYPOTHESES.get(hyp_name, 0.5)
                prev_prob = prev_snapshot.get(hyp_name, baseline)
                delta = prob - prev_prob
                hypothesis_card(
                    name=hyp_name,
                    probability=prob,
                    baseline=baseline,
                    momentum=delta,
                )
        else:
            st.markdown(
                f"<div style='color:#52525b; font-size:0.78em; padding:12px 0;'>"
                f"No data for turn {selected_turn}</div>",
                unsafe_allow_html=True,
            )
    else:
        # Current turn display (no scrubber needed for 0-1 turns)
        st.markdown("#### Hypotheses")
        for hyp_name, hyp in current_state.active_hypotheses.items():
            baseline = DEFAULT_HYPOTHESES.get(hyp_name, 0.5)
            hypothesis_card(
                name=hyp_name,
                probability=hyp.probability,
                baseline=baseline,
                momentum=hyp.momentum,
            )
        if current_state.current_turn > 0:
            for hyp_name, hyp in current_state.active_hypotheses.items():
                display = hyp_name.replace("target_is_", "").replace("_", " ").title()
                delta = hyp.momentum
                color = "#e74c3c" if delta > 0.005 else "#2ecc71" if delta < -0.005 else "#888"
                arrow = "+" if delta > 0 else ""
                st.markdown(
                    f"<span style='color:{color}; font-size:0.85em; font-family:monospace;'>"
                    f"  {display}: {arrow}{delta:.4f}</span>",
                    unsafe_allow_html=True,
                )

    # History chart — fills the lower center
    st.markdown(
        "<span style='color:#71717a; font-size:0.62em; text-transform:uppercase; "
        "letter-spacing:2.5px; font-family: DM Mono, SF Mono, monospace;'>Trajectory</span>",
        unsafe_allow_html=True,
    )
    if st.session_state.history:
        chart = hypothesis_history_chart(st.session_state.history)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(
            "<div style='height:200px; display:flex; align-items:center; justify-content:center; "
            "background:#0e0e10; border:1px solid rgba(255,255,255,0.04); border-radius:12px; "
            "color:#52525b; font-size:0.78em; font-family: DM Mono, SF Mono, monospace;'>"
            "Awaiting analysis</div>",
            unsafe_allow_html=True,
        )

    # --- Replay Analysis Ledger ---
    ledger = st.session_state.get("replay_ledger", [])
    if ledger:
        st.markdown("#### Replay Ledger")
        for le in ledger:
            role_color = {"User": "#3498db", "Target": "#e74c3c", "Other": "#95a5a6"}.get(le.speaker_role, "#888")
            processed_icon = "●" if le.processed else "○"

            # Compact one-line summary
            delta_str = ""
            if le.bluffing_delta is not None and le.speaker_role == "Target":
                delta_str = f" Δbluff={le.bluffing_delta:+.3f}"
            osint_str = ""
            pattern_str = f" [{le.patterns_active}pat]" if le.patterns_active > 0 else ""

            # Show text length indicator when full text is longer than excerpt
            len_str = ""
            if le.text_full_length > 80:
                len_str = f" <span style='color:#555; font-size:0.7em;'>[{le.text_full_length} chars]</span>"

            st.markdown(
                f"<span style='color:#555;'>{processed_icon} T{le.turn_index}</span> "
                f"<span style='color:{role_color}; font-weight:600;'>[{le.speaker_role}]</span> "
                f"<span style='color:#888;'>{le.speaker_raw}:</span> "
                f"<span style='color:#ccc;'>{_html.escape(le.text_excerpt[:50])}</span>{len_str}"
                f"<span style='color:#888; font-size:0.8em;'>{delta_str}{osint_str}{pattern_str}</span>",
                unsafe_allow_html=True,
            )

        # Replay summary
        target_turns = [l for l in ledger if l.speaker_role == "Target"]
        st.markdown("---")
        st.markdown(
            f"**Summary:** {len(ledger)} turns | "
            f"Target: {len(target_turns)} | "
            f"Patterns: {max((l.patterns_active for l in ledger), default=0)}"
        )


# ===== RIGHT ZONE: Diagnostics / Trace =====
with right_col:
    st.markdown(
        "<span style='color:#71717a; font-size:0.62em; text-transform:uppercase; "
        "letter-spacing:2.5px; font-family: DM Mono, SF Mono, monospace;'>Diagnostics</span>",
        unsafe_allow_html=True,
    )

    # Warnings
    if st.session_state.last_warnings:
        for w in st.session_state.last_warnings:
            if "error" in w.lower():
                warning_banner(w, "error")
            else:
                warning_banner(w, "warning")

    # Signal values
    last_sigs = st.session_state.last_signals
    if last_sigs is not None:
        sig_values = {name: getattr(last_sigs, name).value for name in SIGNAL_NAMES}
        sig_reliabilities = {name: getattr(last_sigs, name).signal_reliability for name in SIGNAL_NAMES}

        st.markdown("**Signal Values** (current turn)")
        st.altair_chart(signal_bar_chart(sig_values, "Signal Values"), use_container_width=True)

        st.markdown("**Signal Reliability**")
        st.altair_chart(signal_bar_chart(sig_reliabilities, "Reliability"), use_container_width=True)

        # Top contributors
        st.markdown("**Top Contributors**")
        sorted_sigs = sorted(sig_values.items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_sigs[:3]:
            rel = sig_reliabilities[name]
            eff = val * rel
            display = name.replace("_", " ").title()
            st.markdown(
                f"<span style='color:#ccc; font-size:0.85em;'>"
                f"{display}: value={val:.2f}, rel={rel:.2f}, eff={eff:.3f}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No signals extracted yet.")

    # Speech Act Analysis
    speech_act = st.session_state.get("last_speech_act", {})
    if speech_act.get("target_act"):
        st.markdown("---")
        st.markdown("**Speech Act Analysis**")
        act = speech_act["target_act"]
        act_colors = {
            "INFORM": "#2ecc71", "ACKNOWLEDGE": "#3498db",
            "EVADE": "#e74c3c", "DISMISS": "#e74c3c",
            "DEFLECT": "#e67e22", "DEFEND": "#f39c12",
            "CHALLENGE": "#e67e22", "QUALIFY": "#f1c40f",
        }
        act_color = act_colors.get(act, "#888")
        violation_tag = ""
        if speech_act.get("violation"):
            violation_tag = (
                f" <span style='color:#ff4444;font-weight:bold;'>"
                f"STRUCTURAL VIOLATION (severity: {speech_act.get('severity', 0):.1f})</span>"
            )
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:4px; "
            f"border-left:4px solid {act_color}; margin-bottom:4px;'>"
            f"<span style='color:{act_color}; font-weight:bold; font-size:1.1em;'>{act}</span>"
            f"{violation_tag}<br/>"
            f"<span style='color:#888; font-size:0.8em;'>{speech_act.get('rationale', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Interaction context
    st.markdown("---")
    st.markdown("**Interaction Context**")
    pressure = st.session_state.get("last_user_pressure")
    if pressure is not None and pressure.aggregate > 0.01:
        agg_color = "#e74c3c" if pressure.aggregate > 0.5 else "#f39c12" if pressure.aggregate > 0.2 else "#2ecc71"
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.02); padding:10px 14px; border-radius:4px; "
            f"border-left:4px solid {agg_color}; margin-bottom:8px;'>"
            f"<span style='color:#aaa; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;'>User Pressure</span><br/>"
            f"<span style='color:{agg_color}; font-size:1.1em; font-weight:600;'>Aggregate: {pressure.aggregate:.2f}</span><br/>"
            f"<span style='color:#888; font-size:0.8em;'>"
            f"Accusation: {pressure.accusation:.2f} &nbsp;|&nbsp; "
            f"Repetition: {pressure.repetition:.2f} &nbsp;|&nbsp; "
            f"Hostility: {pressure.hostility:.2f}</span></div>",
            unsafe_allow_html=True,
        )

        # Discount factors applied
        _dbg = getattr(pressure, "debug", None)
        if _dbg and getattr(_dbg, "discount_factors", None):
            disc_parts = " &nbsp;|&nbsp; ".join(
                f"{k.replace('_', ' ').title()}: -{v*100:.0f}% rel"
                for k, v in _dbg.discount_factors.items()
            )
            st.markdown(
                f"<span style='color:#e67e22; font-size:0.8em;'>Discounts applied: {disc_parts}</span>",
                unsafe_allow_html=True,
            )

        # Per-turn detail expander
        if _dbg and getattr(_dbg, "per_turn_hits", None):
            with st.expander("Pressure Trace"):
                for hit in _dbg.per_turn_hits:
                    st.markdown(f"**\"{hit.get('text', '')}\"** (weight: {hit.get('weight', 0):.1f})")
                    parts = []
                    if hit.get("accusation_hits"):
                        parts.append(f"Acc [{hit.get('accusation_score', 0)}]: {', '.join(hit['accusation_hits'])}")
                    if hit.get("repetition_hits"):
                        parts.append(f"Rep [{hit.get('repetition_score', 0)}]: {', '.join(hit['repetition_hits'])}")
                    if hit.get("hostility_hits"):
                        parts.append(f"Hos [{hit.get('hostility_score', 0)}]: {', '.join(hit['hostility_hits'])}")
                    if parts:
                        st.markdown(
                            f"<span style='color:#888; font-size:0.8em;'>{'  |  '.join(parts)}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No patterns matched.")
    else:
        st.caption("No user pressure detected.")

    # Semantic review diagnostics
    sr = st.session_state.get("last_semantic_review")
    if sr and getattr(sr, "ran", False):
        st.markdown("---")
        st.markdown("**Semantic Review**")
        if sr.fallback_used:
            st.caption(f"Rule-based: {sr.fallback_reason[:60]}")
        elif getattr(sr, "slur_path", False):
            ctx_class = getattr(sr, "slur_context_class", "unknown")
            class_colors = {
                "explicit_hostile_attack": "#e74c3c",
                "demeaning_contemptuous": "#e67e22",
                "ambiguous": "#f39c12",
                "quoted_referential": "#3498db",
                "affiliative_banter": "#2ecc71",
            }
            cls_color = class_colors.get(ctx_class, "#888")
            mapped_hos = sr.final_merged.get("hostility", 0.0)
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:4px; "
                f"border-left:4px solid {cls_color}; margin-bottom:8px;'>"
                f"<span style='color:{cls_color}; font-weight:600;'>"
                f"{ctx_class.replace('_', ' ').title()}</span> "
                f"<span style='color:#888; font-size:0.8em;'>(conf: {sr.confidence:.2f})</span><br/>"
                f"<span style='color:#999; font-size:0.8em;'>Hostility: {mapped_hos:.2f}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            sr_color = "#3498db" if sr.confidence > 0.6 else "#f39c12" if sr.confidence > 0.3 else "#888"
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:4px; "
                f"border-left:4px solid {sr_color}; margin-bottom:8px;'>"
                f"<span style='color:{sr_color}; font-size:0.9em;'>Trigger: {sr.trigger_reason}</span> "
                f"<span style='color:#888; font-size:0.8em;'>(conf: {sr.confidence:.2f})</span></div>",
                unsafe_allow_html=True,
            )

    # Evidence trace
    st.markdown("---")
    st.markdown("**Evidence Trace**")
    if current_state.active_hypotheses:
        for hyp_name, hyp in current_state.active_hypotheses.items():
            display = hyp_name.replace("target_is_", "").replace("_", " ").title()
            recent = hyp.evidence_trace[-3:] if len(hyp.evidence_trace) > 3 else hyp.evidence_trace
            st.markdown(f"*{display}*")
            for entry in recent:
                st.markdown(
                    f"<span style='color:#888; font-size:0.8em;'>{_html.escape(str(entry))}</span>",
                    unsafe_allow_html=True,
                )
        with st.expander("Full Evidence Log"):
            for hyp_name, hyp in current_state.active_hypotheses.items():
                st.markdown(f"**{hyp_name}**")
                for entry in hyp.evidence_trace:
                    st.text(entry)

    # Signal Rationale (collapsed by default — clean but accessible)
    sensor_debug = st.session_state.get("sensor_debug", {})
    if sensor_debug:
        with st.expander("Signal Rationale", expanded=False):
            for sig_name, info in sensor_debug.items():
                rationale = info.get("rationale", "")
                if rationale:
                    st.markdown(
                        f"<span style='color:#7ec8e3;font-size:0.85em;'><b>{sig_name.replace('_', ' ').title()}</b>: "
                        f"{rationale}</span>",
                        unsafe_allow_html=True,
                    )
