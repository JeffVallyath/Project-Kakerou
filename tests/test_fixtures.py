"""Fixture-based conversation test harness for the B-ToM Engine.

=== PASS / FAIL SEMANTICS ===

PASS means: the observed outputs satisfied the fixture's bounded assertions.
FAIL means: one or more assertions were violated.

PASS does NOT mean the model is universally "correct."
FAIL does NOT mean the model is universally "wrong."

These fixtures define bounded regression expectations — they establish
a baseline surface that should remain stable as the system evolves.
If a fixture starts failing after a change, the change either broke
something that was working, or the fixture needs updating to reflect
intentional new behavior.

=== HOW TO RUN ===

All fast tests (rules-only, no LLM needed):
    py -m pytest tests/test_fixtures.py -v -m "not live" -s

All live tests (requires LM Studio running):
    py -m pytest tests/test_fixtures.py -v -m live -s

One specific fixture:
    py -m pytest tests/test_fixtures.py -v -k "blame_hostility" -s

All tests (fast + live):
    py -m pytest tests/test_fixtures.py -v -s

Reset + math tests (always fast):
    py -m pytest tests/ -v -m "not live" -s
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from btom_engine.config import DEFAULT_HYPOTHESES
from btom_engine.engine import BTOMEngine, ConversationTurn, TurnResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Per-turn diagnostic record
# ---------------------------------------------------------------------------

@dataclass
class TurnDiag:
    """Per-turn diagnostic snapshot for readable output."""

    turn_num: int
    speaker: str
    text: str
    pressure_acc: float = 0.0
    pressure_rep: float = 0.0
    pressure_hos: float = 0.0
    pressure_agg: float = 0.0
    bluffing: float = 0.0
    withholding: float = 0.0
    review_ran: bool = False
    slur_path: bool = False
    slur_class: str = ""
    discount_factors: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------

@dataclass
class ConversationResult:
    """Collected results from running a conversation through the engine."""

    turn_diags: list[TurnDiag] = field(default_factory=list)
    turn_results: list[TurnResult] = field(default_factory=list)
    final_pressure_acc: float = 0.0
    final_pressure_rep: float = 0.0
    final_pressure_hos: float = 0.0
    final_pressure_agg: float = 0.0
    final_bluffing: float = 0.0
    final_withholding: float = 0.0
    review_ran: bool = False
    slur_path: bool = False
    slur_context_class: str = ""


def _run_conversation(turns: list[dict], tmp_path: Path) -> ConversationResult:
    """Run a list of turns through a fresh engine and collect results."""
    state_path = tmp_path / "fixture_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    engine = BTOMEngine(state_path=state_path, hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    results: list[TurnResult] = []
    diags: list[TurnDiag] = []
    turn_num = 0

    for turn in turns:
        speaker = turn["speaker"].lower()
        text = turn["text"]

        if speaker == "user":
            engine.record_user_turn(text)
            diags.append(TurnDiag(
                turn_num=turn_num,
                speaker="user",
                text=text,
            ))
        elif speaker == "target":
            turn_num += 1
            ct = ConversationTurn(target_text=text)
            result = engine.process_turn(ct)
            results.append(result)

            hyps = result.state.active_hypotheses
            bluff = hyps.get("target_is_bluffing")
            withhold = hyps.get("target_is_withholding_info")

            dbg = getattr(result.user_pressure, "debug", None)
            disc = getattr(dbg, "discount_factors", {}) if dbg else {}

            diags.append(TurnDiag(
                turn_num=turn_num,
                speaker="target",
                text=text,
                pressure_acc=result.user_pressure.accusation,
                pressure_rep=result.user_pressure.repetition,
                pressure_hos=result.user_pressure.hostility,
                pressure_agg=result.user_pressure.aggregate,
                bluffing=bluff.probability if bluff else 0.0,
                withholding=withhold.probability if withhold else 0.0,
                review_ran=result.semantic_review.ran,
                slur_path=getattr(result.semantic_review, "slur_path", False),
                slur_class=getattr(result.semantic_review, "slur_context_class", ""),
                discount_factors=disc,
            ))

    cr = ConversationResult(turn_diags=diags, turn_results=results)

    if results:
        last = results[-1]
        cr.final_pressure_acc = last.user_pressure.accusation
        cr.final_pressure_rep = last.user_pressure.repetition
        cr.final_pressure_hos = last.user_pressure.hostility
        cr.final_pressure_agg = last.user_pressure.aggregate

        hyps = last.state.active_hypotheses
        cr.final_bluffing = hyps.get("target_is_bluffing", type("", (), {"probability": 0})).probability
        cr.final_withholding = hyps.get("target_is_withholding_info", type("", (), {"probability": 0})).probability
        cr.review_ran = last.semantic_review.ran
        cr.slur_path = getattr(last.semantic_review, "slur_path", False)
        cr.slur_context_class = getattr(last.semantic_review, "slur_context_class", "")

    return cr


# ---------------------------------------------------------------------------
# Diagnostic printer
# ---------------------------------------------------------------------------

def _print_diags(name: str, result: ConversationResult) -> None:
    """Print readable per-turn diagnostics."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'Speaker':<7}  {'Text':<40}  {'Acc':>5} {'Rep':>5} {'Hos':>5} {'Agg':>5}  {'Blf':>6} {'Wth':>6}  {'Rev':>3} {'Slur':>4}")
    print(f"  {'---':>3}  {'-------':<7}  {'----':<40}  {'---':>5} {'---':>5} {'---':>5} {'---':>5}  {'---':>6} {'---':>6}  {'---':>3} {'----':>4}")

    for d in result.turn_diags:
        short_text = d.text[:38] + ".." if len(d.text) > 40 else d.text
        if d.speaker == "user":
            print(f"  {'':>3}  {'USER':<7}  {short_text:<40}")
        else:
            rev = "Y" if d.review_ran else "-"
            slr = d.slur_class[:4] if d.slur_path else "-"
            print(
                f"  {d.turn_num:>3}  {'TRGT':<7}  {short_text:<40}  "
                f"{d.pressure_acc:>5.2f} {d.pressure_rep:>5.2f} {d.pressure_hos:>5.2f} {d.pressure_agg:>5.2f}  "
                f"{d.bluffing:>6.3f} {d.withholding:>6.3f}  "
                f"{rev:>3} {slr:>4}"
            )
            if d.discount_factors:
                disc_str = ", ".join(f"{k}: -{v*100:.0f}%" for k, v in d.discount_factors.items())
                print(f"  {'':>3}  {'':>7}  {'Discounts: ' + disc_str:<40}")

    print(f"  {'---':>3}")
    print(f"  Final: bluff={result.final_bluffing:.4f} withhold={result.final_withholding:.4f} "
          f"pressure_agg={result.final_pressure_agg:.3f} review={result.review_ran} slur={result.slur_path}")
    print()


# ---------------------------------------------------------------------------
# Assertion engine
# ---------------------------------------------------------------------------

_PRESSURE_FIELDS = {
    "accusation": "final_pressure_acc",
    "repetition": "final_pressure_rep",
    "hostility": "final_pressure_hos",
    "aggregate": "final_pressure_agg",
}


def _check_assertion(assertion: dict, result: ConversationResult, label: str = "") -> None:
    """Check a single assertion against a conversation result."""
    atype = assertion["type"]
    prefix = f"[{label}] " if label else ""

    if atype == "final_hypothesis_min":
        hyp = assertion["hypothesis"]
        threshold = assertion["value"]
        actual = result.final_bluffing if "bluffing" in hyp else result.final_withholding
        assert actual >= threshold, (
            f"{prefix}final {hyp} = {actual:.4f}, expected >= {threshold}"
        )

    elif atype == "final_hypothesis_max":
        hyp = assertion["hypothesis"]
        threshold = assertion["value"]
        actual = result.final_bluffing if "bluffing" in hyp else result.final_withholding
        assert actual <= threshold, (
            f"{prefix}final {hyp} = {actual:.4f}, expected <= {threshold}"
        )

    elif atype == "final_pressure_min":
        field = assertion["field"]
        threshold = assertion["value"]
        attr = _PRESSURE_FIELDS.get(field, f"final_pressure_{field[:3]}")
        actual = getattr(result, attr, 0.0)
        assert actual >= threshold, (
            f"{prefix}pressure.{field} = {actual:.4f}, expected >= {threshold}"
        )

    elif atype == "final_pressure_max":
        field = assertion["field"]
        threshold = assertion["value"]
        attr = _PRESSURE_FIELDS.get(field, f"final_pressure_{field[:3]}")
        actual = getattr(result, attr, 0.0)
        assert actual <= threshold, (
            f"{prefix}pressure.{field} = {actual:.4f}, expected <= {threshold}"
        )

    elif atype == "review_ran":
        expected = assertion["expected"]
        assert result.review_ran == expected, (
            f"{prefix}review_ran = {result.review_ran}, expected {expected}"
        )

    elif atype == "slur_path":
        expected = assertion["expected"]
        assert result.slur_path == expected, (
            f"{prefix}slur_path = {result.slur_path}, expected {expected}"
        )

    elif atype == "slur_context_class":
        expected = assertion["expected"]
        assert result.slur_context_class == expected, (
            f"{prefix}slur_context_class = '{result.slur_context_class}', expected '{expected}'"
        )

    elif atype == "per_turn_rising":
        # Check that the specified field generally rises across target turns
        target_diags = [d for d in result.turn_diags if d.speaker == "target"]
        if len(target_diags) < 2:
            return
        field = assertion.get("field", "pressure_aggregate")
        attr_map = {
            "pressure_aggregate": "pressure_agg",
            "pressure_accusation": "pressure_acc",
            "pressure_hostility": "pressure_hos",
        }
        attr = attr_map.get(field, field)
        values = [getattr(d, attr, 0.0) for d in target_diags]
        # "Generally rising" = last value > first value
        assert values[-1] >= values[0], (
            f"{prefix}{field} should generally rise: first={values[0]:.4f} last={values[-1]:.4f} "
            f"series={[f'{v:.3f}' for v in values]}"
        )

    elif atype in ("paired_pressure_gt", "paired_hypothesis_comment"):
        pass  # handled in paired test logic

    else:
        pytest.fail(f"{prefix}Unknown assertion type: {atype}")


# ---------------------------------------------------------------------------
# Test: single conversation fixtures (LIVE — with LLM)
# ---------------------------------------------------------------------------

def _single_fixture_data() -> list[tuple[str, dict]]:
    items = []
    for p in _all_fixture_paths():
        data = _load_fixture(p)
        if not data.get("paired"):
            items.append((p.stem, data))
    return items


@pytest.mark.live
@pytest.mark.parametrize(
    "fixture_id,fixture",
    _single_fixture_data(),
    ids=[x[0] for x in _single_fixture_data()],
)
def test_single_fixture_live(fixture_id, fixture, tmp_path):
    """Run a single-conversation fixture with live LLM."""
    result = _run_conversation(fixture["turns"], tmp_path)
    _print_diags(fixture["name"], result)

    for assertion in fixture.get("assertions", []):
        _check_assertion(assertion, result, label=fixture_id)


# ---------------------------------------------------------------------------
# Test: single conversation fixtures (FAST — rules-only, no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_id,fixture",
    _single_fixture_data(),
    ids=[x[0] for x in _single_fixture_data()],
)
def test_single_fixture_rules_only(fixture_id, fixture, tmp_path):
    """Run pressure assertions using rules-only (no LLM sensor for target signals).

    Tests the user-pressure detection and assertion framework
    without requiring LM Studio. Target-side signals are zeroed.
    """
    from btom_engine.interaction_context import compute_pressure
    from btom_engine.semantic_review import _should_trigger

    user_texts = [t["text"] for t in fixture["turns"] if t["speaker"].lower() == "user"]
    pressure = compute_pressure(user_texts)

    print(f"\n--- {fixture['name']} (rules-only) ---")
    print(f"  acc={pressure.accusation:.3f} rep={pressure.repetition:.3f} "
          f"hos={pressure.hostility:.3f} agg={pressure.aggregate:.3f}")
    if pressure.debug.per_turn_hits:
        for h in pressure.debug.per_turn_hits:
            all_hits = h.get("accusation_hits", []) + h.get("repetition_hits", []) + h.get("hostility_hits", [])
            if all_hits:
                print(f"  -> \"{h.get('text', '')[:50]}\" : {all_hits}")

    tags = fixture.get("tags", [])

    for assertion in fixture.get("assertions", []):
        atype = assertion["type"]

        if atype in ("final_pressure_min", "final_pressure_max"):
            # Skip pressure assertions for live-tagged fixtures in rules-only mode
            # (these expect LLM semantic rescue that rules alone can't provide)
            if "live" in tags:
                continue
            field = assertion["field"]
            threshold = assertion["value"]
            actual = {"accusation": pressure.accusation, "repetition": pressure.repetition,
                      "hostility": pressure.hostility, "aggregate": pressure.aggregate}.get(field, 0.0)
            if atype == "final_pressure_min":
                assert actual >= threshold, (
                    f"[{fixture_id} rules-only] pressure.{field} = {actual:.4f}, expected >= {threshold}"
                )
            else:
                assert actual <= threshold, (
                    f"[{fixture_id} rules-only] pressure.{field} = {actual:.4f}, expected <= {threshold}"
                )

        elif atype == "review_ran" and assertion.get("expected"):
            last_user = user_texts[-1] if user_texts else ""
            should, reason = _should_trigger(last_user, pressure)
            tags = fixture.get("tags", [])
            if "live" not in tags:
                assert should, (
                    f"[{fixture_id} rules-only] review should trigger but didn't "
                    f"(pressure agg={pressure.aggregate:.3f})"
                )

        elif atype == "slur_path" and assertion.get("expected"):
            last_user = user_texts[-1] if user_texts else ""
            should, reason = _should_trigger(last_user, pressure)
            if assertion["expected"]:
                assert reason == "slur_identity_attack", (
                    f"[{fixture_id} rules-only] expected slur trigger, got '{reason}'"
                )


# ---------------------------------------------------------------------------
# Test: paired conversation fixtures
# ---------------------------------------------------------------------------

def _paired_fixture_data() -> list[tuple[str, dict]]:
    items = []
    for p in _all_fixture_paths():
        data = _load_fixture(p)
        if data.get("paired"):
            items.append((p.stem, data))
    return items


@pytest.mark.live
@pytest.mark.parametrize(
    "fixture_id,fixture",
    _paired_fixture_data(),
    ids=[x[0] for x in _paired_fixture_data()],
)
def test_paired_fixture_live(fixture_id, fixture, tmp_path):
    """Run a paired conversation fixture comparing two conversations."""
    result_a = _run_conversation(fixture["conversation_a"]["turns"], tmp_path / "a")
    result_b = _run_conversation(fixture["conversation_b"]["turns"], tmp_path / "b")

    label_a = fixture["conversation_a"].get("label", "A")
    label_b = fixture["conversation_b"].get("label", "B")

    _print_diags(f"{fixture['name']} — {label_a}", result_a)
    _print_diags(f"{fixture['name']} — {label_b}", result_b)

    for assertion in fixture.get("assertions", []):
        atype = assertion["type"]
        if atype == "paired_pressure_gt":
            field = assertion.get("field", "aggregate")
            attr = _PRESSURE_FIELDS.get(field, f"final_pressure_{field[:3]}")
            val_a = getattr(result_a, attr, 0.0)
            val_b = getattr(result_b, attr, 0.0)
            assert val_b > val_a, (
                f"[{fixture_id}] expected {label_b} pressure.{field} ({val_b:.4f}) > "
                f"{label_a} ({val_a:.4f})"
            )
        elif atype == "paired_hypothesis_comment":
            print(f"  NOTE: {assertion.get('note', '')}")


# ---------------------------------------------------------------------------
# Test: reset cold start
# ---------------------------------------------------------------------------

def test_reset_cold_start(tmp_path):
    """Fixture F: verify reset produces true cold start with no carryover."""
    from btom_engine.interaction_context import compute_pressure

    state_path = tmp_path / "reset_test.json"
    engine = BTOMEngine(state_path=state_path, hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()

    # Phase 1: build up pressure
    engine.record_user_turn("Stop lying to me.")
    engine.record_user_turn("Do not play dumb.")
    engine.record_user_turn("Answer clearly for once.")

    pressure_before = compute_pressure(list(engine._recent_user_turns))
    assert pressure_before.aggregate > 0.2, (
        f"Pre-reset pressure should be >0.2, got {pressure_before.aggregate:.4f}"
    )

    # Phase 2: reset
    engine.reset()

    # Phase 3: verify clean state
    pressure_after = compute_pressure(list(engine._recent_user_turns))
    assert pressure_after.aggregate == 0.0
    assert len(engine._recent_user_turns) == 0
    assert engine.state.current_turn == 0
    assert engine._last_target_text == ""

    # Phase 4: first turn after reset
    engine.record_user_turn("How are you?")
    ct = ConversationTurn(target_text="I'm fine.")
    result = engine.process_turn(ct)
    assert result.user_pressure.aggregate < 0.05, (
        f"First turn after reset should have near-zero pressure, "
        f"got {result.user_pressure.aggregate:.4f}"
    )

    print(f"\n--- F: Reset Cold Start ---")
    print(f"  Pre-reset pressure: {pressure_before.aggregate:.3f}")
    print(f"  Post-reset pressure: {pressure_after.aggregate:.3f}")
    print(f"  Post-reset user turns: {list(engine._recent_user_turns)}")
    print(f"  First turn pressure: {result.user_pressure.aggregate:.3f}")
    print(f"  PASS: true cold start confirmed")
