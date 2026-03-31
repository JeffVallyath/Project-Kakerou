"""Data schema — Pydantic models for the State Ledger (Layer 4)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SignalReading(BaseModel):
    """A single extracted behavioral signal with its reliability weight."""

    value: float = Field(0.0, ge=0.0, le=1.0)
    signal_reliability: float = Field(0.8, ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    """A tracked hypothesis about the target's hidden motive."""

    probability: float = Field(0.10, ge=0.0, le=1.0)
    momentum: float = 0.0
    evidence_trace: list[str] = Field(default_factory=lambda: ["Initial baseline"])

    @field_validator("probability", mode="after")
    @classmethod
    def clamp_probability(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ExtractedSignals(BaseModel):
    """All signals extracted from the current conversational turn."""

    syntactic_fragmentation: SignalReading = Field(default_factory=SignalReading)
    defensive_justification: SignalReading = Field(default_factory=SignalReading)
    emotional_intensity: SignalReading = Field(default_factory=SignalReading)
    evasive_deflection: SignalReading = Field(default_factory=SignalReading)
    direct_answer_compliance: SignalReading = Field(default_factory=SignalReading)


class StateLedger(BaseModel):
    """The complete persistent state for one session."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    current_turn: int = 0
    system_status: str = "active"

    extracted_signals_current_turn: ExtractedSignals = Field(
        default_factory=ExtractedSignals
    )
    active_hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)

    # Internal bookkeeping
    turns_since_last_signal: dict[str, int] = Field(default_factory=dict)

    def check_system_status(self) -> None:
        """Shift status to 'insufficient_evidence' if avg reliability < 0.4."""
        signals = self.extracted_signals_current_turn
        reliabilities = [
            signals.syntactic_fragmentation.signal_reliability,
            signals.defensive_justification.signal_reliability,
            signals.emotional_intensity.signal_reliability,
            signals.evasive_deflection.signal_reliability,
            signals.direct_answer_compliance.signal_reliability,
        ]
        avg = sum(reliabilities) / len(reliabilities)
        self.system_status = "active" if avg >= 0.4 else "insufficient_evidence"

    # --- Persistence ---

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StateLedger:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def new_session(cls, hypotheses: dict[str, float] | None = None) -> StateLedger:
        from btom_engine.config import DEFAULT_HYPOTHESES

        hyps = hypotheses or DEFAULT_HYPOTHESES
        return cls(
            active_hypotheses={
                name: Hypothesis(probability=prob)
                for name, prob in hyps.items()
            }
        )
