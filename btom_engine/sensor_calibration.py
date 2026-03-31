"""Sensor Calibration — corrects systematic LLM sensor biases.

The LLM sensor returns subjective 0-1 scores for 5 behavioral signals.
These scores are not calibrated: the LLM may consistently over/under-score
certain signals, or compress the range (everything between 0.3-0.7).

This module provides:
1. A labeled calibration dataset (sentences with expected signal values)
2. A function to run the sensor on the dataset and compute corrections
3. A thin correction layer that applies per-signal affine transforms

The corrections are: corrected = clamp(slope * raw + intercept, 0, 1)
Fitted via least-squares against the calibration labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btom_engine.schema import ExtractedSignals, SignalReading

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "sensor_calibration.json"


# ---------------------------------------------------------------------------
# Calibration dataset — expert-labeled signal values
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSample:
    text: str
    expected: dict[str, float]  # signal_name -> expected value


# Each sample has expected values for all 5 signals.
# These are hand-labeled based on the signal definitions in sensor.py.
CALIBRATION_DATASET: list[CalibrationSample] = [
    # === HIGH EVASION ===
    CalibrationSample(
        text="I don't want to talk about that.",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.05,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.85, "direct_answer_compliance": 0.05},
    ),
    CalibrationSample(
        text="Why do you even want to know?",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.05,
                  "emotional_intensity": 0.3, "evasive_deflection": 0.80, "direct_answer_compliance": 0.05},
    ),
    CalibrationSample(
        text="That's not really the issue here.",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.1,
                  "emotional_intensity": 0.15, "evasive_deflection": 0.75, "direct_answer_compliance": 0.05},
    ),
    CalibrationSample(
        text="Can we just move on?",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.0,
                  "emotional_intensity": 0.2, "evasive_deflection": 0.85, "direct_answer_compliance": 0.0},
    ),
    CalibrationSample(
        text="I think you should ask someone else about that.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.80, "direct_answer_compliance": 0.0},
    ),

    # === HIGH DEFENSE ===
    CalibrationSample(
        text="I already told you, I had no choice. The system was down and nobody else was around.",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.85,
                  "emotional_intensity": 0.3, "evasive_deflection": 0.1, "direct_answer_compliance": 0.3},
    ),
    CalibrationSample(
        text="If you had been in my position you would've done the exact same thing.",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.80,
                  "emotional_intensity": 0.3, "evasive_deflection": 0.15, "direct_answer_compliance": 0.1},
    ),
    CalibrationSample(
        text="I was trying to help! Nobody else was stepping up.",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.75,
                  "emotional_intensity": 0.5, "evasive_deflection": 0.1, "direct_answer_compliance": 0.2},
    ),

    # === HIGH COMPLIANCE (truthful, specific answers) ===
    CalibrationSample(
        text="I sent the report at 3pm to Sarah in accounting.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.95},
    ),
    CalibrationSample(
        text="The meeting covered Q3 results. Revenue was up 12%.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.90},
    ),
    CalibrationSample(
        text="I left at 5:30pm, took the 6pm train home.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.90},
    ),
    CalibrationSample(
        text="Yes, I approved it on March 15th after reviewing with legal.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.90},
    ),
    CalibrationSample(
        text="I was in Building 3, Room 214, from 2pm to 4:30pm. The access log will confirm it.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.05,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.95},
    ),

    # === HIGH EMOTIONAL INTENSITY ===
    CalibrationSample(
        text="Fuck you",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.0,
                  "emotional_intensity": 0.95, "evasive_deflection": 0.3, "direct_answer_compliance": 0.0},
    ),
    CalibrationSample(
        text="I DEFINITELY was not there. I swear on my life I was home all night.",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.3,
                  "emotional_intensity": 0.80, "evasive_deflection": 0.1, "direct_answer_compliance": 0.4},
    ),
    CalibrationSample(
        text="Honestly? I'm pissed. They changed the scope twice and then blamed us for being late.",
        expected={"syntactic_fragmentation": 0.15, "defensive_justification": 0.4,
                  "emotional_intensity": 0.75, "evasive_deflection": 0.05, "direct_answer_compliance": 0.4},
    ),

    # === HIGH FRAGMENTATION ===
    CalibrationSample(
        text="I mean... look... it's just... things got complicated.",
        expected={"syntactic_fragmentation": 0.85, "defensive_justification": 0.1,
                  "emotional_intensity": 0.2, "evasive_deflection": 0.5, "direct_answer_compliance": 0.05},
    ),
    CalibrationSample(
        text="So basically, like, what happened was, honestly, it's kind of hard to explain.",
        expected={"syntactic_fragmentation": 0.80, "defensive_justification": 0.1,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.4, "direct_answer_compliance": 0.05},
    ),
    CalibrationSample(
        text="Well, I mean, you know, at the end of the day it is what it is, right?",
        expected={"syntactic_fragmentation": 0.70, "defensive_justification": 0.0,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.5, "direct_answer_compliance": 0.0},
    ),

    # === NEUTRAL / LOW EVERYTHING ===
    CalibrationSample(
        text="ok",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.1},
    ),
    CalibrationSample(
        text="Yes.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.3},
    ),
    CalibrationSample(
        text="sure lets do it",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.0, "direct_answer_compliance": 0.3},
    ),
    CalibrationSample(
        text="nah im good thanks",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.1, "direct_answer_compliance": 0.3},
    ),

    # === MIXED SIGNALS ===
    CalibrationSample(
        text="lol idk",
        expected={"syntactic_fragmentation": 0.3, "defensive_justification": 0.0,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.6, "direct_answer_compliance": 0.0},
    ),
    CalibrationSample(
        text="whatever bro",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.0,
                  "emotional_intensity": 0.3, "evasive_deflection": 0.5, "direct_answer_compliance": 0.0},
    ),
    CalibrationSample(
        text="Mistakes were made. The situation was handled.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.1,
                  "emotional_intensity": 0.05, "evasive_deflection": 0.6, "direct_answer_compliance": 0.15},
    ),
    CalibrationSample(
        text="That was all Mike's idea. I just went along with it.",
        expected={"syntactic_fragmentation": 0.0, "defensive_justification": 0.7,
                  "emotional_intensity": 0.1, "evasive_deflection": 0.2, "direct_answer_compliance": 0.3},
    ),
    CalibrationSample(
        text="I'm not 100% sure, but I think it was around $5,000. I can check the exact number.",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.0,
                  "emotional_intensity": 0.0, "evasive_deflection": 0.05, "direct_answer_compliance": 0.70},
    ),
    CalibrationSample(
        text="Yeah I screwed up. I missed the deadline because I underestimated the integration work. That's on me.",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.3,
                  "emotional_intensity": 0.2, "evasive_deflection": 0.0, "direct_answer_compliance": 0.70},
    ),
    CalibrationSample(
        text="nah",
        expected={"syntactic_fragmentation": 0.05, "defensive_justification": 0.0,
                  "emotional_intensity": 0.05, "evasive_deflection": 0.3, "direct_answer_compliance": 0.1},
    ),
    CalibrationSample(
        text="bruh",
        expected={"syntactic_fragmentation": 0.1, "defensive_justification": 0.0,
                  "emotional_intensity": 0.3, "evasive_deflection": 0.3, "direct_answer_compliance": 0.0},
    ),
]


# ---------------------------------------------------------------------------
# Calibration parameters (per-signal affine transform)
# ---------------------------------------------------------------------------

@dataclass
class SignalCalibration:
    """Per-signal affine correction: corrected = clamp(slope * raw + intercept, 0, 1)."""
    slope: float = 1.0
    intercept: float = 0.0


@dataclass
class SensorCalibrationParams:
    """Calibration parameters for all 5 signals."""
    syntactic_fragmentation: SignalCalibration = field(default_factory=SignalCalibration)
    defensive_justification: SignalCalibration = field(default_factory=SignalCalibration)
    emotional_intensity: SignalCalibration = field(default_factory=SignalCalibration)
    evasive_deflection: SignalCalibration = field(default_factory=SignalCalibration)
    direct_answer_compliance: SignalCalibration = field(default_factory=SignalCalibration)

    def save(self, path: Path | None = None) -> None:
        p = path or _CALIBRATION_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for sig in ["syntactic_fragmentation", "defensive_justification",
                     "emotional_intensity", "evasive_deflection", "direct_answer_compliance"]:
            cal = getattr(self, sig)
            data[sig] = {"slope": cal.slope, "intercept": cal.intercept}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "SensorCalibrationParams":
        p = path or _CALIBRATION_PATH
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            params = cls()
            for sig in ["syntactic_fragmentation", "defensive_justification",
                         "emotional_intensity", "evasive_deflection", "direct_answer_compliance"]:
                if sig in data:
                    setattr(params, sig, SignalCalibration(
                        slope=data[sig].get("slope", 1.0),
                        intercept=data[sig].get("intercept", 0.0),
                    ))
            return params
        except Exception:
            return cls()


# ---------------------------------------------------------------------------
# Apply calibration to signals
# ---------------------------------------------------------------------------

def apply_calibration(
    signals: ExtractedSignals,
    params: SensorCalibrationParams | None = None,
) -> ExtractedSignals:
    """Apply per-signal affine correction to raw LLM sensor output."""
    if params is None:
        params = SensorCalibrationParams.load()

    def _correct(reading: SignalReading, cal: SignalCalibration) -> SignalReading:
        corrected = cal.slope * reading.value + cal.intercept
        return SignalReading(
            value=max(0.0, min(1.0, corrected)),
            signal_reliability=reading.signal_reliability,
        )

    signals.syntactic_fragmentation = _correct(signals.syntactic_fragmentation, params.syntactic_fragmentation)
    signals.defensive_justification = _correct(signals.defensive_justification, params.defensive_justification)
    signals.emotional_intensity = _correct(signals.emotional_intensity, params.emotional_intensity)
    signals.evasive_deflection = _correct(signals.evasive_deflection, params.evasive_deflection)
    signals.direct_answer_compliance = _correct(signals.direct_answer_compliance, params.direct_answer_compliance)

    return signals


# ---------------------------------------------------------------------------
# Fit calibration from data
# ---------------------------------------------------------------------------

def fit_calibration(
    raw_outputs: list[dict[str, float]],
    expected: list[dict[str, float]],
) -> SensorCalibrationParams:
    """Fit per-signal linear calibration from raw LLM outputs vs expected labels.

    Uses numpy least-squares. Returns SensorCalibrationParams ready to save.
    """
    import numpy as np

    params = SensorCalibrationParams()
    signals = ["syntactic_fragmentation", "defensive_justification",
               "emotional_intensity", "evasive_deflection", "direct_answer_compliance"]

    for sig in signals:
        raw = np.array([o.get(sig, 0.0) for o in raw_outputs])
        exp = np.array([e.get(sig, 0.0) for e in expected])

        # Fit y = slope * x + intercept via least squares
        A = np.vstack([raw, np.ones(len(raw))]).T
        result = np.linalg.lstsq(A, exp, rcond=None)
        slope, intercept = result[0]

        setattr(params, sig, SignalCalibration(slope=float(slope), intercept=float(intercept)))

    return params


def run_calibration(rate_limit: float = 0.3) -> SensorCalibrationParams:
    """Run the full calibration pipeline: sensor -> compare -> fit -> save.

    Calls the LLM sensor on each calibration sample, compares to expected,
    fits per-signal corrections, and saves to disk.
    """
    import time
    from btom_engine.sensor import extract_signals_sync

    print(f"Running sensor calibration on {len(CALIBRATION_DATASET)} samples...")

    raw_outputs = []
    expected_list = []

    for i, sample in enumerate(CALIBRATION_DATASET):
        signals = extract_signals_sync(sample.text)
        raw = {
            "syntactic_fragmentation": signals.syntactic_fragmentation.value,
            "defensive_justification": signals.defensive_justification.value,
            "emotional_intensity": signals.emotional_intensity.value,
            "evasive_deflection": signals.evasive_deflection.value,
            "direct_answer_compliance": signals.direct_answer_compliance.value,
        }
        raw_outputs.append(raw)
        expected_list.append(sample.expected)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(CALIBRATION_DATASET)}] processed")

        if rate_limit > 0:
            time.sleep(rate_limit)

    # Fit calibration
    params = fit_calibration(raw_outputs, expected_list)

    # Print diagnostics
    import numpy as np
    signals = ["syntactic_fragmentation", "defensive_justification",
               "emotional_intensity", "evasive_deflection", "direct_answer_compliance"]
    print("\nCalibration results:")
    print(f"{'Signal':<30s} {'Slope':>8s} {'Intercept':>10s} {'MAE_raw':>8s} {'MAE_cal':>8s}")
    print("-" * 70)
    for sig in signals:
        cal = getattr(params, sig)
        raw = np.array([o[sig] for o in raw_outputs])
        exp = np.array([e[sig] for e in expected_list])
        corrected = np.clip(cal.slope * raw + cal.intercept, 0, 1)
        mae_raw = float(np.mean(np.abs(raw - exp)))
        mae_cal = float(np.mean(np.abs(corrected - exp)))
        print(f"{sig:<30s} {cal.slope:>8.3f} {cal.intercept:>10.3f} {mae_raw:>8.3f} {mae_cal:>8.3f}")

    # Save
    params.save()
    print(f"\nCalibration saved to {_CALIBRATION_PATH}")
    return params
