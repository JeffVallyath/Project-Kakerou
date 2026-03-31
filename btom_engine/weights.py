"""Tunable Weight Configuration — all engine parameters in one place.

Every number that affects hypothesis probabilities lives here.
These values can be optimized by Optuna against ground truth datasets
instead of being hand-tuned.

Load order:
1. Defaults defined here
2. Overridden by weights.json if it exists
3. Overridden by Optuna during optimization runs
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "weights.json"


@dataclass
class EngineWeights:
    """All tunable parameters for the B-ToM engine."""

    # --- Bayesian engine parameters ---
    learning_rate: float = 0.18
    temporal_decay: float = 0.78
    covariance_threshold: float = 0.7
    covariance_penalty: float = 0.6
    signal_dead_zone: float = 0.05
    neutral_band: float = 0.12
    diminishing_returns_threshold: float = 0.7
    diminishing_weak_signal_cap: float = 0.2
    pre_turn_decay: float = 0.95
    ema_smoothing_alpha: float = 0.6

    # --- Signal-hypothesis directions (+1 or -1) ---
    # These define whether a signal SUPPORTS or CONTRADICTS each hypothesis
    frag_bluff_dir: int = 1
    frag_withhold_dir: int = 1
    defense_bluff_dir: int = 1
    defense_withhold_dir: int = -1
    emotion_bluff_dir: int = 1
    evasion_bluff_dir: int = 1
    evasion_withhold_dir: int = 1
    compliance_bluff_dir: int = -1
    compliance_withhold_dir: int = -1

    # --- Speech act weights (bluffing delta per act) ---
    act_inform_bluff: float = -0.15
    act_evade_bluff: float = 0.20
    act_dismiss_bluff: float = 0.15
    act_deflect_bluff: float = 0.18
    act_defend_bluff: float = 0.10
    act_challenge_bluff: float = 0.12
    act_acknowledge_bluff: float = -0.10
    act_qualify_bluff: float = 0.05

    # --- Speech act weights (withholding delta per act) ---
    act_inform_withhold: float = -0.15
    act_evade_withhold: float = 0.20
    act_dismiss_withhold: float = 0.18
    act_deflect_withhold: float = 0.15
    act_defend_withhold: float = 0.05
    act_challenge_withhold: float = 0.10
    act_acknowledge_withhold: float = -0.08
    act_qualify_withhold: float = 0.12

    # --- Adjacency pair violation boost ---
    violation_boost_factor: float = 0.15

    # --- Speech act integration weight (vs Bayesian sensor signals) ---
    speech_act_weight: float = 0.5  # 0 = ignore speech acts, 1 = full weight

    # --- LIWC psycholinguistic weights ---
    # Each weight controls how much a per-word-rate of that category affects bluff probability
    liwc_cognitive_weight: float = 0.10      # cognitive mechanism words (think, know, cause)
    liwc_exclusive_weight: float = -0.08     # exclusive words (but, except) — DROP = deception
    liwc_certainty_weight: float = 0.12      # overcompensation (always, never, definitely)
    liwc_tentative_weight: float = 0.05      # uncertainty (maybe, probably)
    liwc_concrete_weight: float = -0.15      # concreteness (gym, office, monday) — HIGH = truth
    liwc_filler_weight: float = 0.08         # filler padding (basically, honestly, like)
    liwc_self_ref_weight: float = -0.10      # first-person pronouns — DROP = distancing
    liwc_integration_weight: float = 0.5     # overall LIWC weight vs other signals

    # --- Baseline deviation weights ---
    dev_calibration_window: int = 5         # turns to establish baseline
    dev_threshold: float = 1.5              # z-score threshold for significance
    dev_word_count_weight: float = 0.08
    dev_word_length_weight: float = 0.05
    dev_sentence_length_weight: float = 0.10
    dev_unique_ratio_weight: float = 0.05
    dev_self_ref_weight: float = 0.15       # pronoun drop = distancing
    dev_cognitive_weight: float = 0.15      # cognitive spike = rationalizing
    dev_certainty_weight: float = 0.12      # certainty spike = overcompensating
    dev_concrete_weight: float = 0.12       # concreteness drop = abstracting
    dev_filler_weight: float = 0.08         # filler spike = buying time
    dev_integration_weight: float = 0.5     # overall deviation weight

    # --- Claim tracker ---
    claim_contradiction_weight: float = 0.35   # how much contradictions affect bluff probability

    # --- Preference inference (ToM Level 2) ---
    preference_divergence_weight: float = 0.50  # how much action-claim divergence affects bluff prob
    preference_reliability: float = 0.70        # signal_reliability for divergence output
    priority_mismatch_weight: float = 0.30      # how much claim-vs-true-priority mismatch affects bluff

    # --- Hypothesis baselines ---
    baseline_bluffing: float = 0.10
    baseline_withholding: float = 0.40

    # --- Text mode calibration ---
    text_mode_frag_cap: float = 0.25
    text_mode_frag_reinforced_cap: float = 0.50
    text_mode_cosignal_threshold: float = 0.15

    # --- Novelty discount ---
    novelty_exact_threshold: float = 0.80
    novelty_paraphrase_threshold: float = 0.45
    novelty_exact_floor: float = 0.15
    novelty_paraphrase_floor: float = 0.40

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path | None = None) -> None:
        """Save weights to JSON."""
        p = path or _WEIGHTS_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Weights saved to %s", p)

    @classmethod
    def load(cls, path: Path | None = None) -> "EngineWeights":
        """Load weights from JSON, falling back to defaults."""
        p = path or _WEIGHTS_PATH
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                # Only use keys that exist in the dataclass
                valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                w = cls(**valid)
                logger.info("Loaded %d weights from %s", len(valid), p)
                return w
            except Exception as e:
                logger.warning("Failed to load weights: %s", e)
        return cls()

    def get_act_bluff_map(self) -> dict[str, float]:
        """Reconstruct the speech act bluffing map from individual weights."""
        return {
            "INFORM": self.act_inform_bluff,
            "EVADE": self.act_evade_bluff,
            "DISMISS": self.act_dismiss_bluff,
            "DEFLECT": self.act_deflect_bluff,
            "DEFEND": self.act_defend_bluff,
            "CHALLENGE": self.act_challenge_bluff,
            "ACKNOWLEDGE": self.act_acknowledge_bluff,
            "QUALIFY": self.act_qualify_bluff,
        }

    def get_act_withhold_map(self) -> dict[str, float]:
        """Reconstruct the speech act withholding map from individual weights."""
        return {
            "INFORM": self.act_inform_withhold,
            "EVADE": self.act_evade_withhold,
            "DISMISS": self.act_dismiss_withhold,
            "DEFLECT": self.act_deflect_withhold,
            "DEFEND": self.act_defend_withhold,
            "CHALLENGE": self.act_challenge_withhold,
            "ACKNOWLEDGE": self.act_acknowledge_withhold,
            "QUALIFY": self.act_qualify_withhold,
        }

    def get_signal_hypothesis_map(self) -> dict[str, dict[str, int]]:
        """Reconstruct signal-hypothesis direction map."""
        m = {}
        if self.frag_bluff_dir: m.setdefault("syntactic_fragmentation", {})["target_is_bluffing"] = self.frag_bluff_dir
        if self.frag_withhold_dir: m.setdefault("syntactic_fragmentation", {})["target_is_withholding_info"] = self.frag_withhold_dir
        if self.defense_bluff_dir: m.setdefault("defensive_justification", {})["target_is_bluffing"] = self.defense_bluff_dir
        if self.defense_withhold_dir: m.setdefault("defensive_justification", {})["target_is_withholding_info"] = self.defense_withhold_dir
        if self.emotion_bluff_dir: m.setdefault("emotional_intensity", {})["target_is_bluffing"] = self.emotion_bluff_dir
        if self.evasion_bluff_dir: m.setdefault("evasive_deflection", {})["target_is_bluffing"] = self.evasion_bluff_dir
        if self.evasion_withhold_dir: m.setdefault("evasive_deflection", {})["target_is_withholding_info"] = self.evasion_withhold_dir
        if self.compliance_bluff_dir: m.setdefault("direct_answer_compliance", {})["target_is_bluffing"] = self.compliance_bluff_dir
        if self.compliance_withhold_dir: m.setdefault("direct_answer_compliance", {})["target_is_withholding_info"] = self.compliance_withhold_dir
        return m


# Global instance — loaded once, used everywhere
WEIGHTS = EngineWeights.load()
