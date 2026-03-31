"""Strategic Deception Classifier — domain-aware learned decision layer.

Architecture:
  1. Domain Detector — classifies conversation type (negotiation, interrogation, social_deduction, general)
  2. Feature Extractor — builds a 41-feature vector from Bayesian engine outputs
  3. Domain Classifier — per-domain trained model (LR/RF) for final deception probability
  4. Router — if domain classifier exists and detector is confident, use it; otherwise fall back to Bayesian

The Bayesian engine always runs. The classifier can only improve, never degrade.
"""

from __future__ import annotations

import itertools
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MODELS_DIR = _DATA_DIR / "classifiers"

# ---------------------------------------------------------------------------
# Feature names (41 features, v2)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Trajectory (10)
    "max_prob", "mean_prob", "std_prob", "last_prob", "slope",
    "n_turns", "frac_above_50", "frac_above_60", "max_minus_min", "late_mean",
    # Deltas (5)
    "max_delta", "min_delta", "mean_abs_delta", "n_positive_jumps", "max_consecutive_rise",
    # Speech acts (8)
    "act_EVADE_rate", "act_DISMISS_rate", "act_DEFLECT_rate", "act_DEFEND_rate",
    "act_INFORM_rate", "act_ACKNOWLEDGE_rate", "act_CHALLENGE_rate", "act_QUALIFY_rate",
    # Speech act bigrams (4)
    "bigram_evade_then_defend", "bigram_inform_then_evade",
    "bigram_defend_then_defend", "act_switches",
    # LIWC (6)
    "liwc_cognitive_rate", "liwc_certainty_rate", "liwc_tentative_rate",
    "liwc_concrete_rate", "liwc_filler_rate", "liwc_self_ref_rate",
    # Text statistics (8)
    "avg_msg_length", "std_msg_length", "total_words", "question_rate",
    "exclamation_rate", "avg_sentence_count", "first_person_density", "hedge_density",
]


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

DOMAINS = ["negotiation", "diplomacy", "interrogation", "general"]

_NEGOTIATION_PATTERNS = re.compile(
    r"\b(trade|offer|deal|give you|how about|package|items?|food|water|firewood|"
    r"i.?ll take|you can have|split|negotiate|priority|willing to)\b",
    re.IGNORECASE,
)
_DIPLOMACY_PATTERNS = re.compile(
    r"\b(alliance|support|betray|stab|move to|fleet|army|convoy|"
    r"austria|england|france|germany|italy|russia|turkey|"
    r"spring|fall|retreat|disband|hold|dmz)\b",
    re.IGNORECASE,
)
_INTERROGATION_PATTERNS = re.compile(
    r"\b(where were you|did you|alibi|suspect|witness|crime|accused|"
    r"confession|lawyer|evidence|testimony|interrogat|detective)\b",
    re.IGNORECASE,
)
_SOCIAL_DEDUCTION_PATTERNS = re.compile(
    r"\b(vote|werewolf|mafia|villager|seer|doctor|role|night phase|"
    r"i.?m the|claim|suspicious|town|wolf|eliminate|lynch)\b",
    re.IGNORECASE,
)


def detect_domain(texts: list[str], confidence_threshold: float = 0.3) -> tuple[str, float]:
    """Detect conversation domain from message texts.

    Returns (domain_name, confidence) where confidence is 0-1.
    Only returns a specific domain if confidence >= threshold,
    otherwise returns ("general", 0.0).
    """
    all_text = " ".join(texts).lower()
    n_words = max(len(all_text.split()), 1)

    scores = {
        "negotiation": len(_NEGOTIATION_PATTERNS.findall(all_text)) / n_words * 50,
        "diplomacy": len(_DIPLOMACY_PATTERNS.findall(all_text)) / n_words * 50,
        "interrogation": len(_INTERROGATION_PATTERNS.findall(all_text)) / n_words * 50,
    }

    best_domain = max(scores, key=scores.get)
    best_score = min(scores[best_domain], 1.0)

    if best_score >= confidence_threshold:
        return best_domain, best_score
    return "general", 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    """Output from the strategic classifier."""
    probability: float = 0.5
    bayesian_probability: float = 0.5
    domain: str = "general"
    domain_confidence: float = 0.0
    classifier_used: bool = False
    top_features: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature extraction (41 features)
# ---------------------------------------------------------------------------

def extract_features(
    predictions: list[float],
    turns: list[Any],
) -> dict[str, float]:
    """Extract the 41-feature vector from engine outputs for one conversation."""
    from btom_engine.speech_acts import analyze_turn
    from btom_engine.liwc_signals import extract_liwc_signals

    p = np.array(predictions) if predictions else np.array([0.5])

    feat: dict[str, float] = {}

    # === Trajectory (10) ===
    feat["max_prob"] = float(p.max())
    feat["mean_prob"] = float(p.mean())
    feat["std_prob"] = float(p.std())
    feat["last_prob"] = float(p[-1])
    feat["slope"] = float(np.polyfit(np.arange(len(p)), p, 1)[0]) if len(p) > 1 else 0.0
    feat["n_turns"] = float(len(p))
    feat["frac_above_50"] = float((p > 0.5).mean())
    feat["frac_above_60"] = float((p > 0.6).mean())
    feat["max_minus_min"] = float(p.max() - p.min())
    feat["late_mean"] = float(p[-3:].mean()) if len(p) >= 3 else float(p.mean())

    # === Deltas (5) ===
    if len(p) > 1:
        deltas = np.diff(p)
        feat["max_delta"] = float(deltas.max())
        feat["min_delta"] = float(deltas.min())
        feat["mean_abs_delta"] = float(np.abs(deltas).mean())
        feat["n_positive_jumps"] = float((deltas > 0.02).sum() / len(deltas))
        feat["max_consecutive_rise"] = float(max(
            (len(list(g)) for k, g in itertools.groupby(deltas > 0) if k), default=0
        ))
    else:
        feat["max_delta"] = 0.0
        feat["min_delta"] = 0.0
        feat["mean_abs_delta"] = 0.0
        feat["n_positive_jumps"] = 0.0
        feat["max_consecutive_rise"] = 0.0

    # === Speech acts (8) + bigrams (4) ===
    acts = []
    for turn in turns:
        text = turn.text if hasattr(turn, "text") else turn.get("text", "")
        context = turn.context if hasattr(turn, "context") else turn.get("context", "")
        act = analyze_turn(target_text=text, user_text=context)
        acts.append(act.target_act)
    total_acts = max(len(acts), 1)
    for act_name in ["EVADE", "DISMISS", "DEFLECT", "DEFEND", "INFORM", "ACKNOWLEDGE", "CHALLENGE", "QUALIFY"]:
        feat[f"act_{act_name}_rate"] = acts.count(act_name) / total_acts

    if len(acts) > 1:
        bigrams = list(zip(acts[:-1], acts[1:]))
        n_bi = max(len(bigrams), 1)
        feat["bigram_evade_then_defend"] = sum(
            1 for a, b in bigrams if a in ("EVADE", "DEFLECT", "DISMISS") and b == "DEFEND"
        ) / n_bi
        feat["bigram_inform_then_evade"] = sum(
            1 for a, b in bigrams if a == "INFORM" and b in ("EVADE", "DEFLECT", "DISMISS")
        ) / n_bi
        feat["bigram_defend_then_defend"] = sum(
            1 for a, b in bigrams if a == "DEFEND" and b == "DEFEND"
        ) / n_bi
        feat["act_switches"] = sum(1 for a, b in bigrams if a != b) / n_bi
    else:
        feat["bigram_evade_then_defend"] = 0.0
        feat["bigram_inform_then_evade"] = 0.0
        feat["bigram_defend_then_defend"] = 0.0
        feat["act_switches"] = 0.0

    # === LIWC (6) ===
    liwc_sums: dict[str, float] = {}
    all_texts = []
    for turn in turns:
        text = turn.text if hasattr(turn, "text") else turn.get("text", "")
        all_texts.append(text)
        liwc = extract_liwc_signals(text)
        for f in ["cognitive_rate", "certainty_rate", "tentative_rate", "concrete_rate", "filler_rate", "self_ref_rate"]:
            liwc_sums[f] = liwc_sums.get(f, 0) + getattr(liwc, f, 0)
    n_t = max(len(turns), 1)
    for f in ["cognitive_rate", "certainty_rate", "tentative_rate", "concrete_rate", "filler_rate", "self_ref_rate"]:
        feat[f"liwc_{f}"] = liwc_sums.get(f, 0) / n_t

    # === Text statistics (8) ===
    all_words = [len(re.findall(r"\b\w+\b", t)) for t in all_texts]
    feat["avg_msg_length"] = float(np.mean(all_words)) if all_words else 0.0
    feat["std_msg_length"] = float(np.std(all_words)) if len(all_words) > 1 else 0.0
    feat["total_words"] = float(sum(all_words))
    feat["question_rate"] = sum(1 for t in all_texts if "?" in t) / max(len(all_texts), 1)
    feat["exclamation_rate"] = sum(1 for t in all_texts if "!" in t) / max(len(all_texts), 1)
    feat["avg_sentence_count"] = float(np.mean([len(re.split(r"[.!?]+", t)) for t in all_texts])) if all_texts else 0.0
    total_w = max(sum(all_words), 1)
    first_person = sum(len(re.findall(r"\b(i|me|my|mine|myself)\b", t.lower())) for t in all_texts)
    feat["first_person_density"] = first_person / total_w
    hedges = sum(len(re.findall(r"\b(maybe|perhaps|possibly|might|could|sort of|kind of|i think|i guess)\b", t.lower())) for t in all_texts)
    feat["hedge_density"] = hedges / total_w

    return feat


# ---------------------------------------------------------------------------
# Predict (domain-routed)
# ---------------------------------------------------------------------------

def predict(
    features: dict[str, float],
    domain: str | None = None,
    domain_confidence: float = 0.0,
    texts: list[str] | None = None,
) -> ClassifierResult:
    """Run the domain-routed classifier.

    If domain is not provided but texts are, auto-detects domain.
    Falls back to Bayesian max_prob if no classifier exists for the domain.
    """
    bayesian_prob = features.get("max_prob", 0.5)

    # Auto-detect domain if not provided
    if domain is None and texts:
        domain, domain_confidence = detect_domain(texts)
    elif domain is None:
        domain = "general"
        domain_confidence = 0.0

    # Try to load domain-specific classifier
    model_path = _MODELS_DIR / f"{domain}.pkl"
    if not model_path.exists() or domain == "general":
        return ClassifierResult(
            probability=bayesian_prob,
            bayesian_probability=bayesian_prob,
            domain=domain,
            domain_confidence=domain_confidence,
            classifier_used=False,
        )

    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        model = bundle["model"]
        scaler = bundle["scaler"]
        feat_names = bundle.get("feature_names", FEATURE_NAMES)
        if set(feat_names) != set(FEATURE_NAMES):
            logger.warning("Model feature names don't match current FEATURE_NAMES — "
                           "predictions may be degraded. Model: %d features, Current: %d",
                           len(feat_names), len(FEATURE_NAMES))
        coefs = bundle.get("feature_importance", {})

        X = np.array([[features.get(name, 0.0) for name in feat_names]])
        X_s = scaler.transform(X)
        proba = float(model.predict_proba(X_s)[0, 1])

        top = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return ClassifierResult(
            probability=proba,
            bayesian_probability=bayesian_prob,
            domain=domain,
            domain_confidence=domain_confidence,
            classifier_used=True,
            top_features=top,
        )
    except Exception as e:
        logger.warning("Classifier failed for domain %s: %s", domain, e)
        return ClassifierResult(
            probability=bayesian_prob,
            bayesian_probability=bayesian_prob,
            domain=domain,
            domain_confidence=domain_confidence,
            classifier_used=False,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_domain_classifier(
    domain: str,
    features_path: Path | None = None,
) -> dict[str, float]:
    """Train a classifier for a specific domain and save to disk."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    fp = features_path or (_DATA_DIR / f"classifier_features_{domain}.json")
    data = json.loads(fp.read_text())

    feat_names = data.get("feature_names", FEATURE_NAMES)
    X = np.array([[f.get(k, 0.0) for k in feat_names] for f in data["features"]])
    y = np.array(data["labels"], dtype=int)

    print(f"Training {domain} classifier: {X.shape[0]} samples, {X.shape[1]} features")

    # Cross-validate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, bas, precs, recs = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = LogisticRegression(max_iter=1000, C=0.05, class_weight="balanced")
        model.fit(X_train, y[train_idx])
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        aucs.append(roc_auc_score(y[test_idx], proba))
        bas.append(balanced_accuracy_score(y[test_idx], pred))
        precs.append(precision_score(y[test_idx], pred, zero_division=0))
        recs.append(recall_score(y[test_idx], pred, zero_division=0))

    # Train on full data
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=0.05, class_weight="balanced")
    model.fit(X_s, y)

    coefs = dict(zip(feat_names, model.coef_[0].tolist()))

    # Save
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS_DIR / f"{domain}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "feature_names": feat_names,
            "feature_importance": coefs,
            "domain": domain,
        }, f)

    metrics = {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_balanced_acc_mean": float(np.mean(bas)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_recall_mean": float(np.mean(recs)),
    }
    print(f"  AUC={metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}  "
          f"BalAcc={metrics['cv_balanced_acc_mean']:.1%}  "
          f"Prec={metrics['cv_precision_mean']:.1%}  "
          f"Rec={metrics['cv_recall_mean']:.1%}")
    print(f"  Saved to {model_path}")
    return metrics
