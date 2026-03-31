"""Behavioral Baseline — builds a psychological profile from digital footprint data.

Uses MOSAIC extractors (MIT licensed) to collect real user content from platforms,
then builds a statistical behavioral baseline in pure Python. No LLM needed for
baseline construction — it's all word frequency, sentiment math, and pattern detection.

The baseline tells us HOW this person normally communicates, so we can detect
deviations during live conversation.

Baseline dimensions:
- Vocabulary complexity (avg word length, unique word ratio, sentence length)
- Emotional range (positive/negative word ratios, intensity markers)
- Communication style (formal vs casual, question frequency, assertion patterns)
- Topic distribution (what they talk about unprompted)
- Temporal patterns (posting frequency, time-of-day distribution)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline data structures
# ---------------------------------------------------------------------------

@dataclass
class BehavioralBaseline:
    """Statistical behavioral profile built from a person's digital footprint."""
    target_name: str = ""
    total_posts_analyzed: int = 0
    platforms_sampled: list[str] = field(default_factory=list)

    # Vocabulary complexity
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0      # words per sentence
    unique_word_ratio: float = 0.0        # vocabulary diversity (0-1)
    avg_words_per_post: float = 0.0

    # Emotional markers
    exclamation_rate: float = 0.0         # exclamations per post
    question_rate: float = 0.0            # questions per post
    caps_rate: float = 0.0                # ALL CAPS words per post
    profanity_rate: float = 0.0           # profanity per post
    positive_ratio: float = 0.0           # positive vs total sentiment words
    negative_ratio: float = 0.0

    # Communication style
    formality_score: float = 0.0          # 0 = very casual, 1 = very formal
    assertion_rate: float = 0.0           # declarative statements per post
    hedging_rate: float = 0.0             # "I think", "maybe", "probably" per post
    self_reference_rate: float = 0.0      # "I", "me", "my" per post

    # LIWC-derived psycholinguistic metrics
    cognitive_load_rate: float = 0.0      # cognitive mechanism words per post
    exclusive_word_rate: float = 0.0      # exclusionary logic words per post
    certainty_rate: float = 0.0           # certainty/conviction words per post
    tentative_rate: float = 0.0           # uncertainty words per post
    distancing_rate: float = 0.0          # third-person/distancing words per post
    concreteness_rate: float = 0.0        # concrete/sensory words per post
    filler_rate: float = 0.0             # filler/padding words per post

    # Topic distribution (top 10 topic keywords)
    top_topics: list[str] = field(default_factory=list)
    topic_distribution: dict[str, float] = field(default_factory=dict)

    # Temporal patterns
    posts_per_day_avg: float = 0.0
    most_active_hours: list[int] = field(default_factory=list)  # 0-23

    # Raw stats for comparison
    word_length_std: float = 0.0
    sentence_length_std: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Human-readable baseline summary."""
        parts = [f"Behavioral Baseline: {self.target_name}"]
        parts.append(f"  Posts analyzed: {self.total_posts_analyzed} across {', '.join(self.platforms_sampled)}")
        parts.append(f"  Vocabulary: avg word length {self.avg_word_length:.1f}, "
                     f"diversity {self.unique_word_ratio:.2f}, "
                     f"avg {self.avg_words_per_post:.0f} words/post")
        parts.append(f"  Emotional range: excl={self.exclamation_rate:.2f}/post, "
                     f"caps={self.caps_rate:.2f}/post, "
                     f"positive={self.positive_ratio:.2f}, negative={self.negative_ratio:.2f}")
        parts.append(f"  Style: formality={self.formality_score:.2f}, "
                     f"hedging={self.hedging_rate:.2f}/post, "
                     f"self-reference={self.self_reference_rate:.2f}/post")
        if self.top_topics:
            parts.append(f"  Top topics: {', '.join(self.top_topics[:7])}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment / style word lists (pure Python, no external deps)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "good", "great", "awesome", "amazing", "excellent", "love", "best", "happy",
    "wonderful", "fantastic", "beautiful", "perfect", "brilliant", "nice", "cool",
    "thanks", "thank", "appreciate", "glad", "excited", "impressive", "helpful",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "angry", "sad",
    "annoying", "stupid", "ugly", "disgusting", "pathetic", "trash", "garbage",
    "useless", "broken", "fail", "wrong", "sucks", "disappointed", "frustrating",
}

_HEDGING_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "might", "could", "i think",
    "i guess", "i suppose", "kind of", "sort of", "somewhat", "arguably",
    "seems like", "not sure", "i believe", "in my opinion",
}

_FORMAL_MARKERS = {
    "however", "therefore", "furthermore", "additionally", "consequently",
    "nevertheless", "regarding", "concerning", "accordingly", "moreover",
    "particularly", "specifically", "essentially", "fundamentally",
}

_CASUAL_MARKERS = {
    "lol", "lmao", "bruh", "bro", "nah", "yeah", "yep", "nope", "idk",
    "tbh", "imo", "smh", "af", "omg", "wtf", "btw", "fr", "lowkey",
    "highkey", "vibes", "sus", "cap", "lit", "goat", "fam", "haha",
}

_PROFANITY = {
    "fuck", "shit", "damn", "ass", "hell", "bitch", "bastard", "crap",
    "dick", "piss", "bullshit",
}

# ---------------------------------------------------------------------------
# LIWC-derived categories (from published psycholinguistic research)
# Sources: Pennebaker et al. (2015), Newman et al. (2003), Tausczik & Pennebaker (2010)
# ---------------------------------------------------------------------------

# Cognitive mechanism words — spike during high cognitive load (rationalizing, lying)
_COGNITIVE_WORDS = {
    "think", "know", "believe", "consider", "understand", "realize", "remember",
    "recognize", "assume", "imagine", "expect", "suppose", "figure", "wonder",
    "cause", "because", "reason", "therefore", "hence", "thus", "since",
    "effect", "result", "consequence", "meaning", "imply", "suggest",
    "ought", "should", "must", "need", "require",
    # Slang cognitive
    "recall", "recollect", "forgot", "forget",
}

# Exclusive words — require complex cognition, DROP during deception
# (liars can't maintain exclusionary logic under cognitive load)
_EXCLUSIVE_WORDS = {
    "but", "except", "without", "however", "rather", "instead", "although",
    "unless", "despite", "whereas", "nevertheless", "unlike", "excluding",
    "aside", "besides", "otherwise", "nonetheless", "yet", "still",
}

# Certainty words — overuse signals overcompensation (trying too hard to convince)
_CERTAINTY_WORDS = {
    "always", "never", "definitely", "absolutely", "certainly", "clearly",
    "obviously", "undoubtedly", "surely", "totally", "completely", "entirely",
    "guaranteed", "positive", "exact", "precisely", "literally", "honestly",
    "truthfully", "swear", "promise", "100",
    # Slang certainty
    "deadass", "frfr", "nocap", "ongod", "ong", "istg", "facts",
}

# Tentative words — uncertainty markers, can indicate genuine thought or evasion
_TENTATIVE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "somewhat", "somehow", "something", "anything", "whatever",
    "guess", "hope", "almost", "nearly", "roughly", "approximately",
    "around", "about", "kinda", "sorta",
    # Slang hedging
    "lowkey", "idk", "idrk", "ig", "ion", "tbh", "imo", "ngl",
}

# Distancing language — liars subconsciously distance from the lie
_DISTANCING_WORDS = {
    "that", "those", "there", "their", "them", "they", "it", "its",
    "one", "someone", "something", "people", "everyone", "anybody",
}

# Self-reference words (first-person) — DROP indicates psychological distancing
_SELF_REFERENCE = {"i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"}

# Concrete words — high concreteness = truthful (episodic memory present)
# Low concreteness = abstract deflection (no real memory to draw from)
_CONCRETE_WORDS = {
    # Physical locations/objects
    "house", "office", "car", "room", "door", "table", "phone", "computer",
    "street", "building", "store", "restaurant", "gym", "school", "home",
    "kitchen", "bathroom", "desk", "chair", "bed", "window",
    # Temporal specifics
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "yesterday", "today", "tomorrow",
    # Sensory details
    "saw", "heard", "felt", "smelled", "touched", "looked", "watched",
    "red", "blue", "green", "black", "white", "loud", "quiet", "cold", "hot",
    # Numbers/specifics
    "first", "second", "third", "once", "twice", "three", "four", "five",
}

# Filler/padding words — used to buy thinking time or pad deceptive statements
_FILLER_WORDS = {
    "like", "basically", "actually", "honestly", "literally", "seriously",
    "really", "just", "well", "so", "anyway", "right", "okay", "look",
    "listen", "mean", "know", "see", "thing", "stuff",
    "to be honest", "to be fair", "at the end of the day", "the thing is",
    # Slang fillers
    "bruh", "bro", "dude", "fam", "yo", "dawg", "chill", "vibes",
    "per my last email", "with all due respect", "no offense but",
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "about", "up", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "this", "that", "these", "those", "and", "but", "or", "if",
}


# ---------------------------------------------------------------------------
# Text analysis functions (pure Python math)
# ---------------------------------------------------------------------------

def _analyze_text(text: str) -> dict:
    """Analyze a single text block. Returns raw metrics."""
    if not text or len(text.strip()) < 5:
        return {}

    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if not words:
        return {}

    word_lengths = [len(w) for w in words]
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences] if sentences else [len(words)]

    # Count markers
    exclamations = text.count('!')
    questions = text.count('?')
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)

    word_set = set(words)
    positive = len(word_set & _POSITIVE_WORDS)
    negative = len(word_set & _NEGATIVE_WORDS)
    hedging = sum(1 for h in _HEDGING_WORDS if h in text.lower())
    formal = len(word_set & _FORMAL_MARKERS)
    casual = len(word_set & _CASUAL_MARKERS)
    profanity = len(word_set & _PROFANITY)
    self_refs = sum(1 for w in words if w in _SELF_REFERENCE)

    # LIWC-derived metrics
    cognitive = len(word_set & _COGNITIVE_WORDS)
    exclusive = len(word_set & _EXCLUSIVE_WORDS)
    certainty = len(word_set & _CERTAINTY_WORDS)
    tentative = len(word_set & _TENTATIVE_WORDS)
    distancing = len(word_set & _DISTANCING_WORDS)
    concrete = len(word_set & _CONCRETE_WORDS)
    filler = sum(1 for f in _FILLER_WORDS if f in text.lower())

    # Derived ratios
    n_words = len(words)
    self_ref_rate = self_refs / n_words if n_words > 0 else 0
    concrete_ratio = concrete / n_words if n_words > 0 else 0
    cognitive_ratio = cognitive / n_words if n_words > 0 else 0
    certainty_ratio = certainty / n_words if n_words > 0 else 0
    filler_ratio = filler / n_words if n_words > 0 else 0

    # Topic words (non-stopwords, length > 3)
    topic_words = [w for w in words if w not in _STOPWORDS and len(w) > 3]

    return {
        "word_count": n_words,
        "avg_word_length": sum(word_lengths) / len(word_lengths),
        "word_length_values": word_lengths,
        "sentence_count": len(sentences),
        "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
        "sentence_length_values": sentence_lengths,
        "unique_ratio": len(set(words)) / len(words),
        "exclamations": exclamations,
        "questions": questions,
        "caps_words": caps_words,
        "positive": positive,
        "negative": negative,
        "hedging": hedging,
        "formal_markers": formal,
        "casual_markers": casual,
        "profanity": profanity,
        "self_refs": self_refs,
        "self_ref_rate": self_ref_rate,
        # LIWC-derived
        "cognitive": cognitive,
        "cognitive_ratio": cognitive_ratio,
        "exclusive": exclusive,
        "certainty": certainty,
        "certainty_ratio": certainty_ratio,
        "tentative": tentative,
        "distancing": distancing,
        "concrete": concrete,
        "concrete_ratio": concrete_ratio,
        "filler": filler,
        "filler_ratio": filler_ratio,
        "topic_words": topic_words,
    }


def _std_dev(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Baseline builder
# ---------------------------------------------------------------------------

def build_baseline_from_texts(
    texts: list[str],
    target_name: str = "",
    platform: str = "",
) -> BehavioralBaseline:
    """Build a behavioral baseline from a list of text samples.

    Each text is one post/comment/message from the target.
    Pure Python — no LLM calls.
    """
    baseline = BehavioralBaseline(target_name=target_name)
    if platform:
        baseline.platforms_sampled.append(platform)

    if not texts:
        return baseline

    all_metrics = []
    all_topic_words: list[str] = []
    all_word_lengths: list[float] = []
    all_sentence_lengths: list[float] = []

    for text in texts:
        metrics = _analyze_text(text)
        if metrics:
            all_metrics.append(metrics)
            all_topic_words.extend(metrics.get("topic_words", []))
            all_word_lengths.extend(metrics.get("word_length_values", []))
            all_sentence_lengths.extend(metrics.get("sentence_length_values", []))

    if not all_metrics:
        return baseline

    n = len(all_metrics)
    baseline.total_posts_analyzed = n

    # Averages
    baseline.avg_word_length = sum(m["avg_word_length"] for m in all_metrics) / n
    baseline.avg_sentence_length = sum(m["avg_sentence_length"] for m in all_metrics) / n
    baseline.unique_word_ratio = sum(m["unique_ratio"] for m in all_metrics) / n
    baseline.avg_words_per_post = sum(m["word_count"] for m in all_metrics) / n

    # Standard deviations
    baseline.word_length_std = _std_dev(all_word_lengths) if all_word_lengths else 0.0
    baseline.sentence_length_std = _std_dev(all_sentence_lengths) if all_sentence_lengths else 0.0

    # Emotional markers (per post rates)
    baseline.exclamation_rate = sum(m["exclamations"] for m in all_metrics) / n
    baseline.question_rate = sum(m["questions"] for m in all_metrics) / n
    baseline.caps_rate = sum(m["caps_words"] for m in all_metrics) / n
    baseline.profanity_rate = sum(m["profanity"] for m in all_metrics) / n

    total_pos = sum(m["positive"] for m in all_metrics)
    total_neg = sum(m["negative"] for m in all_metrics)
    total_sentiment = total_pos + total_neg
    baseline.positive_ratio = total_pos / total_sentiment if total_sentiment > 0 else 0.5
    baseline.negative_ratio = total_neg / total_sentiment if total_sentiment > 0 else 0.5

    # Communication style
    total_formal = sum(m["formal_markers"] for m in all_metrics)
    total_casual = sum(m["casual_markers"] for m in all_metrics)
    total_style = total_formal + total_casual
    baseline.formality_score = total_formal / total_style if total_style > 0 else 0.5

    baseline.hedging_rate = sum(m["hedging"] for m in all_metrics) / n
    baseline.self_reference_rate = sum(m["self_refs"] for m in all_metrics) / n

    # LIWC-derived psycholinguistic metrics
    baseline.cognitive_load_rate = sum(m["cognitive"] for m in all_metrics) / n
    baseline.exclusive_word_rate = sum(m["exclusive"] for m in all_metrics) / n
    baseline.certainty_rate = sum(m["certainty"] for m in all_metrics) / n
    baseline.tentative_rate = sum(m["tentative"] for m in all_metrics) / n
    baseline.distancing_rate = sum(m["distancing"] for m in all_metrics) / n
    baseline.concreteness_rate = sum(m["concrete"] for m in all_metrics) / n
    baseline.filler_rate = sum(m["filler"] for m in all_metrics) / n

    # Topic distribution
    topic_counter = Counter(all_topic_words)
    top_20 = topic_counter.most_common(20)
    total_topics = sum(c for _, c in top_20) if top_20 else 1
    baseline.top_topics = [word for word, _ in top_20[:10]]
    baseline.topic_distribution = {word: count / total_topics for word, count in top_20}

    return baseline


# ---------------------------------------------------------------------------
# MOSAIC integration — extract texts from MOSAIC JSON output
# ---------------------------------------------------------------------------

def extract_texts_from_mosaic_json(json_path: str | Path) -> list[str]:
    """Extract user-written text samples from a MOSAIC extractor JSON file.

    Handles GitHub (commit messages, repo descriptions), Reddit (posts, comments),
    StackOverflow (answers), Medium (articles), etc.
    """
    path = Path(json_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    texts = []

    # Reddit: posts + comments
    for post in data.get("posts", []):
        title = post.get("title", "")
        body = post.get("selftext", post.get("body", ""))
        if title:
            texts.append(title)
        if body and len(body) > 10:
            texts.append(body)

    for comment in data.get("comments", []):
        body = comment.get("body", comment.get("text", ""))
        if body and len(body) > 10:
            texts.append(body)

    # GitHub: repo descriptions + commit messages
    for repo in data.get("repositories", []):
        desc = repo.get("description", "")
        if desc and len(desc) > 10:
            texts.append(desc)

    for event in data.get("events", []):
        # Push events have commit messages
        payload = event.get("payload", {})
        for commit in payload.get("commits", []):
            msg = commit.get("message", "")
            if msg and len(msg) > 10:
                texts.append(msg)

    # StackOverflow: answers
    for answer in data.get("answers", []):
        body = answer.get("body", answer.get("body_markdown", ""))
        if body and len(body) > 20:
            texts.append(body)

    # Medium: articles
    for article in data.get("articles", []):
        title = article.get("title", "")
        content = article.get("content", article.get("subtitle", ""))
        if title:
            texts.append(title)
        if content and len(content) > 20:
            texts.append(content)

    # Generic: any "content" or "text" fields
    for item in data.get("items", data.get("messages", [])):
        text = item.get("text", item.get("content", item.get("body", "")))
        if text and len(text) > 10:
            texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# Deviation detection
# ---------------------------------------------------------------------------

def compute_deviation(baseline: BehavioralBaseline, live_text: str) -> dict:
    """Compare a live conversation turn against the behavioral baseline.

    Returns a dict of deviations — how much each metric differs from
    the person's established baseline. Positive = higher than normal.

    This is where the Bayesian engine gets informed: large deviations
    in specific dimensions suggest the person is behaving unusually.
    """
    metrics = _analyze_text(live_text)
    if not metrics or baseline.total_posts_analyzed == 0:
        return {"has_baseline": False}

    word_count = metrics["word_count"]

    deviations = {"has_baseline": True, "signals": {}}

    # Vocabulary deviation
    wl_dev = (metrics["avg_word_length"] - baseline.avg_word_length) / max(baseline.word_length_std, 0.5)
    deviations["signals"]["vocabulary_complexity"] = {
        "deviation": round(wl_dev, 2),
        "live": round(metrics["avg_word_length"], 2),
        "baseline": round(baseline.avg_word_length, 2),
        "interpretation": "simpler than usual" if wl_dev < -1 else "more complex than usual" if wl_dev > 1 else "normal",
    }

    # Sentence length deviation
    if baseline.sentence_length_std > 0:
        sl_dev = (metrics["avg_sentence_length"] - baseline.avg_sentence_length) / max(baseline.sentence_length_std, 1.0)
    else:
        sl_dev = 0.0
    deviations["signals"]["sentence_complexity"] = {
        "deviation": round(sl_dev, 2),
        "live": round(metrics["avg_sentence_length"], 2),
        "baseline": round(baseline.avg_sentence_length, 2),
        "interpretation": "shorter than usual" if sl_dev < -1 else "longer than usual" if sl_dev > 1 else "normal",
    }

    # Emotional intensity deviation
    excl_dev = (metrics["exclamations"] - baseline.exclamation_rate)
    caps_dev = (metrics["caps_words"] - baseline.caps_rate)
    emotional_dev = excl_dev + caps_dev
    deviations["signals"]["emotional_intensity"] = {
        "deviation": round(emotional_dev, 2),
        "interpretation": "more emotional than usual" if emotional_dev > 1 else "calmer than usual" if emotional_dev < -0.5 else "normal",
    }

    # Formality deviation
    formal = metrics["formal_markers"]
    casual = metrics["casual_markers"]
    live_formality = formal / (formal + casual) if (formal + casual) > 0 else 0.5
    formality_dev = live_formality - baseline.formality_score
    deviations["signals"]["formality_shift"] = {
        "deviation": round(formality_dev, 2),
        "live": round(live_formality, 2),
        "baseline": round(baseline.formality_score, 2),
        "interpretation": "more formal than usual" if formality_dev > 0.2 else "more casual than usual" if formality_dev < -0.2 else "normal",
    }

    # Self-reference deviation
    self_ref_live = metrics["self_refs"] / max(word_count, 1) * 100
    self_ref_baseline = baseline.self_reference_rate / max(baseline.avg_words_per_post, 1) * 100
    self_ref_dev = self_ref_live - self_ref_baseline
    deviations["signals"]["self_focus"] = {
        "deviation": round(self_ref_dev, 2),
        "interpretation": "more self-focused than usual" if self_ref_dev > 1 else "less self-focused than usual" if self_ref_dev < -1 else "normal",
    }

    # Hedging deviation
    hedge_live = metrics["hedging"]
    hedge_dev = hedge_live - baseline.hedging_rate
    deviations["signals"]["uncertainty"] = {
        "deviation": round(hedge_dev, 2),
        "interpretation": "more uncertain than usual" if hedge_dev > 0.5 else "more assertive than usual" if hedge_dev < -0.5 else "normal",
    }

    # --- LIWC-derived deviations ---

    # Cognitive load (spikes during rationalization/deception)
    cog_dev = metrics["cognitive"] - baseline.cognitive_load_rate
    deviations["signals"]["cognitive_load"] = {
        "deviation": round(cog_dev, 2),
        "interpretation": "high cognitive load — rationalizing?" if cog_dev > 1 else "low cognitive engagement" if cog_dev < -1 else "normal",
    }

    # Concreteness (drops during deception — no real memory to draw from)
    conc_dev = metrics["concrete_ratio"] - baseline.concreteness_rate / max(baseline.avg_words_per_post, 1)
    deviations["signals"]["concreteness"] = {
        "deviation": round(conc_dev, 4),
        "interpretation": "less concrete than usual — abstract deflection?" if conc_dev < -0.02 else "more concrete than usual" if conc_dev > 0.02 else "normal",
    }

    # Certainty overcompensation (overuse of "always", "never", "definitely")
    cert_dev = metrics["certainty"] - baseline.certainty_rate
    deviations["signals"]["certainty_overcompensation"] = {
        "deviation": round(cert_dev, 2),
        "interpretation": "overcompensating with certainty language" if cert_dev > 0.5 else "normal",
    }

    # Pronoun distancing (drop in first-person = distancing from statement)
    self_ref_rate_live = metrics["self_ref_rate"]
    self_ref_rate_base = baseline.self_reference_rate / max(baseline.avg_words_per_post, 1)
    pronoun_dev = self_ref_rate_live - self_ref_rate_base
    deviations["signals"]["pronoun_distancing"] = {
        "deviation": round(pronoun_dev, 4),
        "interpretation": "pronoun drop — distancing from statement" if pronoun_dev < -0.02 else "more self-referential than usual" if pronoun_dev > 0.03 else "normal",
    }

    # Filler padding (buying time, padding deceptive statements)
    filler_dev = metrics["filler"] - baseline.filler_rate
    deviations["signals"]["filler_padding"] = {
        "deviation": round(filler_dev, 2),
        "interpretation": "excessive filler — buying time?" if filler_dev > 1 else "normal",
    }

    # Overall anomaly score (how many dimensions deviate significantly)
    significant_deviations = sum(
        1 for sig in deviations["signals"].values()
        if sig.get("interpretation", "normal") != "normal"
    )
    deviations["anomaly_score"] = significant_deviations / len(deviations["signals"])
    deviations["anomaly_label"] = (
        "HIGH ANOMALY" if deviations["anomaly_score"] > 0.5
        else "MODERATE ANOMALY" if deviations["anomaly_score"] > 0.25
        else "NORMAL"
    )

    return deviations
