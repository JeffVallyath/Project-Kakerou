"""Engine configuration constants."""

import os
from pathlib import Path

# Load .env file if it exists (for local development)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_FILE = DATA_DIR / "state_ledger.json"

# --- Bayesian parameters ---
# MOVED TO weights.py (EngineWeights dataclass) — single source of truth
# math_engine.py reads from weights.WEIGHTS, not config.py
# Retained here ONLY for backward compatibility with modules that import them directly
# TODO: Remove after all importers are migrated to weights.py

# --- Text-mode calibration ---
TEXT_MODE = True                          # True = evaluating online text, not live speech
TEXT_MODE_FRAG_RELIABILITY_CAP = 0.25     # standalone fragmentation reliability cap in text mode
TEXT_MODE_FRAG_REINFORCED_CAP = 0.50      # reliability cap when co-signals corroborate
TEXT_MODE_COSIGNAL_THRESHOLD = 0.15       # effective signal threshold for co-signal reinforcement

# --- Novelty / repetition discount ---
NOVELTY_EXACT_THRESHOLD = 0.80      # Jaccard >= this = exact/near-exact duplicate
NOVELTY_PARAPHRASE_THRESHOLD = 0.45 # Jaccard >= this = paraphrase overlap
NOVELTY_EXACT_FLOOR = 0.15          # novelty factor for exact duplicates (hard discount)
NOVELTY_PARAPHRASE_FLOOR = 0.40     # novelty factor floor for paraphrase overlap
NOVELTY_ESCALATION_BONUS = 0.25     # max bonus for escalatory restatements

# ===========================================================================
# LLM Configuration — Composite Model Architecture
#
# LOCAL model: fast signal extraction (sensor, semantic review)
#   Runs on LM Studio, no API key needed, low latency
#
# REMOTE model: heavy reasoning (investigation, claim extraction)
#   Runs on Gemini/Claude/OpenAI API, higher capability
#
# Set GEMINI_API_KEY env var or paste it below to enable remote model.
# When no remote key is set, all calls fall back to the local model.
# ===========================================================================

# --- Local LLM (sensor, semantic review, transcript parsing) ---
LLM_BASE_URL = "http://127.0.0.1:1234"   # LM Studio default
LLM_MODEL = "qwen3.5-9b"
LLM_TIMEOUT_SECONDS = 60

# --- Remote LLM (investigation, claim extraction, scenario simulation) ---
REMOTE_LLM_PROVIDER = "gemini"            # "gemini", "openai", or "anthropic"
REMOTE_LLM_API_KEY = os.environ.get("GEMINI_API_KEY", "")
REMOTE_LLM_MODEL = "gemini-2.5-flash"
REMOTE_LLM_TIMEOUT_SECONDS = 120

# --- Evaluation thresholds ---
CONVERGENCE_CONFIDENCE = 0.85
MAX_VOLATILITY = 0.08

# --- Hypothesis baselines ---
DEFAULT_HYPOTHESES = {
    "target_is_bluffing": 0.10,
    "target_is_withholding_info": 0.40,
}
