# Project Kakerou — B-ToM Engine

## What This Is

“Life is just like a gamble that no one can win. In the end, everyone ends up dying. In the end, everyone ends up losing. It is especially because there is an end, that people, during a gamble ... shine" - Toshio Sako, Usogui.

Now, some of you may ask, where does the name ‘Project Kakerou’ come from? Kakerou is a powerful underground gambling organization in the world of Usogui, which is a story obsessed with fittingly, judgment under uncertainty. The narrative revolves around characters who willingly step into environments of extreme uncertainty, brutally stripping away social facades to find the absolute truth hidden beneath human deception. Baku Madarame, the protagonist, is able to make sense of a world that is chaotic and messy and still ... smile. He is able to understand that there exists beauty in this chaos and conquering it. Even in a world drifting toward entropy, structure can still be extracted, and signals can still be separated from the noise. And that's where this project was born. A Bayesian Theory of Mind engine that infers hidden motives (bluffing, withholding, deception) from conversational text. Processes conversations turn-by-turn through a signal extraction pipeline, updates hypothesis probabilities via Bayesian inference, and outputs a deception probability with explainable feature breakdowns. Validated on CaSiNo (negotiation), Diplomacy, DOLOS (courtroom), and Mafiascum (social deduction) datasets.


## Architecture

```
                          ┌─────────────────────┐
                          │   Conversation Turn  │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────┐    ┌──────────────┐   ┌──────────┐
            │ Speech    │    │ LIWC Word    │   │ LLM      │
            │ Act       │    │ Rates        │   │ Sensor   │
            │ Classifier│    │ (cognitive,  │   │ (Gemini) │
            │ (regex)   │    │  certainty,  │   │ 5 axes   │
            └─────┬─────┘    │  tentative)  │   └────┬─────┘
                  │          └──────┬───────┘        │
                  ▼                 ▼                 ▼
            ┌─────────────────────────────────────────────┐
            │         Synthetic Signal Builder             │
            │   Maps acts + LIWC → 5 behavioral axes      │
            └──────────────────┬──────────────────────────┘
                               ▼
            ┌─────────────────────────────────────────────┐
            │         Bayesian Math Engine                 │
            │   Temporal decay · EMA · Covariance penalty  │
            └──────────────────┬──────────────────────────┘
                               ▼
          ┌────────────┬───────┴───────┬────────────────┐
          ▼            ▼               ▼                ▼
   ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
   │ Claim      │ │ Baseline │ │ Preference │ │ Domain       │
   │ Tracker    │ │ Deviation│ │ Inference  │ │ Classifier   │
   │ (contra-   │ │ (z-score │ │ (action vs │ │ (41-feature  │
   │  dictions) │ │  shift)  │ │  claim)    │ │  logistic)   │
   └────────────┘ └──────────┘ └────────────┘ └──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    State Ledger     │
                    │  (hypothesis probs, │
                    │   momentum, turn #) │
                    └─────────────────────┘
```

## Benchmark Numbers

| Domain | Dataset | AUC | Classifier |
|--------|---------|-----|------------|
| Negotiation | CaSiNo (1,200 conversations) | **0.739** | 41-feature logistic regression |
| Diplomacy | Diplomacy (496 messages) | **0.694** | Domain-routed classifier |
| Interrogation | DOLOS courtroom (121 transcripts) | **0.657** | Domain-routed classifier |
| Social Deduction | Mafiascum (500 games) | **0.617** (lower than our default zero-shot AUC at **.636** so thus not added as a classifier) | Relational features |

Measured on Fast mode (no LLM, pure Python) - LLM Sensor scored .785 on CaSiNo but took 2 hours to parse through 5000+ turns over a sample size of 500 conversations. 



## How a Turn is Processed

1. **Input** — Target text enters via cockpit or CLI
2. **Speech Act Classification** — Regex classifies the turn (INFORM, EVADE, DISMISS, DEFEND, DEFLECT, CHALLENGE, QUALIFY, ACKNOWLEDGE)
3. **LIWC Analysis** — Word rates computed (cognitive, certainty, tentative, filler, self-reference, concrete)
4. **Synthetic Signal Builder** — Maps speech acts + LIWC onto 5 behavioral axes (evasion, defensiveness, fragmentation, emotion, compliance)
5. **Bayesian Update** — Prior probabilities updated via likelihood ratios with temporal decay
6. **Claim Tracking** — Extracts factual claims, checks for contradictions with prior claims
7. **Baseline Deviation** — Compares current turn metrics against the target's established baseline (z-scores)
8. **Preference Inference** — Detects action-claim divergence (says one thing, does another)
9. **Domain Classifier** — 41-feature logistic regression produces final deception probability
10. **State Ledger** — All hypothesis probabilities, momentum, and signals persisted

In **fast mode** (default for replay), steps 2-10 run in pure Python at ~1,700 turns/sec. In **full mode**, step 2 is replaced by an LLM sensor call via Gemini API (~0.4s/turn).

## Key Commands

```bash
# Run the cockpit dashboard
.venv/Scripts/python -m streamlit run cockpit/app.py

# Run tests (340/347 pass without LLM connectivity)
.venv/Scripts/python -m pytest tests/ -v
```

## Environment Setup

```bash
# Python 3.13 (tested), 3.9+ compatible
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e .

# Copy .env.example to .env and add your Gemini API key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# Fast mode (no LLM) works without any API key
```

## Key Design Decisions

- **Regex-first**: Speech acts and LIWC run as pure Python. The LLM sensor is optional.
- **Domain routing**: Each domain has its own trained classifier with domain-specific feature weights.
- **Behavioral baselines**: The cockpit can build a per-target linguistic profile from Reddit comment history or pasted transcripts.
- **No training data leakage**: Classifiers are validated with strict holdout protocols and leakage audits.
