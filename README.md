# 🤖 Customer Support AI Knowledge Loop
### Built for: RTX 50 Series Laptop · 6GB VRAM · Zero Budget

> **What we're building:** A self-improving customer support assistant that runs 100% locally. It learns from past tickets, drafts replies for agents, captures structured feedback on why suggestions were accepted or rejected, extracts machine-readable signals from that feedback, rebuilds its knowledge base weekly, and shows a live dashboard of how the system improves over time. No cloud. No paid APIs. No fine-tuning required.

---

## 📋 Table of Contents

1. [Hardware Reality Check](#1-hardware-reality-check)
2. [How the Full System Works](#2-how-the-full-system-works)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Phase 0 — Environment Setup (Day 0–3)](#5-phase-0--environment-setup-day-03)
6. [Phase 1 — Data Pipeline & PII (Day 4–10)](#6-phase-1--data-pipeline--pii-day-410)
7. [Phase 2 — RAG Core & Hallucination Guardrails (Day 11–25)](#7-phase-2--rag-core--hallucination-guardrails-day-1125)
8. [Phase 3 — Agent UI & Onboarding (Day 26–45)](#8-phase-3--agent-ui--onboarding-day-2645)
9. [Phase 4 — Feedback Pipeline (Day 46–80)](#9-phase-4--feedback-pipeline-day-4680)
10. [Phase 5 — Signal Extractor (Day 81–100)](#10-phase-5--signal-extractor-day-81100)
11. [Phase 6 — Metrics Dashboard & Human Audits (Day 101–120)](#11-phase-6--metrics-dashboard--human-audits-day-101120)
12. [Versioning & Rollback](#12-versioning--rollback)
13. [Retrain Schedule & Orchestration](#13-retrain-schedule--orchestration)
14. [Data Governance & Consent Policy](#14-data-governance--consent-policy)
15. [Access Controls & Encryption](#15-access-controls--encryption)
16. [Quantization & Memory Plan](#16-quantization--memory-plan)
17. [Testing Strategy](#17-testing-strategy)
18. [Backup & Restore](#18-backup--restore)
19. [Evaluation Targets](#19-evaluation-targets)
20. [Failure Mode Playbook](#20-failure-mode-playbook)
21. [Complete Data Schema](#21-complete-data-schema)

---

## 1. Hardware Reality Check

| Property | Your Spec | What It Means |
|---|---|---|
| GPU | RTX 50 series laptop | Blackwell arch, GDDR7, CUDA 12.8+ |
| VRAM | 6 GB | Text-only tier. No vision models. |
| Best model | Phi-4-mini Q4_K_M (~2.5 GB) | ~80 tok/s, 128K context, MIT license |
| Runner-up | Gemma 4 E4B Q4_K_M (~2.8 GB) | ~40–60 tok/s, strong instruction following |
| Fine-tuning | ❌ Not feasible at 6GB | Use RAG + prompt iteration instead |
| Multimodal | ❌ Vision encoder fills remaining VRAM | Text only |
| Embedding model | ✅ all-MiniLM-L6-v2 (~90 MB) | Runs on CPU — leaves all VRAM for LLM |

> ⚠️ Do NOT attempt 7B/8B models on 6GB. They need ~5.5GB weights + KV cache and will either crash or spill to RAM at 1–2 tok/s (unusable). Stick to 3B–4B fully in VRAM.

---

## 2. How the Full System Works

```
Raw Ticket (customer text)
        │
        ▼
┌──────────────────────────┐
│  1. PII Anonymizer       │  presidio + spaCy — strip at ingest, never store raw
└───────────┬──────────────┘
            │ clean text
            ▼
┌──────────────────────────┐
│  2. SQLite Database      │  tickets, feedback, signal_tags, audit_log, versions
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  3. Embedding Engine     │  all-MiniLM-L6-v2 on CPU → 384-dim vectors
└───────────┬──────────────┘
            │ vectors
            ▼
┌──────────────────────────┐
│  4. FAISS Vector Index   │  top-5 most similar past tickets
└───────────┬──────────────┘
            │ retrieved evidence
            ▼
┌──────────────────────────┐
│  5. RAG Prompt Builder   │  inject evidence + confidence check
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  6. Ollama (phi4-mini)   │  Q4_K_M, ctx=4096, temp=0.3 — 40–80 tok/s
└───────────┬──────────────┘
            │ draft + evidence snippets + confidence score
            ▼
┌──────────────────────────┐
│  7. Output PII Filter    │  re-scan model output before showing agent
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  8. Agent UI             │  suggestion + evidence + structured rationale form
│                          │  Accept / Edit / Reject + escalation if rejected
└───────────┬──────────────┘
            │ structured rationale (reason_type, severity, suggested_change, root_cause)
            ▼
┌──────────────────────────┐
│  9. Feedback Pipeline    │  batch job: dedup, quality filter, convert to training signals
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 10. Signal Extractor     │  phi4-mini classifies rationale → structured tags
└───────────┬──────────────┘
            │ tags stored in signal_tags table
            ▼
┌──────────────────────────┐
│ 11. Weekly Orchestrator  │  snapshot → rebuild index → prompt update → report
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 12. Metrics Dashboard    │  acceptance trend, rejection breakdown,
│     + Human Audit        │  retrieval quality, random sample review
└──────────────────────────┘
```

**The self-reinforcing loop:** agent-corrected responses (step 8) flow through the feedback pipeline (step 9) and get re-embedded into FAISS (step 4). The next similar query automatically retrieves better evidence. No retraining needed — the index itself improves.

---

## 3. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM runtime | Ollama | One-command install, CUDA auto-detected |
| LLM model | phi4-mini Q4_K_M | 2.5GB VRAM, 128K ctx, MIT license |
| Embeddings | sentence-transformers | CPU-based, free, no API key |
| Vector DB | FAISS | Local disk, zero cost, fast |
| PII detection | presidio-analyzer + spaCy | Microsoft OSS, pluggable entity types |
| Web backend | FastAPI | Simple, async, easy to extend |
| Frontend | React + Vite + recharts | Fast dev, component-based, charts built in |
| Database | SQLite + sqlcipher | Zero setup, single file, encryption support |
| Scheduling | cron (Linux) | Weekly automation for all batch jobs |
| Versioning | shell snapshots + git tags | Snapshot index + DB before each update |
| Testing | pytest + playwright | Unit tests for pipeline, E2E for UI |

---

## 4. Project Structure

```
support-ai/
├── README.md
├── requirements.txt
├── .gitignore                     ← includes data/raw/, data/clean/, backups/
├── CONSENT_AND_RETENTION.md       ← data governance policy (see section 14)
│
├── data/
│   ├── raw/                       ← original exports (NEVER commit, NEVER log)
│   ├── clean/                     ← anonymized CSVs
│   │   └── audit_log.json         ← ticket IDs + timestamps that entered pipeline
│   ├── faiss_index/
│   │   ├── index.faiss
│   │   └── metadata.json
│   └── holdout/
│       └── holdout.csv            ← 50–100 tickets never seen by the model
│
├── pipeline/
│   ├── anonymize.py               ← PII scrubbing at ingest
│   ├── embed.py                   ← build / rebuild FAISS index
│   ├── rag.py                     ← retrieval + confidence check + generation
│   ├── output_filter.py           ← re-scan model output for PII
│   ├── feedback_pipeline.py       ← batch: dedup, quality filter, re-index
│   └── signal_extractor.py        ← classify rationale into structured tags
│
├── api/
│   ├── main.py                    ← FastAPI app, auth middleware
│   ├── db.py                      ← SQLite schema + helpers
│   └── routes/
│       ├── suggest.py             ← POST /suggest
│       ├── feedback.py            ← POST /feedback
│       ├── escalate.py            ← POST /escalate (rejection routing)
│       ├── metrics.py             ← GET /metrics
│       └── admin.py               ← POST /admin/rebuild (auth-gated)
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── TicketView.jsx         ← suggestion + structured rationale form
│       ├── EscalationView.jsx     ← senior reviewer queue
│       └── Dashboard.jsx          ← live metrics charts
│
├── evaluation/
│   ├── eval.py                    ← weekly accuracy + delta report
│   ├── audit_sampler.py           ← random sample picker for human review
│   └── ab_compare.py              ← compare two index versions on holdout
│
├── scripts/
│   ├── weekly_pipeline.sh         ← full weekly cron job
│   ├── snapshot.sh                ← snapshot index + DB before any update
│   └── restore.sh                 ← roll back to a named snapshot
│
├── backups/                       ← gitignored; snapshots live here
│   └── YYYY-MM-DD/
│       ├── index.faiss
│       ├── metadata.json
│       └── support_ai.db
│
└── tests/
    ├── test_anonymize.py
    ├── test_retrieval.py
    ├── test_rag_prompt.py
    ├── test_feedback_pipeline.py
    └── e2e/
        └── test_agent_ui.py       ← playwright end-to-end
```

---

## 5. Phase 0 — Environment Setup (Day 0–3)

**Goal:** Get the LLM running on GPU and confirm VRAM headroom before writing any application code.

**Steps:**

1. Install Ollama from ollama.com (one-line installer for Linux/WSL/Windows)
2. Run `ollama info` — confirm it detects your RTX 50 series GPU
3. Pull phi4-mini with Q4_K_M quantization (~2.2 GB download)
4. Send a test prompt — confirm response arrives in under 3 seconds
5. Run `nvidia-smi` with the model loaded — confirm ~2.5 GB VRAM used, ~2.5 GB free
6. Set up a Python 3.11+ virtual environment
7. Install all dependencies from requirements.txt
8. Download spaCy transformer NER model (`en_core_web_trf`)
9. Scaffold the React + Vite frontend, install recharts and axios
10. Confirm FastAPI starts on port 8000 and React on port 5173

**Done when:** `ollama run phi4-mini "test"` returns a response under 3 seconds, nvidia-smi shows GPU usage between 2–3 GB, and both servers start without errors.

---

## 6. Phase 1 — Data Pipeline & PII (Day 4–10)

**Goal:** Get clean, anonymized ticket data into SQLite and FAISS with a full audit trail.

### 6.1 Export your tickets

Minimum required columns in the CSV export:

```
ticket_id | created_at | customer_text | agent_response | category | resolution_time_min
```

Place raw files in `data/raw/` — this folder must be in `.gitignore` and never logged.

### 6.2 PII anonymization pipeline

```
PSEUDOCODE: pipeline/anonymize.py

entities to detect and replace:
  PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME,
  CREDIT_CARD, IP_ADDRESS, URL,
  IN_PAN, IN_AADHAAR, IN_VEHICLE       ← India-specific
  ORDER_ID (custom regex: ORD-\d{6})   ← add your own domain patterns

for each ticket row:
    scrub customer_text  → replace all entity spans with <ENTITY_TYPE> tags
    scrub agent_response → same
    write anonymized row to data/clean/tickets_clean.csv

write audit_log.json:
    { ticket_ids: [...all IDs processed...], processed_at: ISO timestamp }

MANUAL SPOT-CHECK (required after every run):
    open output CSV, read 10 random rows
    if you can re-identify any customer in under 5 minutes → pipeline needs tuning
    add missing entity types as custom PatternRecognizer rules
```

### 6.3 Database schema

```
TABLE tickets
  id TEXT PK, customer_text TEXT (anonymized), agent_response TEXT (anonymized),
  category TEXT, resolution_time_min INT, created_at TEXT, source TEXT

TABLE feedback
  id INT PK, ticket_id TEXT, model_suggestion TEXT,
  agent_action TEXT (accept | modify | reject),
  final_response TEXT, agent_rationale TEXT,
  reason_type TEXT, severity TEXT, suggested_change TEXT, root_cause TEXT,
  confidence INT (1–5), dwell_seconds INT, logged_at TEXT

TABLE signal_tags
  id INT PK, feedback_id INT FK, category TEXT, severity TEXT,
  fixable BOOL, fix_type TEXT, summary TEXT, extracted_at TEXT

TABLE escalations
  id INT PK, feedback_id INT FK, reviewer_id TEXT,
  reviewer_decision TEXT, reviewer_notes TEXT, resolved_at TEXT

TABLE retrieval_log
  id INT PK, ticket_id TEXT, retrieved_ids TEXT (JSON),
  similarity_scores TEXT (JSON), logged_at TEXT

TABLE versions
  id INT PK, snapshot_name TEXT, index_path TEXT,
  db_path TEXT, prompt_hash TEXT, created_at TEXT, notes TEXT
```

---

## 7. Phase 2 — RAG Core & Hallucination Guardrails (Day 11–25)

**Goal:** Build the search and generation pipeline with built-in safety against hallucination.

### 7.1 FAISS index build

```
PSEUDOCODE: pipeline/embed.py

load anonymized tickets CSV
for each ticket:
    text = customer_text + " " + agent_response
    if tagged as gold_example: prepend "VERIFIED: " to boost retrieval weight

embed all texts using all-MiniLM-L6-v2 (CPU) → 384-dim float32 vectors
normalize each vector (enables cosine similarity via inner product)
add to FAISS IndexFlatIP
save index.faiss and metadata.json to data/faiss_index/

log: how many total vectors, how many gold examples, rebuild timestamp
```

### 7.2 RAG generation with confidence scoring

```
PSEUDOCODE: pipeline/rag.py

function generate(customer_query):

    # Step 1: Retrieve
    embed query → search FAISS → get top-5 results with similarity scores

    # Step 2: Confidence check
    if top-1 similarity score < 0.65:
        return { suggestion: None, low_confidence: true,
                 message: "No strong match found — agent should write from scratch" }

    # Step 3: Build prompt
    evidence_block = top-3 results formatted as:
        [Doc 1 | similarity: 0.87 | category: billing]
        Past customer: <anonymized question>
        Agent reply: <anonymized response>
        [Doc 2 | ...]

    system_prompt rules (must be explicit):
        - use ONLY information from the evidence block
        - do NOT invent policy names, dates, order numbers, or amounts
        - every reply must include one "Evidence:" line citing which doc supports the answer
        - if evidence does not cover the question, say so explicitly

    # Step 4: Generate
    send to Ollama: phi4-mini, temperature=0.3, num_ctx=4096

    # Step 5: Safety filter
    if response does not contain "Evidence:" line → flag as low_quality
    run output through PII filter
    scan output for any number that looks like an invented ID → flag if found

    return { suggestion, retrieved_docs, confidence_score, low_confidence flag }
```

### 7.3 Hallucination guardrails summary

| Guardrail | How It Works |
|---|---|
| Low similarity threshold | If best match < 0.65 cosine sim, don't generate — tell agent to write manually |
| Evidence citation required | System prompt demands an "Evidence:" line in every reply; missing = flagged |
| Temperature 0.3 | Low temperature reduces creative invention |
| Output ID scan | Regex scan of output for invented order numbers, dates in wrong format |
| Output PII re-scan | Re-run presidio on output before showing to agent |
| Agent dwell time | Log how long agent reads suggestion — < 5 seconds = likely not read |

---

## 8. Phase 3 — Agent UI & Onboarding (Day 26–45)

**Goal:** Give agents a simple, well-explained interface with a structured rationale schema that makes feedback machine-readable from day one.

### 8.1 Structured rationale schema

The rationale form has four fixed fields instead of a free-text box. This is what makes feedback processable at scale.

```
reason_type    (dropdown, required)
  options: wrong_policy | missing_context | wrong_tone |
           hallucination | too_long | too_short |
           correct_but_edited | other

severity       (dropdown, required)
  options: low | medium | high

suggested_change  (short text, required if action = modify or reject)
  example: "Should mention 7-day return window, not 14-day"

root_cause     (dropdown, optional but encouraged)
  options: retrieval_miss | prompt_gap | model_limit | policy_outdated | none
```

All four fields write directly to the `feedback` table. No free-text rationale to parse later.

### 8.2 Agent UI layout

```
LAYOUT: TicketView.jsx

[ Ticket ID ]  [ Category ]
[ Customer query — read only, pre-filled ]

──── AI Suggestion ────────────────────────────────

[ Editable textarea — light blue background ]
  "Based on our records: [Evidence: Doc 1 similarity 0.87]..."

[ Evidence used ]
  Doc 1 (0.87, billing): "Customer asked about refund..." → "Agent replied..."
  Doc 2 (0.82, billing): ...
  Doc 3 (0.76, returns): ...

⚠️  If low_confidence flag is true:
  banner: "No strong match found. Model confidence is low.
           Please write your own response below."

──── Your Decision ────────────────────────────────

[ Accept as-is ]  [ Accept with edits ]  [ Reject — escalate ]

  Accept as-is: 5-second countdown before button activates
                (prevents mindless clicking)

  Accept with edits: textarea becomes editable, rationale form appears

  Reject — escalate: rationale form appears + routes to senior reviewer queue

──── Rationale (required for modify / reject) ─────

  Reason type:        [ dropdown ]
  Severity:           [ dropdown ]
  Suggested change:   [ text input ]
  Root cause:         [ dropdown ]

  Confidence in your reply:  [ slider 1–5 ]

[ Submit ]   ← disabled until all required fields filled

──── In-UI Tips (always visible, collapsible) ──────
  💡 "A good 'suggested change' describes the correct policy, not just the problem."
  💡 "High severity = customer would receive wrong information if this went out."
  💡 "Retrieval miss = the right answer exists in our docs but didn't appear."
```

### 8.3 Agent onboarding script

Before launch, run a 30-minute session covering:

1. What the system does and does not do (it drafts, it does not decide)
2. Walk through each rationale field with a real example for each reason_type
3. Explain why the rationale matters — "your edit today becomes the AI's example tomorrow"
4. Show what a low-confidence flag looks like and what to do when it appears
5. Explain the 5-second delay on Accept (and why it exists)
6. Q&A — field common objections

Leave a one-page cheat sheet pinned in the UI at all times.

### 8.4 Escalation flow

```
PSEUDOCODE: when agent clicks "Reject — escalate"

1. Save feedback row with agent_action = "reject"
2. Create escalation row:
   { feedback_id, status: "pending", created_at }
3. Senior reviewer sees a queue in EscalationView.jsx
4. Reviewer reads original ticket + model suggestion + agent rationale
5. Reviewer makes decision:
   - Confirm reject: write correct response, log as gold_example
   - Override: accept model suggestion with notes
6. Update escalation row with reviewer_decision + reviewer_notes + resolved_at
7. If confirmed reject: corrected response enters feedback pipeline as gold_example
```

---

## 9. Phase 4 — Feedback Pipeline (Day 46–80)

**Goal:** Systematically convert agent edits and structured rationale into retrieval improvements. This is the engine of the self-reinforcing loop.

### 9.1 Batch job — what it does weekly

```
PSEUDOCODE: pipeline/feedback_pipeline.py

run every Monday at 02:00 via cron

Step 1 — Collect new feedback
  query feedback table: all rows since last_run where signal not yet processed

Step 2 — Deduplication
  group by (category, customer_text similarity > 0.90)
  if 3+ near-identical edits exist for same issue:
      merge into one representative gold_example
      flag as "high_signal" — will get priority in FAISS

Step 3 — Quality filter
  drop rows where:
      dwell_seconds < 5     (agent didn't read it)
      suggested_change is empty on a modify row
      confidence < 2        (agent unsure of their own correction)

Step 4 — Create training examples
  for each accepted-with-edits row:
      new_doc = { customer_text, final_response (agent's version) }
      tag as gold_example, add to embed queue

  for each confirmed-reject escalation:
      new_doc = { customer_text, reviewer_correct_response }
      tag as gold_example + high_signal, add to embed queue

Step 5 — Re-embed and update FAISS
  run embed.py on embed queue
  add new vectors to existing index (or full rebuild if >20% new docs)
  log: how many new gold_examples added, new total index size

Step 6 — Prompt update check
  count signal_tags by category for the past 2 weeks
  if any category count > 5:
      generate a suggested system_prompt addition (see section 13)
      write to logs/prompt_suggestions.txt for human review

Step 7 — Mark rows as processed
  update processed_at timestamp on all handled feedback rows
```

### 9.2 Weekly improvement cycle (full sequence)

```
Every Monday 02:00:
  01. scripts/snapshot.sh           → backup index + DB before touching anything
  02. pipeline/feedback_pipeline.py → dedup, filter, re-embed gold examples
  03. pipeline/signal_extractor.py  → tag unprocessed rationale rows
  04. evaluation/eval.py            → print weekly delta report
  05. review logs/prompt_suggestions.txt manually → apply if agreed
  06. evaluation/audit_sampler.py   → pick 10 random accepted replies for human review
```

### 9.3 Weekly acceptance targets

| Week | Target Accept Rate | Action if Below |
|---|---|---|
| 2 | > 20% | Normal. Model is cold. Keep collecting. |
| 4 | > 35% | Review top reason_type counts. Fix prompt. |
| 8 | > 55% | Check gold_example count in index. Re-embed. |
| 12 | > 65% | Consider category-specific prompt variants. |
| 16 | > 70% | System is self-reinforcing. Plan expansion. |

---

## 10. Phase 5 — Signal Extractor (Day 81–100)

**Goal:** Because agents now fill structured rationale fields, the signal extractor's job is simpler — validate, normalize, and store the tags rather than parsing free text.

### 10.1 What the extractor does

```
PSEUDOCODE: pipeline/signal_extractor.py

run nightly at 01:00 via cron (off-peak — model is free)

for each feedback row WHERE extracted_at IS NULL:

    input is already structured (reason_type, severity, suggested_change, root_cause)

    Step 1 — Validate
        confirm reason_type is in allowed values
        confirm severity is in allowed values
        if either is missing or invalid → log warning, skip row

    Step 2 — Enrich with phi4-mini (optional, for summary field only)
        prompt: "In one sentence, describe the core issue:
                 Reason: {reason_type}. Suggested fix: {suggested_change}."
        store result as signal_tags.summary

    Step 3 — Determine fix_type
        wrong_policy, hallucination → fix_type = "prompt_update"
        missing_context, retrieval_miss → fix_type = "index_update"
        wrong_tone, too_long → fix_type = "prompt_update"
        model_limit → fix_type = "none" (document and accept)
        correct_but_edited → fix_type = "none"

    Step 4 — Insert into signal_tags
        { feedback_id, category: reason_type, severity, fixable, fix_type, summary, extracted_at }

    Step 5 — Mark feedback row as extracted
        update extracted_at = now()

error handling:
    wrap each row in try/except — one bad row must not stop the batch
    cap at 200 rows per nightly run to avoid timeout
    log all skipped rows with reason
```

### 10.2 Signal categories and their fix types

| Category | What It Means | Fix Type | Priority |
|---|---|---|---|
| wrong_policy | Model cited wrong or outdated policy | Prompt update + index update | High |
| missing_context | Answer was too generic | Index update (add examples) | High |
| hallucination | Model invented a fact not in evidence | Prompt update (strengthen guardrails) | Critical |
| retrieval_miss | Retrieved docs were irrelevant | Rebuild index, check TOP_K | High |
| wrong_tone | Too formal / too casual / too long | Prompt update | Medium |
| too_long / too_short | Length issue only | Prompt update | Low |
| correct_but_edited | Agent changed style, not substance | Monitor only | Low |
| model_limit | Issue is beyond 3B model capability | Document, accept | Info |

---

## 11. Phase 6 — Metrics Dashboard & Human Audits (Day 101–120)

**Goal:** Prove the system is improving. Show the trend, catch silent failures, and build team trust.

### 11.1 Backend — metrics endpoint

```
PSEUDOCODE: GET /metrics

query SQLite, return JSON:

acceptance_by_week:
    [ { week, accept_pct, modify_pct, reject_pct, total } ... ]
    → feeds 16-week line chart

rejection_breakdown:
    [ { category, count, pct, trend_vs_last_week } ... ]
    → feeds bar chart

retrieval_quality_trend:
    [ { week, avg_top1_similarity, avg_top3_similarity, gold_example_count } ... ]
    → feeds line chart — should trend upward

agent_confidence_trend:
    [ { week, avg_confidence } ... ]

system_health:
    { total_interactions, gold_examples_in_index, last_rebuild,
      unresolved_escalations, low_confidence_flagged_this_week }

audit_queue:
    10 randomly sampled accepted replies from this week (for human review panel)
```

### 11.2 Dashboard layout

```
LAYOUT: Dashboard.jsx

┌──────────────────────────────────────────────────────────┐
│  Customer Support AI — System Health                      │
│  Last rebuilt: Mon 02:05 AM  |  Next rebuild: Mon 02:00  │
├──────────────┬────────────────┬──────────────────────────┤
│  Accept Rate │  Avg Confidence│  Total Interactions      │
│  68%  ↑ +6%  │  3.8 / 5  ↑  │  412  (+38 this week)    │
├──────────────┴────────────────┴──────────────────────────┤
│  ACCEPTANCE TREND — 16 weeks (line chart)                 │
│  accept%  ───────────────────────────────── ↗            │
│  modify%  ─────────────────────────── →                  │
│  reject%  ─────────────────────── ↘                      │
├─────────────────────────┬────────────────────────────────┤
│  REJECTION BREAKDOWN    │  SIGNAL CATEGORY TREND         │
│  (horizontal bar chart) │  (stacked area — 8 weeks)      │
│  wrong_policy     ████  │  shows how each failure type   │
│  missing_context  ███   │  rises or falls week over week │
│  hallucination    ██    │                                │
│  wrong_tone       █     │                                │
├─────────────────────────┴────────────────────────────────┤
│  RETRIEVAL QUALITY TREND (line chart)                     │
│  avg_top1_similarity over time — should trend upward      │
│  gold_example_count in index — should grow every week     │
├──────────────────────────────────────────────────────────┤
│  ⚠️  NEEDS ATTENTION                                      │
│  Unresolved escalations: 3   Low-confidence flags: 7      │
│  [ View Escalation Queue ]   [ View Flagged Replies ]     │
├──────────────────────────────────────────────────────────┤
│  HUMAN AUDIT QUEUE — 10 random accepted replies this week │
│  [ ticket_id | category | suggestion preview | ✅ / ❌ ]  │
│  Reviewer marks each: "Good" or "Silent failure"          │
│  Silent failures feed back into the rejection pipeline    │
└──────────────────────────────────────────────────────────┘
```

### 11.3 Human review sampling (random audits)

Silent failures are replies the agent accepted but that were actually wrong. They are the most dangerous failure mode because they never appear in rejection stats.

```
PSEUDOCODE: evaluation/audit_sampler.py

weekly:
    pick 10 random rows WHERE agent_action = "accept"
    present to a senior reviewer in the dashboard audit panel
    reviewer marks each: "Good" or "Silent failure"

    if "Silent failure":
        create a feedback row with agent_action = "reject"
        route through normal feedback pipeline
        this becomes a gold_example with the correct response

track: silent_failure_rate (silent failures / audited samples)
target: < 5% by week 12
```

---

## 12. Versioning & Rollback

**Rule:** Always snapshot before any update. Never modify the live index or DB without a named backup you can restore in under 5 minutes.

### 12.1 What to snapshot

```
PSEUDOCODE: scripts/snapshot.sh

inputs: snapshot_name (e.g. "2026-06-23-pre-rebuild")

create directory: backups/{snapshot_name}/
copy: data/faiss_index/index.faiss      → backups/{name}/index.faiss
copy: data/faiss_index/metadata.json   → backups/{name}/metadata.json
copy: data/support_ai.db               → backups/{name}/support_ai.db
write: backups/{name}/manifest.json    → { created_at, prompt_hash, index_size,
                                           total_tickets, gold_example_count, notes }

insert into versions table in DB:
    { snapshot_name, paths, prompt_hash, created_at }

print: "Snapshot {name} complete. Restore with: ./scripts/restore.sh {name}"
```

### 12.2 Rollback procedure

```
PSEUDOCODE: scripts/restore.sh

inputs: snapshot_name

stop FastAPI server
stop Ollama (to free VRAM before restore)

copy: backups/{name}/index.faiss     → data/faiss_index/index.faiss
copy: backups/{name}/metadata.json  → data/faiss_index/metadata.json
copy: backups/{name}/support_ai.db  → data/support_ai.db

restart Ollama
restart FastAPI

verify: GET /metrics returns data, GET /suggest returns a response

log: "Rolled back to {name} at {timestamp}"
```

### 12.3 When to snapshot

- Before every weekly feedback pipeline run
- Before any system prompt change
- Before re-embedding a large batch (> 100 new docs)
- Before any schema migration to the DB

---

## 13. Retrain Schedule & Orchestration

**There is no model fine-tuning at 6GB VRAM.** "Retrain" here means index rebuild + prompt update.

### 13.1 Weekly cron job

```
PSEUDOCODE: scripts/weekly_pipeline.sh

schedule: every Monday at 02:00 (agents not working)
log all output to: logs/weekly_{date}.log

02:00  snapshot.sh              → backup before touching anything
02:05  feedback_pipeline.py     → dedup, filter, re-embed gold examples
02:25  signal_extractor.py      → tag unprocessed rationale rows (max 200)
02:45  embed.py                 → full index rebuild with all tickets + gold examples
03:10  eval.py                  → weekly delta report to logs/
03:15  audit_sampler.py         → pick 10 random accepted replies for audit queue

resource limits:
    embed.py: single-threaded, batch_size=32 (keeps RAM under 4GB)
    signal_extractor.py: max 200 rows per run (prevents all-night batch)
    all jobs: if any step fails, halt and send alert to logs/errors.log
              do NOT proceed to the next step on failure
```

### 13.2 Prompt update process (manual gate)

Prompt changes are not automated. The pipeline generates suggestions; a human applies them.

```
PSEUDOCODE: prompt update workflow

weekly_pipeline.sh generates: logs/prompt_suggestions_{date}.txt
  format:
    Category: wrong_policy (count: 12 in last 2 weeks)
    Suggestion: Add to system prompt — "Always cite the 7-day return window for electronics"

developer reviews the file:
    if suggestion looks correct → manually edit pipeline/rag.py system_prompt
    snapshot before applying
    run one test query to verify the change behaves as expected
    commit the change with a git tag: "prompt-v1.3-wrong-policy-fix"
```

### 13.3 Index rebuild triggers

| Trigger | Action |
|---|---|
| Weekly cron | Always rebuild — standard cycle |
| Gold examples > 20 new this week | Rebuild mid-week if quality drop detected |
| Acceptance rate drops > 10% vs prior week | Emergency rebuild — check what changed |
| Schema migration | Rebuild from scratch with new schema |

---

## 14. Data Governance & Consent Policy

Create `CONSENT_AND_RETENTION.md` in the repo root. It must answer these questions:

### What data is collected

- Anonymized customer ticket text (no raw PII ever stored)
- Agent decisions and structured rationale
- Model suggestions (anonymized output)
- Timing data (dwell time, response latency)

### How long data is kept

| Data Type | Retention Period | Deletion Method |
|---|---|---|
| Anonymized tickets | 12 months rolling | Delete from SQLite + rebuild FAISS |
| Feedback / rationale | 24 months | Delete from feedback table |
| Signal tags | 24 months | Delete from signal_tags |
| Audit log (ticket IDs) | 36 months | Required for erasure requests |
| Raw exports | Delete immediately after anonymize.py runs | Overwrite file, empty data/raw/ |

### Agent opt-in / opt-out

- Agents must be briefed before the system goes live (see onboarding, section 8.3)
- Any agent can request their feedback entries be excluded from gold_examples
- Provide a simple script: `scripts/exclude_agent.py {agent_id}` that marks their rows as excluded

### Customer erasure requests

- Maintain audit_log.json mapping ticket_id → processing timestamp
- Provide: `scripts/erase_ticket.py {ticket_id}` that:
  - deletes the row from SQLite
  - removes the corresponding vector from FAISS by rebuilding the index excluding that ID
  - appends to erasure_log.json: { ticket_id, erased_at }

---

## 15. Access Controls & Encryption

### Database encryption

Use SQLCipher (drop-in SQLite replacement with AES-256 encryption at rest).

```
PSEUDOCODE: db.py

on startup:
    connect to support_ai.db using sqlcipher
    key = read from environment variable DB_ENCRYPTION_KEY
          (set in .env file, never hardcoded)
    if key missing → refuse to start, log error

.env file must contain:
    DB_ENCRYPTION_KEY=<random 32-char string, generated once at setup>
    ADMIN_PASSWORD=<password for /admin/* endpoints>
```

### UI authentication

The UI is local-only (localhost:5173) but still needs a password for the admin routes.

```
PSEUDOCODE: api/main.py auth middleware

all routes require: Authorization header with local API key
key is set in .env as LOCAL_API_KEY
frontend reads key from localStorage (set at first login)
admin routes (/admin/*) require a second ADMIN_PASSWORD check

never expose the API on 0.0.0.0 — bind to 127.0.0.1 only
if running on a shared machine: add OS-level user restriction (chmod 700 on data/)
```

### Access log

```
every API request logs:
    timestamp, route, agent_id (from auth header), response_time_ms
    stored in access_log table in SQLite
    retained 90 days, then deleted

never log: request body, customer text, model suggestion content
```

---

## 16. Quantization & Memory Plan

### Exact settings for Phi-4-mini on RTX 50 series 6GB

```
Model:         phi4-mini
Quantization:  Q4_K_M   (GGUF format via Ollama)
  - "K" = smarter per-layer quantization
  - "M" = medium quality/size tradeoff
  - quality loss vs FP16: ~1–2% on benchmarks
  - reason to prefer over Q5: Q5 needs ~3.1GB, cuts headroom below 2GB

Ollama options (set in rag.py):
  num_ctx:     4096    (context window — 4K is sufficient for support tickets)
  temperature: 0.3     (low = consistent, less hallucination)
  num_predict: 512     (max output tokens — support replies don't need more)
  num_gpu:     99      (offload all layers to GPU — critical for speed)
```

### VRAM budget breakdown

```
GPU VRAM total:                     6,144 MB
────────────────────────────────────────────
OS + drivers + CUDA runtime:         ~400 MB
phi4-mini Q4_K_M weights:          ~2,500 MB
KV cache (ctx=4096, 512 output):     ~600 MB
Embedding model (all-MiniLM-L6-v2):    0 MB  ← runs on CPU
FAISS index:                            0 MB  ← runs on CPU RAM
────────────────────────────────────────────
Safe headroom:                      ~2,044 MB ✅

If you switch to Gemma 4 E4B Q4_K_M:
  weights ~2,800 MB → headroom ~1,744 MB     ✅ still fine

If you try Mistral 7B (any quantization):
  weights ~5,200 MB → headroom ~ -600 MB     ❌ OOM crash — do not attempt
```

### Memory management rules

- Embedding model and FAISS must always stay on CPU — never load them to GPU
- Do not run embed.py while Ollama is serving live requests — it spikes RAM
- Nightly batch jobs (signal_extractor, feedback_pipeline) run after Ollama is idle
- If you see `nvidia-smi` showing > 5,500 MB used, restart Ollama immediately

---

## 17. Testing Strategy

### Unit tests (run before every weekly pipeline)

```
tests/test_anonymize.py
  ✓ known PII strings are replaced correctly
  ✓ anonymized output contains no original entity spans
  ✓ custom ORDER_ID pattern is detected
  ✓ clean text passes through unchanged

tests/test_retrieval.py
  ✓ FAISS returns top-5 results for a known query
  ✓ similarity scores are between 0 and 1
  ✓ low-confidence flag triggers when top-1 < 0.65
  ✓ gold_examples rank higher than regular examples for same query

tests/test_rag_prompt.py
  ✓ evidence block is included in every prompt
  ✓ prompt does not include raw PII
  ✓ output contains "Evidence:" line
  ✓ low-confidence path returns None suggestion (not a hallucinated reply)

tests/test_feedback_pipeline.py
  ✓ near-duplicate edits are merged correctly (similarity > 0.90)
  ✓ low-dwell rows (< 5s) are filtered out
  ✓ low-confidence rows (< 2) are filtered out
  ✓ gold_examples are added to embed queue
  ✓ processed_at is set after successful run
```

### End-to-end tests (run weekly)

```
tests/e2e/test_agent_ui.py  (Playwright)
  ✓ submit a ticket → receive a suggestion within 8 seconds
  ✓ accept-with-edits flow saves feedback correctly
  ✓ reject flow creates an escalation row
  ✓ Submit button stays disabled until all required rationale fields filled
  ✓ Accept button has 5-second delay before activating
```

### Regression test (holdout set)

```
PSEUDOCODE: evaluation/eval.py (weekly)

load data/holdout/holdout.csv  ← 50–100 tickets NEVER added to FAISS
for each holdout ticket:
    run generate(customer_text)
    have a reviewer rate suggestion: 0 (wrong), 1 (acceptable), 2 (good)

compute:
    precision@1: % of suggestions rated acceptable or good
    delta vs last week: how much did it change?

log: holdout_results_{date}.json
alert if precision@1 drops > 5 points week over week
```

### A/B comparison (before/after index rebuild)

```
PSEUDOCODE: evaluation/ab_compare.py

inputs: snapshot_name_A (old), snapshot_name_B (new)

for each holdout ticket:
    query index A → get suggestion
    query index B → get suggestion
    present both (anonymized, unlabeled) to a reviewer
    reviewer picks: A better | B better | same

output: win rate for B vs A
accept new index if B win rate > 55%
roll back if B win rate < 45%
```

---

## 18. Backup & Restore

### What to back up

| Asset | Location | Backup Frequency |
|---|---|---|
| FAISS index | data/faiss_index/ | Before every weekly rebuild |
| SQLite database | data/support_ai.db | Before every weekly rebuild |
| System prompt | pipeline/rag.py | git commit on every change |
| .env file | project root | Manually, encrypted, off-machine |
| audit_log.json | data/clean/ | Weekly copy to backups/ |

### Backup structure

```
backups/
├── 2026-06-16-pre-rebuild/
│   ├── index.faiss
│   ├── metadata.json
│   ├── support_ai.db
│   └── manifest.json       ← { created_at, index_size, gold_count, notes }
├── 2026-06-23-pre-rebuild/
│   └── ...
└── latest -> 2026-06-23-pre-rebuild/  ← symlink to most recent
```

### Restore in under 5 minutes

```
PSEUDOCODE: scripts/restore.sh {snapshot_name}

1. stop FastAPI (kill uvicorn process)
2. stop Ollama (free VRAM)
3. copy snapshot files to live locations
4. restart Ollama, confirm GPU load
5. restart FastAPI, confirm /suggest returns a response
6. log restore event to access_log

total time: < 5 minutes
```

### Off-machine backup (weekly)

Copy the `backups/` folder to an external drive or encrypted cloud folder weekly. The DB contains anonymized data only, but still treat it as sensitive.

---

## 19. Evaluation Targets

Track these in the dashboard and a weekly spreadsheet:

| Metric | How to Measure | Target |
|---|---|---|
| Suggestion acceptance rate | accepts ÷ total | > 65% by week 12 |
| Edit distance ratio | avg(char diff / suggestion length) | < 0.20 |
| Rejection rate | rejects ÷ total | < 20% by week 8 |
| Silent failure rate | audit failures ÷ audited samples | < 5% by week 12 |
| Avg agent confidence | mean of 1–5 scores | > 3.5 |
| Retrieval avg similarity | mean top-1 cosine score | > 0.80 |
| Signal extraction coverage | tagged rows ÷ total feedback rows | > 90% weekly |
| Gold examples in index | count of verified docs | grows every week |
| Holdout precision@1 | acceptable + good ratings ÷ total holdout | > 70% by week 16 |
| Response latency | POST /suggest → response | < 8 seconds |
| Escalation resolution time | escalation created → resolved | < 48 hours |

---

## 20. Failure Mode Playbook

| Failure | Signal | Response |
|---|---|---|
| Hallucination | Agent catches invented facts or policies | Lower temperature to 0.1. Add explicit "do not invent" rules to system prompt. Check if Evidence: line is present. |
| Retrieval miss | Low similarity scores, irrelevant docs returned | Increase TOP_K from 5 to 8. Confirm embed.py indexed the right text columns. Rebuild index. |
| PII in output | Agent sees name or email in model response | Confirm output_filter.py is called after generation. Add missed entity type to PII_ENTITIES. |
| Slow response > 15s | Agent waits too long | Model spilling to RAM. Restart Ollama. Reduce num_ctx to 2048. Check nvidia-smi. |
| Agent gaming | Accept rate spikes, confidence drops | Check dwell_time logs. Enforce 5-second delay. Raise in team meeting. |
| Index stale | Acceptance rate drops unexpectedly | Check last_rebuild date in dashboard. Run embed.py manually. |
| Signal extractor gaps | signal_tags missing for recent feedback | Check extractor logs. Confirm reason_type values are in allowed list. Re-run manually. |
| Silent failures detected | Audit queue shows > 5% bad accepted replies | Route all audited failures through feedback pipeline. Investigate which category they fall into. |
| Escalation queue growing | Unresolved escalations > 5 | Assign senior reviewer time. Check if rejection rate has spiked — may indicate prompt issue. |
| DB encryption failure | FastAPI won't start | Confirm DB_ENCRYPTION_KEY is set in .env. Never hardcode it. Check .env is not gitignored-but-deleted. |
| Snapshot missing | Need to roll back but no backup exists | Restore from last successful off-machine backup. Add snapshot step to pipeline and enforce it. |
| Ollama crash | API returns 503 | Restart Ollama. Add health-check to FastAPI startup that pings Ollama and refuses to start if it's down. |

---

## 21. Complete Data Schema

Three classes of data make the loop work: **historical tickets** (memory), **agent interactions** (feedback), and **metadata + labels** (signals). All three must be collected in a consistent, structured way from day one. Below is the full column spec — save as CSV for import into SQLite.

Every column has a clear owner: the support tool exports ticket columns automatically, the agent UI writes interaction columns in real time, and the signal extractor fills evaluation columns in the weekly batch.

---

### Class A — Ticket (Memory)

These are your training corpus. They feed FAISS and are used as evidence in RAG retrieval.

| Column | Type | Description | Example |
|---|---|---|---|
| `ticket_id` | text | Unique ticket identifier | `TKT-20260624-001` |
| `created_at` | ISO datetime | When ticket was opened | `2026-06-20T09:12:00` |
| `closed_at` | ISO datetime | When ticket was resolved or closed | `2026-06-20T10:05:00` |
| `channel` | text | Origin channel | `chat` / `email` / `phone` |
| `product` | text | Product or service area | `ProPlan` |
| `issue_type` | text | Top-level category | `billing` / `returns` / `technical` |
| `issue_subtype` | text | Finer-grained category | `duplicate_charge` / `refund_request` |
| `priority` | text | Ticket urgency | `low` / `medium` / `high` / `critical` |
| `sla_breach` | int | 1 if SLA was missed, 0 if met | `0` |
| `customer_text` | text | Full customer message thread (anonymized) | `"I was charged twice for order ORD-001"` |
| `agent_response` | text | Final reply sent to customer (anonymized) | `"Refund issued for duplicate charge..."` |
| `resolution_status` | text | Final outcome | `resolved` / `escalated` / `pending` |
| `resolution_time_seconds` | int | Total time from open to close | `3600` |
| `first_contact_resolved` | int | 1 if resolved without escalation | `1` |
| `escalated_to` | text | Role or team if escalated, else null | `senior_support` / `null` |
| `pii_flag` | int | 1 if PII was detected and scrubbed in this row | `1` |
| `pii_entities_found` | text | Semicolon list of entity types scrubbed | `PERSON;PHONE_NUMBER` |
| `anonymized_at` | ISO datetime | When anonymizer ran on this row | `2026-06-20T08:00:00` |
| `source` | text | Where this ticket came from | `zendesk_export` / `csv_import` / `feedback_pipeline` |
| `is_gold_example` | int | 1 if agent-verified or reviewer-confirmed | `0` |
| `gold_signal` | text | Why it was promoted to gold | `agent_corrected` / `reviewer_confirmed` / `null` |
| `kb_refs` | text | Semicolon list of KB / FAQ article IDs used | `KB-45;KB-12` |
| `doc_refs` | text | Semicolon list of policy doc names cited | `return_policy_v3;refund_sop` |
| `language` | text | Language of the ticket | `en` / `hi` / `null` |
| `data_consent` | int | 1 if consent to use in training is confirmed | `1` |
| `retention_expires_at` | ISO date | When this row must be deleted | `2027-06-20` |

---

### Class B — Agent Interaction (Feedback)

These are written by the agent UI in real time when a suggestion is reviewed. They are the training signal.

| Column | Type | Description | Example |
|---|---|---|---|
| `interaction_id` | int | Auto-increment primary key | `1042` |
| `ticket_id` | text | FK → tickets.ticket_id | `TKT-20260624-001` |
| `agent_id` | text | Anonymized agent identifier | `AGT-07` |
| `suggested_at` | ISO datetime | When model suggestion was generated | `2026-06-20T09:13:45` |
| `model_version` | text | Snapshot name or prompt hash at suggestion time | `snapshot-2026-06-16` |
| `model_suggestion` | text | Draft produced by the model (anonymized) | `"We can refund the duplicate charge..."` |
| `retrieval_top1_id` | text | ticket_id of the top retrieved evidence doc | `TKT-20260101-055` |
| `retrieval_top1_score` | float | Cosine similarity of top-1 result | `0.87` |
| `retrieval_top3_avg_score` | float | Average similarity of top-3 results | `0.81` |
| `low_confidence_flag` | int | 1 if top-1 score was below threshold (0.65) | `0` |
| `agent_action` | text | What the agent did | `accept` / `modify` / `reject` |
| `agent_final` | text | The actual reply sent (agent's version, anonymized) | `"Refund processed; expect 3–5 days."` |
| `edit_distance` | int | Character-level edit distance between suggestion and final | `42` |
| `edit_distance_ratio` | float | edit_distance ÷ len(model_suggestion) | `0.08` |
| `dwell_seconds` | int | How long agent viewed the suggestion before deciding | `18` |
| `agent_confidence` | int | Agent's self-rated confidence in their reply (1–5) | `4` |
| `rationale_type` | text | Semicolon list of categorical reason codes | `tone;policy` |
| `rationale_severity` | text | How bad was the model's error | `low` / `medium` / `high` |
| `rationale_suggested_change` | text | What the agent says should change | `"Include 7-day return window per electronics policy"` |
| `rationale_root_cause` | text | Why the model failed | `retrieval_miss` / `prompt_gap` / `model_limit` / `policy_outdated` / `none` |
| `submitted_at` | ISO datetime | When agent clicked Submit | `2026-06-20T09:15:22` |
| `excluded_from_training` | int | 1 if agent opted out of this row being used as a training example | `0` |

---

### Class C — Evaluation Labels (Signals)

These are written by the signal extractor (weekly batch) and human auditors. They make feedback machine-readable and queryable.

| Column | Type | Description | Example |
|---|---|---|---|
| `signal_id` | int | Auto-increment primary key | `308` |
| `interaction_id` | int | FK → interactions.interaction_id | `1042` |
| `signal_category` | text | Normalized failure category | `wrong_policy` / `hallucination` / `retrieval_miss` / `missing_context` / `wrong_tone` / `correct_but_edited` / `model_limit` / `other` |
| `signal_severity` | text | Severity of the issue | `low` / `medium` / `high` / `critical` |
| `fixable` | int | 1 if fix is within reach (prompt or index), 0 if model limit | `1` |
| `fix_type` | text | What kind of fix addresses this | `prompt_update` / `index_update` / `both` / `none` |
| `signal_summary` | text | One-sentence description of the issue (LLM-generated) | `"Model omitted the 7-day electronics return window"` |
| `extracted_by` | text | Who or what created this signal | `signal_extractor_v1` / `human_auditor` |
| `extracted_at` | ISO datetime | When signal was written | `2026-06-23T01:35:00` |
| `human_quality_rating` | int | Auditor rating of the final agent reply (0=wrong, 1=ok, 2=good) | `2` |
| `silent_failure` | int | 1 if auditor marked an accepted reply as actually wrong | `0` |
| `audited_by` | text | Auditor agent ID, null if not yet audited | `AGT-02` |
| `audited_at` | ISO datetime | When human audit was performed | `2026-06-23T10:00:00` |
| `used_in_index_rebuild` | int | 1 if this signal led to a FAISS index update | `1` |
| `used_in_prompt_update` | int | 1 if this signal triggered a prompt change | `0` |
| `prompt_version_after` | text | Prompt hash after any resulting change, else null | `prompt-v1.4` / `null` |

---

### Auxiliary Documents Table

KB articles, FAQs, SOPs, and policy docs that the RAG pipeline retrieves alongside tickets.

| Column | Type | Description | Example |
|---|---|---|---|
| `doc_id` | text | Unique document identifier | `KB-45` |
| `doc_type` | text | Type of document | `faq` / `policy` / `sop` / `template` |
| `title` | text | Document title | `"Electronics Return Policy v3"` |
| `content` | text | Full document text (anonymized if needed) | `"Items must be returned within 7 days..."` |
| `product` | text | Which product area this applies to | `ProPlan` / `all` |
| `valid_from` | ISO date | When this version became active | `2026-01-01` |
| `valid_to` | ISO date | When this version expires, null if current | `null` |
| `version` | text | Document version string | `v3.1` |
| `indexed_at` | ISO datetime | When this doc was last embedded into FAISS | `2026-06-16T02:15:00` |

---

### Data Class Summary

| Class | Who Writes It | When | Used For |
|---|---|---|---|
| A — Tickets | Support tool export + anonymizer | At ingest | FAISS retrieval memory |
| B — Interactions | Agent UI (real time) | Every suggestion reviewed | Feedback pipeline input |
| C — Signals | Signal extractor + human auditor | Weekly batch + audit queue | Dashboard, prompt updates, index improvement |
| Aux Docs | Manual upload + version control | When policy changes | RAG retrieval alongside tickets |

---

### Column Counts by Class

| Class | Columns | Required at Launch | Can Add Later |
|---|---|---|---|
| A — Tickets | 25 | 15 core columns | pii_entities_found, doc_refs, language, data_consent, retention_expires_at |
| B — Interactions | 21 | 12 core columns | retrieval scores, edit_distance_ratio, dwell_seconds, excluded_from_training |
| C — Signals | 16 | 8 core columns | human_quality_rating, silent_failure, used_in_* columns |
| Aux Docs | 9 | 5 core columns | valid_from, valid_to, version |

Start with the core columns on day 1. Add the rest as each phase goes live.

---



| Days | Phase | Key Output |
|---|---|---|
| 0–3 | Environment setup | Ollama on GPU, both servers running |
| 4–10 | Data pipeline + PII | Anonymized tickets in SQLite, FAISS built, audit log created |
| 11–25 | RAG core + guardrails | generate() working with confidence scoring and PII output filter |
| 26–45 | Agent UI + onboarding | Structured rationale form live, agents trained, escalation flow wired |
| 46–80 | Feedback pipeline | Weekly batch: dedup, filter, re-embed gold examples, prompt suggestions |
| 81–100 | Signal extractor | Structured tags in DB, fix_type assigned to every rejection |
| 101–120 | Dashboard + audits | Live charts, human audit queue, silent failure tracking |

---

*Last updated: June 2026 | Built for RTX 50 series 6GB VRAM laptops*
*All tools in this plan are free and open source. Zero cloud dependency.*
