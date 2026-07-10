"""
pipeline/feedback_pipeline.py

The batch job that actually closes the self-improving loop. Weekly (via
scripts/weekly_pipeline.sh), this:
  1. Reads agent feedback (interactions table) since the last run.
  2. Quality-filters + dedupes it.
  3. Appends the good corrections to data/clean/tickets.csv as new gold
     examples — so the NEXT run of pipeline/embed.py picks them up and
     they become retrievable context for future suggestions.

IMPORTANT DESIGN CHOICE: this does NOT call pipeline/embed.py directly or
touch the FAISS index itself. It only prepares the corpus file. Rebuilding
the index is a separate, explicit step (scripts/weekly_pipeline.sh runs
feedback_pipeline.py THEN embed.py). Keeping these decoupled means you can
inspect/audit what's about to be added to the corpus before it's embedded
and searchable — re-indexing bad data is much harder to undo than
appending a bad row to a CSV you can still edit.

QUALITY FILTER (what counts as "good enough to learn from"):
  - excluded_from_training must be False (agents/auditors can flag a row
    as unfit — e.g. a reject made in error — via this existing column).
  - agent_final must be non-empty (nothing to learn from an empty reply).
  - agent_action must be 'accept' or 'modify' by default. 'reject' rows
    are excluded by default because a fully-rejected suggestion with a
    hand-written replacement often reflects a different situation
    entirely, not a correction of the model's approach — include them
    with include_rejects=True if you decide otherwise later.

DEDUP: exact-text dedup only (normalized whitespace/case) against both
the current batch and the existing tickets.csv. This is NOT semantic
dedup — two differently-worded corrections that mean the same thing will
both get indexed. Good enough for prototype scale; upgrade to embedding-
based dedup (via the same Embedder from pipeline/embed.py) if duplicate
near-identical corrections start bloating the index.

PROCESSED-TRACKING: interaction_ids already folded into the corpus are
recorded in data/clean/feedback_processed.json so re-running this job
doesn't duplicate entries. This mirrors data/clean/audit_log.json's role
for the anonymization step.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from api.db import SessionLocal, Interaction, Ticket

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "data" / "clean"
TICKETS_CSV = CLEAN_DIR / "tickets.csv"
PROCESSED_LOG = CLEAN_DIR / "feedback_processed.json"

TARGET_FIELDNAMES = [
    "ticket_id", "created_at", "closed_at", "channel", "product", "issue_type",
    "issue_subtype", "priority", "sla_breach", "customer_text", "agent_response",
    "resolution_status", "resolution_time_seconds", "first_contact_resolved",
    "escalated_to", "pii_flag", "pii_entities_found", "anonymized_at", "source",
    "is_gold_example", "gold_signal", "kb_refs", "doc_refs", "language",
    "data_consent", "retention_expires_at",
]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


class FeedbackPipeline:
    def __init__(self, session_factory=SessionLocal, include_rejects: bool = False):
        self.session_factory = session_factory
        self.include_rejects = include_rejects

    # --- Step 1: fetch eligible feedback -------------------------------

    def fetch_eligible(self, db, already_processed: set):
        allowed_actions = {"accept", "modify"}
        if self.include_rejects:
            allowed_actions.add("reject")

        rows = (
            db.query(Interaction, Ticket)
            .join(Ticket, Interaction.ticket_id == Ticket.ticket_id)
            .filter(Interaction.excluded_from_training.is_(False))
            .all()
        )

        eligible = []
        for interaction, ticket in rows:
            if interaction.interaction_id in already_processed:
                continue
            if interaction.agent_action not in allowed_actions:
                continue
            if not interaction.agent_final or not interaction.agent_final.strip():
                continue
            eligible.append((interaction, ticket))
        return eligible

    # --- Step 2: dedup ---------------------------------------------------

    def dedup(self, eligible: list, existing_agent_responses: set):
        seen_in_batch = set()
        deduped = []
        skipped = 0
        for interaction, ticket in eligible:
            fingerprint = _normalize(interaction.agent_final)
            if fingerprint in seen_in_batch or fingerprint in existing_agent_responses:
                skipped += 1
                continue
            seen_in_batch.add(fingerprint)
            deduped.append((interaction, ticket))
        return deduped, skipped

    # --- Step 3: build corpus rows ---------------------------------------

    def to_corpus_rows(self, deduped: list):
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for interaction, ticket in deduped:
            gold_signal = "agent_corrected" if interaction.agent_action == "modify" else \
                          "agent_accepted" if interaction.agent_action == "accept" else \
                          "agent_rejected"
            rows.append({
                # New synthetic ticket_id derived from the interaction, so it
                # never collides with the original ticket_id (that ticket
                # already exists in the corpus with the ORIGINAL response —
                # this row captures what the agent actually ended up sending).
                "ticket_id": f"{ticket.ticket_id}-FB-{interaction.interaction_id}",
                "created_at": ticket.created_at.isoformat() if ticket.created_at else "",
                "closed_at": ticket.closed_at.isoformat() if ticket.closed_at else "",
                "channel": ticket.channel or "",
                "product": ticket.product or "",
                "issue_type": ticket.issue_type or "",
                "issue_subtype": ticket.issue_subtype or "",
                "priority": ticket.priority or "",
                "sla_breach": int(bool(ticket.sla_breach)),
                "customer_text": ticket.customer_text or "",
                "agent_response": interaction.agent_final,
                "resolution_status": ticket.resolution_status or "",
                "resolution_time_seconds": ticket.resolution_time_seconds or "",
                "first_contact_resolved": int(bool(ticket.first_contact_resolved)) if ticket.first_contact_resolved is not None else "",
                "escalated_to": ticket.escalated_to or "",
                "pii_flag": 0,  # agent_final is written by a human agent post-generation, not
                                 # re-scanned here — pipeline/output_filter.py already screened
                                 # it before the agent ever saw/edited the suggestion.
                "pii_entities_found": "",
                "anonymized_at": now,
                "source": "feedback_loop",
                "is_gold_example": 1,
                "gold_signal": gold_signal,
                "kb_refs": "",
                "doc_refs": "",
                "language": ticket.language or "en",
                "data_consent": int(bool(ticket.data_consent)) if ticket.data_consent is not None else 1,
                "retention_expires_at": ticket.retention_expires_at.isoformat() if ticket.retention_expires_at else "",
            })
        return rows

    # --- Step 4: append to corpus + update processed log -----------------

    def append_to_corpus(self, rows: list):
        if not rows:
            return
        file_exists = TICKETS_CSV.exists()
        with open(TICKETS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TARGET_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def load_processed(self) -> set:
        if not PROCESSED_LOG.exists():
            return set()
        with open(PROCESSED_LOG, encoding="utf-8") as f:
            return set(json.load(f))

    def save_processed(self, processed: set):
        PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
            json.dump(sorted(processed), f, indent=2)

    def load_existing_agent_responses(self) -> set:
        if not TICKETS_CSV.exists():
            return set()
        existing = set()
        with open(TICKETS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(_normalize(row.get("agent_response", "")))
        return existing

    # --- End-to-end run ---------------------------------------------------

    def run(self):
        db = self.session_factory()
        try:
            already_processed = self.load_processed()
            eligible = self.fetch_eligible(db, already_processed)
            existing_responses = self.load_existing_agent_responses()
            deduped, n_dupe_skipped = self.dedup(eligible, existing_responses)
            rows = self.to_corpus_rows(deduped)
            self.append_to_corpus(rows)

            newly_processed = already_processed | {i.interaction_id for i, t in deduped}
            self.save_processed(newly_processed)

            print(f"Feedback pipeline: {len(eligible)} eligible, "
                  f"{n_dupe_skipped} skipped as duplicates, "
                  f"{len(rows)} appended to {TICKETS_CSV.name}")
            return rows
        finally:
            db.close()


def main():
    pipeline = FeedbackPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()