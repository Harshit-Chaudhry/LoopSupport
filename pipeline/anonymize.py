"""
pipeline/pii_anonymizer.py
 
Transforms raw ticket data (which contains direct identifiers: name, email,
phone, address, card number, account_id, Aadhaar, PAN) into the anonymized
`tickets` schema used everywhere else in the system (retrieval index,
prompts, DB).
 
CLASS-BASED, because this needs to run in two different modes:
  1. BATCH — processing tickets_raw.csv / holdout_raw.csv (500+ rows) once,
     offline, to build the initial retrieval index.
  2. LIVE — when the real system is running, a new customer message comes
     in through the FastAPI endpoint and needs to be anonymized in-flight,
     one message at a time, before it's shown to the agent or sent to the
     LLM/retriever. There's no "row" with closed_at/resolution_time yet at
     that point — just raw text.
 
`PIIAnonymizer` is instantiated ONCE (same pattern as the LLM in
pipeline/llm.py — loaded once at FastAPI startup, reused across requests).
The regex patterns are compiled once at construction, not per-call.
 
Design decisions (read before changing behavior):
 
1. STABLE ID POLICY: `ticket_id` is preserved byte-for-byte from raw.
   This pipeline never regenerates or reformats ticket_id. If raw and
   anonymized IDs don't match 1:1, retention/deletion requests can't be
   traced end-to-end.
 
2. STRUCTURED PII COLUMNS ARE FULLY DROPPED, NOT MASKED:
   customer_name, customer_email, customer_phone, customer_address,
   card_number, card_last4, account_id, order_id, aadhaar, pan are never
   written to the output.
 
3. pii_flag / pii_entities_found REFLECT FREE-TEXT LEAKAGE ONLY — not
   "did this raw record have a name/email column" (trivially always true),
   but "did we find and redact a PII pattern inside the actual message
   text." That's the signal that matters for a system indexing this text.
 
4. order_id is explicitly NOT treated as PII by default (it's a business
   identifier, not a personal one). Toggle via order_id_is_pii at
   construction if your compliance policy disagrees.
 
5. resolution_time_seconds is RECOMPUTED from created_at/closed_at when
   both are available, never passed through from raw.
 
6. Fields that don't exist in raw exports (is_gold_example, gold_signal,
   kb_refs, doc_refs, language, data_consent) are set to safe defaults
   here — they're filled in by a separate labeling/curation step, not by
   this class. Anonymization and labeling are different jobs.
"""



import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine


class PIIAnonymizer:
    """
    Usage — batch (offline):
        anonymizer = PIIAnonymizer()
        anonymizer.anonymize_csv(Path("data/raw/tickets_raw.csv"),
                                  Path("data/raw/tickets.csv"),
                                  source_label="csv_import")
 
    Usage — live (FastAPI, instantiate once at startup):
        anonymizer = PIIAnonymizer()   # e.g. in api/dependencies.py
 
        # per incoming message, before it touches retrieval/LLM:
        clean_text, entities = anonymizer.redact_text(incoming_message)
 
        # once a ticket record needs to be written to the DB/index:
        row = anonymizer.anonymize_ticket(raw_ticket_dict, source_label="live_chat")
    """

    TARGET_FIELDNAMES = [
        "ticket_id", "created_at", "closed_at", "channel", "product", "issue_type",
        "issue_subtype", "priority", "sla_breach", "customer_text", "agent_response",
        "resolution_status", "resolution_time_seconds", "first_contact_resolved",
        "escalated_to", "pii_flag", "pii_entities_found", "anonymized_at", "source",
        "is_gold_example", "gold_signal", "kb_refs", "doc_refs", "language",
        "data_consent", "retention_expires_at",
    ]
 
    RAW_TO_TARGET_RENAME = {
        "category": "issue_type",
        "status": "resolution_status",
    }
 
    DIRECT_ID_COLUMNS = {
        "customer_name", "customer_email", "customer_phone", "customer_address",
        "card_number", "card_last4", "account_id", "order_id", "aadhaar", "pan",
    }


    def __init__(self, order_id_is_pii: bool = False,
                 default_retention_days: int = 365,
                 default_language: str = "en"):
        self.order_id_is_pii = order_id_is_pii
        self.default_retention_days = default_retention_days
        self.default_language = default_language
 
        # Compiled once per instance, reused across every call — matters
        # for the live path where this runs per-message, not just once
        # per batch job.
        self._email_re = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self._phone_re = re.compile(r'(?:\+\d{1,3}[-\s]?)?\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}')
        self._card_re = re.compile(r'\b\d(?:[ -]?\d){12,18}\b')
        self._aadhaar_re = re.compile(r'\b\d{4}-\d{4}-\d{4}\b')
        self._pan_re = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
        self._order_id_re = re.compile(r'\bORD-\d+\b')


    
     # --- Core redaction, safe to call per live message ------------------
 
    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        """Luhn checksum, to avoid flagging arbitrary long numbers as card numbers."""
        digits = re.sub(r'[ -]', '', digits)
        if not digits.isdigit() or not (13 <= len(digits) <= 19):
            return False
        total = 0
        for i, d in enumerate(digits[::-1]):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0
    
    def redact_text(self, text: str):
        """
        Scans free text for PII patterns, replaces them with typed
        placeholder tokens, and returns (redacted_text, sorted entity list).
 
        This only catches PII with a reliable machine-checkable pattern
        (email, phone, card, Aadhaar, PAN). Names and free-text addresses
        have no such pattern and are NOT caught here — a production system
        needs an NER model for that. Structured columns (customer_name,
        customer_address) are dropped wholesale in anonymize_ticket(),
        which is the actual safety net for those two.
        """
        if not text:
            return text, []
 
        found = set()
        result = text
 
        def repl(pattern, label, validator=None):
            nonlocal result
            def _sub(m):
                val = m.group(0)
                if validator and not validator(val):
                    return val
                found.add(label)
                return f"[{label}_REDACTED]"
            result = pattern.sub(_sub, result)
 
        repl(self._email_re, "EMAIL")
        repl(self._aadhaar_re, "AADHAAR")
        repl(self._pan_re, "PAN")
        repl(self._card_re, "CARD", validator=self._luhn_valid)
        repl(self._phone_re, "PHONE")
 
        if self.order_id_is_pii:
            repl(self._order_id_re, "ORDER_ID")
 
        return result, sorted(found)
    

    # --- Full ticket-row transform (batch or live-on-ticket-close) ------
 
    def anonymize_ticket(self, raw_row: dict, source_label: str,
                          retention_days: int = None) -> dict:
        retention_days = retention_days or self.default_retention_days
        row = dict(raw_row)
 
        for old, new in self.RAW_TO_TARGET_RENAME.items():
            if old in row:
                row[new] = row.pop(old)
 
        customer_text, entities_1 = self.redact_text(row.get("customer_text", ""))
        agent_response, entities_2 = self.redact_text(row.get("agent_response", ""))
        entities_found = sorted(set(entities_1) | set(entities_2))
 
        resolution_time_seconds = row.get("resolution_time_seconds")
        try:
            created = datetime.fromisoformat(row["created_at"])
            closed = datetime.fromisoformat(row["closed_at"]) if row.get("closed_at") else None
            if closed:
                resolution_time_seconds = int((closed - created).total_seconds())
        except (KeyError, ValueError):
            pass  # leave raw value if timestamps missing/malformed; caller should audit
 
        anonymized_at = datetime.now(timezone.utc).isoformat()
        try:
            retention_expires_at = (
                datetime.fromisoformat(row["created_at"]) + timedelta(days=retention_days)
            ).isoformat()
        except (KeyError, ValueError):
            retention_expires_at = None
 
        out = {
            "ticket_id": row.get("ticket_id"),
            "created_at": row.get("created_at"),
            "closed_at": row.get("closed_at"),
            "channel": row.get("channel"),
            "product": row.get("product"),
            "issue_type": row.get("issue_type"),
            "issue_subtype": row.get("issue_subtype"),
            "priority": row.get("priority"),
            "sla_breach": row.get("sla_breach"),
            "customer_text": customer_text,
            "agent_response": agent_response,
            "resolution_status": row.get("resolution_status"),
            "resolution_time_seconds": resolution_time_seconds,
            "first_contact_resolved": row.get("first_contact_resolved"),
            "escalated_to": row.get("escalated_to"),
            "pii_flag": 1 if entities_found else 0,
            "pii_entities_found": ";".join(entities_found) if entities_found else "",
            "anonymized_at": anonymized_at,
            "source": source_label,
            "is_gold_example": "",
            "gold_signal": "",
            "kb_refs": "",
            "doc_refs": "",
            "language": self.default_language,
            "data_consent": 1,
            "retention_expires_at": retention_expires_at,
        }


    # Safety net: if a future edit ever accidentally reintroduces a raw
        # direct-identifier field into the output dict, fail loudly instead
        # of silently leaking PII into the anonymized dataset.
        leaked = self.DIRECT_ID_COLUMNS & out.keys()
        if leaked:
            raise ValueError(
                f"Direct identifier column(s) {leaked} present in anonymized "
                f"output for ticket_id={out.get('ticket_id')} — this should "
                f"never happen, check anonymize_ticket()."
            )
 
        return out
    
    # --- Batch CSV job ----------------------------------------------------
 
    def anonymize_csv(self, input_path: Path, output_path: Path, source_label: str):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        n_flagged = 0
        skipped = []
 
        with open(input_path, newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            rows_out = []
            for i, raw_row in enumerate(reader, start=2):  # header is line 1
                if not raw_row.get("ticket_id"):
                    skipped.append(i)
                    continue
                try:
                    out_row = self.anonymize_ticket(raw_row, source_label=source_label)
                except Exception as e:
                    skipped.append(i)
                    print(f"  ⚠ row {i} ({raw_row.get('ticket_id')}) failed: {e}")
                    continue
                rows_out.append(out_row)
                n_rows += 1
                if out_row["pii_flag"]:
                    n_flagged += 1
 
        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=self.TARGET_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows_out)
 
        print(f"{input_path.name} -> {output_path.name}: {n_rows} rows written, "
              f"{n_flagged} flagged for free-text PII"
              + (f", {len(skipped)} rows skipped (malformed): lines {skipped}" if skipped else ""))
        return n_rows, n_flagged, skipped


 
def main():
    """CLI entry point for the batch job — separate from the class so the
    class stays import-friendly for api/dependencies.py without pulling in
    argparse/CLI concerns."""
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "raw"
 
    anonymizer = PIIAnonymizer()  # same class, same defaults, batch mode
 
    jobs = [
        ("tickets_raw.csv", "tickets.csv", "csv_import"),
        ("holdout_raw.csv", "holdout.csv", "holdout"),
    ]
 
    for raw_name, out_name, source_label in jobs:
        in_path = raw_dir / raw_name
        out_path = out_dir / out_name
        if not in_path.exists():
            print(f"  ⚠ {raw_name} not found in {raw_dir}, skipping")
            continue
        anonymizer.anonymize_csv(in_path, out_path, source_label)
 
 
if __name__ == "__main__":
    main()