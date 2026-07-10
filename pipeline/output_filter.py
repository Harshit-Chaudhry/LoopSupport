"""
pipeline/output_filter.py

Second line of defense: re-scans the MODEL'S GENERATED OUTPUT for PII
before it's shown to the agent. This is separate from pipeline/anonymize.py
(which scrubs incoming raw ticket data) — this catches PII that could leak
INTO a generated suggestion, which can happen even with clean input data,
because:
  1. Retrieved context (past tickets/KB docs) could itself contain PII that
     slipped through anonymization — this is the safety net for that.
  2. The LLM can hallucinate what looks like a plausible phone number,
     email, or account-style ID even if none was in the prompt.

Reuses PIIAnonymizer's regex detection (composition, not duplication) —
same patterns, same entity types, so "what counts as PII" is defined in
exactly one place in the codebase.

POLICY: not all detected entities are treated the same way.
  - EMAIL / PHONE: redacted inline, suggestion still goes to the agent,
    flagged so it's visible that a redaction happened.
  - CARD / AADHAAR / PAN: these are high-severity — finding one of these
    in a *generated* reply is anomalous (it means something upstream
    leaked), not just noisy. These BLOCK the suggestion outright rather
    than silently redacting and passing it through, and should route to
    the escalation queue for human review rather than the agent's normal
    suggestion flow. See api/routes/escalate.py.
"""

from dataclasses import dataclass, field

from pipeline.anonymize import PIIAnonymizer

HIGH_SEVERITY_ENTITIES = {"CARD", "AADHAAR", "PAN"}


@dataclass
class FilterResult:
    safe_text: str
    entities_found: list
    was_modified: bool
    blocked: bool
    block_reason: str = field(default="")


class OutputFilter:
    """
    Usage (api/main.py, once at startup — same singleton pattern as the
    other pipeline components):
        anonymizer = PIIAnonymizer()          # already instantiated for anonymize.py
        output_filter = OutputFilter(anonymizer)   # reuses the same instance

    Usage (per request, after rag.generate()):
        result = output_filter.scan(rag_result.suggestion)
        if result.blocked:
            # route to api/routes/escalate.py instead of returning to agent
            ...
        else:
            # return result.safe_text to the agent (possibly redacted, still usable)
            ...
    """

    def __init__(self, anonymizer: PIIAnonymizer = None):
        # Accept an existing instance so callers reuse the same compiled
        # regexes rather than constructing a second PIIAnonymizer — mirrors
        # how Embedder is shared between embed.py and rag.py.
        self.anonymizer = anonymizer or PIIAnonymizer()

    def scan(self, text: str) -> FilterResult:
        if not text:
            return FilterResult(safe_text=text, entities_found=[], was_modified=False, blocked=False)

        safe_text, entities_found = self.anonymizer.redact_text(text)
        was_modified = safe_text != text

        high_severity_hits = HIGH_SEVERITY_ENTITIES & set(entities_found)
        if high_severity_hits:
            return FilterResult(
                safe_text=safe_text,
                entities_found=entities_found,
                was_modified=was_modified,
                blocked=True,
                block_reason=(
                    f"High-severity PII ({', '.join(sorted(high_severity_hits))}) "
                    f"detected in generated output — routing to human review "
                    f"instead of auto-suggesting."
                ),
            )

        return FilterResult(
            safe_text=safe_text,
            entities_found=entities_found,
            was_modified=was_modified,
            blocked=False,
        )