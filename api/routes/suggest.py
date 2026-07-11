"""
api/routes/suggest.py

POST /suggest — an agent's ticket view calls this to get a draft reply.

Flow:
  1. Redact PII from the incoming customer_text (live per-message
     anonymization — see pipeline/anonymize.py's PIIAnonymizer.redact_text,
     the live-call path, not the batch CSV path).
  2. Ensure a Ticket row exists for this ticket_id (create a minimal one
     if this is the first time we've seen it).
  3. Run retrieval + generation (pipeline/rag.py).
  4. Re-scan the generated suggestion for PII (pipeline/output_filter.py).
     If high-severity PII is found, DO NOT return the suggestion — return
     blocked=True instead, so the frontend routes this to the escalation
     queue rather than showing it as a normal suggestion.
  5. Log an Interaction row with the suggestion + retrieval metadata.
     agent_action/agent_final/etc. are left NULL — they get filled in
     later by POST /feedback once the agent actually acts on it.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db, Ticket, Interaction
from api.main import get_anonymizer, get_output_filter, get_rag
from pipeline.anonymize import PIIAnonymizer
from pipeline.output_filter import OutputFilter
from pipeline.rag import RAGPipeline

router = APIRouter()


class SuggestRequest(BaseModel):
    ticket_id: str
    customer_text: str
    agent_id: str
    channel: str = "chat"
    product: str = ""


class SuggestResponse(BaseModel):
    interaction_id: int
    ticket_id: str
    suggestion: str | None
    blocked: bool
    block_reason: str = ""
    retrieval_top1_id: str | None
    retrieval_top1_score: float | None
    retrieval_top3_avg_score: float | None
    low_confidence_flag: bool


@router.post("/suggest", response_model=SuggestResponse)
def suggest(
    req: SuggestRequest,
    db: Session = Depends(get_db),
    anonymizer: PIIAnonymizer = Depends(get_anonymizer),
    output_filter: OutputFilter = Depends(get_output_filter),
    rag: RAGPipeline = Depends(get_rag),
):
    if not req.customer_text.strip():
        raise HTTPException(status_code=400, detail="customer_text cannot be empty")

    # 1. Redact PII from the live incoming message before it touches
    # retrieval or the LLM.
    clean_text, entities_found = anonymizer.redact_text(req.customer_text)

    # 2. Ensure a Ticket row exists.
    ticket = db.query(Ticket).filter(Ticket.ticket_id == req.ticket_id).first()
    if ticket is None:
        ticket = Ticket(
            ticket_id=req.ticket_id,
            created_at=datetime.now(timezone.utc),
            channel=req.channel,
            product=req.product,
            issue_type="unclassified",
            priority="medium",
            sla_breach=False,
            customer_text=clean_text,
            pii_flag=bool(entities_found),
            pii_entities_found=";".join(entities_found),
            anonymized_at=datetime.now(timezone.utc),
            source="live_chat",
            is_gold_example=False,
            language="en",
            data_consent=True,
        )
        db.add(ticket)
        db.commit()

    # 3. Retrieval + generation.
    rag_result = rag.generate(clean_text)

    # 4. Re-scan the generated output.
    filter_result = output_filter.scan(rag_result.suggestion)

    # 5. Log the interaction. agent_action stays NULL until /feedback.
    interaction = Interaction(
        ticket_id=req.ticket_id,
        agent_id=req.agent_id,
        suggested_at=datetime.now(timezone.utc),
        model_version="phi-4-mini-nf4",  # TODO: pull from actual loaded model config once versioned
        model_suggestion=rag_result.suggestion,
        retrieval_top1_id=rag_result.retrieval_top1_id,
        retrieval_top1_score=rag_result.retrieval_top1_score,
        retrieval_top3_avg_score=rag_result.retrieval_top3_avg_score,
        low_confidence_flag=rag_result.low_confidence_flag,
        excluded_from_training=filter_result.blocked,  # blocked suggestions shouldn't feed back into training
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    return SuggestResponse(
        interaction_id=interaction.interaction_id,
        ticket_id=req.ticket_id,
        suggestion=None if filter_result.blocked else filter_result.safe_text,
        blocked=filter_result.blocked,
        block_reason=filter_result.block_reason,
        retrieval_top1_id=rag_result.retrieval_top1_id,
        retrieval_top1_score=rag_result.retrieval_top1_score,
        retrieval_top3_avg_score=rag_result.retrieval_top3_avg_score,
        low_confidence_flag=rag_result.low_confidence_flag,
    )