"""
api/db.py

SQLite schema + engine/session management, merged into one file per the
LoopSupport project structure (README.md). This replaces the earlier
two-file split (db/models.py + db/session.py) from the prototype's first
pass — same tables, same session behavior, just co-located here since
that's where api/main.py and api/routes/ expect to import it from.

Four tables:
  - Ticket        (Class A: historical ticket / memory)
  - Interaction    (Class B: agent feedback on model suggestions)
  - SignalTag      (Class C: evaluation labels / structured signals)
  - KBDoc          (auxiliary: knowledge base documents used for retrieval)

ENTERPRISE MIGRATION NOTE: swapping SQLite for Postgres/managed DB in
production means changing DATABASE_URL (env var) and dropping the
SQLite-only connect_args below. Table definitions do not need to change —
same pattern as pipeline/llm.py being the single lever for the LLM swap.
"""

import os
from pathlib import Path

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


# --- Schema ----------------------------------------------------------

class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    channel = Column(String, nullable=False)
    product = Column(String, nullable=False)
    issue_type = Column(String, nullable=False)
    issue_subtype = Column(String, nullable=True)
    priority = Column(String, nullable=False)
    sla_breach = Column(Boolean, nullable=False, default=False)
    customer_text = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=True)
    resolution_status = Column(String, nullable=True)
    resolution_time_seconds = Column(Integer, nullable=True)
    first_contact_resolved = Column(Boolean, nullable=True)
    escalated_to = Column(String, nullable=True)
    pii_flag = Column(Boolean, nullable=False, default=False)
    pii_entities_found = Column(String, nullable=True)
    anonymized_at = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    is_gold_example = Column(Boolean, nullable=False, default=False)
    gold_signal = Column(String, nullable=True)
    kb_refs = Column(String, nullable=True)
    doc_refs = Column(String, nullable=True)
    language = Column(String, nullable=False, default="en")
    data_consent = Column(Boolean, nullable=False, default=True)
    retention_expires_at = Column(DateTime, nullable=True)

    interactions = relationship("Interaction", back_populates="ticket")


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id = Column(Integer, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=False)
    agent_id = Column(String, nullable=False)
    suggested_at = Column(DateTime, nullable=False)
    model_version = Column(String, nullable=False)
    model_suggestion = Column(Text, nullable=False)
    retrieval_top1_id = Column(String, nullable=True)
    retrieval_top1_score = Column(Float, nullable=True)
    retrieval_top3_avg_score = Column(Float, nullable=True)
    low_confidence_flag = Column(Boolean, nullable=False, default=False)
    agent_action = Column(String, nullable=False)
    agent_final = Column(Text, nullable=True)
    edit_distance = Column(Integer, nullable=True)
    edit_distance_ratio = Column(Float, nullable=True)
    dwell_seconds = Column(Integer, nullable=True)
    agent_confidence = Column(Integer, nullable=True)
    rationale_type = Column(String, nullable=True)
    rationale_severity = Column(String, nullable=True)
    rationale_suggested_change = Column(Text, nullable=True)
    rationale_root_cause = Column(String, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    excluded_from_training = Column(Boolean, nullable=False, default=False)

    ticket = relationship("Ticket", back_populates="interactions")
    signal_tags = relationship("SignalTag", back_populates="interaction")


class SignalTag(Base):
    __tablename__ = "signal_tags"

    signal_id = Column(Integer, primary_key=True)
    interaction_id = Column(Integer, ForeignKey("interactions.interaction_id"), nullable=False)
    signal_category = Column(String, nullable=False)
    signal_severity = Column(String, nullable=False)
    fixable = Column(Boolean, nullable=False, default=False)
    fix_type = Column(String, nullable=True)
    signal_summary = Column(Text, nullable=True)
    extracted_by = Column(String, nullable=True)
    extracted_at = Column(DateTime, nullable=True)
    human_quality_rating = Column(Integer, nullable=True)
    silent_failure = Column(Boolean, nullable=False, default=False)
    audited_by = Column(String, nullable=True)
    audited_at = Column(DateTime, nullable=True)
    used_in_index_rebuild = Column(Boolean, nullable=False, default=False)
    used_in_prompt_update = Column(Boolean, nullable=False, default=False)
    prompt_version_after = Column(String, nullable=True)

    interaction = relationship("Interaction", back_populates="signal_tags")


class KBDoc(Base):
    __tablename__ = "kb_docs"

    doc_id = Column(String, primary_key=True)
    doc_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    product = Column(String, nullable=True)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    version = Column(String, nullable=True)
    indexed_at = Column(DateTime, nullable=True)


# --- Engine / session --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "support_ai.db"

# Env var override — this is the enterprise migration lever for the DB
# layer, same role as pipeline/llm.py plays for the model.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create tables if they don't exist. Does NOT drop/recreate — safe to
    call on every API startup without wiping live agent feedback data."""
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, guarantees it's closed after
    the request even if the request raises.

    Usage in a route:
        from fastapi import Depends
        from api.db import get_db

        @router.post("/suggest")
        def suggest(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()