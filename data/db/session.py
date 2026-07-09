"""
db/session.py

Engine + session management for the SQLite database. This is what
api/dependencies.py will import to get a DB session per request — nothing
in here is FastAPI-specific, so it's equally usable from scripts
(data/seeds/load_csvs_to_db.py could be pointed at this instead of building
its own engine) and from the retrain/eval jobs.

SQLite-specific note: check_same_thread=False is required because FastAPI
can (and will) serve a request on a different thread than the one that
created the connection. This is safe here because each request gets its
own Session from SessionLocal() — sessions are not shared across requests/
threads, only the underlying engine's connection pool is.

ENTERPRISE MIGRATION NOTE: this file is scoped the same way as
pipeline/llm.py — swapping SQLite for Postgres/managed DB in production
means changing DATABASE_URL (env var) and dropping the SQLite-only
connect_args below. db/models.py does not need to change.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "db" / "support.db"

# Env var override, so this doubles as the enterprise migration lever for
# the DB layer — same pattern as pipeline/llm.py's model config.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create tables if they don't exist. Does NOT drop/recreate — unlike
    the CSV loader's full-reload behavior, this is safe to call every time
    the API starts up without wiping live agent feedback data."""
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, guarantees it's closed after
    the request, even if the request raises."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()