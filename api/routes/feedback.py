"""
api/routes/feedback.py

POST /feedback — called once the agent accepts/modifies/rejects a
suggestion. Fills in the Interaction row that /suggest created with
agent_action=None (pending).

edit_distance / edit_distance_ratio are computed HERE, server-side, from
model_suggestion vs agent_final — never trusted from the client. This is
the exact bug class found in the original sample data (edit_distance that
didn't match the actual text pair) — computing it ourselves at write time
is what prevents that from ever happening again.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db, Interaction

router = APIRouter()


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


class FeedbackRequest(BaseModel):
    interaction_id: int
    agent_action: str  # accept | modify | reject
    agent_final: str
    dwell_seconds: int | None = None
    agent_confidence: int | None = None
    rationale_type: str | None = None
    rationale_severity: str | None = None
    rationale_suggested_change: str | None = None
    rationale_root_cause: str | None = None


class FeedbackResponse(BaseModel):
    interaction_id: int
    agent_action: str
    edit_distance: int
    edit_distance_ratio: float


VALID_ACTIONS = {"accept", "modify", "reject"}


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.agent_action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"agent_action must be one of {VALID_ACTIONS}")

    interaction = db.query(Interaction).filter(
        Interaction.interaction_id == req.interaction_id
    ).first()
    if interaction is None:
        raise HTTPException(status_code=404, detail="interaction_id not found")
    if interaction.agent_action is not None:
        raise HTTPException(status_code=409, detail="feedback already submitted for this interaction")

    model_suggestion = interaction.model_suggestion or ""
    edit_distance = _levenshtein(model_suggestion, req.agent_final)
    max_len = max(len(model_suggestion), len(req.agent_final), 1)  # avoid div-by-zero on two empty strings
    edit_distance_ratio = edit_distance / max_len

    interaction.agent_action = req.agent_action
    interaction.agent_final = req.agent_final
    interaction.edit_distance = edit_distance
    interaction.edit_distance_ratio = edit_distance_ratio
    interaction.dwell_seconds = req.dwell_seconds
    interaction.agent_confidence = req.agent_confidence
    interaction.rationale_type = req.rationale_type
    interaction.rationale_severity = req.rationale_severity
    interaction.rationale_suggested_change = req.rationale_suggested_change
    interaction.rationale_root_cause = req.rationale_root_cause
    interaction.submitted_at = datetime.now(timezone.utc)

    db.commit()

    return FeedbackResponse(
        interaction_id=interaction.interaction_id,
        agent_action=interaction.agent_action,
        edit_distance=edit_distance,
        edit_distance_ratio=edit_distance_ratio,
    )