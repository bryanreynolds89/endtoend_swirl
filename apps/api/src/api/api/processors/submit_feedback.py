from fastapi import HTTPException
from typing import Optional
from langsmith import Client

client = Client()

def submit_feedback(
    trace_id: str,
    feedback_score: Optional[int] = None,
    feedback_text: str = "",
    feedback_source_type: str = "api",
):
    if not trace_id or not trace_id.strip():
        raise HTTPException(status_code=422, detail="trace_id is required to submit feedback")

    if feedback_score is not None:
        client.create_feedback(
            trace_id=trace_id,  # change to run_id=... only if this is truly a run id
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type,
        )

    if feedback_text.strip():
        client.create_feedback(
            trace_id=trace_id,
            key="comment",
            value=feedback_text.strip(),
            feedback_source_type=feedback_source_type,
        )