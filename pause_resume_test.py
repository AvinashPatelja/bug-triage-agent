import os
import uuid
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

jobs = {}

class ReviewerDecision(BaseModel):
    decision: str

async def pipeline(job_id: str):
    job = jobs[job_id]

    job["status"] = "step1_running"
    await asyncio.sleep(2)
    job["status"] = "step2_running"
    await asyncio.sleep(2)

    job["status"] = "waiting_for_review"
    job["review_event"].clear()
    await job["review_event"].wait()

    decision = job["review_event"]
    if decision == "approve":
        job["status"] = "completed"
    else:
        job["status"] = "rejected"

@app.post("/start")
async def start_job():
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting",
        "review_event": asyncio.Event(),
        "review_decision": None,
    }
    asyncio.create_task(pipeline(job_id))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_dtatus(job_id: str):
    job = jobs[job_id]
    return {"job_id": job_id, "status": job["status"]}

@app.post("/review/job{job_id}")
async def submit_review(job_id, payload: ReviewerDecision):
    job = jobs[job_id]
    job["review_decision"] = payload.decision
    job["review_event"].set()
    return {"job_id" : job_id, "decision" : payload.decision }