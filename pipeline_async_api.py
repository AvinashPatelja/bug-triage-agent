import asyncio
import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import StopMessage, StructuredMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()
app = FastAPI()

jobs = {}


class ReviewDecision(BaseModel):
    decision: str


class TriageResult(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    priority: Literal["P0", "P1", "P2", "P3"]
    root_cause: str
    assigned_team: Literal["Backend", "Frontend", "DevOps", "FullStack"]
    confidence: int


class FixProposal(BaseModel):
    explanation: str
    file_path: str
    code_fix: str
    unit_test: str
    risk_level: Literal["Low", "Medium", "High"]


class PRResult(BaseModel):
    branch_name: str
    pr_title: str
    pr_description: str
    files_changed: list[str]


class PipelineComplete(TerminationCondition):
    def __init__(self):
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages):
        for msg in messages:
            if msg.source == "human_review_2":
                self._terminated = True
                return StopMessage(content="done", source="PipelineComplete")
            if msg.source == "human_review_1" and "approve" not in str(msg.content).lower():
                self._terminated = True
                return StopMessage(content="rejected", source="PipelineComplete")
        return None

    async def reset(self) -> None:
        self._terminated = False


def custom_selector(messages):
    last_message = messages[-1]
    last_speaker = last_message.source

    if last_speaker == "user":
        return "fetcher_agent"
    if last_speaker == "fetcher_agent":
        return "triage_agent"
    if last_speaker == "triage_agent":
        return "investigation_agent"
    if last_speaker == "investigation_agent":
        return "fix_agent"
    if last_speaker == "fix_agent":
        return "human_review_1"
    if last_speaker == "human_review_1":
        content = str(last_message.content).lower()
        return "pr_agent" if "approve" in content else None
    if last_speaker == "pr_agent":
        return "human_review_2"
    if last_speaker == "human_review_2":
        return None
    return None


def make_review_handler(job_id: str, status_label: str):
    async def handler(prompt: str, cancellation_token=None) -> str:
        job = jobs[job_id]
        job["status"] = status_label
        job["review_event"].clear()
        await job["review_event"].wait()
        return job["review_decision"]
    return handler


async def run_pipeline_job(job_id: str, issue_key: str):
    job = jobs[job_id]

    jira_server_params = StdioServerParams(
        command="mcp-atlassian",
        args=[],
        env={
            "JIRA_URL": os.environ["JIRA_URL"],
            "JIRA_USERNAME": os.environ["JIRA_USERNAME"],
            "JIRA_API_TOKEN": os.environ["JIRA_API_TOKEN"],
        },
    )
    jira_tools = await mcp_server_tools(jira_server_params)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    fetcher_agent = AssistantAgent(
        name="fetcher_agent", model_client=model_client,
        system_message="""Use the jira_get_issue tool to fetch the requested Jira issue.
        Report back the issue's key, summary, and full description as plain text.""",
        tools=jira_tools,
    )
    triage_agent = AssistantAgent(
        name="triage_agent", model_client=model_client,
        system_message="""You are a Triage Agent. Analyze the provided bug report
        and classify it according to the required schema.""",
        output_content_type=TriageResult,
    )
    investigation_agent = AssistantAgent(
        name="investigation_agent", model_client=model_client,
        system_message="""You are an Investigation Agent. Hypothesize which files/methods
        are likely involved and what evidence would confirm the root cause.""",
    )
    fix_agent = AssistantAgent(
        name="fix_agent", model_client=model_client,
        system_message="""You are a Fix Generator Agent. Propose a concrete code fix
        and unit test as a realistic C# snippet. Assess risk level honestly.""",
        output_content_type=FixProposal,
    )

    human_review_1 = UserProxyAgent(
        name="human_review_1",
        input_func=make_review_handler(job_id, "waiting_for_review_1"),
    )

    pr_agent = AssistantAgent(
        name="pr_agent", model_client=model_client,
        system_message="""You are a PR Creation Agent. Propose a branch name, PR title,
        description (referencing the bug ID), and files changed. Simulation only.""",
        output_content_type=PRResult,
    )

    human_review_2 = UserProxyAgent(
        name="human_review_2",
        input_func=make_review_handler(job_id, "waiting_for_review_2"),
    )

    team = SelectorGroupChat(
        participants=[fetcher_agent, triage_agent, investigation_agent, fix_agent,
                      human_review_1, pr_agent, human_review_2],
        model_client=model_client,
        selector_func=custom_selector,
        termination_condition=PipelineComplete(),
        custom_message_types=[
            StructuredMessage[TriageResult],
            StructuredMessage[FixProposal],
            StructuredMessage[PRResult],
        ],
    )

    job["status"] = "running"
    task_result = await team.run(task=f"Fetch and triage Jira issue {issue_key}")

    job["status"] = "completed"
    job["messages"] = [
        {"agent": m.source, "content": str(m.content)}
        for m in task_result.messages
    ]

    await model_client.close()


@app.post("/triage/{issue_key}")
async def start_triage(issue_key: str):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting",
        "review_event": asyncio.Event(),
        "review_decision": None,
        "messages": [],
    }
    asyncio.create_task(run_pipeline_job(job_id, issue_key))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "messages": job.get("messages", []),
    }


@app.post("/review/{job_id}")
async def submit_review(job_id: str, payload: ReviewDecision):
    job = jobs[job_id]
    job["review_decision"] = payload.decision
    job["review_event"].set()
    return {"job_id": job_id, "decision_submitted": payload.decision}