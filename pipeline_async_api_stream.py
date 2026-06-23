import asyncio
import os
import uuid
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}


# ---------- Request/response schemas ----------

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


# ---------- GitHub tool (Phase D) ----------

async def get_github_file_contents(file_path: str) -> str:
    """Fetches the contents of a file from the configured GitHub repository.
    Use this to read real source code before proposing a fix.
    file_path should be relative to the repo root, e.g. 'PaymentService.cs'."""

    owner = os.environ["GITHUB_OWNER"]
    repo = os.environ["GITHUB_REPO"]
    token = os.environ["GITHUB_TOKEN"]

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error fetching file: {response.status_code} - {response.text}"

    return response.text


# ---------- Termination condition ----------

class PipelineComplete(TerminationCondition):
    def __init__(self):
        self._terminated = False
        self.outcome = "completed"  # default; overwritten if rejected

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages):
        for msg in messages:
            if msg.source == "human_review_2":
                self._terminated = True
                content = str(msg.content).lower()
                self.outcome = "completed" if "approve" in content else "rejected"
                return StopMessage(content=self.outcome, source="PipelineComplete")
            if msg.source == "human_review_1" and "approve" not in str(msg.content).lower():
                self._terminated = True
                self.outcome = "rejected"
                return StopMessage(content="rejected", source="PipelineComplete")
        return None

    async def reset(self) -> None:
        self._terminated = False
        self.outcome = "completed"


# ---------- Selector (controls speaking order) ----------

def custom_selector(messages):
    last_message = messages[-1]
    last_speaker = last_message.source

    if last_speaker == "user":
        return "fetcher_agent"
    if last_speaker == "fetcher_agent":
        return "triage_agent"
    if last_speaker == "triage_agent":
        return "jira_commenter_agent"
    if last_speaker == "jira_commenter_agent":
        return "jira_transitioner_agent"
    if last_speaker == "jira_transitioner_agent":
        return "investigation_agent"
    if last_speaker == "investigation_agent":
        return "code_reader_agent"
    if last_speaker == "code_reader_agent":
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


# ---------- Human review handler factory ----------

def make_review_handler(job_id: str, status_label: str):
    async def handler(prompt: str, cancellation_token=None) -> str:
        job = jobs[job_id]
        job["status"] = status_label
        job["review_event"].clear()
        await job["review_event"].wait()
        return job["review_decision"]
    return handler


# ---------- The pipeline itself ----------

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
    jira_tools = None
    for attempt in range(3):
        try:
            jira_tools = await mcp_server_tools(jira_server_params)
            print(f"[DEBUG] Got {len(jira_tools)} jira tools (attempt {attempt + 1})")
            break
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            await asyncio.sleep(1)
    
    if jira_tools is None:
        job["status"] = "failed"
        return
        
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

    jira_commenter_agent = AssistantAgent(
        name="jira_commenter_agent",
        model_client=model_client,
        system_message="""Call jira_add_comment right now to post the given triage summary
        as a comment on the given Jira issue key. Do this immediately, take no other action.""",
        tools=jira_tools,
    )

    # jira_transitioner_agent = AssistantAgent(
    #     name="jira_transitioner_agent",
    #     model_client=model_client,
    #     system_message="""You move a Jira issue to "In Progress". You will be told the issue key.

    #     Step 1: Call jira_get_transitions with that issue key.
    #     Step 2: From the result, find the entry whose "name" is exactly "In Progress" and note its "id".
    #     Step 3: Call jira_transition_issue with that issue key and transition_id set to that id (as a string).

    #     Do NOT call jira_add_comment. Do NOT call any other tool. Your only job is the transition,
    #     nothing else. You must complete Step 3 - calling jira_get_transitions alone is not enough.""",
    #     tools=transition_only_tools,
    # )
    jira_transitioner_agent = AssistantAgent(
        name="jira_transitioner_agent",
        model_client=model_client,
        system_message="""You move a Jira issue to In Progress.

        Step 1: Call jira_get_transitions for the issue key.
        Step 2: Find the transition whose name is exactly "In Progress".
        Step 3: Call jira_transition_issue with that same issue key and the matching transition id.

        Do not stop after listing transitions. Your job is not done until the issue is transitioned successfully.""",
        tools=[t for t in jira_tools if t.name in ("jira_get_transitions", "jira_transition_issue")],
    )

    investigation_agent = AssistantAgent(
        name="investigation_agent", model_client=model_client,
        system_message="""You are an Investigation Agent. Hypothesize which files/methods
        are likely involved and what evidence would confirm the root cause.""",
    )

    code_reader_agent = AssistantAgent(
        name="code_reader_agent", model_client=model_client,
        system_message="""You are a Code Reader Agent. Given an investigation's hypothesis
        about which file is likely involved in a bug, use the get_github_file_contents tool
        to fetch that file's real contents. Report back the exact file contents you retrieved,
        plus the filename. If you're unsure of the exact filename, try 'PaymentService.cs'.""",
        tools=[get_github_file_contents],
    )

    fix_agent = AssistantAgent(
        name="fix_agent", model_client=model_client,
        system_message="""You are a Fix Generator Agent. You will be given the REAL source code
        of the file involved in this bug. Propose a concrete code fix based on the ACTUAL code shown
        to you do not invent class names or methods that aren't in the real code. 
        Mention the file, method and line of code where the changes are required.
        Also propose a unit test. Assess risk level honestly.
        
        IMPORTANT: Your code_fix and unit_test fields must be valid C# code, properly escaped as
        JSON string values. Avoid using string interpolation with embedded double quotes (e.g. avoid
        $"text {variable}" patterns with nested quotes) - prefer string.Format or simple concatenation
        if it helps avoid quote-escaping issues. Keep code concise.""",
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

    termination_condition = PipelineComplete()

    team = SelectorGroupChat(
        participants=[
            fetcher_agent, triage_agent, investigation_agent, code_reader_agent,
            fix_agent, human_review_1, pr_agent, human_review_2, jira_commenter_agent,
            jira_transitioner_agent
        ],
        model_client=model_client,
        selector_func=custom_selector,
        termination_condition=termination_condition,   # pass the SAME object
        custom_message_types=[
            StructuredMessage[TriageResult],
            StructuredMessage[FixProposal],
            StructuredMessage[PRResult],
        ],
    )

    job["status"] = "running"
    job["current_agent"] = "fetcher_agent"
    job["messages"] = []

    final_outcome = "completed"  # default

    async for message in team.run_stream(task=f"Fetch and triage Jira issue {issue_key}. After posting the triage comment, transition this issue to In Progress. Issue key: {issue_key}"):
        source = getattr(message, "source", None)
        if source is None:
            continue

        content = getattr(message, "content", "")

        if hasattr(content, "model_dump_json"):
            content_str = content.model_dump_json()
        else:
            content_str = str(content)

        job["messages"].append({"agent": source, "content": content_str})
        job["current_agent"] = source

        # Determine the real outcome ourselves, from the messages we're already seeing
        if source == "human_review_2":
            final_outcome = "completed" if "approve" in content_str.lower() else "rejected"
        elif source == "human_review_1" and "approve" not in content_str.lower():
            final_outcome = "rejected"

    job["status"] = final_outcome

    await model_client.close()

# ---------- API endpoints ----------

@app.post("/triage/{issue_key}")
async def start_triage(issue_key: str):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting",
        "current_agent": None,
        "review_event": asyncio.Event(),
        "review_decision": None,
        "messages": [],
    }
    asyncio.create_task(run_pipeline_job(job_id, issue_key))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (server may have restarted)")
    return {
        "job_id": job_id,
        "status": job["status"],
        "current_agent": job.get("current_agent"),
        "messages": job.get("messages", []),
    }


@app.post("/review/{job_id}")
async def submit_review(job_id: str, payload: ReviewDecision):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (server may have restarted)")
    job["review_decision"] = payload.decision
    job["review_event"].set()
    return {"job_id": job_id, "decision_submitted": payload.decision}
