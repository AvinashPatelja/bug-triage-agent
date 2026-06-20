import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
import sys

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage, BaseChatMessage, StopMessage
from autogen_agentchat.base import TerminationCondition

load_dotenv()


class TriageResult(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    priority: Literal["P0", "P1", "P2", "P3"]
    root_cause: str
    assigned_team: Literal["Backend", "Frontend", "DevOps", "FullStack"]
    confidence: int

class FixProposal(BaseModel):
    explanation: str          # plain-English summary for the human reviewer
    file_path: str            # which file the fix targets
    code_fix: str             # the actual code change (snippet or diff)
    unit_test: str            # a test that verifies the fix
    risk_level: Literal["Low", "Medium", "High"]  # how risky is this change

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
                content = str(msg.content).lower()
                if "approve" in content:
                    return StopMessage(content="PR approved and merged. Pipeline complete.", source="PipelineComplete")
                else:
                    return StopMessage(content="PR rejected at final review. Pipeline stopped.", source="PipelineComplete")
            if msg.source == "human_review_1" and "approve" not in str(msg.content).lower():
                self._terminated = True
                return StopMessage(content="Rejected by human reviewer. Pipeline stopped.", source="PipelineComplete")
        return None

    async def reset(self) -> None:
        self._terminated = False

def custom_selector(messages):
    last_message = messages[-1]
    last_speaker = last_message.source

    # Fixed sequence for first 4 turns
    if last_speaker == "user":
        return "triage_agent"
    if last_speaker == "triage_agent":
        return "investigation_agent"
    if last_speaker == "investigation_agent":
        return "fix_agent"
    if last_speaker == "fix_agent":
        return "human_review_1"

    # The one real decision point
    if last_speaker == "human_review_1":
        content = str(last_message.content).lower()
        if "approve" in content:
            return "pr_agent"
        else:
            return None  # None tells AutoGen: no one speaks next, stop here

    if last_speaker == "pr_agent":
        return "human_review_2"          # <-- goes to second gate instead of stopping
    if last_speaker == "human_review_2":
        return None

    return None

def get_human_input(prompt: str) -> str:
    print("\n" + "-"*60)
    print("HUMAN REVIEW REQUIRED")
    print("-"*60)
    sys.stdout.flush()
    return input("Your decision (approve / reject): ")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Agent 1: Triage (same as before)
    triage_agent = AssistantAgent(
        name="triage_agent",
        model_client=model_client,
        system_message="""You are a Triage Agent for a software bug remediation system.
        Analyze the bug report and classify it according to the required schema.""",
        output_content_type=TriageResult,
    )

    # Agent 2: Investigation
    investigation_agent = AssistantAgent(
        name="investigation_agent",
        model_client=model_client,
        system_message="""You are an Investigation Agent for a software bug remediation system.
        You receive a Triage Agent's classification of a bug.
        Your job: hypothesize WHICH specific files/methods are likely involved,
        and describe what evidence (logs, code patterns) would confirm the root cause.
        Be specific and technical. List 2-4 files you'd check, with reasoning.""",
    )

    # Agent 3: Implementation - Code Fix - Test
    fix_agent = AssistantAgent(
        name="fix_agent",
        model_client=model_client,
        system_message="""You are a Fix Generator Agent for a software bug remediation system.
        You receive a bug triage classification and an investigation analysis.
        Your job: propose a concrete code fix and a unit test for it.
        Write the fix as a realistic C# code snippet appropriate for a .NET codebase.
        Be specific and production-quality. Assess the risk level of your proposed change honestly.
        End your message with the word TERMINATE.""",
        output_content_type=FixProposal,
    )

     # Human Review #1 — pauses for real human input
    human_review_1 = UserProxyAgent(
        name="human_review_1",
        input_func=get_human_input,
    )

    # Agent 4: Raise PR
    pr_agent = AssistantAgent(
        name="pr_agent",
        model_client=model_client,
        system_message="""You are a PR Creation Agent for a software bug remediation system.
        You receive an approved fix proposal.
        Your job: propose a git branch name, a PR title, a PR description (referencing the bug ID),
        and list the files that would be changed.
        This is a simulation - you are not actually creating a PR, just describing what it would look like.""",
        output_content_type=PRResult,
    )

    # Human Review #2 — PR approval - Merge
    human_review_2 = UserProxyAgent(
    name="human_review_2",
    input_func=get_human_input,
)

    # We are not using selector_prompt now rather we are have custom_selector function written
    selector_prompt = """You are coordinating a bug remediation pipeline with these roles:
    {roles}

    Conversation so far:
    {history}

    Decide who speaks next, following this EXACT sequence:
    1. triage_agent speaks first (always)
    2. investigation_agent speaks second (always)
    3. fix_agent speaks third (always)
    4. human_review_1 speaks fourth (always)
    5. AFTER human_review_1 speaks:
       - If their message contains the word "approve", select pr_agent next
       - If their message contains the word "reject", select nobody - the conversation should end
    6. After pr_agent speaks, the conversation should end

    Only return the name of the next speaker. Choose from: {participants}
    """

    # The termination condition: stop after the word TERMINATE appears,
    # OR after 4 messages total, whichever comes first (safety net)    
    #termination = TextMentionTermination("reject") | MaxMessageTermination(7)
    termination = PipelineComplete()

    # The TEAM: defines turn order + when to stop
    team = SelectorGroupChat(
        participants=[triage_agent, investigation_agent, fix_agent, human_review_1, pr_agent, human_review_2],
        model_client=model_client, # <- model_client NOT REQUIRED FOR "RoundRobinGroupChat"
        #selector_prompt=selector_prompt,
        selector_func=custom_selector,
        termination_condition=termination,
        custom_message_types=[StructuredMessage[TriageResult], StructuredMessage[FixProposal], StructuredMessage[PRResult]]
    )

    bug_report = """
    BUG-1042: NullReferenceException in PaymentService.ProcessRefund()
    Production error. Stack trace shows orderId exists but refund record is null.
    Affects ~12% of refund attempts since deploy v2.4.1.
    """

    print("\n" + "="*60)
    print("Starting Bug Remediation Pipeline for BUG-1042")
    print("="*60 + "\n")

    # Console() streams each agent's message live as it happens
    #await Console(team.run_stream(task=bug_report))
    result = await Console(team.run_stream(task=bug_report))

    # Check the final outcome and print a clear merge status
    #last_msg = result.messages[-1]

    # Find human_review_2's actual decision, not just the last message in the list
    final_decision = None
    for msg in result.messages:
        if msg.source == "human_review_2":
            final_decision = str(msg.content).lower()

    if final_decision and "approve" in final_decision:
        print("\n" + "="*60)
        print("✅ MERGED TO QA — Bug remediation complete!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ NOT MERGED — Pipeline stopped before completion.")
        print("="*60)

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())