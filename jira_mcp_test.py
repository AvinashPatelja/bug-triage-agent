import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()

class TriageResult(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    priority: Literal["P0", "P1", "P2", "P3"]
    root_cause: str
    assigned_team: Literal["Backend", "Frontend", "DevOps", "FullStack"]
    confidence: int

async def main():
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

    # Step 1: An agent that can use tools, NO structured output constraint
    fetcher_agent = AssistantAgent(
        name="fetcher_agent",
        model_client=model_client,
        system_message="""Use the jira_get_issue tool to fetch the requested Jira issue.
        Report back the issue's key, summary, and full description as plain text.""",
        tools=jira_tools,
    )

    fetch_response = await fetcher_agent.run(task="Fetch Jira issue ABR-1")
    raw_ticket_info = fetch_response.messages[-1].content
    print("--- Raw fetched ticket info ---")
    print(raw_ticket_info)

    # Step 2: A SEPARATE agent, no tools at all, just structures the text
    triage_agent = AssistantAgent(
        name="triage_agent",
        model_client=model_client,
        system_message="""You are a Triage Agent for a software bug remediation system.
        Analyze the provided bug report and classify it according to the required schema.""",
        output_content_type=TriageResult,   # safe now - no tools attached to this agent
    )

    triage_response = await triage_agent.run(task=raw_ticket_info)
    result = triage_response.messages[-1].content

    print("\n--- Structured triage result ---")
    print(result.model_dump_json(indent=2))

    await model_client.close()

asyncio.run(main())