import os
from dotenv import load_dotenv
from fastapi import FastAPI

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()

app = FastAPI()

@app.get('/fetch/{issue_key}')
async def fetch_issue(issue_key: str):
    client = OpenAIChatCompletionClient(model='gpt-4o-mini')

    jira_server_params = StdioServerParams(
        command='mcp-atlassian',
        args=[],
        env={
            "JIRA_URL": os.environ["JIRA_URL"],
            "JIRA_USERNAME": os.environ["JIRA_USERNAME"],
            "JIRA_API_TOKEN": os.environ["JIRA_API_TOKEN"],
        },        
    )
    jira_tools = await mcp_server_tools(jira_server_params)

    fetcher_agent = AssistantAgent(
        name = 'fetcher_agent',
        model_client = client,
        system_message="""Use the jira_get_issue tool to fetch the requested Jira issue.
        Report back the issue's key, summary, and full description as plain text.""",
        tools=jira_tools
    )

    response = await fetcher_agent.run(task=f"Fetch jira issue {issue_key}")

    await client.close()

    return {
        "issue_key" : issue_key,
        "agent_response" : response.messages[-1].content
    } 