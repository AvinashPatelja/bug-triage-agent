import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()


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

    # Print every tool's name AND its parameter schema for jira_transition_issue specifically
    for tool in jira_tools:
        if tool.name == "jira_transition_issue":
            print("Tool name:", tool.name)
            print("Tool description:", tool.description)
            print("Tool schema:", tool.schema if hasattr(tool, "schema") else "no schema attr")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    transition_agent = AssistantAgent(
        name="transition_agent",
        model_client=model_client,
        system_message="""Call the jira_transition_issue tool right now to transition
        issue ABR-1 to transition id 21. Do this immediately, do not ask questions,
        do not check anything else first. Just call the tool with issue_key='ABR-1'
        and the appropriate transition parameter set to 21.""",
        tools=jira_tools,
    )

    response = await transition_agent.run(task="Transition ABR-1 using transition id 21 now.")

    for msg in response.messages:
        print(f"--- {msg.source} ---")
        print(msg.content)
        print()

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())