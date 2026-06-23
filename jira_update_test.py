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

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    jira_updater_agent = AssistantAgent(
        name="jira_updater_agent",
        model_client=model_client,
        system_message="""You update Jira issues with AI triage findings. Follow these steps in order,
        and actually perform each one - do not stop after just checking:
    
        1. Use jira_add_comment to post the triage summary as a comment on the issue.
        2. Use jira_get_transitions to see available transitions for the issue.
        3. Find the transition with name "In Progress" in that list, and call jira_transition_issue
           using its id to actually move the issue to that state. You MUST call jira_transition_issue -
           checking the list alone is not sufficient.
    
        Report back exactly what you did, including the comment posted and the transition ID used.""",
        tools=jira_tools,
    )

    response = await jira_updater_agent.run(
        task="""Issue key: ABR-1
        Triage summary: Severity High, Priority P1, assigned to Backend team.
        Root cause: NullReferenceException in PaymentService.ProcessRefund() due to
        missing refund record. AI pipeline has generated a proposed fix, pending human review."""
    )

    print(response.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())