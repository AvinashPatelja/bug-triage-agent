import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

def get_bug_priority_policy(severity: str) -> str:
    """Returns the company policy for how fast a bug of a given severity must be fixed."""
    policy = {
        "Critical": "Must be fixed within 4 hours.",
        "High": "Must be fixed within 1-2 business day.",
        "Medium": "Must be fixed within current sprint",
        "Low": "No fixed SLA, fix when convenient.",
    }
    return policy.get(severity, "Unknown severity level.")

async def main():

    client = OpenAIChatCompletionClient(model='gpt-4o-mini')
    
    agent = AssistantAgent(
        name="policy_agent",
        model_client=client,
        system_message="""You answer questions about bug-fixing SLAs.
        Use the get_bug_priority_policy tool to check the actual policy - do not guess.""",
        tools=[get_bug_priority_policy],
    )

    response = await agent.run(task="A Low severity bug just came in. What's our SLA for it?")
    print(response.messages[-1].content)

    await client.close()

asyncio.run(main())