import asyncio
import os
import httpx
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


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


async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    code_reader_agent = AssistantAgent(
        name="code_reader_agent", model_client=model_client,
        system_message="""You are a Code Reader Agent. Given an investigation's hypothesis
        about which file is likely involved in a bug, use the get_github_file_contents tool
        to fetch that file's real contents. Report back the exact file contents you retrieved,
        plus the filename. If you're unsure of the exact filename, try 'PaymentService.cs'.""",
        tools=[get_github_file_contents],
    )

    response = await code_reader_agent.run(
        task="The investigation suggests the bug is in the payment/refund processing file."
    )

    print(response.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())