# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os
from typing import Literal

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI, DefaultAioHttpClient

load_dotenv(override=True)


@asynccontextmanager
async def create_client(
    provider: Literal["openai", "azure_openai"] = "azure_openai",
) -> AsyncGenerator[AsyncOpenAI | AsyncAzureOpenAI, None]:
    match provider:
        case "openai":
            client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                http_client=DefaultAioHttpClient(),
            )
        case "azure_openai":
            # Automatically choose between API key and Entra ID based on environment
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure OpenAI")

            azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if azure_api_key:
                client = AsyncAzureOpenAI(
                    api_key=azure_api_key,
                    api_version="2025-03-01-preview",
                    azure_endpoint=azure_endpoint,
                    http_client=DefaultAioHttpClient(),
                )
            else:
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                client = AsyncAzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version="2025-03-01-preview",
                    http_client=DefaultAioHttpClient(),
                )
        case _:
            raise ValueError(f"Unsupported provider: {provider}")

    try:
        yield client
    finally:
        await client.close()


if __name__ == "__main__":

    async def main() -> None:
        async with create_client() as client:
            response = await client.responses.create(
                model="gpt-5",
                instructions="You are a coding assistant that talks like a pirate.",
                input="How do I check if a Python object is an instance of a class?",
            )

            print(response.output_text)

    asyncio.run(main())
