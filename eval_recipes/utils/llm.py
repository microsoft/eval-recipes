# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os
from typing import Literal

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI, DefaultAioHttpClient
import tiktoken

load_dotenv()


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
            azure_endpoint_version = os.environ.get("AZURE_OPENAI_VERSION", "2024-10-01-preview")
            if azure_api_key:
                client = AsyncAzureOpenAI(
                    api_key=azure_api_key,
                    api_version=azure_endpoint_version,
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
                    api_version=azure_endpoint_version,
                    http_client=DefaultAioHttpClient(),
                )
        case _:
            raise ValueError(f"Unsupported provider: {provider}")

    try:
        yield client
    finally:
        await client.close()


def truncate_reports_to_token_limit(reports: list[str], max_total_tokens: int = 120000) -> list[str]:
    """Truncate reports so their combined token count fits within max_total_tokens.

    Each report is allocated an equal share of the token budget. Reports shorter than
    their allocation keep their full content; reports longer are truncated from the
    beginning to keep recent content.

    Args:
        reports: List of report strings to truncate
        max_total_tokens: Maximum total tokens for all reports combined

    Returns:
        List of (possibly truncated) report strings
    """
    if not reports:
        return reports

    encoding = tiktoken.get_encoding("o200k_base")

    # Count tokens for each report
    report_tokens = [encoding.encode(report) for report in reports]
    total_tokens = sum(len(tokens) for tokens in report_tokens)

    # If within limit, return as-is
    if total_tokens <= max_total_tokens:
        return reports

    # Calculate fair share per report
    tokens_per_report = max_total_tokens // len(reports)

    # Truncate each report to its fair share
    truncated_reports = []
    for tokens in report_tokens:
        if len(tokens) <= tokens_per_report:
            truncated_reports.append(encoding.decode(tokens))
        else:
            # Keep the last tokens_per_report tokens (truncate from beginning)
            truncated_tokens = tokens[-tokens_per_report:]
            truncated_reports.append(encoding.decode(truncated_tokens))

    return truncated_reports


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
