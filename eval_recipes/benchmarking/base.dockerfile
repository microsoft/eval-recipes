FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git

# Install UV: https://docs.astral.sh/uv/getting-started/installation/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /project

{{agent_installation}}

{{task_installation}}

# Most tests currently require the Claude Agent SDK
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install -y nodejs

RUN npm install -g @anthropic-ai/claude-code

ENV BASH_DEFAULT_TIMEOUT_MS=300000
ENV BASH_MAX_TIMEOUT_MS=600000

RUN claude --version
