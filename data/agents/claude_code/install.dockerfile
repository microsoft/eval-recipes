# Install Claude Code
# Based on: https://github.com/anthropics/claude-code

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install -y nodejs

RUN npm install -g @anthropic-ai/claude-code

ENV BASH_DEFAULT_TIMEOUT_MS=300000
ENV BASH_MAX_TIMEOUT_MS=600000

RUN claude --version
