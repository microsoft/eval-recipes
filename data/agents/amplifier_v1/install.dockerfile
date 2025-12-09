# Install pnpm
RUN wget -qO- https://get.pnpm.io/install.sh | ENV="$HOME/.bashrc" SHELL="$(which bash)" bash -

# Set up pnpm environment variables
ENV PNPM_HOME="/root/.local/share/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

# Install make (required for Amplifier setup)
RUN apt-get update && apt-get install -y --no-install-recommends make

# Clone into a temp repo and then move Amplifier into /project
RUN git clone -b amplifier-claude https://github.com/microsoft/amplifier.git /tmp/amplifier && \
    mv /tmp/amplifier/* /tmp/amplifier/.[!.]* /project/ 2>/dev/null || true

RUN make install

# Add virtual environment to PATH so it's automatically used
ENV PATH="/project/.venv/bin:$PATH"
