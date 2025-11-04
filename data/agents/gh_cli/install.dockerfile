# Node.js v22 is installed in base.dockerfile
RUN npm install -g @github/copilot

# Verify installation
RUN copilot --version
