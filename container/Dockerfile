# NanoClaw Agent Container (Python 3.11)
# Runs Claude Agent SDK (Python) in isolated Linux VM with browser automation
# NOTE: claude-agent-sdk wraps the claude CLI, so Node.js + @anthropic-ai/claude-code
#       must still be present even though the agent runner itself is Python.

FROM python:3.11-slim

# Install system dependencies for Chromium and Node.js
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    libdrm2 \
    libxshmfence1 \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Install agent-browser and claude-code globally (required by claude-agent-sdk)
RUN npm install -g agent-browser @anthropic-ai/claude-code

# Create app directory
RUN mkdir -p /app/src

# Install Python dependencies
COPY agent-runner/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/extra /workspace/ipc/messages /workspace/ipc/tasks /workspace/ipc/input

# Create entrypoint script
# Python agent runner reads stdin directly via sys.stdin — no temp file needed
RUN printf '#!/bin/bash\nset -e\ncd /app && exec python /app/src/main.py\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
# Keep node user for compatibility with --dangerously-skip-permissions
RUN chown -R node:node /workspace /app/src

# Switch to non-root user
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
