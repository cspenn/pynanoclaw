# Python Migration Reference

## Overview

NanoClaw was migrated from TypeScript (Node.js) to Python 3.11. The core architecture is unchanged: a single async process connects to WhatsApp, routes messages to Claude agents running in Apple Container (Linux VMs), with per-group isolated filesystems and memory. The migration replaces every Node.js dependency with a Python equivalent while preserving the same SQLite schema, file-based IPC protocol, and container interface.

## Stack Changes

| Component | TypeScript | Python 3.11 |
|---|---|---|
| Package manager | npm | uv |
| Database | better-sqlite3 (raw) | SQLAlchemy Core + Alembic |
| Config | .env + process.env | config.yml + credentials.yml + Pydantic models |
| WhatsApp | @whiskeysockets/baileys | neonize (Go/whatsmeow Python bindings) |
| Agent SDK | @anthropic-ai/claude-agent-sdk | claude-agent-sdk (Python) |
| Scheduling | cron-parser + custom loop | croniter + asyncio polling |
| Container mgmt | Node.js spawn() | asyncio.create_subprocess_exec() |
| Testing | vitest | pytest + pytest-asyncio |
| Linting | (none) | ruff + mypy + bandit |

## Key Decisions

**ClaudeSDKClient vs query()**: The Python SDK exposes two interfaces. `query()` is simpler but does not support hooks. `ClaudeSDKClient` is required wherever hooks are needed (e.g., `PreCompact`, `PreToolUse`). The container agent runner uses `ClaudeSDKClient` for this reason.

**Sync SQLAlchemy**: Database calls use synchronous SQLAlchemy Core, matching the better-sqlite3 pattern. SQLite operations are fast enough that async DB drivers add complexity without benefit.

**asyncio throughout**: The entire process runs on a single asyncio event loop. This avoids shared-state race conditions without requiring locks on most data structures.

**File-based IPC preserved**: The same IPC protocol is used — the container writes task files to a watched directory and the host polls for them. Only the polling implementation changed (Node fs.watch → Python pathlib polling).

**Same SQLite schema**: Alembic creates and migrates the database on startup. Existing data is preserved across upgrades.

## Project Structure

```
src/nanoclaw/
  main.py           # Entry point, event loop setup
  config.py         # Pydantic config/credentials models
  db.py             # SQLAlchemy Core operations
  channels/
    whatsapp.py     # neonize connection, auth, send/receive
  ipc.py            # Pathlib polling, task processing
  router.py         # Message formatting and outbound routing
  container_runner.py  # Spawns agent containers
  task_scheduler.py    # croniter-based scheduled tasks
  group_queue.py       # Per-group concurrency via asyncio.Semaphore
```

## Getting Started

```bash
uv sync                                    # Install deps
alembic upgrade head                       # Create/migrate database
cp credentials.yml.dist credentials.yml   # Add API keys
uv run python -m nanoclaw.main            # Start
uv run pytest tests/                      # Run tests
./checkpython.sh                          # Full quality gate
```

## Architecture Notes

**GroupQueue**: Each group has an `asyncio.Semaphore` enforcing a concurrency limit. Slots are claimed eagerly before task creation — this prevents a race where two messages arrive simultaneously and both bypass the limit before either task is recorded.

**IPC**: `pathlib.Path` polling runs every 1 second. Incoming task files are validated with Pydantic, authorization is checked (main group vs non-main group permissions), and tasks are dispatched to the appropriate handler.

**Container agent runner**: The container process reads a JSON payload from stdin, runs `ClaudeSDKClient` with hooks attached, emits `OUTPUT_START`...`OUTPUT_END` delimiters around agent output, then polls the IPC directory for follow-up messages. This loop continues until a `_close` sentinel is received.

## Legacy Code

The original TypeScript source is archived in `_archive/`. It is kept for reference and is excluded from all quality tools (ruff, mypy, bandit, pytest).
