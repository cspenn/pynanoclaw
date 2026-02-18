# NanoClaw — Product Requirements Document

## Overview

NanoClaw is a personal Claude assistant that connects WhatsApp to Claude agent containers.
A single Python process orchestrates WhatsApp message routing, agent container lifecycle,
scheduled task execution, and inter-process communication.

## Architecture

```
WhatsApp (Neonize) → Orchestrator (main.py) → Apple Container (asyncio.subprocess)
                          ↓                          ↓
                    SQLite (SQLAlchemy)        Agent (claude-agent-sdk)
                          ↓                          ↓
                    IPC Watcher (ipc.py)       Filesystem (/workspace/group/)
                          ↓
                    Task Scheduler (APScheduler)
```

## Core Requirements

### R1 — WhatsApp Integration
- Connect to WhatsApp using neonize (Go/whatsmeow backend)
- Receive messages and route them to registered groups
- Send responses back to the originating chat
- Support typing indicators while processing
- Persist QR auth state to disk (re-auth only on first run)
- Sync group metadata (names, JIDs) on startup and every 24 hours

### R2 — Agent Container Runner
- Spawn isolated Apple Container instances per group
- Pass input via stdin as JSON (prompt, session_id, group_folder, chat_jid, secrets)
- Parse streaming output via `---NANOCLAW_OUTPUT_START---` / `---NANOCLAW_OUTPUT_END---` markers
- Enforce hard timeout (default 30 minutes) with idle reset on each output
- Build volume mounts: group workspace, IPC dir, sessions dir, skills dir
- Pass secrets (ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN) via stdin only — never written to disk

### R3 — Group Queue (Concurrency Control)
- Limit concurrent containers to configurable max (default 5)
- Queue pending messages when a group's container is active
- Pipe follow-up messages to active containers via IPC file
- Retry failed processing with exponential backoff (max 5 retries)
- Drain waiting groups when concurrency slot opens

### R4 — IPC Watcher
- Poll per-group IPC directories every 1 second
- Process message files (send WhatsApp messages on behalf of agent)
- Process task files (schedule, pause, resume, cancel scheduled tasks)
- Enforce authorization: non-main groups can only act on themselves
- Move failed files to errors/ directory

### R5 — Task Scheduler
- Poll for due scheduled tasks every 60 seconds
- Support cron, interval, and once schedule types
- Run tasks in isolated or group context (context_mode)
- Log all runs to task_run_logs table
- Update next_run after each execution

### R6 — Database
- SQLite database at store/messages.db
- Managed via SQLAlchemy Core + Alembic migrations
- Tables: chats, messages, scheduled_tasks, task_run_logs, router_state, sessions, registered_groups
- All schema changes via Alembic (no inline ALTER TABLE)

### R7 — Configuration
- Non-secrets in config.yml (assistant name, timing, paths, container settings)
- Secrets in credentials.yml (gitignored; template at credentials.yml.dist)
- All config parsed into Pydantic models at startup — any missing key raises at boot

### R8 — Group Isolation
- Each group gets its own filesystem namespace (/workspace/group/)
- Each group has isolated Claude session state (data/sessions/{group}/.claude/)
- Each group has isolated IPC namespace (data/ipc/{group}/)
- Global memory directory (groups/global/) is read-only for non-main groups
- Main group gets full project mount for management capabilities

### R9 — Security
- Additional mounts validated against external allowlist (~/.config/nanoclaw/mount-allowlist.json)
- Allowlist is NOT mounted into containers (tamper-proof from agents)
- Secrets passed via stdin only — never written to disk or mounted
- Bash tool in containers strips secrets from subprocess environments
- Non-main groups cannot register other groups or see all available groups

## Non-Functional Requirements

### Performance
- Message polling interval: 2 seconds (configurable)
- Scheduler poll interval: 60 seconds (configurable)
- Container timeout: 30 minutes (configurable per group)
- Idle timeout: 30 minutes (resets on each agent output)
- Max concurrent containers: 5 (configurable)

### Reliability
- Startup recovery: check for unprocessed messages on boot
- Cursor rollback on agent error (prevents message loss)
- Outgoing message queue with flush on reconnect
- Graceful shutdown: detach (not kill) active containers

### Observability
- Structured logging to file (logs/nanoclaw.log) and console
- Per-container log files in groups/{name}/logs/
- Task run logs in database (task_run_logs table)

## Technology Choices

| Domain | Choice | Reason |
|--------|--------|--------|
| WhatsApp | neonize | Go/whatsmeow backend, active maintenance, Python bindings |
| Agent SDK | claude-agent-sdk (Python) | Official Python SDK for Claude agents |
| Database ORM | SQLAlchemy Core | Type-safe, no raw SQL, migration support |
| Migrations | Alembic | Industry standard, integrates with SQLAlchemy |
| Config | Pydantic + PyYAML | Runtime validation, type safety |
| Container mgmt | asyncio.create_subprocess_exec | Native async, no subprocess wrapper needed |
| Retry | tenacity | Exponential backoff, configurable |
| Scheduling | APScheduler | Production-grade, multiple trigger types |
| Cron parsing | croniter | Accurate timezone-aware cron calculation |

## Deployment

- macOS launchd service via ~/Library/LaunchAgents/com.nanoclaw.plist
- Managed with: `launchctl load/unload ~/Library/LaunchAgents/com.nanoclaw.plist`
- Start command: `uv run python -m nanoclaw.main`
- Working directory: project root

## File Structure

```
src/nanoclaw/
├── main.py          # Entry point + orchestrator
├── config.py        # Pydantic config models
├── types.py         # Pydantic domain models
├── router.py        # Message formatting + outbound routing
├── container.py     # Apple Container subprocess management
├── queue.py         # Concurrency control (GroupQueue)
├── ipc.py           # IPC file watcher + authorization
├── scheduler.py     # APScheduler task runner
├── db/
│   ├── models.py    # SQLAlchemy ORM models
│   └── operations.py # All DB functions
└── channels/
    ├── base.py      # Channel ABC
    └── whatsapp.py  # Neonize WhatsApp channel
```
