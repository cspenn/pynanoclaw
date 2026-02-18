# NanoClaw

Personal Claude assistant. See [README.md](README.md) for philosophy and setup. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for architecture decisions.

## Quick Context

Single Python 3.11 asyncio process that connects to WhatsApp, routes messages to Claude Agent SDK running in Apple Container (Linux VMs). Each group has isolated filesystem and memory.

## Key Files

| File | Purpose |
|------|---------|
| `src/nanoclaw/main.py` | Orchestrator: state, message loop, agent invocation |
| `src/nanoclaw/channels/whatsapp.py` | WhatsApp connection, auth, send/receive |
| `src/nanoclaw/ipc.py` | IPC watcher and task processing |
| `src/nanoclaw/router.py` | Message formatting and outbound routing |
| `src/nanoclaw/config.py` | Configuration (config.yml + credentials.yml) |
| `src/nanoclaw/container.py` | Spawns agent containers with mounts |
| `src/nanoclaw/scheduler.py` | Runs scheduled tasks |
| `src/nanoclaw/db/operations.py` | SQLite operations (via SQLAlchemy) |
| `groups/{name}/CLAUDE.md` | Per-group memory (isolated) |
| `container/skills/agent-browser.md` | Browser automation tool (available to all agents via Bash) |

## Skills

| Skill | When to Use |
|-------|-------------|
| `/setup` | First-time installation, authentication, service configuration |
| `/customize` | Adding channels, integrations, changing behavior |
| `/debug` | Container issues, logs, troubleshooting |

## Development

Run commands directly—don't tell the user to run them.

```bash
uv run python -m nanoclaw.main  # Run directly
uv sync                          # Install dependencies
uv run pytest tests/             # Run tests
./checkpython.sh                 # Quality gate
./container/build.sh             # Rebuild agent container
```

Service management (the plist uses `uv run python -m nanoclaw.main`):
```bash
launchctl load ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.plist
```

## Container Build Cache

Apple Container's buildkit caches the build context aggressively. `--no-cache` alone does NOT invalidate COPY steps — the builder's volume retains stale files. To force a truly clean rebuild:

```bash
container builder stop && container builder rm && container builder start
./container/build.sh
```

Always verify after rebuild: `container run -i --rm --entrypoint wc nanoclaw-agent:latest -l /app/src/main.py`
