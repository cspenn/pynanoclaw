<p align="center">
  <img src="assets/nanoclaw-logo.png" alt="NanoClaw" width="400">
</p>

<p align="center">
  Christopher Penn's Python fork of NanoClaw — a personal Claude assistant running securely in containers.
</p>

<p align="center">
  <a href="https://github.com/cspenn/pynanoclaw/issues"><img src="https://img.shields.io/github/issues/cspenn/pynanoclaw" alt="Issues" valign="middle"></a>&nbsp; • &nbsp;
  <img src="repo-tokens/badge.svg" alt="Token count" valign="middle">
</p>

## Fork of NanoClaw

[NanoClaw](https://github.com/gavrielc/nanoclaw) is a great project — lightweight, auditable, and designed around OS-level container isolation rather than application-level permission checks. I forked it to create a pure Python 3.11 version. The original is TypeScript/Node.js; this version uses Python throughout, enabling Python-native tooling, strict type checking with mypy, and a test suite at 100% line coverage enforced by pytest.

## Quick Start

```bash
git clone https://github.com/cspenn/pynanoclaw.git
cd pynanoclaw
claude
```

Then run `/setup`. Claude Code handles everything: dependencies, authentication, container setup, service configuration.

## Philosophy

**Small enough to understand.** One process, a few source files. No microservices, no message queues, no abstraction layers. Have Claude Code walk you through it.

**Secure by isolation.** Agents run in Linux containers (Apple Container on macOS, or Docker). They can only see what's explicitly mounted. Bash access is safe because commands run inside the container, not on your host.

**Built for one user.** This isn't a framework. It's working software that fits one user's exact needs. You fork it and have Claude Code make it match your exact needs.

**Customization = code changes.** No configuration sprawl. Want different behavior? Modify the code. The codebase is small enough that this is safe.

**AI-native.** No installation wizard; Claude Code guides setup. No monitoring dashboard; ask Claude what's happening. No debugging tools; describe the problem, Claude fixes it.

**Skills over features.** Contributors shouldn't add features (e.g. Slack support) to the codebase. Instead, they contribute [Claude Code skills](https://code.claude.com/docs/en/skills) like `/add-slack` that transform your fork. You end up with clean code that does exactly what you need.

**Best harness, best model.** This runs on Claude Agent SDK, which means you're running Claude Code directly. The harness matters. A bad harness makes even smart models seem dumb, a good harness gives them superpowers. Claude Code is the best harness available.

## What It Supports

- **WhatsApp I/O** - Message Claude from your phone
- **Isolated group context** - Each group has its own `CLAUDE.md` memory, isolated filesystem, and runs in its own container sandbox with only that filesystem mounted
- **Main channel** - Your private channel (self-chat) for admin control; every other group is completely isolated
- **Scheduled tasks** - Recurring jobs that run Claude and can message you back
- **Web access** - Search and fetch content
- **Container isolation** - Agents sandboxed in Apple Container (macOS) or Docker (macOS/Linux)
- **Agent Swarms** - Spin up teams of specialized agents that collaborate on complex tasks
- **Optional integrations** - Add Gmail (`/add-gmail`), Telegram (`/add-telegram`), and more via skills
- **Voice note transcription** - WhatsApp voice messages auto-transcribed via Whisper (`/add-voice-transcription`)
- **X (Twitter) posting** - Post, like, reply from any group (`/x-integration`)
- **Telegram Agent Swarms** - Each subagent gets its own bot identity (`/add-telegram-swarm`)

## Usage

Talk to your assistant with the trigger word (default: `@chris`):

```
@chris send an overview of the sales pipeline every weekday morning at 9am
@chris review the git history for the past week each Friday and update the README if there's drift
@chris every Monday at 8am, compile news on AI developments from Hacker News and TechCrunch and message me a briefing
```

From the main channel (your self-chat), you can manage groups and tasks:
```
@chris list all scheduled tasks across groups
@chris pause the Monday briefing task
@chris join the Family Chat group
```

## Customizing

There are no configuration files to learn. Just tell Claude Code what you want:

- "Change the trigger word to @Bob"
- "Remember in the future to make responses shorter and more direct"
- "Add a custom greeting when I say good morning"
- "Store conversation summaries weekly"

Or run `/customize` for guided changes.

The codebase is small enough that Claude can safely modify it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, quality standards, and the Request for Skills list.

## Requirements

- macOS or Linux
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Claude Code](https://claude.ai/download)
- [Apple Container](https://github.com/apple/container) (macOS) or [Docker](https://docker.com/products/docker-desktop) (macOS/Linux)

## Architecture

```
WhatsApp (neonize) --> SQLite --> Polling loop --> Container (Claude Agent SDK) --> Response
```

Single Python 3.11 asyncio process. Agents execute in isolated Linux containers with mounted directories. Per-group message queue with concurrency control. IPC via filesystem.

Key files:
- `src/nanoclaw/main.py` - Orchestrator: state, message loop, agent invocation
- `src/nanoclaw/channels/whatsapp.py` - WhatsApp connection, auth, send/receive
- `src/nanoclaw/ipc.py` - IPC watcher and task processing
- `src/nanoclaw/router.py` - Message formatting and outbound routing
- `src/nanoclaw/queue.py` - Per-group queue with global concurrency limit
- `src/nanoclaw/container.py` - Spawns streaming agent containers
- `src/nanoclaw/scheduler.py` - Runs scheduled tasks
- `src/nanoclaw/db/operations.py` - SQLite operations (via SQLAlchemy)
- `groups/*/CLAUDE.md` - Per-group memory

## FAQ

**Why WhatsApp and not Telegram/Signal/etc?**

WhatsApp is what I use. Fork it and run `/customize` to switch channels — Telegram support is available via `/add-telegram`.

**Why Apple Container instead of Docker?**

On macOS, Apple Container is lightweight, fast, and optimized for Apple silicon. But Docker is also fully supported — during `/setup`, you can choose which runtime to use. On Linux, Docker is used automatically.

**Can I run this on Linux?**

Yes. Run `/setup` and it will automatically configure Docker as the container runtime.

**Is this secure?**

Agents run in containers, not behind application-level permission checks. They can only access explicitly mounted directories. You should still review what you're running, but the codebase is small enough that you actually can. See [docs/SECURITY.md](docs/SECURITY.md) for the full security model.

**Why no configuration files?**

There's no configuration sprawl by design. Every user should customize the code to match exactly what they want rather than configuring a generic system. If you like having config files, tell Claude to add them.

**How do I debug issues?**

Ask Claude Code. "Why isn't the scheduler running?" "What's in the recent logs?" "Why did this message not get a response?" That's the AI-native approach.

**Why isn't the setup working for me?**

Run `claude`, then run `/debug`. If the issue is likely affecting other users, open an issue on GitHub at https://github.com/cspenn/pynanoclaw/issues so it can be fixed in the setup skill.

**What changes will be accepted into the codebase?**

Security fixes, bug fixes, and clear improvements to the base configuration. That's it.

Everything else (new capabilities, OS compatibility, hardware support, enhancements) should be contributed as skills.

This keeps the base system minimal and lets every user customize their installation without inheriting features they don't want.

## Community

Questions? Ideas? [Open an issue on GitHub](https://github.com/cspenn/pynanoclaw/issues).

## License

MIT
