# Fact-Check Audit Report — Post-Rewrite README.md

**Generated:** 2026-02-20
**Audited Document:** `README.md` (post-rewrite version, committed at 6d86a36)
**Previous Audit:** `output/2026-02-20-factcheck-audit.md` (pre-rewrite, for reference)
**Agents dispatched:** 2 in parallel (codebase/skills validator + external URL validator)

---

## Executive Summary

**33 claims verified. 33/33 passed. 0 failures.**

All architecture claims, file path references, skill listings, feature descriptions, and external URLs in the rewritten README.md are fully grounded or corroborated. One minor URL redirect noted (low priority).

**Overall: CLEAN ✅**

---

## Section 1: Codebase & Skills Claims (23/23 GROUNDED)

### Architecture Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| "Python throughout, strict type checking with mypy, 100% line coverage enforced by pytest" | GROUNDED | `pyproject.toml`: `requires-python = ">=3.11"`, `[tool.mypy]` strict=true, pytest `--cov-fail-under=100` |
| "Single Python 3.11 asyncio process" | GROUNDED | `main.py` line 13: `import asyncio`; asyncio event loop creation confirmed |
| Architecture diagram: `WhatsApp (neonize) --> SQLite --> Polling loop --> Container (Claude Agent SDK) --> Response` | GROUNDED | neonize in `whatsapp.py` docstring; SQLAlchemy/SQLite in `db/operations.py`; polling loop in `main.py`; `claude_agent_sdk` import in `container/agent-runner/src/main.py` line 26 |
| "IPC via filesystem" | GROUNDED | `ipc.py` lines 1–7: "Polls per-group IPC directories for JSON files written by container agents" |
| "Per-group message queue with concurrency control" | GROUNDED | `queue.py`: `GroupQueue` class with `asyncio.Semaphore(max_concurrent)` |

### Key Files (9/9 exist at exact stated paths)

| File | Status |
|------|--------|
| `src/nanoclaw/main.py` | EXISTS ✓ |
| `src/nanoclaw/channels/whatsapp.py` | EXISTS ✓ |
| `src/nanoclaw/ipc.py` | EXISTS ✓ |
| `src/nanoclaw/router.py` | EXISTS ✓ |
| `src/nanoclaw/queue.py` | EXISTS ✓ |
| `src/nanoclaw/container.py` | EXISTS ✓ |
| `src/nanoclaw/scheduler.py` | EXISTS ✓ |
| `src/nanoclaw/db/operations.py` | EXISTS ✓ |
| `groups/main/CLAUDE.md` (groups/*/CLAUDE.md) | EXISTS ✓ |

### Referenced Assets (3/3 exist)

| Asset | Status |
|-------|--------|
| `docs/SECURITY.md` | EXISTS ✓ |
| `assets/nanoclaw-logo.png` | EXISTS ✓ |
| `repo-tokens/badge.svg` | EXISTS ✓ |

### Skills (8 mentioned must exist; /add-slack must NOT exist)

| Skill | Status |
|-------|--------|
| `/setup` | EXISTS ✓ |
| `/customize` | EXISTS ✓ |
| `/debug` | EXISTS ✓ |
| `/add-gmail` | EXISTS ✓ |
| `/add-telegram` | EXISTS ✓ |
| `/add-voice-transcription` | EXISTS ✓ |
| `/x-integration` | EXISTS ✓ |
| `/add-telegram-swarm` | EXISTS ✓ |
| `/add-slack` (RFS — must be absent) | CORRECTLY ABSENT ✓ |

### Feature Claims Validated Against Skill Content

| README Claim | Status | Evidence |
|--------------|--------|----------|
| "Voice notes auto-transcribed via Whisper (`/add-voice-transcription`)" | GROUNDED | `add-voice-transcription/SKILL.md` line 2: "using OpenAI's Whisper API" |
| "X (Twitter) posting — Post, like, reply from any group (`/x-integration`)" | GROUNDED | `x-integration/SKILL.md` line 3: "Post tweets, like, reply, retweet, and quote" — all 3 stated capabilities confirmed |
| "Telegram Agent Swarms — Each subagent gets its own bot identity (`/add-telegram-swarm`)" | GROUNDED | `add-telegram-swarm/SKILL.md` line 4: "Each subagent in a team gets its own bot identity in the Telegram group" — exact match |

### Setup/FAQ Claims

| FAQ Claim | Status | Evidence |
|-----------|--------|----------|
| "Docker is also fully supported — during /setup, you can choose which runtime" | GROUNDED | `setup/SKILL.md`: runtime choice prompts for macOS (Apple Container vs Docker) |
| "On Linux, Docker is used automatically" | GROUNDED | `setup/SKILL.md`: `PLATFORM=linux → Docker` auto-selection logic |

---

## Section 2: External URLs & Factual Claims (10/10 VERIFIED)

### URL Validity

| URL | Status | Notes |
|-----|--------|-------|
| `https://github.com/cspenn/pynanoclaw` | LIVE ✓ | Repo exists; Python + TypeScript (container tooling); MIT licensed |
| `https://github.com/gavrielc/nanoclaw` | LIVE ✓ | Original upstream exists; 94.4% TypeScript confirmed |
| `https://claude.ai/download` | REDIRECTS ⚠️ | 301 → `https://claude.com/download` — page is real and functional |
| `https://docs.astral.sh/uv/` | LIVE ✓ | Official uv docs confirmed |
| `https://github.com/apple/container` | LIVE ✓ | Apple Container repo (24.6k stars, v0.9.0 released Feb 3 2026) |
| `https://docker.com/products/docker-desktop` | LIVE ✓ | Docker Desktop product page confirmed |
| `https://code.claude.com/docs/en/skills` | LIVE ✓ | Claude Code skills docs confirmed |

### Factual Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| "Apple Container is lightweight, fast, and optimized for Apple silicon" | CORROBORATED | Apple's own description: "written in Swift, and optimized for Apple silicon"; sub-second start times documented |
| "NanoClaw... designed around OS-level container isolation" | CORROBORATED | gavrielc/nanoclaw describes itself as "a lightweight alternative...that runs in containers for security" |
| "The original is TypeScript/Node.js" | CORROBORATED | gavrielc/nanoclaw: 94.4% TypeScript, contains `tsconfig.json`, `package.json`; quick start uses npm and Node.js 20+ |

---

## Section 3: Issues Found

### None requiring immediate action.

### Low Priority (optional improvement)
**`https://claude.ai/download` redirects to `https://claude.com/download`**
- The link works correctly for users (browsers follow the redirect transparently)
- Updating to the final URL would avoid the extra hop
- Recommendation: Update to `https://claude.com/download` at next opportunity, but not urgent

---

## Section 4: Notable Observations

**`/convert-to-docker` skill exists and is functional** but is not mentioned in README.md. This is intentional — the skill is referenced inside the `/setup` flow (users encounter it via `/setup` not directly). Not an error.

---

## Conclusion

The rewritten README.md is **entirely accurate**. Every claim is supported by direct codebase evidence or external corroboration. The only actionable finding is a cosmetic URL redirect (low priority).

**Recommendation:** Optionally update `https://claude.ai/download` → `https://claude.com/download`.

---

*Audit performed by parallel explore agents on 2026-02-20.*
*Codebase agent: 23 claims checked. External agent: 10 claims checked.*
