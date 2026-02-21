# Fact-Check Audit Report

**Generated:** 2026-02-20
**Audited Document:** `README.md`
**Audit Scope:** Architecture claims, skill availability, "Agent Swarms first" superlative, file paths, URLs

---

## Executive Summary

20 claims analyzed. 18 fully grounded. 1 claim is wrong (skill listed as wanted but already implemented). 1 claim is accurate but belongs to the upstream project, not the fork.

---

## Section 1: Claims Grounded in Source Materials

### Verified Claims (Full Match)

| Claim | Source | Status |
|-------|--------|--------|
| "Single Python 3.11 asyncio process" | `pyproject.toml` (requires-python ≥3.11), `main.py` (asyncio throughout) | GROUNDED |
| Architecture diagram: neonize → SQLite → polling → Container → SDK | `whatsapp.py` (neonize imports), `db/operations.py` (SQLAlchemy/SQLite), `main.py` (polling loop), `container/agent-runner/src/main.py` (claude_agent_sdk import) | GROUNDED |
| All 8 key files exist at stated paths | Glob confirmed all paths | GROUNDED |
| "IPC via filesystem" | `ipc.py` docstring + directory polling pattern | GROUNDED |
| "Per-group message queue with concurrency control" | `queue.py` — `GroupQueue` with `asyncio.Semaphore` | GROUNDED |
| `docs/SECURITY.md` link | File confirmed at that path | GROUNDED |
| `assets/nanoclaw-logo.png` | File exists (217KB) | GROUNDED |
| `repo-tokens/badge.svg` | File exists (1.1KB SVG) | GROUNDED |
| `groups/*/CLAUDE.md` pattern | `groups/main/CLAUDE.md` and `groups/global/CLAUDE.md` confirmed | GROUNDED |
| `/setup` skill available | `.claude/skills/setup/` exists | GROUNDED |
| `/customize` skill available | `.claude/skills/customize/` exists | GROUNDED |
| `/debug` skill available | `.claude/skills/debug/` exists | GROUNDED |
| `/add-gmail` skill available | `.claude/skills/add-gmail/` exists | GROUNDED |
| `/convert-to-docker` skill available | `.claude/skills/convert-to-docker/` exists | GROUNDED |
| `/add-slack` correctly listed as RFS (not yet implemented) | Not found in `.claude/skills/` | GROUNDED |
| `/add-discord` correctly listed as RFS | Not found | GROUNDED |
| `/setup-windows` correctly listed as RFS | Not found | GROUNDED |
| `/add-clear` correctly listed as RFS | Not found | GROUNDED |

---

## Section 2: Claims Requiring Correction

### Wrong Claims

| Claim | Finding | Recommendation |
|-------|---------|----------------|
| `/add-telegram` listed as RFS (wanted, not yet implemented) | `/add-telegram` EXISTS at `.claude/skills/add-telegram/` | **Remove from RFS list.** Also note that `/add-telegram-swarm`, `/add-voice-transcription`, `/x-integration` all exist and are undocumented. |

### Claim That Requires Context Adjustment

| Claim | Finding | Recommendation |
|-------|---------|----------------|
| "First AI assistant to support Agent Swarms" | CORROBORATED — the original NanoClaw (upstream: gavrielc/nanoclaw) was confirmed by Hacker News and VentureBeat as the first personal AI assistant to support Claude Agent Swarms, releasing January 31, 2026. However, this claim belongs to the **upstream project**, not to this Python fork. | Rephrase to credit the upstream: "Supports Agent Swarms (a NanoClaw-original feature)" or simply list it as a feature without the "first" superlative, since the "first" credit belongs to gavrielc. |

---

## Section 3: Undocumented Skills (Present but Not in README)

These skills exist but are not mentioned anywhere in README.md or CONTRIBUTING.md:

| Skill | Location | Action |
|-------|----------|--------|
| `/add-telegram-swarm` | `.claude/skills/add-telegram-swarm/` | Add to "What It Supports" as optional integration |
| `/add-voice-transcription` | `.claude/skills/add-voice-transcription/` | Add to "What It Supports" as optional integration |
| `/x-integration` | `.claude/skills/x-integration/` | Add to "What It Supports" as optional integration |
| `/add-parallel` | `.claude/skills/add-parallel/` | Evaluate whether to document |
| `/factcheck`, `/qa-*`, `/writing-style-analysis` | dev/QA skills | These are development tools, not user-facing; no documentation needed in README |

---

## Section 4: Recommendations for Revision

### High Priority (Immediate Action)
1. **Remove `/add-telegram` from RFS** — it's already implemented
2. **Rephrase "First AI assistant to support Agent Swarms"** — the "first" claim belongs to the upstream project (gavrielc/nanoclaw). In Chris's fork, it should read: "Supports Agent Swarms — spin up teams of specialized agents that collaborate on complex tasks" with a note that NanoClaw pioneered this feature.

### Medium Priority (Should Address)
1. **Add implemented optional integrations to "What It Supports"** — `/add-voice-transcription` (voice note transcription), `/x-integration` (Twitter/X posting), `/add-telegram-swarm` (multi-bot Telegram) all exist and add meaningful value.

### Low Priority (Nice to Have)
1. **Update badge alt text** — "34.9k tokens, 17% of context window" in `repo-tokens/badge.svg` is a static value that may drift from reality. The badge itself auto-updates; consider removing the hardcoded alt text.

---

## Audit Methodology

- **Source Materials Checked:** pyproject.toml, main.py, whatsapp.py, db/operations.py, ipc.py, queue.py, container.py, scheduler.py, .claude/skills/ directory tree
- **Web Searches Conducted:** 4 (Agent Swarms claim, NanoClaw history, competing projects, Anthropic feature release timeline)
- **Total Claims Analyzed:** 20
- **Verification Rate:** 18/20 fully grounded; 1 wrong (add-telegram RFS); 1 contextually inaccurate (first claim belongs to upstream)

---

*Audit generated by /factcheck skill.*
