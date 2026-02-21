# Contributing

## Source Code Changes

**Accepted:** Bug fixes, security fixes, simplifications, reducing code.

**Not accepted:** Features, capabilities, compatibility, enhancements. These should be skills.

## Quality Standards

All source code changes must pass the full quality gate before merging:

```bash
./checkpython.sh   # Runs all Tier 1–3 checks
```

The gate enforces:

| Check | Requirement |
|-------|-------------|
| `ruff` | Zero lint errors |
| `ruff format` | Consistent formatting |
| `mypy` | No type errors (strict) |
| `pytest` | All tests pass, **100% line coverage** |
| `bandit` | No high-severity security issues |
| `radon cc` | No Grade C functions (CC ≥ 11) |
| `xenon` | `--max-absolute B --max-modules A --max-average A` |
| `vulture` | No unused code at 80% confidence |
| `interrogate` | ≥ 80% docstring coverage |
| `deptry` | No missing or unused dependencies |
| `pip-audit` | No known CVEs |

When adding or modifying source code, include tests that keep coverage at 100%. Run `uv run pytest tests/ --cov=src --cov-report=term-missing` to find gaps.

## Skills

A [skill](https://code.claude.com/docs/en/skills) is a markdown file under `.claude/skills/<skill-name>/` that teaches Claude Code how to transform a NanoClaw installation.

A skill PR should contain only the skill file(s), not source code changes.

Your skill should contain the **instructions** Claude follows to add the feature—not pre-built code. Look at existing skills in `.claude/skills/` for reference on structure and style.

### Why?

Every user should have clean and minimal code that does exactly what they need. Skills let users selectively add features to their fork without inheriting code for features they don't want.

## RFS (Request for Skills)

The following skills are wanted. If you build one, open a PR at [github.com/cspenn/pynanoclaw/pulls](https://github.com/cspenn/pynanoclaw/pulls).

### Communication Channels

- `/add-slack` — Add Slack as input/output channel
- `/add-discord` — Add Discord as input/output channel
- `/add-sms` — Add SMS via Twilio or similar

### Platform Support

- `/setup-windows` — Windows via WSL2 + Docker

### Session Management

- `/add-clear` — Add a `/clear` command that compacts the conversation (summarizes context while preserving critical information). Requires figuring out how to trigger compaction programmatically via the Claude Agent SDK.

## Testing

Test your skill by running it on a fresh clone before submitting.
