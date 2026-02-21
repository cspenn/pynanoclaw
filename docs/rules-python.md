# RULES.MD - Python 3.11+ Development Standards

Target: Python 3.11+ on macOS with zsh. This document governs all code generation and modification.

---

## PART 1: FIRST PRINCIPLES

These principles take precedence over all other rules.

### P1: Fix Over Create
- Always update/fix existing code rather than creating new code
- **EXCEPTION**: When `radon cc` shows grade C or worse (CC > 10), refactor into smaller files
- Check complexity before deciding: `radon cc src/path/to/file.py -a`

### P2: Reusable Testing Infrastructure
- NO one-off diagnostic scripts
- Build a single static quality utility in `src/scripts` for all quality checks
- Add new checks to the master utility, don't create new scripts
- Unit tests belong in `tests/`

### P3: Documentation Location
- ALL documentation goes in `docs/` folder
- ONLY exception: CLAUDE.md (AI assistant guidance at project root)

### P4: Never Defer Necessary Work
- Clean, high-quality code is ALWAYS first priority
- No "we'll fix this later" or "good enough for now"
- Difficult/laborious work must be done immediately
- Never defer or ignore necessary work

### P5: Use Agents Whenever Possible
- Agents allow for separate context windows
- Agents allow for parallel work
- Always use agents for tasks without dependencies on a single context window

---

## PART 2: CORE STANDARDS

### 2.1 Project Structure

| Requirement | Rule |
|-------------|------|
| File markers | Start and end files with path comment: `# start src/example/file.py` |
| Imports | Absolute from source root only. No relative imports. |
| Entry point | `main.py` orchestrates only; no core logic. Invoke: `python -m {project}.main` |
| Complexity | No file exceeds radon cc grade B. Grades C/D/E MUST be refactored. |

**Required files:**
- `docs/prd.md`
- `checkpython.sh` (never modify)
- `.pre-commit-config.yaml`

### 2.2 Configuration

- **NEVER use environment variables** for configuration
- Use `config.yml` for general settings
- Use `credentials.yml` for secrets (must be in `.gitignore`)
- Provide `credentials.yml.dist` as placeholder
- **Never overwrite** user-set tokens/keys/passwords
- **Runtime validation**: All YAML config must parse into Pydantic models at startup
- Never hardcode variables or data in code

### 2.3 Command-Line Interface

- Use **Typer** for CLI
- `config.yml` is primary config source
- CLI flags **override** YAML settings (e.g., `--dry-run`)
- Other CLI argument forms for configuration are forbidden

### 2.4 Code Quality Principles

| Principle | Meaning |
|-----------|---------|
| DRY | Don't Repeat Yourself |
| SPOT | Single Point of Truth |
| SOLID | Single responsibility, Open/closed, Liskov, Interface segregation, Dependency inversion |
| GRASP | General Responsibility Assignment Software Patterns |
| YAGNI | You Aren't Gonna Need It |

**Rules:**
- Google Python Style for docstrings (PEP 257)
- Max two levels of nesting
- Clear, descriptive, unambiguous names
- Single responsibility per function/method
- Inline comments max two lines

### 2.5 Environment & Tooling

| Tool | Purpose |
|------|---------|
| Python | 3.11+ required |
| uv | Package/environment management (preferred over Poetry) |
| pyproject.toml | Dependency specification |
| pre-commit | Automated quality checks via `checkpython.sh` |

**uv commands:**
```zsh
uv init              # Initialize project
uv add <package>     # Add dependency
uv run <command>     # Run in virtual environment
uv sync              # Sync dependencies
```

---

## PART 3: QUALITY GATE CHECKLIST

### Tier 1 - Gate Checks (Must Pass Before Commit)

| Tool | Command | Purpose |
|------|---------|---------|
| ruff | `ruff check src/` | Fast linting |
| ruff | `ruff format src/` | Code formatting |
| mypy | `mypy src/` | Type checking |
| pytest | `pytest tests/` | Unit tests |
| deptry | `deptry src/` | Dependency analysis |

### Tier 2 - Quality Analysis

| Tool | Command | Purpose |
|------|---------|---------|
| radon | `radon cc src/ -a -nb` | Cyclomatic complexity |
| bandit | `bandit -r src/` | Security scanning |
| interrogate | `interrogate src/` | Docstring coverage |
| pylint | `pylint src/` | Deep linting |

### Tier 3 - Advanced

| Tool | Command | Purpose |
|------|---------|---------|
| xenon | `xenon src/ --max-absolute B` | Complexity enforcement |
| semgrep | `semgrep --config auto src/` | Security patterns |
| pip-audit | `pip-audit` | Dependency vulnerabilities |

### Specialized Tools

| Tool | Purpose |
|------|---------|
| hypothesis | Property-based testing |
| mutmut | Mutation testing |
| py-spy | Performance profiling |
| wily | Complexity trends |
| cohesion | Class cohesion metrics |
| pydeps | Dependency visualization |
| python-rope | Refactoring |
| pandera | DataFrame validation |
| instructor | LLM structured outputs |
| tenacity | Retry logic |
| docformatter | Docstring formatting |
| autopep8 | Targeted PEP8 fixes (preserves style) |

---

## PART 4: 8-DIMENSION QA FRAMEWORK

Evaluate code on these 8 dimensions:

| Dimension | Question |
|-----------|----------|
| Good | What's working correctly? |
| Bad | What's broken or incorrect? |
| Missing | What functionality is absent? |
| Unnecessary | What code/features are superfluous? |
| Fixed | What was repaired in this change? |
| Newly Broken | What previously worked but now fails? |
| Silent Errors | What hidden failures exist? |
| Overengineered | What is unnecessarily complex? |

---

## PART 5: IMPLEMENTATION STANDARDS

### 5.1 Type Hints (Python 3.11+ Syntax)

**Modern syntax (USE THIS):**
```python
def process(items: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    return result

def fetch(url: str) -> bytes | None:
    ...

from typing import Self
class Builder:
    def add(self, item: str) -> Self:
        return self
```

**Deprecated syntax (AVOID):**
```python
# DON'T USE:
from typing import List, Dict, Union, Optional
List[str]           # use list[str]
Dict[str, int]      # use dict[str, int]
Union[X, Y]         # use X | Y
Optional[X]         # use X | None
```

### 5.2 Core Libraries

| Domain | Library | Rule |
|--------|---------|------|
| Database | SQLAlchemy | All DB ops via Core or ORM. No raw SQL strings. |
| HTTP | HTTPX | All external HTTP requests. Sync client only. |
| Migrations | Alembic | All schema changes. No manual DB modifications. |
| Validation | Pydantic | All config/data validation at runtime. |

### 5.3 Logging & Error Handling

```python
import logging

# Log to timestamped file in logs/ AND console
# Configurable level in config.yml

# Emoji indicators:
logging.info("Processing complete")     # Prefix with appropriate indicator
logging.warning("Rate limit near")      # Use consistent markers
logging.error("Connection failed")      # for visual scanning
```

**Rules:**
- Define custom project-specific exceptions
- Use exponential backoff for network calls (tenacity)
- Log level must be configurable via config.yml

### 5.4 Testing

**pytest test categories:**
- Pydantic model validation
- SQLAlchemy database logic (use test database)
- HTTPX API interactions (use pytest-httpx for mocking)
- Alembic migrations
- Typer CLI overrides

**Critical rule:** When tests produce no output, assume complete failure. Never assume success from silence.

### 5.5 Progress & Reporting

- Use **tqdm** for loops with >5 steps or >10 seconds duration
- Reports: HTML, CSS, Tailwind, d3.js
- **Client-side rendering ONLY** - no server-side or external API calls

---

## PART 6: VUW METHODOLOGY

**Verifiable Units of Work** - micro-plans for disciplined debugging.

### Core Principles

1. **Extreme Granularity**: One file or one specific error per VUW
2. **Verification = Done**: Task incomplete until checklist passes
3. **Sequential Execution**: One VUW at a time; complete before next
4. **Clarity Over Conciseness**: Literal instructions, assume nothing

### VUW Template

```markdown
**VUW_ID:** [e.g., BUGFIX-001]

**Objective:** [One sentence explaining WHY this matters]

**Files:** [List of files to modify]

**Pre-Work Checkpoint:** git commit before any changes

**Steps:**
1. [Literal instruction with exact code/paths]
2. [Show changes as git diff format]
3. [...]

**Verification:**
- [ ] `./checkpython.sh` reports zero errors
- [ ] All tests pass
- [ ] `ruff check src/` clean
- [ ] `mypy src/` clean

**Post-Work Checkpoint:** git commit after verification passes
```

### Campaign Structure

Organize VUWs into campaigns by priority:

1. **Application Stability** - Fix blockers preventing `pytest` from running
2. **Type Safety** - Achieve zero `mypy` errors
3. **Code Quality** - Achieve zero `ruff` errors

---

## Python 3.11+ Features Reference

| Feature | Usage |
|---------|-------|
| `tomllib` | TOML parsing (stdlib) |
| `match-case` | Pattern matching |
| `except*` | Exception groups |
| Zero-cost exceptions | Performance-critical error handling |
| `Self` type | Class method return types |
| `dataclass(slots=True)` | Memory-efficient dataclasses |
| `contextlib.chdir` | Context manager for directory changes |

---

## Anti-Patterns (NEVER DO)

- Mutable default arguments
- Bare `except:` clauses
- Circular imports
- Global variable overuse
- Magic numbers without constants
- Hardcoded secrets
- Unvalidated external inputs
- Debugger remnants in production
- `typing.List`, `typing.Dict`, `typing.Union` imports (use builtins + `|`)
