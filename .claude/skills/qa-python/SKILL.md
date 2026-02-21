---
name: qa-python
description: Run code quality audit when user asks to check code quality, run QA, verify code before commit, check for antipatterns, run linting/type checks, or assess code health. Executes tiered quality checks (ruff, mypy, pytest, deptry, radon, bandit) and scans for antipatterns.
---

# Python Codebase Quality Assessment (8 Dimensions)

You are performing a comprehensive quality assessment of this Python codebase. Generate a detailed report evaluating 8 key dimensions.

## Setup

First, capture the timestamp and branch information for the report:

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BRANCH=$(git branch --show-current)
echo "Timestamp: $TIMESTAMP"
echo "Branch: $BRANCH"
```

## Step 1: Gather Context

### Project Structure
```bash
find src/ -name "*.py" | head -30
```

### Files Changed from Main
```bash
git diff main...HEAD --name-only -- '*.py' 2>/dev/null || echo "Unable to diff against main (branch may not exist or no changes)"
```

### Git Diff Summary
```bash
git diff main...HEAD --stat -- '*.py' 2>/dev/null || echo "No diff available"
```

## Step 2: Run Quality Tools

Run each tool and capture output. Tools that aren't installed will be skipped gracefully.

### Ruff (Linting & Style)
```bash
ruff check src/ 2>&1 || true
```

### Ruff Statistics
```bash
ruff check src/ --statistics 2>&1 || true
```

### MyPy (Type Checking)
```bash
mypy src/ 2>&1 || true
```

### Pyright (Additional Type Checking)
```bash
pyright src/ 2>&1 || echo "Pyright not installed"
```

### Bandit (Security)
```bash
bandit -r src/ -f txt 2>&1 || true
```

### Radon Cyclomatic Complexity
```bash
radon cc src/ -a -s 2>&1 || echo "Radon not installed"
```

### Radon Maintainability Index
```bash
radon mi src/ -s 2>&1 || echo "Radon not installed"
```

### Cohesion (God Class Detection)
```bash
cohesion --below=50 src/ 2>&1 || echo "Cohesion not installed"
```

### Refurb (Modernization)
```bash
refurb src/ 2>&1 || echo "Refurb not installed"
```

### Deptry (Dependencies)
```bash
deptry . 2>&1 || echo "Deptry not installed"
```

### Vulture (Dead Code)
```bash
vulture src/ --min-confidence 80 2>&1 || echo "Vulture not installed"
```

### Interrogate (Docstring Coverage)
```bash
interrogate src/ -v 2>&1 || echo "Interrogate not installed"
```

### Pytest Coverage
```bash
pytest tests/ --cov=src --cov-report=term-missing -q 2>&1 || echo "Tests failed or pytest not configured"
```

## Step 3: Silent Error Detection

### Bare Except Clauses
```bash
grep -rn "except:" src/ --include="*.py" 2>&1 || echo "None found"
```

### Empty Exception Handlers (except followed by pass)
```bash
grep -rn -A1 "except" src/ --include="*.py" | grep -B1 "pass$" 2>&1 || echo "None found"
```

### TODO/FIXME/HACK Comments
```bash
grep -rn "TODO\|FIXME\|XXX\|HACK" src/ --include="*.py" 2>&1 || echo "None found"
```

### Print Statements (should use logging)
```bash
grep -rn "print(" src/ --include="*.py" 2>&1 || echo "None found"
```

### Unused Variables (underscore prefix check)
```bash
ruff check src/ --select F841 2>&1 || true
```

## Step 4: Analysis Instructions

Based on all the tool outputs above, create a comprehensive quality assessment report following the 8-dimension framework below. For each dimension:

- Provide specific findings with file paths and line numbers when available
- Assess severity (Critical/High/Medium/Low/Info)
- Give concrete, actionable recommendations

### Dimension Analysis Framework

1. **What's Good**: Identify passing checks, good patterns, well-tested code, proper type hints, clean architecture decisions, high coverage areas

2. **What's Bad**: List all errors, violations, security issues, and failures from the tools. Categorize by severity.

3. **What's Missing**: Note missing docstrings, missing tests, missing type hints, missing error handling, gaps in coverage

4. **What's Unnecessary**: Dead code, unused imports, unused dependencies, redundant patterns, obsolete code

5. **What's Fixed (vs main)**: Based on git diff, identify what issues were resolved on this branch compared to main

6. **What's Newly Broken (vs main)**: Based on git diff, identify new issues introduced since diverging from main

7. **Silent Errors**: Bare excepts, swallowed exceptions, ignored return values, missing error handling, code that fails silently

8. **Overengineered**: Functions with high cyclomatic complexity (C, D, E, F grades), God classes (low cohesion), unnecessary abstractions, overly complex patterns

## Step 5: Generate and Save Report

Create the report in the following markdown format and save it to `docs/reports/qa-{TIMESTAMP}.md`:

```markdown
# Quality Assessment Report

**Project:** tube-vacuum
**Date:** {TIMESTAMP}
**Branch:** {BRANCH}
**Compared Against:** main

---

## Executive Summary

[2-3 sentence overall assessment of codebase health]

**Overall Health Score:** [A/B/C/D/F] - [Brief justification]

| Dimension | Status | Issues Found |
|-----------|--------|--------------|
| Good | [status emoji] | [summary] |
| Bad | [status emoji] | [count] |
| Missing | [status emoji] | [count] |
| Unnecessary | [status emoji] | [count] |
| Fixed | [status emoji] | [count] |
| Newly Broken | [status emoji] | [count] |
| Silent Errors | [status emoji] | [count] |
| Overengineered | [status emoji] | [count] |

---

## 1. What's Good

[List positive findings - passing checks, good patterns, well-documented code, high test coverage areas]

## 2. What's Bad

### Critical
[Critical issues requiring immediate attention]

### High
[High severity issues]

### Medium
[Medium severity issues]

### Low
[Low severity issues]

## 3. What's Missing

[Missing documentation, tests, type hints, error handling]

## 4. What's Unnecessary

[Dead code, unused imports, unused dependencies]

## 5. What's Fixed (since main)

[Issues resolved on this branch - or "No comparison available" if on main]

## 6. What's Newly Broken (since main)

[New issues introduced - or "No comparison available" if on main]

## 7. Silent Errors

[Bare excepts, swallowed exceptions, ignored return values]

## 8. Overengineered

[High complexity functions, God classes, unnecessary abstractions]

---

## Recommendations

### High Priority
1. [Most critical fixes needed]

### Medium Priority
1. [Important improvements]

### Low Priority
1. [Nice-to-have enhancements]

---

## Tool Output Summary

| Tool | Status | Issues/Notes |
|------|--------|--------------|
| ruff | [pass/fail] | [issue count or notes] |
| mypy | [pass/fail] | [issue count or notes] |
| pyright | [pass/fail/skipped] | [issue count or notes] |
| bandit | [pass/fail] | [issue count or notes] |
| radon cc | [pass/fail/skipped] | [average grade] |
| radon mi | [pass/fail/skipped] | [average score] |
| cohesion | [pass/fail/skipped] | [issue count or notes] |
| refurb | [pass/fail/skipped] | [issue count or notes] |
| deptry | [pass/fail/skipped] | [issue count or notes] |
| vulture | [pass/fail/skipped] | [issue count or notes] |
| interrogate | [pass/fail/skipped] | [coverage %] |
| pytest | [pass/fail] | [test count, coverage %] |

---

*Generated by /qa slash command on {TIMESTAMP}*
```

After generating the report content, save it to the file path shown above.

$ARGUMENTS
