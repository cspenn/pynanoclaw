---
name: qa-chrome
description: Run comprehensive Chrome Extension code quality audit. Executes tiered quality checks (Biome, TypeScript, Vitest, Knip) and scans for MV3-specific anti-patterns (CSP violations, blocking webRequests, console logs). Generates an 8-dimension health report.
---

# Chrome Extension Codebase Quality Assessment (8 Dimensions)

You are performing a comprehensive quality assessment of this Chrome Extension (Manifest V3) codebase. Generate a detailed report evaluating 8 key dimensions, specifically focusing on the unique constraints of browser extensions (Service Worker lifecycle, Content Security Policy, and DOM isolation).

## Setup

Capture timestamp and branch info:

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BRANCH=$(git branch --show-current)
echo "Timestamp: $TIMESTAMP"
echo "Branch: $BRANCH"
```

## Step 1: Gather Context

### Extension Structure
```bash
# locate key MV3 entry points and structure
find src/ -name "manifest.json" -o -name "background.ts" -o -name "content.ts" -o -name "*.tsx" | head -30
```

### Files Changed from Main
```bash
git diff main...HEAD --name-only -- '*.ts' '*.tsx' '*.json' 2>/dev/null || echo "Unable to diff against main"
```

## Step 2: Run Quality Tools

Run each tool using `pnpm` (the required package manager). Tools that fail or aren't installed should be skipped gracefully.

### Biome (Linting, Formatting & Import Sorting)
*Replaces Ruff/Black/Isort.*
```bash
pnpm biome check src/ --reporter=summary 2>&1 || echo "Biome checks failed or issues found"
```

### TypeScript Compiler (Strict Type Checking)
*Replaces MyPy.*
```bash
pnpm tsc --noEmit --pretty 2>&1 || true
```

### Knip (Dead Code & Unused Dependencies)
*Replaces Deptry & Vulture.*
```bash
pnpm knip --no-exit-code 2>&1 || echo "Knip not installed"
```

### Vitest (Unit Testing)
*Replaces Pytest.*
```bash
pnpm vitest run --coverage --reporter=verbose 2>&1 || echo "Tests failed"
```

### Dependency Audit (Security)
*Replaces Bandit/Pip-audit.*
```bash
pnpm audit --audit-level=high 2>&1 || true
```

## Step 3: MV3 Anti-Pattern Detection (Grepping)

### Forbidden CSP Violations
*Manifest V3 strictly forbids `eval()` and remote code execution.*
```bash
grep -rn "eval(" src/ --include="*.ts*" 2>&1 || echo "None found"
grep -rn "new Function(" src/ --include="*.ts*" 2>&1 || echo "None found"
grep -rn "innerHTML" src/ --include="*.ts*" 2>&1 || echo "None found (Check for dangerouslySetInnerHTML in React)"
```

### Deprecated MV2 APIs
*Scanning for legacy blocking webRequest API which causes rejection.*
```bash
grep -rn "chrome.webRequest.onBeforeRequest" src/ --include="*.ts" 2>&1 || echo "None found"
grep -rn "chrome.browserAction" src/ --include="*.ts" 2>&1 || echo "None found (Should be chrome.action)"
```

### Console Pollution
*Console logs in production extensions are performance leaks.*
```bash
grep -rn "console.log" src/ --include="*.ts*" 2>&1 || echo "None found"
```

### "Any" Type Usage (Strictness)
```bash
grep -rn ": any" src/ --include="*.ts*" 2>&1 || echo "None found"
```

### Promise Hygiene (Floating Promises)
*Detects .then() usage which is discouraged in favor of async/wait.*
```bash
grep -rn "\.then(" src/ --include="*.ts*" 2>&1 || echo "None found"
```

## Step 4: Analysis Instructions

Create a comprehensive quality assessment report following the 8-dimension framework. Adapt the analysis specifically for Chrome Extensions:

### Dimension Analysis Framework

1.  **What's Good**: Passing Biome checks, Zero `any` types, high Vitest coverage, clean separation of concerns (UI vs Background), use of `zod` for validation.
2.  **What's Bad**: TypeScript errors, Linter violations, Failed tests, High severity vulnerabilities.
3.  **What's Missing**: Missing TSDoc comments on exported functions, Missing `manifest.json` descriptions, Missing unit tests for Reducers/Utils, Missing Error Boundaries in React.
4.  **What's Unnecessary**: Unused exports (Knip results), `console.log` statements, commented-out code, redundant dependencies.
5.  **What's Fixed (vs main)**: Issues resolved in this branch.
6.  **What's Newly Broken**: New TS errors or Lint violations.
7.  **Silent Errors**: Empty `catch (e) {}` blocks, Unawaited Promises (floating promises), ignoring `chrome.runtime.lastError`.
8.  **Overengineered**: React components > 250 lines, "God" `background.ts` file (should be split into modules), excessive abstraction layers for simple Chrome APIs.

## Step 5: Generate and Save Report

Create the report in `docs/reports/qa-{TIMESTAMP}.md`:

```markdown
# QA Report: Chrome Extension

**Project:** {PROJECT_NAME}
**Date:** {TIMESTAMP}
**Branch:** {BRANCH}
**Target:** Manifest V3 / Chrome
**Compared Against:** main

---

## Executive Summary

[2-3 sentence assessment. Is this ready for the Chrome Web Store?]

**Overall Health Score:** [A/B/C/D/F]

| Dimension | Status | Issues Found |
|-----------|--------|--------------|
| Good | [emoji] | [summary] |
| Bad | [emoji] | [count] |
| Missing | [emoji] | [count] |
| Unnecessary | [emoji] | [count] |
| Fixed | [emoji] | [count] |
| Newly Broken | [emoji] | [count] |
| Silent Errors | [emoji] | [count] |
| Overengineered | [emoji] | [count] |

---

## 1. What's Good
[Valid types, clean Biome run, good test coverage, no MV2 legacy code]

## 2. What's Bad
### Critical (Store Rejection Risks)
[CSP violations, eval(), remote code loading, manifest errors]

### High (Runtime Crashes)
[TS Errors, Service Worker keep-alive issues]

### Medium
[Linter warnings, missing keys in React lists]

## 3. What's Missing
[Tests for message passing, error handling for chrome.storage]

## 4. What's Unnecessary
[Dead files found by Knip, console.logs]

## 5. What's Fixed
[...diff analysis...]

## 6. What's Newly Broken
[...diff analysis...]

## 7. Silent Errors
[Ignored promises, empty catch blocks, any types]

## 8. Overengineered
[Giant components, complex redux setup for simple state]

---

## Tool Output Summary

| Tool | Status | Issues/Notes |
|------|--------|--------------|
| Biome | [pass/fail] | [Lint/Format errors] |
| TypeScript | [pass/fail] | [Type errors] |
| Vitest | [pass/fail] | [Coverage %] |
| Knip | [pass/fail] | [Unused files/exports] |
| Audit | [pass/fail] | [Vulns] |
| CSP Check | [pass/fail] | [eval/innerHTML count] |
| MV3 Check | [pass/fail] | [Legacy API count] |

---
*Generated by /qa-extension on {TIMESTAMP}*
```

After generating, save to `docs/reports/qa-{TIMESTAMP}.md`.

$ARGUMENTS
