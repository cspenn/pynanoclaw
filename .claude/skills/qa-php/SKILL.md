---
name: qa-php
description: Run comprehensive PHP 8.4 code quality audit. Executes tiered quality checks (PHPStan, Pest, Laravel Pint, Rector) and scans for PHP-specific anti-patterns (type juggling, SQL injection, XSS vulnerabilities, unserialize). Generates an 8-dimension health report with WordPress/WP Engine compatibility checks.
---

# PHP 8.4 Codebase Quality Assessment (8 Dimensions)

You are performing a comprehensive quality assessment of this PHP 8.4 codebase. Generate a detailed report evaluating 8 key dimensions, specifically focusing on the unique constraints of PHP applications (strict typing, security vulnerabilities, WordPress integration, and WP Engine hosting limits).

## Setup

Capture timestamp and branch info:

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
PHP_VERSION=$(php -v | head -1 | awk '{print $2}')
echo "Timestamp: $TIMESTAMP"
echo "Branch: $BRANCH"
echo "PHP Version: $PHP_VERSION"
```

## Step 1: Gather Context

### Project Structure
```bash
# Locate key entry points and structure
find . -name "composer.json" -o -name "phpstan.neon" -o -name "rector.php" -o -name "phpunit.xml" -o -name "pest.php" 2>/dev/null | grep -v vendor | head -20

# Show src/ structure
find src/ -name "*.php" 2>/dev/null | head -30 || echo "No src/ directory found"
```

### WordPress Detection
```bash
# Check if this is a WordPress plugin/theme
if [ -f "*.php" ] && grep -l "Plugin Name:" *.php 2>/dev/null; then
    echo "WordPress Plugin detected"
elif [ -f "style.css" ] && grep -l "Theme Name:" style.css 2>/dev/null; then
    echo "WordPress Theme detected"
else
    echo "Standalone PHP application"
fi
```

### Files Changed from Main
```bash
git diff main...HEAD --name-only -- '*.php' '*.neon' '*.xml' 2>/dev/null || echo "Unable to diff against main"
```

## Step 2: Run Quality Tools

Run each tool using `composer`. Tools that fail or aren't installed should be skipped gracefully.

### PHP Syntax Check (Fast Fail)
```bash
find src/ -name "*.php" -exec php -l {} \; 2>&1 | grep -v "No syntax errors" || echo "All files pass syntax check"
```

### Laravel Pint (Code Style - PSR-12)
*Replaces Black/Ruff format.*
```bash
vendor/bin/pint --test 2>&1 || echo "Pint found formatting issues (run 'vendor/bin/pint' to fix)"
```

### PHPStan (Static Analysis)
*Replaces MyPy. Level 6+ required.*
```bash
vendor/bin/phpstan analyse --error-format=table 2>&1 || echo "PHPStan found issues"
```

### Rector (Automated Refactoring - Dry Run)
*Identifies code that can be modernized to PHP 8.4.*
```bash
vendor/bin/rector process --dry-run 2>&1 || echo "Rector not configured or found issues"
```

### Pest/PHPUnit (Unit Testing)
*Replaces Pytest.*
```bash
if [ -f "pest.php" ] || [ -f "vendor/bin/pest" ]; then
    vendor/bin/pest --coverage 2>&1 || echo "Pest tests failed"
else
    vendor/bin/phpunit --coverage-text 2>&1 || echo "PHPUnit tests failed"
fi
```

### Composer Audit (Security Vulnerabilities)
*Replaces Pip-audit/Bandit.*
```bash
composer audit 2>&1 || echo "Security vulnerabilities found"
```

### PHPMD (Mess Detection - Optional)
```bash
vendor/bin/phpmd src/ text cleancode,codesize,controversial,design,naming,unusedcode 2>&1 || echo "PHPMD not installed or found issues"
```

## Step 3: PHP Anti-Pattern Detection (Grepping)

### Missing Strict Types Declaration (CRITICAL)
*Every PHP file MUST start with `declare(strict_types=1);`*
```bash
echo "=== Files missing strict_types declaration ==="
find src/ -name "*.php" -exec grep -L "declare(strict_types=1)" {} \; 2>/dev/null || echo "All files have strict_types"
```

### Type Juggling Vulnerabilities (CRITICAL)
*Loose equality operators cause security bugs.*
```bash
echo "=== Loose equality (use === instead) ==="
grep -rn " == " src/ --include="*.php" 2>&1 | grep -v "===" || echo "None found"
grep -rn " != " src/ --include="*.php" 2>&1 | grep -v "!==" || echo "None found"
```

### Object Injection Risk (CRITICAL)
*Never use unserialize() on user input.*
```bash
echo "=== unserialize() usage (potential object injection) ==="
grep -rn "unserialize(" src/ --include="*.php" 2>&1 || echo "None found"
```

### SQL Injection Vectors
*Direct variable interpolation in queries.*
```bash
echo "=== Potential SQL injection (raw queries without prepare) ==="
grep -rn '\$wpdb->query.*\$' src/ --include="*.php" 2>&1 | grep -v "prepare" || echo "None found"
grep -rn "mysql_query\|mysqli_query" src/ --include="*.php" 2>&1 || echo "None found (legacy mysql functions)"
```

### XSS Vulnerabilities
*Output without escaping.*
```bash
echo "=== Unescaped output (potential XSS) ==="
grep -rn "echo \$" src/ --include="*.php" 2>&1 | grep -v "esc_" || echo "None found"
grep -rn "print \$" src/ --include="*.php" 2>&1 | grep -v "esc_" || echo "None found"
```

### Forbidden WordPress Functions
```bash
echo "=== Forbidden WordPress patterns ==="
grep -rn "query_posts(" src/ --include="*.php" 2>&1 || echo "None found (good - use WP_Query)"
grep -rn "extract(" src/ --include="*.php" 2>&1 || echo "None found (good - extract is dangerous)"
```

### Variable Variables (Impossible to Analyze)
```bash
echo "=== Variable variables (\$\$var) ==="
grep -rn '\$\$' src/ --include="*.php" 2>&1 || echo "None found"
```

### Error Suppression Operator
*The @ operator hides errors silently.*
```bash
echo "=== Error suppression (@) usage ==="
grep -rn '@\$\|@fopen\|@file\|@include\|@require' src/ --include="*.php" 2>&1 || echo "None found"
```

### Debug/Dev Code in Production
```bash
echo "=== Debug code that should be removed ==="
grep -rn "var_dump\|print_r\|dd(\|dump(" src/ --include="*.php" 2>&1 || echo "None found"
grep -rn "error_log(" src/ --include="*.php" 2>&1 || echo "None found (review if intentional)"
```

### Mixed Type Usage (Weak Typing)
```bash
echo "=== 'mixed' type usage (should be explicit) ==="
grep -rn ": mixed" src/ --include="*.php" 2>&1 || echo "None found"
```

### Legacy Array Syntax
```bash
echo "=== Legacy array() syntax (use [] instead) ==="
grep -rn "array(" src/ --include="*.php" 2>&1 | head -10 || echo "None found"
```

## Step 4: WordPress/WP Engine Specific Checks

*Run only if WordPress plugin/theme detected.*

### WP Engine Query Length Risk
```bash
echo "=== Complex meta_query (WP Engine 1024 char limit risk) ==="
grep -rn "meta_query" src/ --include="*.php" 2>&1 || echo "None found"
```

### Missing Nonce Verification
```bash
echo "=== Form handlers without nonce verification ==="
grep -rn "wp_ajax_\|admin_post_" src/ --include="*.php" -A 5 2>&1 | grep -v "wp_verify_nonce\|check_ajax_referer" | head -20 || echo "Review AJAX handlers manually"
```

### Direct Database Access
```bash
echo "=== Direct \$wpdb usage (should use prepare()) ==="
grep -rn '\$wpdb->' src/ --include="*.php" 2>&1 | grep -v "prepare\|prefix\|posts\|postmeta" | head -10 || echo "None found"
```

### Session Usage (Breaks WP Engine Cache)
```bash
echo "=== \$_SESSION usage (breaks page cache) ==="
grep -rn '\$_SESSION' src/ --include="*.php" 2>&1 || echo "None found"
```

## Step 5: File Size and Complexity Check

### Large Files (Potential God Classes)
```bash
echo "=== Files over 250 lines (refactoring candidates) ==="
find src/ -name "*.php" -exec wc -l {} \; 2>/dev/null | awk '$1 > 250 {print $1, $2}' | sort -rn | head -10 || echo "All files under 250 lines"
```

### Classes Without Type Hints
```bash
echo "=== Functions missing return types ==="
grep -rn "function [a-zA-Z_]*(" src/ --include="*.php" 2>&1 | grep -v "): " | head -10 || echo "All functions have return types"
```

## Step 6: Analysis Instructions

Create a comprehensive quality assessment report following the 8-dimension framework. Adapt the analysis specifically for PHP:

### Dimension Analysis Framework

1.  **What's Good**: Passing PHPStan at Level 6+, Zero loose equality, `strict_types` everywhere, high Pest coverage, PSR-4 autoloading, proper escaping, prepared statements.
2.  **What's Bad**: PHPStan errors, Pint violations, Failed tests, Security vulnerabilities from `composer audit`, Missing strict_types.
3.  **What's Missing**: Missing PHPDoc on public methods, Missing return types, Missing validation on user input, Missing nonce checks on forms, Missing error handling.
4.  **What's Unnecessary**: Dead code (PHPMD), `var_dump`/`print_r` statements, commented-out code, legacy `array()` syntax, unused Composer dependencies.
5.  **What's Fixed (vs main)**: Issues resolved in this branch compared to main.
6.  **What's Newly Broken**: New PHPStan errors, New security patterns, New Pint violations.
7.  **Silent Errors**: Empty `catch` blocks, `@` error suppression, `mixed` types hiding real types, unchecked `$_GET`/`$_POST` access.
8.  **Overengineered**: God classes > 250 lines, excessive abstraction for simple CRUD, misuse of magic methods when Property Hooks would suffice.

## Step 7: Generate and Save Report

Create the report in `docs/reports/qa-php-{TIMESTAMP}.md`:

```markdown
# QA Report: PHP 8.4 Application

**Project:** {PROJECT_NAME}
**Date:** {TIMESTAMP}
**Branch:** {BRANCH}
**PHP Version:** {PHP_VERSION}
**Target:** PHP 8.4 / WP Engine
**Compared Against:** main

---

## Executive Summary

[2-3 sentence assessment. Is this production-ready? Are there security concerns?]

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
[Strict types enforced, PHPStan clean, good test coverage, proper escaping, prepared statements]

## 2. What's Bad

### Critical (Security Vulnerabilities)
[SQL injection vectors, XSS vulnerabilities, unserialize on user input, missing nonces]

### High (Runtime Errors)
[PHPStan errors, missing strict_types, type juggling with ==]

### Medium
[Pint violations, missing return types, legacy syntax]

## 3. What's Missing
[Test coverage gaps, PHPDoc on public API, input validation, error boundaries]

## 4. What's Unnecessary
[var_dump statements, commented code, unused dependencies, legacy array() syntax]

## 5. What's Fixed
[...diff analysis...]

## 6. What's Newly Broken
[...diff analysis...]

## 7. Silent Errors
[Empty catch blocks, @ suppression, unchecked $_POST access, ignored return values]

## 8. Overengineered
[God classes, excessive abstraction, magic methods where Property Hooks fit]

---

## Security Checklist

| Check | Status | Notes |
|-------|--------|-------|
| `strict_types` everywhere | [✅/❌] | |
| No loose equality (`==`) | [✅/❌] | |
| No `unserialize()` on user data | [✅/❌] | |
| All queries use `prepare()` | [✅/❌] | |
| All output escaped | [✅/❌] | |
| Nonces on all forms | [✅/❌] | |
| No `extract()` usage | [✅/❌] | |
| No `$_SESSION` (WP Engine) | [✅/❌] | |

---

## Tool Output Summary

| Tool | Status | Issues/Notes |
|------|--------|--------------|
| PHP Syntax | [pass/fail] | [Parse errors] |
| Laravel Pint | [pass/fail] | [Style violations] |
| PHPStan | [pass/fail] | [Level X, Y errors] |
| Rector | [pass/fail] | [Modernization opportunities] |
| Pest/PHPUnit | [pass/fail] | [Coverage %] |
| Composer Audit | [pass/fail] | [Vulnerabilities] |
| Strict Types Check | [pass/fail] | [Files missing declaration] |
| Security Grep | [pass/fail] | [Patterns found] |

---

## WP Engine Compatibility

| Check | Status | Notes |
|-------|--------|-------|
| No `$_SESSION` usage | [✅/❌] | Breaks page cache |
| No complex `meta_query` | [✅/❌] | 1024 char query limit |
| No banned functions | [✅/❌] | `exec`, `system`, `phpinfo` |
| Object cache compatible | [✅/❌] | Handles cache misses |

---

## Recommended Actions

### Immediate (Before Deploy)
1. [Critical security fix]
2. [Missing strict_types]

### Short-term (This Sprint)
1. [PHPStan errors]
2. [Test coverage gaps]

### Long-term (Tech Debt)
1. [Refactor god classes]
2. [Modernize legacy syntax]

---
*Generated by /qa-php on {TIMESTAMP}*
```

After generating, save to `docs/reports/qa-php-{TIMESTAMP}.md`.

$ARGUMENTS
