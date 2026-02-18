#!/usr/bin/env bash
# start checkpython.sh
# NanoClaw Python Quality Gate Runner
# NEVER MODIFY THIS FILE — it is the canonical quality gate definition.
set -euo pipefail

SRC="src/"
TESTS="tests/"

echo "================================================="
echo "  NanoClaw Python Quality Gates"
echo "================================================="

echo ""
echo "--- Tier 1: Gate Checks (must pass before commit) ---"

echo "[1/5] ruff check..."
uv run ruff check "$SRC"

echo "[2/5] ruff format check..."
uv run ruff format --check "$SRC"

echo "[3/5] mypy..."
uv run mypy "$SRC" --strict

echo "[4/5] pytest..."
uv run pytest "$TESTS" -v

echo "[5/5] deptry..."
uv run deptry "$SRC"

echo ""
echo "--- Tier 2: Quality Analysis ---"

echo "[6/8] radon cyclomatic complexity..."
uv run radon cc "$SRC" -a -nb

echo "[7/8] bandit security scan..."
uv run bandit -r "$SRC" -ll

echo "[8/8] interrogate docstring coverage..."
uv run interrogate "$SRC" --fail-under 80

echo ""
echo "================================================="
echo "  All quality gates passed!"
echo "================================================="
# end checkpython.sh
