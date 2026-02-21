#!/bin/bash
set -euo pipefail

# 02-install-deps.sh — Run uv sync and verify key packages

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/setup.log"

mkdir -p "$PROJECT_ROOT/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [install-deps] $*" >> "$LOG_FILE"; }

cd "$PROJECT_ROOT"

log "Running uv sync"

if uv sync >> "$LOG_FILE" 2>&1; then
  log "uv sync succeeded"
else
  log "uv sync failed"
  cat <<EOF
=== NANOCLAW SETUP: INSTALL_DEPS ===
PACKAGES: failed
STATUS: failed
ERROR: uv_sync_failed
LOG: logs/setup.log
=== END ===
EOF
  exit 1
fi

# Verify key packages
MISSING=""
for pkg in neonize sqlalchemy pydantic nanoclaw; do
  uv run python -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
done

if [ -n "$MISSING" ]; then
  log "Missing packages after install:$MISSING"
  cat <<EOF
=== NANOCLAW SETUP: INSTALL_DEPS ===
PACKAGES: failed
STATUS: failed
ERROR: missing_packages:$MISSING
LOG: logs/setup.log
=== END ===
EOF
  exit 1
fi

log "All key packages verified"

cat <<EOF
=== NANOCLAW SETUP: INSTALL_DEPS ===
PACKAGES: installed
STATUS: success
LOG: logs/setup.log
=== END ===
EOF
