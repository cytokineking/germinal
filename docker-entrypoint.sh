#!/usr/bin/env bash
set -euo pipefail

# Raise file descriptor limits if possible
if command -v prlimit >/dev/null 2>&1; then
  prlimit --pid $$ --nofile=65536:65536 || true
fi

# Exec command inside the micromamba environment
exec micromamba run -n germinal --no-capture-output "$@"


