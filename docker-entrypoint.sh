#!/usr/bin/env bash
set -euo pipefail

# Raise file descriptor limits if possible
if command -v prlimit >/dev/null 2>&1; then
  prlimit --pid $$ --nofile=65536:65536 || true
fi

# If no arguments provided, run bash
if [[ $# -eq 0 ]]; then
  exec micromamba run -n germinal --no-capture-output bash
fi

# Filter out any empty arguments and exec command inside the micromamba environment
args=()
for arg in "$@"; do
  if [[ -n "$arg" ]]; then
    args+=("$arg")
  fi
done

if [[ ${#args[@]} -eq 0 ]]; then
  exec micromamba run -n germinal --no-capture-output bash
else
  exec micromamba run -n germinal --no-capture-output "${args[@]}"
fi


