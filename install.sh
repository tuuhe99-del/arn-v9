#!/usr/bin/env bash
# Linux/macOS one-command installer.
set -euo pipefail
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$(dirname "$0")/install.py" "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$(dirname "$0")/install.py" "$@"
else
  echo "Python 3.10+ is required. Install Python, then rerun: bash install.sh" >&2
  exit 1
fi
