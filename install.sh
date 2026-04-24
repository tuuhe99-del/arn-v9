#!/usr/bin/env bash
# ARN — One-line installer
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/tuuhe99-del/arn-v9/main/install.sh | bash
#   bash install.sh [options]
#
# Options:
#   --dir <path>       Install directory (default: ~/arn)
#   --model <tier>     nano | small | base | large | xl  (auto-detected from RAM if omitted)
#   --no-daemon        Skip daemon setup
#   --no-connect       Skip arn connect wizard
#   --skip-tests       Skip post-install tests
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'
ok()  { echo -e "${GRN}✓${NC} $*"; }
err() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
inf() { echo -e "${CYN}→${NC} $*"; }
wrn() { echo -e "${YLW}!${NC} $*"; }
hdr() { echo -e "\n${BLD}$*${NC}"; }

INSTALL_DIR="${HOME}/arn"
DATA_DIR="${HOME}/.arn_data/default"
MODEL_TIER=""          # empty = auto-detect
DAEMON=true
RUN_CONNECT=true
SKIP_TESTS=false
REPO="https://github.com/tuuhe99-del/arn-v9.git"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dir)        INSTALL_DIR="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2";    shift 2 ;;
    --model)      MODEL_TIER="$2";  shift 2 ;;
    --no-daemon)  DAEMON=false;     shift ;;
    --no-connect) RUN_CONNECT=false; shift ;;
    --skip-tests) SKIP_TESTS=true;  shift ;;
    --repo)       REPO="$2";        shift 2 ;;
    -h|--help)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) err "Unknown option: $1" ;;
  esac
done

echo ""
echo -e "${BLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLD}║  ARN — Adaptive Reasoning Network        ║${NC}"
echo -e "${BLD}║  Brain-inspired memory for AI agents     ║${NC}"
echo -e "${BLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 1. prerequisites ───────────────────────────────────────────────────────────
hdr "1. Checking prerequisites"

command -v python3 &>/dev/null || err "python3 not found. Install: sudo apt install python3"
PYOK=$(python3 -c 'import sys; print(int(sys.version_info>=(3,10)))')
[[ "$PYOK" == "1" ]] || err "Python 3.10+ required (found $(python3 --version))"
ok "Python $(python3 --version | cut -d' ' -f2)"

command -v git &>/dev/null || err "git not found. Install: sudo apt install git"
ok "git"

python3 -m pip --version &>/dev/null 2>&1 || python3 -m ensurepip --upgrade 2>/dev/null || err "pip unavailable"
ok "pip"

# ── 2. auto-detect model tier ─────────────────────────────────────────────────
if [[ -z "$MODEL_TIER" ]]; then
  RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 4000000)
  RAM_GB=$(python3 -c "print(round($RAM_KB/1024/1024,1))")
  if (( RAM_KB < 2000000 )); then
    MODEL_TIER="nano"
  elif (( RAM_KB < 6000000 )); then
    MODEL_TIER="small"
  else
    MODEL_TIER="base"
  fi
  inf "RAM detected: ${RAM_GB}GB → model tier: ${MODEL_TIER}"
else
  inf "Model tier: ${MODEL_TIER} (specified)"
fi

case "$MODEL_TIER" in
  nano)  MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2";  MODEL_SIZE="22MB"  ;;
  small) MODEL_NAME="sentence-transformers/all-mpnet-base-v2"; MODEL_SIZE="420MB" ;;
  base)  MODEL_NAME="BAAI/bge-base-en-v1.5";                   MODEL_SIZE="440MB" ;;
  *)     wrn "Unknown tier '$MODEL_TIER', falling back to nano"; MODEL_TIER="nano"
         MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2";  MODEL_SIZE="22MB"  ;;
esac
ok "Embedding model: ${MODEL_NAME} (${MODEL_SIZE})"

# ── 3. clone / update ─────────────────────────────────────────────────────────
hdr "2. Installing ARN"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
  inf "Updating existing install..."
  git -C "${INSTALL_DIR}" pull --ff-only 2>/dev/null || wrn "Could not pull. Continuing with existing version."
elif [[ -f "${INSTALL_DIR}/pyproject.toml" ]]; then
  ok "Already installed (non-git). Skipping clone."
else
  git clone "${REPO}" "${INSTALL_DIR}"
fi
ok "Source at ${INSTALL_DIR}"

# ── 4. pip install ────────────────────────────────────────────────────────────
hdr "3. Installing dependencies"
pip3 install --quiet --upgrade pip 2>/dev/null || true
pip3 install --quiet sentence-transformers numpy
ok "Core dependencies"
pip3 install --quiet fastapi uvicorn pydantic 2>/dev/null && ok "API extras" || wrn "API extras skipped"

inf "Installing ARN package..."
pip3 install --quiet -e "${INSTALL_DIR}" 2>/dev/null || pip3 install --quiet "${INSTALL_DIR}"
ok "ARN package installed"

# ── 5. data directory + env ───────────────────────────────────────────────────
hdr "4. Configuring environment"
mkdir -p "${DATA_DIR}"
ok "Data dir: ${DATA_DIR}"

SHELL_RC="${HOME}/.bashrc"
[[ "${SHELL:-}" == */zsh ]] && SHELL_RC="${HOME}/.zshrc"

for line in \
  "export ARN_DATA_DIR=\"${DATA_DIR}\"" \
  "export ARN_EMBEDDING_MODEL=\"${MODEL_NAME}\""; do
  grep -qF "${line}" "${SHELL_RC}" 2>/dev/null || echo "${line}" >> "${SHELL_RC}"
done
ok "Shell env set in ${SHELL_RC}"

export ARN_DATA_DIR="${DATA_DIR}"
export ARN_EMBEDDING_MODEL="${MODEL_NAME}"

# ── 6. daemon ─────────────────────────────────────────────────────────────────
if [[ "${DAEMON}" == "true" ]]; then
  hdr "5. Daemon setup"
  DAEMON_PY="${INSTALL_DIR}/arn_daemon.py"
  if [[ -f "${DAEMON_PY}" ]]; then
    if command -v systemctl &>/dev/null && systemctl --user status &>/dev/null 2>&1; then
      SERVICE_DIR="${HOME}/.config/systemd/user"
      mkdir -p "${SERVICE_DIR}"
      cat > "${SERVICE_DIR}/arn-daemon.service" <<EOF
[Unit]
Description=ARN Memory Daemon
After=default.target

[Service]
Type=simple
ExecStart=python3 ${DAEMON_PY} start
Restart=on-failure
RestartSec=5
Environment=ARN_DATA_DIR=${DATA_DIR}
Environment=ARN_EMBEDDING_MODEL=${MODEL_NAME}

[Install]
WantedBy=default.target
EOF
      systemctl --user daemon-reload
      systemctl --user enable arn-daemon.service
      systemctl --user start  arn-daemon.service 2>/dev/null || true
      sleep 2
      python3 "${DAEMON_PY}" ping &>/dev/null && ok "Daemon running (systemd)" || wrn "Daemon starting (model loading, ~30s first run)"
    else
      python3 "${DAEMON_PY}" ping &>/dev/null 2>&1 || {
        nohup python3 "${DAEMON_PY}" start >"${HOME}/.arn_daemon.log" 2>&1 &
        wrn "Daemon started in background. Auto-start: add to .bashrc if needed."
      }
    fi
  else
    wrn "arn_daemon.py not found — daemon skipped (recall will be slower on first call)"
  fi
fi

# ── 7. tests ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_TESTS}" == "false" ]]; then
  hdr "6. Verifying install"
  python3 -m pytest "${INSTALL_DIR}/tests/" -q --tb=short 2>/dev/null && ok "Tests passed" || wrn "Some tests failed — ARN may still work"
fi

# ── 8. framework detection + connect ──────────────────────────────────────────
if [[ "${RUN_CONNECT}" == "true" ]]; then
  hdr "7. Framework setup"
  if [[ -d "${HOME}/.openclaw" ]]; then
    inf "OpenClaw detected — installing memory skill to all agents..."
    python3 -c "
from arn.adapters.openclaw import install_skill
from pathlib import Path
n = install_skill()
print(f'  Installed to {n} agents' if n else '  No agents found — install SKILL.md manually')
" 2>/dev/null || wrn "OpenClaw skill install skipped"
  fi

  python3 -c "import langchain" 2>/dev/null && {
    inf "LangChain detected — writing tool snippet..."
    python3 -c "
from arn.adapters.langchain import get_tools
# Just verify it can be called
print('  LangChain adapter ready. Use: from arn.adapters.langchain import get_tools')
" 2>/dev/null
  } || true
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLD}║           ARN ready                      ║${NC}"
echo -e "${BLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Restart your shell, then:"
echo ""
echo -e "  ${CYN}arn store \"User prefers Python\" --importance 0.8${NC}"
echo -e "  ${CYN}arn recall \"what language does user prefer?\"${NC}"
echo -e "  ${CYN}arn context \"project decisions\"${NC}"
echo -e "  ${CYN}arn stats${NC}"
echo ""
echo "  Model tier : ${MODEL_TIER} (${MODEL_NAME})"
echo "  Data dir   : ${DATA_DIR}"
echo "  Docs       : ${INSTALL_DIR}/README.md"
echo ""
echo "To switch model tier later:"
echo "  export ARN_EMBEDDING_MODEL=\"BAAI/bge-base-en-v1.5\"  # upgrade to base"
echo ""
ok "Done."
