#!/usr/bin/env bash
#
# ARN v9 Installer
# =================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/tuuhe99-del/arn-v9/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --tier base --client codex
#
# Options:
#   --tier    nano|small|base|base-e5  (default: nano)
#   --client  codex|claude|kimi|openclaw (default: none)
#   --dir     install directory (default: ~/arn-v9)
#

set -e

# ─── Parse arguments ───
TIER="nano"
CLIENT=""
INSTALL_DIR="$HOME/arn-v9"
BRANCH="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)    TIER="$2";       shift 2 ;;
        --client)  CLIENT="$2";     shift 2 ;;
        --dir)     INSTALL_DIR="$2"; shift 2 ;;
        --branch)  BRANCH="$2";     shift 2 ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "ARN v9 Installer"
echo "  Tier:    $TIER"
echo "  Client:  ${CLIENT:-none}"
echo "  Install: $INSTALL_DIR"
echo ""

# ─── Check Python ───
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Found: $PY_VERSION"
    exit 1
fi
echo "Python: $PY_VERSION"

# ─── Check pip ───
if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null 2>&1; then
    echo "ERROR: pip not found. Install pip first."
    exit 1
fi

# ─── Install/update dependencies ───
echo ""
echo "Installing dependencies..."

PIP_FLAGS=""
# Check if we need --break-system-packages (PEP 668)
if python3 -m pip install --help 2>&1 | grep -q "break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
fi

python3 -m pip install --user $PIP_FLAGS numpy sentence-transformers 2>&1 | tail -1
python3 -m pip install --user $PIP_FLAGS rank_bm25 2>&1 | tail -1 || true

# ─── Download ARN ───
echo ""
echo "Downloading ARN v9..."

if command -v git &>/dev/null; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "  Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull origin "$BRANCH" --quiet
    else
        git clone --depth 1 --branch "$BRANCH" \
            "https://github.com/tuuhe99-del/arn-v9.git" "$INSTALL_DIR" 2>&1 | tail -1
    fi
else
    # No git — download tarball
    mkdir -p "$INSTALL_DIR"
    curl -fsSL "https://github.com/tuuhe99-del/arn-v9/archive/refs/heads/$BRANCH.tar.gz" \
        | tar xz --strip-components=1 -C "$INSTALL_DIR"
fi

echo "  Installed to: $INSTALL_DIR"

# ─── Add to PATH and PYTHONPATH ───
echo ""
echo "Configuring environment..."

BASHRC="$HOME/.bashrc"
ZSHRC="$HOME/.zshrc"

add_to_rc() {
    local RC_FILE="$1"
    local LINE="$2"
    local MARKER="$3"
    
    if [ -f "$RC_FILE" ]; then
        # Remove old ARN lines with this marker
        sed -i "/# ARN:$MARKER/d" "$RC_FILE" 2>/dev/null || true
        echo "$LINE  # ARN:$MARKER" >> "$RC_FILE"
    fi
}

# PYTHONPATH so the arn package is importable
add_to_rc "$BASHRC" "export PYTHONPATH=\"$INSTALL_DIR:\$PYTHONPATH\"" "pythonpath"
add_to_rc "$BASHRC" "export ARN_DATA_DIR=\"$HOME/.arn_data\"" "datadir"
add_to_rc "$BASHRC" "export ARN_EMBEDDING_TIER=\"$TIER\"" "tier"
add_to_rc "$BASHRC" "export ARN_AGENT_ID=\"default\"" "agentid"
add_to_rc "$BASHRC" "alias arn='python3 $INSTALL_DIR/arn/phase2/arn_cli.py'" "alias"

# Also do zsh if it exists
if [ -f "$ZSHRC" ]; then
    add_to_rc "$ZSHRC" "export PYTHONPATH=\"$INSTALL_DIR:\$PYTHONPATH\"" "pythonpath"
    add_to_rc "$ZSHRC" "export ARN_DATA_DIR=\"$HOME/.arn_data\"" "datadir"
    add_to_rc "$ZSHRC" "export ARN_EMBEDDING_TIER=\"$TIER\"" "tier"
    add_to_rc "$ZSHRC" "export ARN_AGENT_ID=\"default\"" "agentid"
    add_to_rc "$ZSHRC" "alias arn='python3 $INSTALL_DIR/arn/phase2/arn_cli.py'" "alias"
fi

# Set for current session
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
export ARN_DATA_DIR="$HOME/.arn_data"
export ARN_EMBEDDING_TIER="$TIER"
export ARN_AGENT_ID="default"

echo "  ARN_DATA_DIR=$HOME/.arn_data"
echo "  ARN_EMBEDDING_TIER=$TIER"

# ─── Run setup ───
echo ""
SETUP_ARGS="--tier $TIER"
if [ -n "$CLIENT" ]; then
    SETUP_ARGS="$SETUP_ARGS --client $CLIENT"
fi

python3 "$INSTALL_DIR/arn/phase2/arn_cli.py" setup $SETUP_ARGS

# ─── Final message ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  ARN memory is installed."
echo "  Model:  $TIER"
echo "  Data:   ~/.arn_data/default"
if [ -n "$CLIENT" ]; then
echo "  Client: $CLIENT"
fi
echo ""
echo "  Restart your terminal, then try:"
echo "    arn store -c \"My name is Mohamed\" -i 0.9"
echo "    arn recall -q \"what is my name\" -k 1"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
